#!/usr/bin/env python3
"""EcoSort AI - end-to-end evaluation on a labelled photo set.

Runs the REAL pipeline (detect -> verify -> advise) on every labelled photo and reports:
  * item detection precision / recall / F1
  * category accuracy of the detector, the second opinion and the final advice (+ 95% CI)
  * per-class precision / recall / F1, macro-F1 and a confusion matrix
  * effect of self-verification and clarifying questions
  * confidence calibration (ECE)
  * safety-alert recall for hazardous items
  * collection-point audit score error (if `batch` is given)
  * advice format validity, answer-language check, latency

Usage (from the project root):
    python evaluation/run_eval.py --dry-run                # test the harness, no API key needed
    python evaluation/run_eval.py --language English       # real run (GEMINI_API_KEY must be set)
    python evaluation/run_eval.py --language Hindi --out evaluation/report_hindi
"""
import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import metrics as M  # noqa: E402
import pipeline as P  # noqa: E402


# ------------------------------------------------------------------ labels
def read_labels(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    rows = []
    for r in csv.DictReader(lines):
        if not (r.get("image") or "").strip():
            continue
        names = [x.strip() for x in r["items"].split("|") if x.strip()]
        cats = [x.strip() for x in r["categories"].split("|") if x.strip()]
        if len(names) != len(cats):
            raise SystemExit(f"{r['image']}: 'items' and 'categories' must have the same number of entries")
        bad = [c for c in cats if c not in P.CATEGORIES]
        if bad:
            raise SystemExit(f"{r['image']}: unknown categories {bad}; use one of {P.CATEGORIES}")
        hz_raw = (r.get("hazards") or "").strip()
        if hz_raw:
            hazards = [x.strip().lower() in ("1", "true", "yes") for x in hz_raw.split("|")]
            if len(hazards) != len(names):
                raise SystemExit(f"{r['image']}: 'hazards' must have one entry per item")
        else:
            hazards = [c in ("domestic_hazardous", "e_waste") for c in cats]
        rows.append({"image": r["image"].strip(), "names": names, "cats": cats, "hazards": hazards,
                     "batch": (r.get("batch") or "").strip() or None,
                     "answer": (r.get("answer") or "Not sure").strip()})
    if not rows:
        raise SystemExit("No rows found in the labels file.")
    return rows


# ------------------------------------------------------------------ pipeline run
def run_pipeline(client, kb, img, row, model, language):
    """Same sequence as app.py. Returns items, results, timings, model actually used."""
    t0 = time.time()
    items, note, used = P.detect(client, img, model)
    t1 = time.time()
    items = P.verify(client, img, items, model) if items else items
    t2 = time.time()
    answers = {it["name"]: row["answer"] for it in items if it.get("needs_clarification")}
    results, _ = P.advise(client, kb, items, answers, language, model) if items else ([], [])
    t3 = time.time()
    return {"items": items, "results": results, "used_model": used,
            "t": {"detect": t1 - t0, "verify": t2 - t1, "advise": t3 - t2, "total": t3 - t0}}


def apply_hazard_guard(items, results):
    """Copy of the guard in app.py: never lose a hazard flag from detection."""
    hz = {it["name"] for it in items if it.get("hazard")}
    out = []
    for r in results:
        r = dict(r)
        r["safety_alert_raw"] = r.get("safety_alert")
        if r.get("name") in hz and not r.get("safety_alert"):
            r["safety_alert"] = "Hazardous item: pack it safely, label it and keep it separate from regular waste."
        out.append(r)
    return out


# ------------------------------------------------------------------ evaluation
def evaluate(records, args):
    thr = args.threshold
    rows_out = []          # one dict per ground-truth item
    TP = FP = FN = 0
    imgs_all_found = 0
    named_images = 0
    fmt_ok = fmt_total = 0
    lang_ok = lang_total = 0
    audit_err = []
    fmt_problems = []

    for rec in records:
        row, items, results = rec["row"], rec["items"], rec["final_results"]
        if row["names"] == ["*"]:
            # single-object photo from a public dataset: judge the category of the most confident detection;
            # detection precision/recall are not meaningful here, so they are not counted
            pairs = [(0, max(range(len(items)), key=lambda i: items[i]["confidence"]), 1.0)] if items else []
        else:
            pairs = M.match_items(row["names"], [i["name"] for i in items], thr)
            TP += len(pairs)
            FP += len(items) - len(pairs)
            FN += len(row["names"]) - len(pairs)
            imgs_all_found += int(len(pairs) == len(row["names"]))
            named_images += 1
        by_gt = {gi: pi for gi, pi, _ in pairs}

        for gi, name in enumerate(row["names"]):
            d = {"image": row["image"], "gt_item": name, "gt_cat": row["cats"][gi], "gt_hazard": row["hazards"][gi],
                 "pred_item": None, "det_cat": "missed", "second_cat": None, "final_cat": "missed",
                 "confidence": None, "agree": None, "flagged": None, "alert": False, "alert_raw": False}
            if gi in by_gt:
                it = items[by_gt[gi]]
                d.update(pred_item=it["name"], det_cat=it["category"], second_cat=it.get("second_opinion"),
                         confidence=it["confidence"], agree=it.get("agree"), flagged=bool(it.get("needs_clarification")))
                best = max(results, key=lambda r: M.name_similarity(it["name"], r.get("name", "")), default=None)
                if best is not None and M.name_similarity(it["name"], best.get("name", "")) >= thr:
                    d["final_cat"] = best.get("category", "missed")
                    d["alert"] = bool(best.get("safety_alert"))
                    d["alert_raw"] = bool(best.get("safety_alert_raw"))
            rows_out.append(d)

        # advice format + language
        for r in results:
            fmt_total += 1
            problems = []
            if r.get("category") not in P.CATEGORIES:
                problems.append("bad category")
            if not str(r.get("reason") or "").strip():
                problems.append("no reason")
            steps = r.get("steps")
            if not isinstance(steps, list) or not 2 <= len(steps) <= 3:
                problems.append("steps not 2-3")
            if not str(r.get("spoken") or "").strip():
                problems.append("no spoken text")
            if problems:
                fmt_problems.append(f"{row['image']} / {r.get('name')}: {', '.join(problems)}")
            fmt_ok += int(not problems)
            text = " ".join([str(r.get("reason") or ""), " ".join(map(str, steps or [])), str(r.get("spoken") or "")])
            lang_total += 1
            lang_ok += int(M.script_ratio(text, args.language) >= 0.5)

        if row["batch"] and results:
            gt_score = round(100 * sum(c == row["batch"] for c in row["cats"]) / len(row["cats"]))
            audit_err.append((rec["row"]["image"], gt_score, P.audit_score(results, row["batch"])[0]))

    matched = [d for d in rows_out if d["pred_item"] is not None]
    n_gt, n_m = len(rows_out), len(matched)

    def acc(rows, key, f=lambda c: c):
        return sum(f(d[key]) == f(d["gt_cat"]) for d in rows)

    det_prec = TP / (TP + FP) if TP + FP else 0.0
    det_rec = TP / (TP + FN) if TP + FN else 0.0
    det_f1 = 2 * det_prec * det_rec / (det_prec + det_rec) if det_prec + det_rec else 0.0

    final_rep = M.classification_report([d["gt_cat"] for d in rows_out], [d["final_cat"] for d in rows_out])
    det_rep = M.classification_report([d["gt_cat"] for d in rows_out], [d["det_cat"] for d in rows_out])

    # self-verification / clarification analysis (matched items only)
    agree = [d for d in matched if d["agree"]]
    disagree = [d for d in matched if d["agree"] is False]
    det_wrong = [d for d in matched if d["det_cat"] != d["gt_cat"]]
    det_right = [d for d in matched if d["det_cat"] == d["gt_cat"]]
    caught = [d for d in det_wrong if d["flagged"]]
    false_alarm = [d for d in det_right if d["flagged"]]
    fixed = [d for d in det_wrong if d["final_cat"] == d["gt_cat"]]
    broken = [d for d in det_right if d["final_cat"] != d["gt_cat"]]

    # calibration of detector confidence
    ece = M.expected_calibration_error([d["confidence"] for d in matched],
                                       [int(d["det_cat"] == d["gt_cat"]) for d in matched])

    # safety
    haz = [d for d in rows_out if d["gt_hazard"]]
    haz_found = [d for d in haz if d["pred_item"] is not None]
    nonhaz_matched = [d for d in matched if not d["gt_hazard"]]

    totals = [r["t"]["total"] for r in records]
    metrics = {
        "images": len(records), "named_images": named_images, "gt_items": n_gt, "matched_items": n_m,
        "detection": {"tp": TP, "fp": FP, "fn": FN, "precision": det_prec, "recall": det_rec, "f1": det_f1,
                      "images_with_all_items_found": imgs_all_found},
        "category_accuracy_on_matched": {
            "detector": acc(matched, "det_cat"),
            "second_opinion": acc([d for d in matched if d["second_cat"]], "second_cat"),
            "second_opinion_coverage": sum(bool(d["second_cat"]) for d in matched),
            "final": acc(matched, "final_cat"), "final_lenient": acc(matched, "final_cat", M.lenient)},
        "end_to_end_accuracy": {"final": acc(rows_out, "final_cat"), "final_lenient": acc(rows_out, "final_cat", M.lenient), "n": n_gt},
        "final_report": {k: v for k, v in final_rep.items()},
        "detector_report": {k: v for k, v in det_rep.items()},
        "verification": {"agree_n": len(agree), "disagree_n": len(disagree),
                         "acc_when_agree": acc(agree, "det_cat"), "acc_when_disagree": acc(disagree, "det_cat"),
                         "flag_rate": sum(bool(d["flagged"]) for d in matched), "detector_errors": len(det_wrong),
                         "errors_caught": len(caught), "false_alarms": len(false_alarm), "detector_right": len(det_right),
                         "fixed_by_pipeline": len(fixed), "broken_by_pipeline": len(broken)},
        "calibration_ece": ece,
        "safety": {"hazard_items": len(haz), "hazard_found": len(haz_found),
                   "alert_recall_with_guard": sum(d["alert"] for d in haz_found),
                   "alert_recall_model_only": sum(d["alert_raw"] for d in haz_found),
                   "alerts_on_non_hazard": sum(d["alert"] for d in nonhaz_matched), "non_hazard_matched": len(nonhaz_matched)},
        "audit": {"images": len(audit_err), "mae": (sum(abs(g - p) for _, g, p in audit_err) / len(audit_err)) if audit_err else None,
                  "details": audit_err},
        "advice_quality": {"format_ok": fmt_ok, "format_total": fmt_total, "language_ok": lang_ok, "language_total": lang_total,
                           "language": args.language, "format_problems": fmt_problems[:20]},
        "latency_seconds": {"mean": statistics.mean(totals) if totals else 0, "median": statistics.median(totals) if totals else 0,
                            "p95": M.percentile(totals, 95),
                            "detect_mean": statistics.mean(r["t"]["detect"] for r in records) if records else 0,
                            "verify_mean": statistics.mean(r["t"]["verify"] for r in records) if records else 0,
                            "advise_mean": statistics.mean(r["t"]["advise"] for r in records) if records else 0},
    }
    return metrics, rows_out


# ------------------------------------------------------------------ reporting
def write_confusion_png(rep, path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    labels, cols = list(rep["confusion"].keys()), rep["columns"]
    data = [[rep["confusion"][t][p] for p in cols] for t in labels]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(data, cmap="Greens")
    ax.set_xticks(range(len(cols)), cols, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(title)
    mx = max(max(r) for r in data) or 1
    for i, r in enumerate(data):
        for j, v in enumerate(r):
            ax.text(j, i, v, ha="center", va="center", color="white" if v > mx / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def write_summary(m, args, path, skipped, failures, dry):
    cat = m["category_accuracy_on_matched"]
    e2e = m["end_to_end_accuracy"]
    det, ver, saf = m["detection"], m["verification"], m["safety"]
    fr = m["final_report"]
    L = []
    if dry:
        L += ["> **DRY RUN with a mock model - these numbers are NOT real results.** They only prove the harness works.", ""]
    L += [f"# EcoSort AI - evaluation summary", "",
          f"Model: `{args.model}` | Answer language: {args.language} | Images: {m['images']} | Ground-truth items: {m['gt_items']}"
          f" | Skipped (missing file): {skipped} | Failed (API error): {failures}", "",
          "## 1. Headline results", "",
          "| Metric | Result |", "|---|---|",
          f"| **End-to-end category accuracy** (missed items count as wrong) | {M.fmt_rate(e2e['final'], e2e['n'])} |",
          f"| End-to-end accuracy, lenient (e-waste = hazardous) | {M.fmt_rate(e2e['final_lenient'], e2e['n'])} |",
          f"| Macro-F1 (final category) | {fr['macro_f1']:.3f} |",
          f"| Item detection recall | {M.fmt_rate(det['tp'], det['tp'] + det['fn'])} |",
          f"| Item detection precision | {M.fmt_rate(det['tp'], det['tp'] + det['fp'])} |",
          f"| Item detection F1 | {det['f1']:.3f} |",
          f"| Photos with every item found | {M.fmt_rate(det['images_with_all_items_found'], m['named_images'])} |",
          f"| Safety-alert recall for hazardous items (with app guard) | {M.fmt_rate(saf['alert_recall_with_guard'], saf['hazard_found'])} |",
          f"| Confidence calibration error (ECE, lower is better) | {m['calibration_ece']:.3f} |",
          f"| Latency per photo (mean / median / p95) | {m['latency_seconds']['mean']:.1f}s / {m['latency_seconds']['median']:.1f}s / {m['latency_seconds']['p95']:.1f}s |",
          "", "## 2. Where each stage stands (matched items only)", "",
          "| Stage | Category accuracy |", "|---|---|",
          f"| Detector (first VLM pass) | {M.fmt_rate(cat['detector'], m['matched_items'])} |",
          f"| Independent second opinion | {M.fmt_rate(cat['second_opinion'], cat['second_opinion_coverage'])} |",
          f"| Final advice (after clarification + RAG) | {M.fmt_rate(cat['final'], m['matched_items'])} |",
          "", "## 3. Per-class results (final category)", "",
          "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for c, v in fr["per_class"].items():
        L.append(f"| {c} | {v['precision']:.2f} | {v['recall']:.2f} | {v['f1']:.2f} | {v['support']} |")
    L += ["", "Confusion matrix (rows = truth, columns = predicted; `missed` = item not detected):", "",
          "| truth \\ pred | " + " | ".join(fr["columns"]) + " |", "|---|" + "---|" * len(fr["columns"])]
    for t, r in fr["confusion"].items():
        L.append(f"| {t} | " + " | ".join(str(r[p]) for p in fr["columns"]) + " |")
    L += ["", "![confusion matrix](confusion_matrix.png)", "",
          "## 4. Does self-verification help?", "",
          "| Question | Answer |", "|---|---|",
          f"| Detector accuracy when the two models AGREE | {M.fmt_rate(ver['acc_when_agree'], ver['agree_n'])} |",
          f"| Detector accuracy when they DISAGREE | {M.fmt_rate(ver['acc_when_disagree'], ver['disagree_n'])} |",
          f"| Detector mistakes flagged for a clarifying question (error-catch rate) | {M.fmt_rate(ver['errors_caught'], ver['detector_errors'])} |",
          f"| Correct detections needlessly flagged (false-alarm rate) | {M.fmt_rate(ver['false_alarms'], ver['detector_right'])} |",
          f"| Detector errors fixed by clarification + RAG advice | {ver['fixed_by_pipeline']} of {ver['detector_errors']} |",
          f"| Correct detections broken by the advice step | {ver['broken_by_pipeline']} of {ver['detector_right']} |",
          "", "## 5. Safety, audit mode and advice quality", "",
          "| Check | Result |", "|---|---|",
          f"| Hazard alert recall - model output only (before app guard) | {M.fmt_rate(saf['alert_recall_model_only'], saf['hazard_found'])} |",
          f"| Hazard items missed by detection | {saf['hazard_items'] - saf['hazard_found']} of {saf['hazard_items']} |",
          f"| Alerts shown on non-hazardous items (false alarms) | {saf['alerts_on_non_hazard']} of {saf['non_hazard_matched']} |"]
    a = m["audit"]
    L.append(f"| Audit score mean absolute error | {('%.1f points over %d photos' % (a['mae'], a['images'])) if a['mae'] is not None else 'n/a (no `batch` rows)'} |")
    q = m["advice_quality"]
    L.append(f"| Advice format valid (category, reason, 2-3 steps, spoken text) | {M.fmt_rate(q['format_ok'], q['format_total'])} |")
    L.append(f"| Advice written in {q['language']} | {M.fmt_rate(q['language_ok'], q['language_total'])} |")
    if q["format_problems"]:
        L += ["", "Format problems (first 20):", ""] + [f"- {p}" for p in q["format_problems"]]
    L += ["", "## 6. How to read this", "",
          "- Detection precision/recall/F1 and 'photos with every item found' only use photos whose items are named in the labels file; public-dataset photos (item `*`) are judged on category only.",
          "- 95% CI = Wilson confidence interval. With fewer than ~100 items the interval is wide: quote the interval, not just the percentage.",
          "- Clarifying questions were answered automatically with the `answer` column of the labels file (default *Not sure*), so the numbers are a conservative estimate of what a real user would get.",
          "- `lenient` merges e-waste and domestic hazardous because batteries sit on that boundary in the knowledge base."]
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", nargs="+", default=[str(HERE / "eval_data" / "labels.csv")],
                    help="one or more label CSV files (e.g. your own + labels_public.csv)")
    ap.add_argument("--images", default=str(HERE / "eval_data" / "images"))
    ap.add_argument("--model", default=P.MODELS[0])
    ap.add_argument("--language", default="English", choices=list(P.TTS_LANG))
    ap.add_argument("--api-key", default=os.environ.get("GEMINI_API_KEY", ""))
    ap.add_argument("--threshold", type=float, default=0.6, help="name-similarity needed to match a detected item to a label")
    ap.add_argument("--sleep", type=float, default=2.0, help="seconds between photos (free-tier rate limits)")
    ap.add_argument("--limit", type=int, default=0, help="only run the first N photos")
    ap.add_argument("--no-cache", action="store_true", help="ignore cached API results")
    ap.add_argument("--dry-run", action="store_true", help="use a mock model and synthetic images (no API key needed)")
    ap.add_argument("--out", default=str(HERE / "report"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [row for p in args.labels for row in read_labels(p)]
    if args.limit:
        rows = rows[: args.limit]

    if args.dry_run:
        import mock_client
        mock_client.install_google_stub()
        client = mock_client.MockClient()
    else:
        if not args.api_key:
            raise SystemExit("Set GEMINI_API_KEY or pass --api-key (or use --dry-run to test the harness).")
        from google import genai
        client = genai.Client(api_key=args.api_key)

    kb = P.KnowledgeBase()
    cache_path = out / "cache.json"
    cache = {} if (args.dry_run or args.no_cache or not cache_path.exists()) else json.loads(cache_path.read_text(encoding="utf-8"))

    records, skipped, failures = [], 0, 0
    for n, row in enumerate(rows, 1):
        path = Path(args.images) / row["image"]
        if path.exists():
            raw = path.read_bytes()
        elif args.dry_run:
            import mock_client
            raw = mock_client.synthetic_image_bytes(seed=n)
        else:
            print(f"[{n}/{len(rows)}] SKIP {row['image']}: file not found in {args.images}")
            skipped += 1
            continue
        img = P.load_image(raw)
        key = hashlib.md5(raw).hexdigest() + "|" + args.model + "|" + args.language + "|" + row["answer"]
        try:
            if key in cache:
                run = cache[key]
                tag = "cached"
            else:
                if args.dry_run:
                    client.set_row(row)
                run = run_pipeline(client, kb, img, row, args.model, args.language)
                if not args.dry_run:
                    cache[key] = run
                    cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
                    time.sleep(args.sleep)
                tag = f"{run['t']['total']:.1f}s"
        except Exception as e:  # keep going; report how many photos failed
            print(f"[{n}/{len(rows)}] FAIL {row['image']}: {e}")
            failures += 1
            continue
        ok, _ = P.quality_check(img)
        records.append({"row": row, "items": run["items"], "t": run["t"], "quality_ok": ok,
                        "final_results": apply_hazard_guard(run["items"], run["results"])})
        print(f"[{n}/{len(rows)}] {row['image']}: {len(run['items'])} item(s) detected ({tag})")

    if not records:
        raise SystemExit("Nothing was evaluated. Check the images folder and labels file.")

    metrics, items_out = evaluate(records, args)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    with open(out / "per_item.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(items_out[0].keys()))
        w.writeheader()
        w.writerows(items_out)
    png = write_confusion_png(metrics["final_report"], out / "confusion_matrix.png", "Final category - confusion matrix")
    write_summary(metrics, args, out / "summary.md", skipped, failures, args.dry_run)

    e2e = metrics["end_to_end_accuracy"]
    print("\n" + "=" * 60)
    print(f"End-to-end accuracy : {M.fmt_rate(e2e['final'], e2e['n'])}")
    print(f"Macro-F1            : {metrics['final_report']['macro_f1']:.3f}")
    print(f"Detection F1        : {metrics['detection']['f1']:.3f}")
    print(f"Report written to   : {out}  (summary.md, metrics.json, per_item.csv{', confusion_matrix.png' if png else ''})")
    if args.dry_run:
        print("DRY RUN: mock model - these numbers are not real.")


if __name__ == "__main__":
    main()
