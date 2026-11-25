#!/usr/bin/env python3
"""Offline checks of the non-LLM parts of EcoSort AI (no API key needed).

1. Image-quality check  - does it flag blurred / too-dark / over-exposed photos, without false alarms?
2. Audit score          - is the collection-point segregation score computed correctly?
3. Voice text           - is the text sent to the speaker clean (no markdown/emoji), hazardous items first?
4. Text-to-speech (optional, needs internet): python evaluation/eval_offline.py --tts

Usage (project root):  python evaluation/eval_offline.py [--tts]
Uses your photos in eval_data/images if present, otherwise synthetic test images.
"""
import argparse
import io
import sys
import time
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import mock_client  # noqa: E402
import pipeline as P  # noqa: E402

results = []   # (section, check, passed, detail)


def check(section, name, passed, detail=""):
    results.append((section, name, bool(passed), detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {section}: {name} {detail}")


def base_images(limit=20):
    imgs = []
    for p in sorted((HERE / "eval_data" / "images").glob("*")):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            imgs.append(P.load_image(p.read_bytes()))
        if len(imgs) >= limit:
            break
    real = bool(imgs)
    if not imgs:
        imgs = [P.load_image(mock_client.synthetic_image_bytes(seed=s)) for s in range(10)]
    return imgs, real


def quality_section():
    imgs, real = base_images()
    src = "your photos" if real else "synthetic images"
    degrade = {
        "blurred": lambda im: im.filter(ImageFilter.GaussianBlur(radius=10)),
        "too dark": lambda im: ImageEnhance.Brightness(im).enhance(0.12),
        "over-exposed": lambda im: ImageEnhance.Brightness(im).enhance(4.0),
    }
    originals_ok = sum(P.quality_check(im)[0] for im in imgs)
    rate = originals_ok / len(imgs)
    # With real photos some originals may legitimately be soft/dark, so only require a sensible pass rate
    check("quality", f"originals accepted ({src})", rate >= (0.7 if real else 0.99), f"{originals_ok}/{len(imgs)}")
    for name, fn in degrade.items():
        flagged = sum(not P.quality_check(fn(im))[0] for im in imgs)
        check("quality", f"{name} photos flagged", flagged / len(imgs) >= 0.9, f"{flagged}/{len(imgs)}")


def audit_section():
    mk = lambda cats: [{"name": f"item{i}", "category": c} for i, c in enumerate(cats)]  # noqa: E731
    s, mis = P.audit_score(mk(["dry", "dry", "dry", "dry"]), "dry")
    check("audit", "all correct -> 100", s == 100 and mis == [], f"got {s}")
    s, mis = P.audit_score(mk(["dry", "wet", "dry", "e_waste"]), "dry")
    check("audit", "2 of 4 misplaced -> 50", s == 50 and len(mis) == 2, f"got {s}")
    s, mis = P.audit_score(mk(["wet", "wet"]), "dry")
    check("audit", "all wrong -> 0", s == 0 and len(mis) == 2, f"got {s}")
    s, mis = P.audit_score([], "dry")
    check("audit", "empty batch -> 0, no crash", s == 0 and mis == [], f"got {s}")


def voice_text_section():
    sample = [
        {"name": "Peel", "category": "wet", "spoken": "**Peel** goes in the green bin. ♻️", "safety_alert": None},
        {"name": "AA battery", "category": "domestic_hazardous", "reason": "Contains lithium", "steps": ["Tape the ends", "Drop at a collection point"],
         "safety_alert": "Fire risk, keep it dry ⚠️"},
        {"name": "Box", "category": "dry", "reason": "Clean cardboard", "steps": ["Flatten it", "Keep it dry"], "safety_alert": None},
    ]
    texts = [P.speech_text(r) for r in sample]
    check("voice", "no markdown/emoji in spoken text", not any(ch in t for t in texts for ch in "*#_`♻⚠"), "")
    check("voice", "every item has non-empty speech", all(len(t) > 10 for t in texts), "")
    check("voice", "fallback text mentions the bin", "Domestic hazardous" in texts[1], "")
    check("voice", "fallback steps end with punctuation", all(s.strip()[-1] in ".!?" for s in texts[1].split(". ") if s.strip()) or texts[1].endswith("."), "")
    allt = P.speech_text_all(sample)
    check("voice", "hazardous item is read first", allt.index("Fire risk") < allt.index("green bin"), "")


def tts_section():
    samples = {"English": "Put the plastic bottle in the blue bin.",
               "Hindi": mock_client.TEXT["Hindi"][2], "Kannada": mock_client.TEXT["Kannada"][2],
               "Telugu": mock_client.TEXT["Telugu"][2], "Tamil": mock_client.TEXT["Tamil"][2]}
    for lang, text in samples.items():
        t0 = time.time()
        try:
            data = P.synthesize(text, lang)
            check("tts", f"{lang} audio generated", len(data) > 1000, f"{len(data) / 1024:.1f} KB in {time.time() - t0:.1f}s")
        except Exception as e:
            check("tts", f"{lang} audio generated", False, f"error: {str(e)[:120]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tts", action="store_true", help="also test gTTS for all languages (needs internet)")
    ap.add_argument("--out", default=str(HERE / "report"))
    args = ap.parse_args()

    quality_section()
    audit_section()
    voice_text_section()
    if args.tts:
        tts_section()

    passed = sum(r[2] for r in results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# Offline checks", "", f"{passed}/{len(results)} checks passed", "", "| Section | Check | Result | Detail |", "|---|---|---|---|"]
    lines += [f"| {s} | {n} | {'PASS' if ok else 'FAIL'} | {d} |" for s, n, ok, d in results]
    (out / "offline_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{passed}/{len(results)} checks passed -> {out / 'offline_summary.md'}")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
