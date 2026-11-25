"""Metric helpers for the EcoSort AI evaluation (pure Python, no API calls)."""
import math
import re
from difflib import SequenceMatcher

CATS = ["wet", "dry", "domestic_hazardous", "e_waste", "sanitary"]

# Unicode blocks used to check that advice is really written in the requested language
SCRIPT_RANGES = {
    "Hindi": (0x0900, 0x097F),
    "Kannada": (0x0C80, 0x0CFF),
    "Telugu": (0x0C00, 0x0C7F),
    "Tamil": (0x0B80, 0x0BFF),
}


# ------------------------------------------------------------------ item matching
def _tokens(s):
    s = re.sub(r"\(\d+\)", " ", str(s).lower())
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return [t for t in s.split() if not t.isdigit()]


def name_similarity(a, b):
    """0..1 similarity between two item names (token overlap, containment, edit ratio)."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    na, nb = " ".join(ta), " ".join(tb)
    if na == nb:
        return 1.0
    sa, sb = set(ta), set(tb)
    jaccard = len(sa & sb) / len(sa | sb)
    contain = 0.85 if (na in nb or nb in na) else 0.0
    return max(jaccard, contain, SequenceMatcher(None, na, nb).ratio())


def match_items(gt_names, pred_names, threshold=0.6):
    """Greedy one-to-one matching, best similarity first. Returns [(gt_idx, pred_idx, sim)]."""
    pairs = sorted(((name_similarity(g, p), gi, pi)
                    for gi, g in enumerate(gt_names) for pi, p in enumerate(pred_names)), reverse=True)
    used_g, used_p, out = set(), set(), []
    for sim, gi, pi in pairs:
        if sim < threshold:
            break
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        out.append((gi, pi, sim))
    return out


# ------------------------------------------------------------------ classification
def classification_report(y_true, y_pred, labels=CATS):
    """Accuracy, per-class precision/recall/F1, macro-F1 and a confusion matrix.

    A prediction of "missed" (item not detected) counts as wrong and appears as an extra column.
    """
    cols = list(labels) + ["missed"]
    cm = {t: {p: 0 for p in cols} for t in labels}
    for t, p in zip(y_true, y_pred):
        if t not in cm:
            continue
        cm[t][p if p in cols else "missed"] += 1
    n = sum(sum(r.values()) for r in cm.values())
    correct = sum(cm[c][c] for c in labels)
    per_class, f1s = {}, []
    for c in labels:
        tp = cm[c][c]
        fp = sum(cm[t][c] for t in labels if t != c)
        fn = sum(v for p, v in cm[c].items() if p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        support = tp + fn
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1, "support": support}
        if support:
            f1s.append(f1)
    return {"n": n, "accuracy": correct / n if n else 0.0,
            "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
            "per_class": per_class, "confusion": cm, "columns": cols}


def wilson_ci(k, n, z=1.96):
    """95% Wilson confidence interval for a proportion (honest with small test sets)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def fmt_rate(k, n):
    """'83.3% (25/30, 95% CI 66-93%)'"""
    if n == 0:
        return "n/a (0 items)"
    lo, hi = wilson_ci(k, n)
    return f"{100 * k / n:.1f}% ({k}/{n}, 95% CI {100 * lo:.0f}-{100 * hi:.0f}%)"


def expected_calibration_error(confidences, corrects, bins=10):
    """ECE: how far stated confidence is from real accuracy (0 = perfectly calibrated)."""
    n = len(confidences)
    if n == 0:
        return 0.0
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confidences) if (lo < c <= hi) or (b == 0 and c == 0)]
        if not idx:
            continue
        acc = sum(corrects[i] for i in idx) / len(idx)
        conf = sum(confidences[i] for i in idx) / len(idx)
        total += len(idx) / n * abs(acc - conf)
    return total


# ------------------------------------------------------------------ misc
def lenient(cat):
    """Treat e_waste and domestic_hazardous as one group (batteries sit on that boundary)."""
    return "domestic_hazardous" if cat == "e_waste" else cat


def script_ratio(text, language):
    """Share of letters in `text` written in the script of `language` (ASCII letters for English)."""
    letters = [ch for ch in str(text) if ch.isalpha()]
    if not letters:
        return 0.0
    if language in SCRIPT_RANGES:
        lo, hi = SCRIPT_RANGES[language]
        return sum(lo <= ord(c) <= hi for c in letters) / len(letters)
    return sum(c.isascii() for c in letters) / len(letters)


def percentile(values, q):
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * q / 100
    f, c = math.floor(k), math.ceil(k)
    return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)
