"""Fake Gemini client so the evaluation harness can be tested WITHOUT an API key or internet.

It imitates the three prompts used by pipeline.py (detect / verify / advise) and answers from the
ground-truth labels with a controlled error rate. Numbers produced with it are NOT real results -
use it only to check that the evaluation scripts run.
"""
import io
import json
import random
import re
import sys
import types
from types import SimpleNamespace

import numpy as np
from PIL import Image

from pathlib import Path  # noqa: E402

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
import metrics as M  # noqa: E402
import pipeline as P  # noqa: E402

TEXT = {
    "English": ("Put it in the correct bin.", ["Keep it clean and dry.", "Hand it to the collector."],
                "Put this item in the correct bin and keep it clean.", "Pack it safely and keep it separate."),
    "Hindi": ("इसे सही कचरे के डिब्बे में डालें।", ["इसे साफ और सूखा रखें।", "कचरा संग्रहकर्ता को दें।"],
              "इस वस्तु को सही डिब्बे में डालें।", "इसे सुरक्षित पैक करें।"),
    "Kannada": ("ಇದನ್ನು ಸರಿಯಾದ ಡಬ್ಬಿಯಲ್ಲಿ ಹಾಕಿ.", ["ಇದನ್ನು ಸ್ವಚ್ಛವಾಗಿ ಇಡಿ.", "ಸಂಗ್ರಹಕರಿಗೆ ಕೊಡಿ."],
                "ಈ ವಸ್ತುವನ್ನು ಸರಿಯಾದ ಡಬ್ಬಿಯಲ್ಲಿ ಹಾಕಿ.", "ಇದನ್ನು ಸುರಕ್ಷಿತವಾಗಿ ಪ್ಯಾಕ್ ಮಾಡಿ."),
    "Telugu": ("దీన్ని సరైన డబ్బాలో వేయండి.", ["దీన్ని శుభ్రంగా ఉంచండి.", "సేకరణదారుకు ఇవ్వండి."],
               "ఈ వస్తువును సరైన డబ్బాలో వేయండి.", "దీన్ని సురక్షితంగా ప్యాక్ చేయండి."),
    "Tamil": ("இதை சரியான தொட்டியில் போடுங்கள்.", ["இதை சுத்தமாக வையுங்கள்.", "சேகரிப்பவரிடம் கொடுங்கள்."],
              "இந்த பொருளை சரியான தொட்டியில் போடுங்கள்.", "இதை பாதுகாப்பாக பேக் செய்யுங்கள்."),
}


def install_google_stub():
    """pipeline.call_gemini imports google.genai.types; fake it if google-genai is not installed."""
    try:
        import google.genai  # noqa: F401
        return
    except Exception:
        pass
    g = types.ModuleType("google")
    genai = types.ModuleType("google.genai")
    t = types.ModuleType("google.genai.types")
    t.GenerateContentConfig = lambda **kw: SimpleNamespace(**kw)
    genai.types = t
    g.genai = genai
    sys.modules.update({"google": g, "google.genai": genai, "google.genai.types": t})


def synthetic_image_bytes(seed=0, size=320):
    """A sharp, mid-brightness noise picture (passes the quality check)."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(60, 200, (size, size, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


class MockClient:
    def __init__(self, seed=0, detect_error=0.20, verify_error=0.15, miss_rate=0.10, fp_rate=0.10, fix_rate=0.5):
        self.models = self
        self.seed, self.row = seed, None
        self.detect_error, self.verify_error = detect_error, verify_error
        self.miss_rate, self.fp_rate, self.fix_rate = miss_rate, fp_rate, fix_rate

    def set_row(self, row):
        self.row = row
        self.rng = random.Random(f"{self.seed}-{row['image']}")

    # -- helpers
    def _gt_for(self, name):
        if self.row["names"] == ["*"]:
            return 0
        best = max(range(len(self.row["names"])), key=lambda i: M.name_similarity(name, self.row["names"][i]), default=None)
        if best is None or M.name_similarity(name, self.row["names"][best]) < 0.5:
            return None
        return best

    def _other(self, cat):
        return self.rng.choice([c for c in P.CATEGORIES if c != cat])

    # -- the API surface used by pipeline.call_gemini
    def generate_content(self, model, contents, config=None):
        prompt = contents[-1]
        if prompt.startswith("You are a waste-identification expert"):
            out = self._detect()
        elif prompt.startswith("Independently classify"):
            out = self._verify(prompt)
        else:
            out = self._advise(prompt)
        return SimpleNamespace(text=json.dumps(out, ensure_ascii=False))

    def _detect(self):
        items = []
        for n, c, h in zip(self.row["names"], self.row["cats"], self.row["hazards"]):
            if self.rng.random() < self.miss_rate:
                continue
            wrong = self.rng.random() < self.detect_error
            items.append({"name": "item" if n == "*" else n, "material": "mixed", "condition": "used",
                          "category": self._other(c) if wrong else c,
                          "confidence": round(self.rng.uniform(0.40, 0.88) if wrong else self.rng.uniform(0.80, 0.99), 2),
                          "question": None, "hazard": bool(h)})
        if self.rng.random() < self.fp_rate:
            items.append({"name": "bottle cap", "material": "plastic", "condition": "clean", "category": "dry",
                          "confidence": 0.9, "question": None, "hazard": False})
        return {"items": items, "visible_note": "mock run"}

    def _verify(self, prompt):
        names = re.search(r"use these exact names\): (.*)\n", prompt).group(1).split(", ")
        out = []
        for n in names:
            gi = self._gt_for(n)
            gt = self.row["cats"][gi] if gi is not None else "dry"
            wrong = self.rng.random() < self.verify_error
            out.append({"name": n, "category": self._other(gt) if wrong else gt, "confidence": 0.8})
        return {"items": out}

    def _advise(self, prompt):
        lang = re.search(r"Write the output text in (\w+)\.", prompt).group(1)
        payload = json.loads(re.search(r"ITEMS \(with user answers to clarifying questions, if any\):\n(.*?)\n\nFor each item return",
                                       prompt, re.S).group(1))
        reason, steps, spoken, alert = TEXT.get(lang, TEXT["English"])
        out = []
        for it in payload:
            gi = self._gt_for(it["name"])
            cat = it["category"]
            if gi is not None and cat != self.row["cats"][gi] and self.rng.random() < self.fix_rate:
                cat = self.row["cats"][gi]
            out.append({"name": it["name"], "category": cat, "reason": reason, "steps": steps,
                        "reuse": None, "sellable": False, "spoken": spoken,
                        "safety_alert": alert if (it.get("hazard") and self.rng.random() < 0.8) else None})
        return {"items": out}
