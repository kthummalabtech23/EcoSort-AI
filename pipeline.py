"""EcoSort AI - core pipeline.

Steps: image quality check -> VLM detection -> independent verification ->
clarifying questions -> rule retrieval (RAG) -> recommendation generation.
"""
import io
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).parent
KB_DIR = BASE / "knowledge_base"
DB_PATH = BASE / "scans.db"

MODELS = ["gemini-3.5-flash", "gemini-3-flash", "gemini-2.5-flash", "gemini-2.0-flash"]
CATEGORIES = ["wet", "dry", "domestic_hazardous", "e_waste", "sanitary"]
BIN_INFO = {
    "wet": ("Wet waste", "Green bin", "#2e7d32"),
    "dry": ("Dry waste", "Blue bin", "#1565c0"),
    "domestic_hazardous": ("Domestic hazardous", "Separate red/black bag", "#c62828"),
    "e_waste": ("E-waste", "E-waste collection point", "#6a1b9a"),
    "sanitary": ("Sanitary waste", "Wrapped, dry bin or as advised", "#ef6c00"),
}
CONFIDENCE_THRESHOLD = 0.75


# ---------------------------------------------------------------- quality check
def quality_check(img: Image.Image):
    """Return (ok, message) using blur (Laplacian variance) and brightness."""
    g = np.asarray(img.convert("L").resize((512, int(512 * img.height / max(img.width, 1)))), dtype=np.float32)
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    blur = float(lap.var())
    bright = float(g.mean())
    issues = []
    if blur < 40:
        issues.append("the photo looks blurry")
    if bright < 40:
        issues.append("the photo is too dark")
    if bright > 235:
        issues.append("the photo is overexposed")
    if issues:
        return False, "Image quality warning: " + " and ".join(issues) + ". Results may be less accurate; consider retaking."
    return True, "Image quality OK"


# ---------------------------------------------------------------- knowledge base (RAG)
class KnowledgeBase:
    def __init__(self, kb_dir=KB_DIR):
        self.chunks, self.sources = [], []
        for f in sorted(kb_dir.glob("*.md")):
            title = f.read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip()
            for para in f.read_text(encoding="utf-8").split("\n"):
                para = para.strip()
                if len(para) > 40 and not para.startswith("#"):
                    self.chunks.append(para)
                    self.sources.append(title)
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vec.fit_transform(self.chunks)

    def retrieve(self, query, k=4):
        sims = cosine_similarity(self.vec.transform([query]), self.matrix)[0]
        idx = sims.argsort()[::-1][:k]
        return [{"text": self.chunks[i], "source": self.sources[i], "score": float(sims[i])} for i in idx if sims[i] > 0]


# ---------------------------------------------------------------- Gemini helpers
def _parse_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    return json.loads(text)


def call_gemini(client, contents, preferred_model=None):
    """Call Gemini with JSON output, falling back through model names."""
    from google.genai import types
    order = [preferred_model] + [m for m in MODELS if m != preferred_model] if preferred_model else MODELS
    last_err = None
    for model in order:
        try:
            resp = client.models.generate_content(
                model=model, contents=contents,
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2),
            )
            return _parse_json(resp.text), model
        except Exception as e:  # model not found / unavailable -> try next
            last_err = e
            msg = str(e).lower()
            if "api key" in msg or "permission" in msg or "unauthenticated" in msg:
                raise
    raise RuntimeError(f"All Gemini models failed. Last error: {last_err}")


DETECT_PROMPT = """You are a waste-identification expert for Indian households.
Look at the photo and list every distinct waste item you can see (max 10).
For each item return:
- name: short name (e.g. "pizza box", "AA battery")
- material: main material
- condition: e.g. clean, food-soiled, broken, wet, swollen
- category: one of wet, dry, domestic_hazardous, e_waste, sanitary
- confidence: 0 to 1, how sure you are of the category from the image alone
- question: if the correct category depends on something not visible (e.g. is it oily, is it empty), a short yes/no question for the user, else null
- hazard: true if it can injure or harm sanitation workers (sharp, toxic, flammable, mercury, needles, lithium battery), else false
Also return "visible_note": one sentence describing anything you could not see clearly.
Return JSON: {"items":[{...}], "visible_note":"..."}"""

VERIFY_PROMPT = """Independently classify the waste items in this photo into Indian source-segregation categories.
Categories: wet, dry, domestic_hazardous, e_waste, sanitary.
Items to classify (use these exact names): {names}
Return JSON: {{"items":[{{"name":"...","category":"...","confidence":0.0}}]}}"""

ADVICE_PROMPT = """You are EcoSort AI, a waste segregation advisor for India.
Use ONLY the retrieved rules below as your source for disposal instructions. If the rules do not cover something, give cautious general advice.
Write the output text in {language}. Keep JSON keys in English.

RETRIEVED RULES:
{rules}

ITEMS (with user answers to clarifying questions, if any):
{items}

For each item return:
- name
- category: one of wet, dry, domestic_hazardous, e_waste, sanitary (update it if the user's answer changes it)
- reason: one sentence explaining why it belongs there
- steps: 2-3 short disposal steps
- reuse: one short reuse or upcycling idea, or null
- sellable: true if scrap dealers usually buy it
- safety_alert: a short warning with safe packing instructions if hazardous, else null
- spoken: a natural 2-3 sentence version of this advice meant to be READ ALOUD in {language}: say the item, which bin it goes in (mention the bin colour), and the key disposal step; if there is a safety alert, end with it. Plain sentences only, no symbols, bullets, emojis or markdown.
Return JSON: {{"items":[{{...}}]}}"""


# ---------------------------------------------------------------- pipeline steps
def detect(client, img, model=None):
    data, used = call_gemini(client, [img, DETECT_PROMPT], model)
    items = data.get("items", [])
    seen = {}
    for it in items:
        n = str(it.get("name") or "item").strip()
        seen[n] = seen.get(n, 0) + 1
        it["name"] = n if seen[n] == 1 else f"{n} ({seen[n]})"
        it["category"] = it.get("category") if it.get("category") in CATEGORIES else "dry"
        it["confidence"] = float(it.get("confidence") or 0)
    return items, data.get("visible_note", ""), used


def verify(client, img, items, model=None):
    """Second independent opinion; mark disagreements / low confidence for clarification."""
    names = ", ".join(i["name"] for i in items)
    data, _ = call_gemini(client, [img, VERIFY_PROMPT.format(names=names)], model)
    second = {d.get("name", "").lower(): d for d in data.get("items", [])}
    for it in items:
        other = second.get(it["name"].lower())
        it["second_opinion"] = other.get("category") if other else None
        it["agree"] = (other is None) or (other.get("category") == it["category"])
        it["needs_clarification"] = (not it["agree"]) or it["confidence"] < CONFIDENCE_THRESHOLD or bool(it.get("question"))
        if it["needs_clarification"] and not it.get("question"):
            it["question"] = f"Is the {it['name']} clean and dry (not soiled with food or liquid)?"
    return items


def advise(client, kb, items, answers, language="English", model=None):
    rules, used_sources = [], set()
    for it in items:
        q = f"{it['name']} {it.get('material','')} {it.get('condition','')} {it['category'].replace('_',' ')}"
        for r in kb.retrieve(q, k=3):
            if r["text"] not in rules:
                rules.append(r["text"])
                used_sources.add(r["source"])
    payload = []
    for it in items:
        entry = {k: it.get(k) for k in ("name", "material", "condition", "category", "hazard")}
        if it["name"] in answers:
            entry["clarifying_question"] = it.get("question")
            entry["user_answer"] = answers[it["name"]]
        payload.append(entry)
    prompt = ADVICE_PROMPT.format(language=language, rules="\n".join(f"- {r}" for r in rules),
                                  items=json.dumps(payload, ensure_ascii=False, indent=1))
    data, _ = call_gemini(client, [prompt], model)
    out = data.get("items", [])
    for o in out:
        if o.get("category") not in CATEGORIES:
            o["category"] = "dry"
    return out, sorted(used_sources)


def audit_score(results, batch_type):
    """Collection-point audit: share of items that belong in the declared batch."""
    if not results:
        return 0, []
    misplaced = [r for r in results if r["category"] != batch_type]
    score = round(100 * (len(results) - len(misplaced)) / len(results))
    return score, misplaced


# ---------------------------------------------------------------- voice output
TTS_LANG = {"English": "en", "Hindi": "hi", "Kannada": "kn", "Telugu": "te", "Tamil": "ta"}
_SPEECH_CLEAN = re.compile(r"[*_`#>\u2022\u267b\U0001F4B0\u26a0\u2705\ufe0f]+")


def _clean_for_speech(text):
    return re.sub(r"\s+", " ", _SPEECH_CLEAN.sub(" ", str(text or ""))).strip()


def speech_text(r):
    """Text to read aloud for one recommendation (uses the model's 'spoken' field, with a fallback)."""
    spoken = _clean_for_speech(r.get("spoken"))
    if spoken:
        return spoken
    label, bin_name, _ = BIN_INFO.get(r.get("category"), BIN_INFO["dry"])
    parts = [f"{r.get('name', 'Item')}. {label}, {bin_name}.", r.get("reason", "")]
    parts += list(r.get("steps") or [])
    if r.get("safety_alert"):
        parts.append(r["safety_alert"])
    # make sure every part ends with punctuation so the voice pauses between them
    parts = [str(x).strip() for x in parts if x]
    return _clean_for_speech(" ".join(x if x[-1] in ".!?\u0964" else x + "." for x in parts))


def speech_text_all(results):
    """Read every recommendation, hazardous items first (their text already ends with the safety alert)."""
    ordered = sorted(results, key=lambda r: not r.get("safety_alert"))
    return " ".join(t for t in (speech_text(r) for r in ordered) if t)


def synthesize(text, language="English"):
    """Convert text to MP3 bytes with gTTS (needs internet). Raises on failure."""
    from gtts import gTTS
    buf = io.BytesIO()
    kwargs = {"tld": "co.in"} if language == "English" else {}  # Indian-accent English
    gTTS(text=text, lang=TTS_LANG.get(language, "en"), **kwargs).write_to_fp(buf)
    return buf.getvalue()


# ---------------------------------------------------------------- storage
def save_scan(mode, results, answers):
    con = sqlite3.connect(DB_PATH)
    con.execute("CREATE TABLE IF NOT EXISTS scans (ts TEXT, mode TEXT, results TEXT, answers TEXT)")
    con.execute("INSERT INTO scans VALUES (?,?,?,?)", (datetime.now().isoformat(timespec="seconds"), mode,
                                                       json.dumps(results, ensure_ascii=False), json.dumps(answers)))
    con.commit()
    con.close()


def load_history(limit=10):
    if not DB_PATH.exists():
        return []
    con = sqlite3.connect(DB_PATH)
    rows = con.execute("SELECT ts, mode, results FROM scans ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
    con.close()
    return rows


def load_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes))
    img = img.convert("RGB")
    img.thumbnail((1280, 1280))
    return img
