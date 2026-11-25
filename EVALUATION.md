# EcoSort AI - Evaluation guide

All commands run from the project root (the folder with `app.py`).
Install once: `pip install -r requirements.txt -r requirements-eval.txt`

## 1. What each script measures

| Script | Needs API key? | What it tells you for the report |
|---|---|---|
| `evaluation/run_eval.py` | Yes (Gemini) | Item detection P/R/F1, **category accuracy** (detector / second opinion / final), per-class P/R/F1, macro-F1, confusion matrix, effect of self-verification, confidence calibration (ECE), safety-alert recall, audit-score error, advice format + language check, latency |
| `evaluation/eval_retrieval.py` | No | RAG quality: Hit@1, Hit@3, Hit@5, MRR over 32 test queries |
| `evaluation/eval_offline.py` | No (`--tts` needs internet) | Image-quality check on blurred/dark/over-exposed photos, audit-score correctness, voice-text cleanliness, text-to-speech works in all 5 languages |
| `evaluation/mock_client.py` | - | Fake model used by `--dry-run` to test the scripts. Its numbers are NOT results. |

## 2. Build the test set (this is the part only you can do)

1. Take **60-100 photos** of waste. Aim for at least 12 items per category (wet, dry, domestic_hazardous, e_waste, sanitary), a mix of single-item and multi-item photos, different lighting, and the tricky cases the project is about: greasy pizza box, used battery, broken glass, chip packet, medicine strip, sanitary pad.
2. Put them in `evaluation/eval_data/images/`.
3. Edit `evaluation/eval_data/labels.csv` (a 12-row sample is included - replace it). One row per photo:
   `image, items (a|b|c), categories (same order), hazards (optional 1/0), batch (optional dry|wet), answer (optional Yes|No|Not sure)`
4. Take 8-10 extra photos of mixed items spread on a table and fill `batch` to test **audit mode**.
5. Label from the rules in `knowledge_base/` and do NOT tune the prompts on this set afterwards - keep it as an untouched test set.

**Labelling policy for batteries:** the knowledge base calls used batteries *domestic hazardous*, but they are also e-waste. Pick one policy and use it for every label. The report also gives a "lenient" accuracy that merges the two.

## 2b. Public datasets you can use (optional, saves photo-taking time)

| Dataset | Good for | Link |
|---|---|---|
| Garbage Classification v2 (10 classes incl. battery, biological, clothes, shoes; CC BY 4.0) | **Best first choice** - has hazardous (battery) and wet (biological) | https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2 |
| Garbage Classification (12 classes, 15,150 images) | More images; web-scraped, so easier than real life | https://www.kaggle.com/datasets/mostafaabla/garbage-classification |
| TrashNet (glass, paper, cardboard, plastic, metal, trash; 2,527 photos on a white background) | Classic benchmark; very easy images, so accuracy will look high | https://huggingface.co/datasets/garythung/trashnet (code: https://github.com/garythung/trashnet) |
| TACO (litter in the wild, many items per photo, COCO annotations, CC BY 4.0) | Stress test for multi-item detection in messy scenes (outdoor litter, not household bins) | http://tacodataset.org/ - https://github.com/pedropro/TACO |

Kaggle downloads need a free Kaggle account.

**Limits you must state in the report:** these sets label by *material*, one object per photo, and contain **no e-waste and no sanitary waste**. So use them for the wet / dry / hazardous part and take your own photos (about 12 each) for e-waste, sanitary and multi-item scenes. `prepare_dataset.py` maps the classes for you:

`biological -> wet` | `battery -> domestic_hazardous` | `cardboard, paper, plastic, metal, glass, clothes, shoes -> dry` | `trash` is skipped (ambiguous; add `--include-trash` to use it as dry).

```
python evaluation/prepare_dataset.py --source "C:/path/to/unzipped/dataset" --per-category 30
```
This copies a balanced sample into `eval_data/images/` and writes `eval_data/labels_public.csv`. Then run with both label files (see section 3).

## 3. Run

```
python evaluation/run_eval.py --dry-run                      # 1) check the scripts work (mock model)
set GEMINI_API_KEY=your_key                                  #    (Windows)   export GEMINI_API_KEY=your_key  (Mac/Linux)
python evaluation/run_eval.py --language English             # 2) real evaluation  -> evaluation/report/
python evaluation/run_eval.py --labels evaluation/eval_data/labels.csv evaluation/eval_data/labels_public.csv --sleep 4   # own + public photos
python evaluation/run_eval.py --language Hindi --out evaluation/report_hindi
python evaluation/run_eval.py --model gemini-2.5-flash --out evaluation/report_25flash   # compare models
python evaluation/eval_retrieval.py
python evaluation/eval_offline.py --tts
```
Useful flags: `--sleep 4` (free-tier rate limits), `--limit 10` (quick trial), `--no-cache` (re-call the API; results are cached by photo + model + language), `--threshold 0.5` (how loosely a detected item name must match your label).

Outputs in `evaluation/report/`: `summary.md` (ready-made tables), `metrics.json`, `per_item.csv` (every item: truth vs detector vs second opinion vs final - use it to find failure cases), `confusion_matrix.png`.

## 4. How to read the numbers

- **End-to-end accuracy** is the main result: share of ground-truth items given the right category *by the whole pipeline*; an item the model never detected counts as wrong.
- **Detector -> second opinion -> final** shows what each stage adds. The claim "self-verification and clarifying questions improve reliability" is supported if (a) detector accuracy is much higher when the two models agree than when they disagree, (b) the error-catch rate is high with a low false-alarm rate, and (c) "fixed by pipeline" > "broken by pipeline".
- **Safety-alert recall** matters most for waste workers: every hazardous item that is detected should carry an alert. Report both "model only" and "with app guard".
- **ECE** near 0 means the confidence the model states can be trusted.
- **95% CI** is shown because a small test set gives a wide range - quote it.
- Clarifying questions are answered automatically from the `answer` column (default *Not sure*), so results are conservative. Run once with real answers in that column for a best case.

## 5. Findings already visible from the offline tests

- Retrieval currently misses 2 of 32 queries (Hit@1 about 66%, Hit@3 about 91%). Both misses come from word forms: TF-IDF has no stemming, so "needle" does not match "needles" and "sharp" does not match "sharps". Adding a simple stemmer to `KnowledgeBase` (or switching to the planned embedding retrieval) is the fix; re-run `eval_retrieval.py` to show before/after.
- The image-quality check flags blurred, dark and over-exposed test images; try it on your own photos too.

## 6. Suggested wording for the project report

> We evaluated EcoSort AI on N labelled photos (M items) covering five categories. The full pipeline reached X% end-to-end category accuracy (95% CI a-b%), macro-F1 Y, item-detection F1 Z. The independent second opinion flagged E% of the detector's mistakes for a clarifying question at a false-alarm rate of F%. Hazardous items received a safety alert in S% of cases. RAG retrieval achieved Hit@3 of H% and MRR of R. Median response time was T seconds per photo.
