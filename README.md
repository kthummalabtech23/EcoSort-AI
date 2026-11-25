# EcoSort AI - Prototype (PS 11)

Upload a photo of waste -> the app detects every item, double-checks the category, asks a question when unsure, retrieves the matching Indian waste rules, and gives bin + disposal steps + reuse ideas + safety alerts.

## 1. Setup (5 minutes)
1. Install Python 3.10 or newer.
2. Get a free Gemini API key: https://aistudio.google.com -> "Get API key".
3. In this folder run:
   ```
   pip install -r requirements.txt
   streamlit run app.py
   ```
   On Windows you can just double-click `run_windows.bat`.
4. The app opens in the browser (http://localhost:8501). Paste the API key in the sidebar.
   (Optional) set it once instead: `set GEMINI_API_KEY=your_key` (Windows) or `export GEMINI_API_KEY=your_key` (Mac/Linux).

## 2. How it works (maps to the report)
| Step | Code | What happens |
|---|---|---|
| Image quality check | `pipeline.quality_check` | Blur (Laplacian variance) and brightness check |
| Item detection (VLM) | `pipeline.detect` | Gemini lists items, material, condition, category, confidence, hazard |
| Self-verification | `pipeline.verify` | Independent second classification; disagreement or low confidence -> clarifying question |
| Knowledge retrieval (RAG) | `pipeline.KnowledgeBase` | TF-IDF retrieval over `knowledge_base/` (summaries of SWM 2016, Plastic, E-Waste rules) |
| Recommendation engine | `pipeline.advise` | Gemini writes bin, reason, steps, reuse, sellable, safety alert using ONLY retrieved rules |
| Voice output | `pipeline.speech_text`, `pipeline.synthesize` | Recommendations are read aloud in the selected language (gTTS; per-item Listen button and Read all; hazardous items first) |
| Pickup requests | `requests_store.py`, `requests_ui.py` | Households raise a request when waste is not cleared for 1-2+ days at a location; collection staff see, filter and update them |
| Collection-point audit | `pipeline.audit_score` | Score for a batch = share of items that belong in the declared batch |
| History | `scans.db` (SQLite) | Every scan and user answer is stored (future retraining data) |

The model name falls back automatically (gemini-3.5-flash -> 3-flash -> 2.5-flash -> 2.0-flash).
In the final version, TF-IDF retrieval can be swapped for ChromaDB + Sentence-Transformers, and the second opinion for a fine-tuned MobileNetV2.

## 3. Demo script for the panel (2 minutes)
1. Household scan: photo of a pizza box + a used battery (or chip packet + broken glass).
2. Show "Detected items" -> point at the "second opinion" line (self-verification).
3. Answer the clarifying question ("Is the box oily?" -> Yes).
4. Show the safety alert for the battery, the bin badges, the "Why" line, reuse and sell tips.
5. Switch language to Hindi/Kannada/Telugu and re-run to show multilingual output.
6. Switch Mode -> Collection-point audit, batch = Dry waste, upload a photo of mixed items spread on a table -> segregation score + misplaced items.
7. Press "Read all recommendations" to play the advice aloud; change the language and press it again to hear Hindi/Kannada/Telugu/Tamil.
8. Open "Rules used" to show RAG sources.

Tips: keep 3-4 test photos ready on the laptop; use good lighting; make sure the laptop has internet.


## 4. Evaluation (accuracy, F1, confusion matrix, RAG, safety)
See `evaluation/EVALUATION.md`. Quick start: `pip install -r requirements-eval.txt` then `python evaluation/run_eval.py --dry-run` to test, and `python evaluation/run_eval.py` on your labelled photos for the real numbers.

## 5. Pickup requests (households <-> collection staff)
Use the menu at the top of the app: **Raise pickup request** and **Collection team**.
- **Household:** fills name, mobile number, area, exact location, waste type and how long it has been uncleared (1 day / 2 days / 3+ days), optionally a photo. They get a request number (REQ-0001) and can follow its status by entering their mobile number.
- **Collection team:** enter the staff PIN (default `1234`; change it with `set STAFF_PIN=yourpin` on Windows or `export STAFF_PIN=yourpin` on Mac/Linux). They see all requests ranked by priority (hazardous / sanitary waste or 3+ days = High), filter by status, area and waste type, download a CSV route sheet, and mark requests Acknowledged or Collected with a note the household can read.
- Requests are stored in `scans.db` (table `pickup_requests`). To fill the dashboard for a demo run `python seed_demo_requests.py` once.
- Tests: `python tests/test_requests.py`.
- Prototype limits: one shared staff PIN and households identified by mobile number only; a real deployment needs OTP login for households, staff accounts and a shared server database.
