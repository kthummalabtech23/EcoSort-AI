import hashlib
import os

import streamlit as st

import pipeline as P
import requests_ui

st.set_page_config(page_title="EcoSort AI", page_icon="♻️", layout="wide")


@st.cache_resource
def get_kb():
    return P.KnowledgeBase()


def get_client(key):
    from google import genai
    return genai.Client(api_key=key)


def badge(category):
    label, bin_name, color = P.BIN_INFO.get(category, P.BIN_INFO["dry"])
    return (f"<span style='background:{color};color:white;padding:4px 10px;border-radius:12px;"
            f"font-weight:600;font-size:0.9rem'>{label} &rarr; {bin_name}</span>")


# ---------------------------------------------------------------- navigation
PAGE_SORT, PAGE_RAISE, PAGE_STAFF = "♻️ Sort waste", "📣 Raise pickup request", "📋 Collection team"
st.title("♻️ EcoSort AI")
page = st.radio("Go to", [PAGE_SORT, PAGE_RAISE, PAGE_STAFF], horizontal=True, label_visibility="collapsed")
if page == PAGE_RAISE:
    requests_ui.household_page()
    st.stop()
if page == PAGE_STAFF:
    requests_ui.staff_page()
    st.stop()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    default_key = os.environ.get("GEMINI_API_KEY", "")
    try:
        default_key = default_key or st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        pass
    api_key = st.text_input("Gemini API key", value=default_key, type="password",
                            help="Get a free key at aistudio.google.com")
    model = st.selectbox("Model", P.MODELS, index=0, help="Falls back to the next model automatically if unavailable")
    language = st.selectbox("Answer language", ["English", "Hindi", "Kannada", "Telugu", "Tamil"])
    mode = st.radio("Mode", ["Household scan", "Collection-point audit"],
                    help="Household: scan items you are unsure about. Audit: photograph a sorted batch spread out at a collection point.")
    voice_on = st.toggle("🔊 Voice output", value=True,
                         help="Read recommendations aloud in the selected language (needs internet)")
    batch_type = None
    if mode == "Collection-point audit":
        batch_type = st.radio("This batch is supposed to be", ["dry", "wet"],
                              format_func=lambda c: P.BIN_INFO[c][0])
    st.divider()
    with st.expander("Recent scans"):
        for ts, m, _ in P.load_history():
            st.caption(f"{ts} · {m}")

# ---------------------------------------------------------------- header
st.caption("Smart waste segregation and recycling advisor · Photo → items → verified category → rule-based advice")

tab_up, tab_cam = st.tabs(["Upload photo", "Use camera"])
with tab_up:
    up = st.file_uploader("Upload a photo of waste", type=["jpg", "jpeg", "png", "webp"])
with tab_cam:
    cam = st.camera_input("Take a photo")
src = up or cam

if not src:
    st.info("Upload or take a photo of one or more waste items to begin.")
    st.stop()

raw = src.getvalue()
img_id = hashlib.md5(raw).hexdigest() + mode + str(batch_type)
def reset():
    for k in list(st.session_state.keys()):
        if k in ("items", "note", "results", "sources", "voice") or str(k).startswith("q"):
            del st.session_state[k]


if st.session_state.get("img_id") != img_id:
    reset()
    st.session_state.img_id = img_id

img = P.load_image(raw)
left, right = st.columns([1, 1.4])
with left:
    st.image(img)
    ok, qmsg = P.quality_check(img)
    (st.success if ok else st.warning)(qmsg)

with right:
    if not api_key:
        st.error("Enter your Gemini API key in the sidebar.")
        st.stop()

    # ---------------- Step 1: detect + verify
    if "items" not in st.session_state:
        if st.button("Analyse waste", type="primary", use_container_width=True):
            try:
                client = get_client(api_key)
                with st.status("Analysing...", expanded=True) as s:
                    st.write("Detecting items with the vision-language model")
                    items, note, used = P.detect(client, img, model)
                    if not items:
                        s.update(label="No waste items found", state="error")
                        st.stop()
                    st.write(f"Verifying with an independent second opinion ({len(items)} items)")
                    items = P.verify(client, img, items, model)
                    s.update(label=f"Found {len(items)} item(s) using {used}", state="complete")
                st.session_state.items, st.session_state.note = items, note
                st.rerun()
            except Exception as e:
                st.error(f"Could not analyse the image: {e}")
        st.stop()

    items = st.session_state["items"]
    st.subheader("Detected items")
    for it in items:
        mark = "✅ models agree" if it["agree"] else f"⚠️ second opinion says {P.BIN_INFO.get(it['second_opinion'], ('?',))[0]}"
        st.markdown(f"**{it['name']}** · {it.get('material','')} · {it.get('condition','')} · "
                    f"confidence {it['confidence']:.0%} · {mark}")
    if st.session_state.get("note"):
        st.caption(f"Note: {st.session_state.note}")

    # ---------------- Step 2: clarifying questions
    unsure = [it for it in items if it.get("needs_clarification")]
    if "results" not in st.session_state:
        with st.form("clarify"):
            answers = {}
            if unsure:
                st.markdown("#### 🤔 Quick question before I decide")
                for i, it in enumerate(unsure):
                    answers[it["name"]] = st.radio(f"{it['name']}: {it['question']}", ["Yes", "No", "Not sure"],
                                                   horizontal=True, key=f"q{i}")
            go = st.form_submit_button("Get recommendations", type="primary", use_container_width=True)
        if go:
            try:
                client = get_client(api_key)
                with st.spinner("Retrieving rules and generating advice..."):
                    results, sources = P.advise(client, get_kb(), items, answers, language, model)
                # never lose a hazard flag from detection
                hz = {it["name"] for it in items if it.get("hazard")}
                for r in results:
                    if r.get("name") in hz and not r.get("safety_alert"):
                        r["safety_alert"] = "Hazardous item: pack it safely, label it and keep it separate from regular waste."
                st.session_state.results, st.session_state.sources = results, sources
                P.save_scan(mode, results, answers)
                st.rerun()
            except Exception as e:
                st.error(f"Could not generate recommendations: {e}")
        st.stop()

# ---------------------------------------------------------------- results
results = st.session_state["results"]
st.divider()

if mode == "Collection-point audit":
    score, misplaced = P.audit_score(results, batch_type)
    c1, c2, c3 = st.columns(3)
    c1.metric("Segregation quality", f"{score}%")
    c2.metric("Items checked", len(results))
    c3.metric("Misplaced items", len(misplaced))
    st.progress(score / 100)
    if misplaced:
        st.warning("Misplaced in this " + P.BIN_INFO[batch_type][0].lower() + " batch: " +
                   ", ".join(f"{m['name']} (→ {P.BIN_INFO[m['category']][0]})" for m in misplaced))
    else:
        st.success("This batch is correctly segregated.")

alerts = [r for r in results if r.get("safety_alert")]
if alerts:
    st.subheader("⚠️ Safety alerts for waste workers")
    for r in alerts:
        st.error(f"**{r['name']}**: {r['safety_alert']}")

@st.cache_data(show_spinner=False)
def tts_cached(text, lang):
    return P.synthesize(text, lang)


def speak(label, text):
    """Button callback: build the audio and keep it in session state so the player survives reruns."""
    try:
        st.session_state.voice = {"label": label, "audio": tts_cached(text, language), "error": None}
    except Exception as e:
        st.session_state.voice = {"label": label, "audio": None, "error": str(e)}


st.subheader("Recommendations")
if voice_on:
    st.button("🔊 Read all recommendations", on_click=speak,
              args=("All recommendations", P.speech_text_all(results)))
    v = st.session_state.get("voice")
    if v:
        if v["audio"]:
            st.caption(f"🔊 Now playing: {v['label']}")
            st.audio(v["audio"], format="audio/mp3", autoplay=True)
        else:
            st.warning("Voice output needs an internet connection and the gTTS package "
                       f"(`pip install gTTS`). Details: {v['error']}")
cols = st.columns(2)
for i, r in enumerate(results):
    with cols[i % 2]:
        with st.container(border=True):
            st.markdown(f"### {r['name']}")
            st.markdown(badge(r["category"]), unsafe_allow_html=True)
            st.markdown(f"**Why:** {r.get('reason','')}")
            for step in r.get("steps") or []:
                st.markdown(f"- {step}")
            extras = []
            if r.get("reuse"):
                extras.append(f"♻️ **Reuse:** {r['reuse']}")
            if r.get("sellable"):
                extras.append("💰 **Sell it:** scrap dealers usually buy this")
            for e in extras:
                st.markdown(e)
            if voice_on:
                st.button("🔊 Listen", key=f"speak{i}", on_click=speak,
                          args=(r["name"], P.speech_text(r)))

with st.expander("Rules used for these recommendations (RAG sources)"):
    for s in st.session_state.get("sources", []):
        st.markdown(f"- {s}")

if st.button("Analyse this photo again"):
    reset()
    st.rerun()
st.caption("To scan a different item, upload or take a new photo above.")
