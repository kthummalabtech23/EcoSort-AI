"""Tests for the pickup-request feature. Run from the project root:

    python tests/test_requests.py        (or: python -m pytest tests -q)

The UI tests use a tiny stand-in for Streamlit that returns scripted answers, so the page logic
(validation, duplicate warning, staff login, status update) is exercised without a browser.
"""
import io
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402


# ------------------------------------------------------------------ fake streamlit
class Rerun(Exception):
    pass


class Stop(Exception):
    pass


class _State(dict):
    __getattr__ = dict.get

    def __setattr__(self, k, v):
        self[k] = v


class _Ctx:
    """Context manager that also forwards widget calls (used for st.columns / st.sidebar)."""
    def __init__(self, fake):
        self._fake = fake

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):
        return getattr(self._fake, name)


class FakeSt:
    def __init__(self, answers=None, clicks=()):
        self.answers, self.clicks = answers or {}, set(clicks)
        self.session_state = _State()
        self.out = []          # (function name, text)
        self.sidebar = _Ctx(self)

    # ---- helpers
    def _rec(self, name, *a, **k):
        self.out.append((name, " ".join(str(x) for x in a if isinstance(x, str))))
        return _Ctx(self)

    def texts(self, *names):
        return [t for n, t in self.out if not names or n in names]

    def _ans(self, kind, label, default):
        return self.answers.get((kind, label), self.answers.get(label, default))

    def __getattr__(self, name):          # any output function (markdown, caption, success, ...)
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: self._rec(name, *a, **k)

    # ---- inputs
    def text_input(self, label, value="", key=None, **k):
        if key is not None and key in self.session_state:
            return self.session_state[key]
        v = self._ans("text_input", label, value)
        if key is not None:
            self.session_state[key] = v
        return v

    def text_area(self, label, **k):
        return self._ans("text_area", label, "")

    def selectbox(self, label, options, index=0, **k):
        return self._ans("selectbox", label, list(options)[index])

    def multiselect(self, label, options, default=None, **k):
        return self._ans("multiselect", label, default or [])

    def radio(self, label, options, index=0, **k):
        return self._ans("radio", label, list(options)[index])

    def file_uploader(self, label, **k):
        return self.answers.get(("file", label))

    def camera_input(self, label, **k):
        return self.answers.get(("file", label))

    def toggle(self, label, value=False, **k):
        return value

    def form_submit_button(self, label, **k):
        return label in self.clicks

    def button(self, label, **k):
        return label in self.clicks

    # ---- layout
    def columns(self, spec, **k):
        n = spec if isinstance(spec, int) else len(spec)
        return [_Ctx(self) for _ in range(n)]

    def tabs(self, labels):
        return [_Ctx(self) for _ in labels]

    def form(self, *a, **k):
        return _Ctx(self)

    def container(self, **k):
        return _Ctx(self)

    def expander(self, *a, **k):
        return _Ctx(self)

    def cache_resource(self, f=None, **k):
        return f if f else (lambda g: g)

    cache_data = cache_resource

    def rerun(self):
        raise Rerun()

    def stop(self):
        raise Stop()


class Upload:
    def __init__(self, data):
        self._d = data

    def getvalue(self):
        return self._d


def jpeg_bytes(size=(1600, 1200)):
    buf = io.BytesIO()
    Image.new("RGB", size, (30, 120, 60)).save(buf, format="JPEG")
    return buf.getvalue()


# ------------------------------------------------------------------ setup helpers
def fresh_store():
    import requests_store as R
    R.DB_PATH = Path(tempfile.mkdtemp()) / "test.db"
    return R


def make(R, phone="98450 12345", address="12, 3rd Cross, near green bin", waste="dry", days=2, area="indiranagar",
         name="Asha", **kw):
    return R.create_request(name, phone, area, address, waste, days, **kw)


def ui(fake):
    import requests_ui
    requests_ui.st = fake
    return requests_ui


VALID_FORM = {"Your name *": "Asha", "Mobile number *": "98450 12345", "Area / ward / locality *": "indiranagar",
              "Exact location *": "12, 3rd Cross, near green bin"}


# ------------------------------------------------------------------ store tests
def test_phone_normalisation():
    R = fresh_store()
    assert R.normalize_phone("98450 12345") == "9845012345"
    assert R.normalize_phone("+91 98450-12345") == "9845012345"
    assert R.normalize_phone("09845012345") == "9845012345"
    assert R.normalize_phone("12345") is None
    assert R.normalize_phone("5845012345") is None      # Indian mobiles start with 6-9
    assert R.normalize_phone(None) is None


def test_validation():
    R = fresh_store()
    assert R.validate("Asha", "9845012345", "Indiranagar", "12 3rd Cross", "dry", 1) == []
    errs = R.validate("", "123", "", "x", "plastic", 7)
    assert len(errs) == 6, errs


def test_create_list_priority_and_order():
    R = fresh_store()
    a = make(R, days=1, waste="dry", address="Normal one, 1st Main")
    b = make(R, days=3, waste="wet", address="Old heap, 2nd Main")
    c = make(R, days=1, waste="sanitary", address="Hazard, 3rd Main")
    d = make(R, days=2, waste="dry", address="Medium, 4th Main")
    rows = R.list_requests()
    pri = {r["id"]: r["priority"] for r in rows}
    assert pri == {a: "Normal", b: "High", c: "High", d: "Medium"}, pri
    assert [r["id"] for r in rows][:2] == [b, c]          # High first, oldest first inside a priority
    assert rows[-1]["id"] == a
    assert rows[0]["ref"].startswith("REQ-") and rows[0]["area"] == "Indiranagar"


def test_photo_is_compressed_and_flagged():
    R = fresh_store()
    big = jpeg_bytes()
    rid = make(R, photo_bytes=big)
    no = make(R, address="No photo here, 5th Main")
    rows = {r["id"]: r for r in R.list_requests()}
    assert rows[rid]["has_photo"] and not rows[no]["has_photo"]
    stored = R.get_photo(rid)
    img = Image.open(io.BytesIO(stored))
    assert max(img.size) <= 800 and len(stored) < len(big) * 2
    assert R.get_photo(no) is None
    assert "photo" not in rows[rid]                        # raw bytes are never put in list results
    rid2 = make(R, address="Bad photo, 6th Main", photo_bytes=b"not an image")
    assert R.get_photo(rid2) is None                       # bad image is ignored, request still saved


def test_duplicate_detection():
    R = fresh_store()
    rid = make(R)
    assert R.find_duplicate("9845012345", "  12, 3rd  cross, NEAR green bin ")["id"] == rid
    assert R.find_duplicate("9845012345", "another address entirely") is None
    assert R.find_duplicate("9000000001", "12, 3rd Cross, near green bin") is None
    R.update_status(rid, "Collected")
    assert R.find_duplicate("9845012345", "12, 3rd Cross, near green bin") is None


def test_update_status_and_stats():
    R = fresh_store()
    a, b, c = make(R, address="addr one, 1st"), make(R, address="addr two, 2nd", days=3), make(R, address="addr three, 3rd")
    assert R.update_status(a, "Acknowledged", "Truck comes tomorrow morning")
    assert R.update_status(b, "Collected")
    assert not R.update_status(9999, "Collected")
    try:
        R.update_status(a, "Done")
        assert False, "invalid status must raise"
    except ValueError:
        pass
    s = R.stats()
    assert (s["open"], s["acknowledged"], s["collected_today"]) == (1, 1, 1), s
    assert s["high_priority"] == 0                          # the 3-day request is collected, others are not high
    got = {r["id"]: r for r in R.list_requests()}
    assert got[a]["staff_note"] == "Truck comes tomorrow morning"
    assert R.list_requests()[-1]["id"] == b                 # collected sorts last


def test_filters():
    R = fresh_store()
    a = make(R, phone="9845012345", area="indiranagar", waste="wet", address="addr one, 1st")
    b = make(R, phone="9845012345", area="koramangala", waste="dry", address="addr two, 2nd")
    c = make(R, phone="9000000001", area="koramangala", waste="e_waste", address="addr three, 3rd")
    R.update_status(c, "Collected")
    ids = lambda **k: sorted(r["id"] for r in R.list_requests(**k))  # noqa: E731
    assert ids(phone="98450 12345") == sorted([a, b])
    assert ids(area="Koramangala") == sorted([b, c])
    assert ids(waste_types=["wet", "e_waste"]) == sorted([a, c])
    assert ids(statuses=["Open"]) == sorted([a, b])
    assert ids(phone="123") == []                            # invalid phone matches nothing
    assert R.areas() == ["Indiranagar", "Koramangala"]


def test_invalid_create_raises():
    R = fresh_store()
    try:
        R.create_request("", "1", "", "", "dry", 1)
        assert False
    except ValueError:
        pass
    assert R.list_requests() == []


# ------------------------------------------------------------------ UI tests
def test_ui_household_submit_duplicate_and_track():
    R = fresh_store()
    f = FakeSt(VALID_FORM, clicks={"Submit request"})
    U = ui(f)
    U.household_page()
    assert any("REQ-0001" in t for t in f.texts("success")), f.out
    assert len(R.list_requests()) == 1
    assert any("REQ-0001" in t for t in f.texts("markdown")), "tracker should list the new request"
    # same request again -> duplicate warning, nothing new saved
    f2 = FakeSt(VALID_FORM, clicks={"Submit request"})
    U = ui(f2)
    U.household_page()
    assert f2.texts("warning") and len(R.list_requests()) == 1


def test_ui_household_validation_errors():
    R = fresh_store()
    f = FakeSt({**VALID_FORM, "Your name *": "", "Mobile number *": "123"}, clicks={"Submit request"})
    ui(f).household_page()
    assert len(f.texts("error")) == 2 and R.list_requests() == []


def test_ui_track_other_phone_shows_only_own():
    R = fresh_store()
    make(R, phone="9845012345", address="addr one, 1st")
    make(R, phone="9000000001", address="addr two, 2nd")
    f = FakeSt({("text_input", "Enter your mobile number to see your requests"): "9000000001"})
    ui(f).household_page()
    cards = [t for t in f.texts("markdown") if "REQ-" in t]
    assert len(cards) == 1 and "REQ-0002" in cards[0], cards
    f = FakeSt({("text_input", "Enter your mobile number to see your requests"): "12"})
    ui(f).household_page()
    assert f.texts("warning")


def test_ui_staff_login_view_and_update():
    R = fresh_store()
    rid = make(R, days=3)
    # wrong PIN
    f = FakeSt({"Staff PIN": "0000"}, clicks={"Open dashboard"})
    ui(f).staff_page()
    assert f.texts("error") and not f.session_state.get("staff_ok")
    # correct PIN -> rerun requested and flag set
    f = FakeSt({"Staff PIN": "1234"}, clicks={"Open dashboard"})
    try:
        ui(f).staff_page()
        assert False, "expected rerun after login"
    except Rerun:
        assert f.session_state["staff_ok"] is True
    # logged-in view shows the household's request, details and a CSV download
    f = FakeSt({})
    f.session_state["staff_ok"] = True
    ui(f).staff_page()
    names = [n for n, _ in f.out]
    assert "dataframe" in names and "download_button" in names
    assert any("9845012345" in t for t in f.texts("markdown")), "staff should see the household contact"
    # update the request
    f = FakeSt({("radio", "Status"): "Collected", ("text_input", "Note for the household (optional)"): "Cleared at 7am"},
               clicks={"Save update"})
    f.session_state["staff_ok"] = True
    try:
        ui(f).staff_page()
        assert False, "expected rerun after save"
    except Rerun:
        pass
    r = R.list_requests()[0]
    assert (r["id"], r["status"], r["staff_note"]) == (rid, "Collected", "Cleared at 7am")
    assert "REQ-0001" in f.session_state["staff_flash"]


def test_ui_staff_empty_and_logout():
    fresh_store()
    f = FakeSt({})
    f.session_state["staff_ok"] = True
    ui(f).staff_page()
    assert f.texts("info") and not [n for n, _ in f.out if n == "dataframe"]
    f = FakeSt({}, clicks={"Log out"})
    f.session_state["staff_ok"] = True
    try:
        ui(f).staff_page()
    except Rerun:
        pass
    assert f.session_state["staff_ok"] is False


def _run_app(page):
    """Execute app.py top to bottom with the fake Streamlit and the given page selected."""
    fresh_store()
    fake = FakeSt({("radio", "Go to"): page})
    sys.modules["streamlit"] = fake
    for m in ("requests_ui", "app"):
        sys.modules.pop(m, None)
    code = compile((ROOT / "app.py").read_text(encoding="utf-8"), "app.py", "exec")
    try:
        exec(code, {"__name__": "__main__"})
    except Stop:
        pass
    return fake


def test_app_routes_to_request_pages():
    f = _run_app("📣 Raise pickup request")
    assert any(n == "subheader" and "Raise a pickup request" in t for n, t in f.out)
    assert not any("Settings" in t for n, t in f.out if n == "header"), "sidebar settings must not show on request pages"
    f = _run_app("📋 Collection team")
    assert any("Collection team dashboard" in t for n, t in f.out)
    f = _run_app("♻️ Sort waste")
    assert any(n == "header" and "Settings" in t for n, t in f.out), "sort page keeps its sidebar settings"
    assert any(n == "info" for n, _ in f.out), "sort page asks for a photo"


# ------------------------------------------------------------------ runner
if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"FAIL  {name}: {e!r}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)
