"""EcoSort AI - pickup requests (data layer).

A household raises a request when waste has not been cleared from a location for a day or two;
collection staff see the requests, acknowledge them and mark them collected.
Stored in the same SQLite file as the scan history (scans.db), table `pickup_requests`.
"""
import io
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image

DB_PATH = Path(__file__).parent / "scans.db"

WASTE_LABEL = {
    "wet": "Wet waste (kitchen / food)",
    "dry": "Dry waste (plastic / paper / metal / glass)",
    "domestic_hazardous": "Domestic hazardous (batteries, bulbs, medicines)",
    "e_waste": "E-waste",
    "sanitary": "Sanitary waste (diapers, pads)",
    "mixed": "Mixed / garbage heap",
}
DAYS_LABEL = {1: "1 day", 2: "2 days", 3: "3 days or more"}
STATUSES = ["Open", "Acknowledged", "Collected"]
PRIORITY_ORDER = {"High": 0, "Medium": 1, "Normal": 2}
DUPLICATE_WINDOW_HOURS = 24


# ---------------------------------------------------------------- helpers
def _con():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS pickup_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
        name TEXT NOT NULL, phone TEXT NOT NULL,
        area TEXT NOT NULL, address TEXT NOT NULL, landmark TEXT,
        waste_type TEXT NOT NULL, days_pending INTEGER NOT NULL, notes TEXT,
        photo BLOB, status TEXT NOT NULL DEFAULT 'Open', staff_note TEXT)""")
    return con


def _now():
    return datetime.now().isoformat(timespec="seconds")


def ref(request_id):
    """Human-friendly request number, e.g. REQ-0007."""
    return f"REQ-{int(request_id):04d}"


def normalize_phone(raw):
    """Return a 10-digit Indian mobile number, or None if it is not valid."""
    digits = re.sub(r"\D", "", str(raw or ""))
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    elif len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]
    return digits if re.fullmatch(r"[6-9]\d{9}", digits) else None


def _norm_text(s):
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def priority(days_pending, waste_type):
    """High: hazardous / sanitary waste or 3+ days; Medium: 2 days; otherwise Normal."""
    if waste_type in ("domestic_hazardous", "sanitary") or int(days_pending) >= 3:
        return "High"
    return "Medium" if int(days_pending) == 2 else "Normal"


def _compress_photo(photo_bytes):
    if not photo_bytes:
        return None
    try:
        img = Image.open(io.BytesIO(photo_bytes)).convert("RGB")
    except Exception:
        return None
    img.thumbnail((800, 800))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def _row_to_dict(r):
    d = {k: r[k] for k in r.keys() if k != "photo"}
    d["ref"] = ref(d["id"])
    d["priority"] = priority(d["days_pending"], d["waste_type"])
    d["waste_label"] = WASTE_LABEL.get(d["waste_type"], d["waste_type"])
    d["has_photo"] = bool(r["photo"]) if "photo" in r.keys() else False
    d["age_hours"] = max(0.0, (datetime.now() - datetime.fromisoformat(d["created_at"])).total_seconds() / 3600)
    return d


# ---------------------------------------------------------------- validation
def validate(name, phone, area, address, waste_type, days_pending):
    """Return a list of problems (empty list = OK)."""
    errors = []
    if not str(name or "").strip():
        errors.append("Please enter your name.")
    if normalize_phone(phone) is None:
        errors.append("Please enter a valid 10-digit mobile number.")
    if not str(area or "").strip():
        errors.append("Please enter your area / ward / locality.")
    if len(str(address or "").strip()) < 5:
        errors.append("Please describe the exact location (street, house or bin) so staff can find it.")
    if waste_type not in WASTE_LABEL:
        errors.append("Please choose the type of waste.")
    if days_pending not in DAYS_LABEL:
        errors.append("Please say for how long the waste has not been cleared.")
    return errors


# ---------------------------------------------------------------- write
def find_duplicate(phone, address):
    """An unresolved request from the same phone for the same address in the last 24 hours, or None."""
    p, a = normalize_phone(phone), _norm_text(address)
    since = (datetime.now() - timedelta(hours=DUPLICATE_WINDOW_HOURS)).isoformat(timespec="seconds")
    con = _con()
    rows = con.execute("SELECT * FROM pickup_requests WHERE phone=? AND status!='Collected' AND created_at>=? "
                       "ORDER BY id DESC", (p, since)).fetchall()
    con.close()
    for r in rows:
        if _norm_text(r["address"]) == a:
            return _row_to_dict(r)
    return None


def create_request(name, phone, area, address, waste_type, days_pending, landmark="", notes="", photo_bytes=None):
    """Save a new request and return its id. Raises ValueError if the input is invalid."""
    errors = validate(name, phone, area, address, waste_type, days_pending)
    if errors:
        raise ValueError(" ".join(errors))
    now = _now()
    con = _con()
    cur = con.execute(
        "INSERT INTO pickup_requests (created_at, updated_at, name, phone, area, address, landmark, waste_type, "
        "days_pending, notes, photo, status) VALUES (?,?,?,?,?,?,?,?,?,?,?, 'Open')",
        (now, now, str(name).strip(), normalize_phone(phone), str(area).strip().title(), str(address).strip(),
         str(landmark or "").strip(), waste_type, int(days_pending), str(notes or "").strip(),
         _compress_photo(photo_bytes)))
    con.commit()
    rid = cur.lastrowid
    con.close()
    return rid


def update_status(request_id, status, staff_note=""):
    """Collection staff: set the status (Open / Acknowledged / Collected) and an optional note for the household."""
    if status not in STATUSES:
        raise ValueError(f"Status must be one of {STATUSES}")
    con = _con()
    cur = con.execute("UPDATE pickup_requests SET status=?, staff_note=?, updated_at=? WHERE id=?",
                      (status, str(staff_note or "").strip(), _now(), int(request_id)))
    con.commit()
    con.close()
    return cur.rowcount == 1


# ---------------------------------------------------------------- read
def list_requests(statuses=None, phone=None, area=None, waste_types=None):
    """Requests as dicts (no photo bytes), most urgent first: priority, then oldest first."""
    q, args = "SELECT * FROM pickup_requests WHERE 1=1", []
    if statuses:
        q += f" AND status IN ({','.join('?' * len(statuses))})"
        args += list(statuses)
    if phone is not None:
        q += " AND phone=?"
        args.append(normalize_phone(phone) or "")
    if area:
        q += " AND area=?"
        args.append(area)
    if waste_types:
        q += f" AND waste_type IN ({','.join('?' * len(waste_types))})"
        args += list(waste_types)
    con = _con()
    rows = [_row_to_dict(r) for r in con.execute(q, args).fetchall()]
    con.close()
    rows.sort(key=lambda d: (d["status"] == "Collected", PRIORITY_ORDER[d["priority"]], d["created_at"]))
    return rows


def get_photo(request_id):
    con = _con()
    r = con.execute("SELECT photo FROM pickup_requests WHERE id=?", (int(request_id),)).fetchone()
    con.close()
    return r["photo"] if r and r["photo"] else None


def areas():
    con = _con()
    out = [r["area"] for r in con.execute("SELECT DISTINCT area FROM pickup_requests ORDER BY area").fetchall()]
    con.close()
    return out


def stats():
    """Counts for the staff dashboard."""
    reqs = list_requests()
    today = datetime.now().date().isoformat()
    open_ = [r for r in reqs if r["status"] == "Open"]
    return {
        "open": len(open_),
        "acknowledged": sum(r["status"] == "Acknowledged" for r in reqs),
        "collected_today": sum(r["status"] == "Collected" and r["updated_at"].startswith(today) for r in reqs),
        "high_priority": sum(r["priority"] == "High" for r in reqs if r["status"] != "Collected"),
        "oldest_open_hours": max((r["age_hours"] for r in open_), default=0.0),
    }
