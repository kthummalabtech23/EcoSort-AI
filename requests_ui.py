"""EcoSort AI - pickup request pages (Streamlit).

household_page(): a household raises a pickup request and tracks its own requests by mobile number.
staff_page():     collection staff (PIN protected) see all households' requests, filter them,
                  download a route sheet and mark requests Acknowledged / Collected.

Prototype note: the staff PIN is a simple shared PIN (env var STAFF_PIN, default 1234) and households are
identified by mobile number only. A real deployment needs proper login (e.g. OTP for households, accounts for staff).
"""
import csv
import io
import os

import streamlit as st

import requests_store as R

STATUS_COLOR = {"Open": "#c62828", "Acknowledged": "#ef6c00", "Collected": "#2e7d32"}
PRIORITY_COLOR = {"High": "#c62828", "Medium": "#ef6c00", "Normal": "#546e7a"}


def _pill(text, color):
    return (f"<span style='background:{color};color:white;padding:2px 10px;border-radius:12px;"
            f"font-weight:600;font-size:0.8rem'>{text}</span>")


def _dur(hours):
    if hours < 1:
        return "under 1 hour"
    if hours < 48:
        return f"{int(hours)} h"
    return f"{int(hours // 24)} days"


def _staff_pin():
    return os.environ.get("STAFF_PIN", "1234")


def _location(r):
    return r["address"] + (f" (near {r['landmark']})" if r.get("landmark") else "")


def _request_card(r, staff=False):
    with st.container(border=True):
        st.markdown(f"**{r['ref']}** &nbsp; {_pill(r['status'], STATUS_COLOR[r['status']])} &nbsp; "
                    f"{_pill(r['priority'] + ' priority', PRIORITY_COLOR[r['priority']])}", unsafe_allow_html=True)
        st.markdown(f"📍 **{r['area']}**: {_location(r)}")
        st.caption(f"{r['waste_label']} · not cleared for {R.DAYS_LABEL[r['days_pending']]} · raised {_dur(r['age_hours'])} ago")
        if staff:
            st.markdown(f"👤 {r['name']} · 📞 {r['phone']}")
        if r.get("notes"):
            st.markdown(f"📝 {r['notes']}")
        if r.get("staff_note"):
            st.info(f"Collection team: {r['staff_note']}")
        if r.get("has_photo"):
            photo = R.get_photo(r["id"])
            if photo:
                st.image(photo, width=260)


# ---------------------------------------------------------------- households
def household_page():
    st.subheader("📣 Raise a pickup request")
    st.caption("Waste not cleared for a day or two? Tell the collection team exactly where to come.")

    with st.form("raise_request"):
        c1, c2 = st.columns(2)
        name = c1.text_input("Your name *")
        phone = c2.text_input("Mobile number *", max_chars=15, placeholder="10-digit number",
                              help="Staff may call you; you also use it to track your requests.")
        area = st.text_input("Area / ward / locality *", placeholder="e.g. Indiranagar, Ward 80")
        address = st.text_input("Exact location *", placeholder="e.g. 12, 3rd Cross - heap beside the green bin at the corner")
        landmark = st.text_input("Landmark (optional)", placeholder="e.g. opposite the temple")
        c3, c4 = st.columns(2)
        waste_type = c3.selectbox("What was not collected? *", list(R.WASTE_LABEL), format_func=R.WASTE_LABEL.get)
        days = c4.selectbox("Not cleared for *", list(R.DAYS_LABEL), format_func=R.DAYS_LABEL.get)
        notes = st.text_area("Anything else staff should know (optional)", max_chars=300)
        photo = st.file_uploader("Photo of the location (optional)", type=["jpg", "jpeg", "png", "webp"])
        submitted = st.form_submit_button("Submit request", type="primary", use_container_width=True)

    if submitted:
        errors = R.validate(name, phone, area, address, waste_type, days)
        if errors:
            for e in errors:
                st.error(e)
        else:
            dup = R.find_duplicate(phone, address)
            if dup:
                st.warning(f"You already raised **{dup['ref']}** for this location in the last 24 hours. "
                           f"Its status is **{dup['status']}**, so the team can already see it. "
                           "Track it below instead of sending a new one.")
                st.session_state["track_phone"] = phone
            else:
                rid = R.create_request(name, phone, area, address, waste_type, days, landmark, notes,
                                       photo.getvalue() if photo else None)
                st.success(f"Request **{R.ref(rid)}** sent to the collection team. "
                           "You can follow its status below using your mobile number.")
                st.session_state["track_phone"] = phone

    st.divider()
    st.subheader("📋 Track your requests")
    phone_q = st.text_input("Enter your mobile number to see your requests", key="track_phone")
    if phone_q:
        if R.normalize_phone(phone_q) is None:
            st.warning("Enter a valid 10-digit mobile number.")
        else:
            mine = R.list_requests(phone=phone_q)
            if not mine:
                st.info("No requests found for this number.")
            for r in mine:
                _request_card(r)


# ---------------------------------------------------------------- collection staff
def staff_page():
    st.subheader("📋 Collection team dashboard")

    if not st.session_state.get("staff_ok"):
        st.caption("For municipal / collection staff only.")
        with st.form("staff_login"):
            pin = st.text_input("Staff PIN", type="password")
            ok = st.form_submit_button("Open dashboard", type="primary")
        if ok:
            if pin == _staff_pin():
                st.session_state["staff_ok"] = True
                st.rerun()
            else:
                st.error("Wrong PIN.")
        return

    if st.button("Log out"):
        st.session_state["staff_ok"] = False
        st.rerun()
    flash = st.session_state.pop("staff_flash", None)
    if flash:
        st.success(flash)

    s = R.stats()
    m = st.columns(5)
    m[0].metric("Open", s["open"])
    m[1].metric("Acknowledged", s["acknowledged"])
    m[2].metric("Collected today", s["collected_today"])
    m[3].metric("High priority", s["high_priority"])
    m[4].metric("Oldest open", _dur(s["oldest_open_hours"]) if s["open"] else "-")

    f1, f2, f3 = st.columns(3)
    statuses = f1.multiselect("Status", R.STATUSES, default=["Open", "Acknowledged"])
    area = f2.selectbox("Area", ["All areas"] + R.areas())
    wtypes = f3.multiselect("Waste type", list(R.WASTE_LABEL), format_func=R.WASTE_LABEL.get)
    reqs = R.list_requests(statuses or None, area=None if area == "All areas" else area, waste_types=wtypes or None)
    if not reqs:
        st.info("No requests match these filters.")
        return

    rows = [{"Request": r["ref"], "Priority": r["priority"], "Status": r["status"], "Area": r["area"],
             "Location": _location(r), "Waste": r["waste_label"], "Not cleared for": R.DAYS_LABEL[r["days_pending"]],
             "Raised (ago)": _dur(r["age_hours"]), "Contact": f"{r['name']} · {r['phone']}",
             "Notes": r.get("notes") or ""} for r in reqs]
    st.dataframe(rows, use_container_width=True, hide_index=True)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    st.download_button("⬇️ Download list as CSV (route sheet)", buf.getvalue(), file_name="pickup_requests.csv",
                       mime="text/csv")

    st.divider()
    st.markdown("#### Update a request")
    labels = {r["id"]: f"{r['ref']} · {r['area']} · {R.DAYS_LABEL[r['days_pending']]} · {r['status']}" for r in reqs}
    rid = st.selectbox("Choose a request", list(labels), format_func=labels.get)
    r = next(x for x in reqs if x["id"] == rid)
    _request_card(r, staff=True)
    with st.form(f"update_{rid}"):
        status = st.radio("Status", R.STATUSES, index=R.STATUSES.index(r["status"]), horizontal=True, key=f"status_{rid}")
        note = st.text_input("Note for the household (optional)", value=r.get("staff_note") or "", key=f"note_{rid}")
        save = st.form_submit_button("Save update", type="primary")
    if save:
        R.update_status(rid, status, note)
        st.session_state["staff_flash"] = f"{r['ref']} is now {status}."
        st.rerun()
