"""Add a few sample pickup requests so the staff dashboard is not empty during a demo.

Usage:  python seed_demo_requests.py          (run once; run it again to add the samples again)
"""
from datetime import datetime, timedelta

import requests_store as R

SAMPLES = [
    # name, phone, area, address, landmark, waste_type, days, notes, hours_ago, status, staff_note
    ("Asha Rao", "9845012345", "Indiranagar", "12, 3rd Cross - heap beside the green bin", "Opposite the temple", "mixed", 2,
     "Bin overflowing, stray dogs spreading it.", 40, "Open", ""),
    ("Ravi Kumar", "9900112233", "Koramangala", "5th Block, Lane 4 - dry waste bags at the gate", "", "dry", 1,
     "", 20, "Open", ""),
    ("Meena S", "9741098765", "Jayanagar", "7th Main, near the park - used diapers in a bag", "Next to the bakery", "sanitary", 1,
     "Please collect today.", 12, "Acknowledged", "Truck will come between 4 and 6 pm."),
    ("Imran Pasha", "9886044556", "Indiranagar", "Block C, ground floor - old batteries and CFL bulbs box", "", "domestic_hazardous", 3,
     "", 80, "Open", ""),
    ("Latha M", "9448123456", "Koramangala", "80 Feet Road - kitchen waste not picked up", "Near the milk booth", "wet", 2,
     "", 50, "Collected", "Collected this morning, sorry for the delay."),
]

for name, phone, area, addr, landmark, wt, days, notes, hours_ago, status, note in SAMPLES:
    rid = R.create_request(name, phone, area, addr, wt, days, landmark, notes)
    stamp = (datetime.now() - timedelta(hours=hours_ago)).isoformat(timespec="seconds")
    con = R._con()                      # backdate so the demo shows realistic ages
    con.execute("UPDATE pickup_requests SET created_at=? WHERE id=?", (stamp, rid))
    con.commit()
    con.close()
    if status != "Open":
        R.update_status(rid, status, note)
print(f"Added {len(SAMPLES)} sample requests. Staff PIN is 1234 unless you set STAFF_PIN.")
