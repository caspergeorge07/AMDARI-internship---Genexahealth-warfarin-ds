import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("BASE_URL")
TOKEN = os.getenv("ACCESS_TOKEN")

if not TOKEN:
    raise SystemExit("❌ ACCESS_TOKEN missing from .env")

def fetch_patient_ids():
    url = f"{BASE_URL}/patient_ids"
    headers = {"Authorization": f"Bearer {TOKEN}"}
    r = requests.get(url, headers=headers, timeout=120)
    r.raise_for_status()
    return r.json()

if __name__ == "__main__":
    ids_raw = fetch_patient_ids()

# Case 1: API returns a list -> keep as-is
if isinstance(ids_raw, list):
    ids = ids_raw

# Case 2: API returns a single string "id1,id2,id3" -> split
elif isinstance(ids_raw, str):
    ids = [x.strip() for x in ids_raw.split(",") if x.strip()]

# Case 3: API returns {"patient_ids": ...}
elif isinstance(ids_raw, dict) and "patient_ids" in ids_raw:
    v = ids_raw["patient_ids"]
    if isinstance(v, list):
        ids = v
    else:
        ids = [x.strip() for x in str(v).split(",") if x.strip()]
else:
    ids = [x.strip() for x in str(ids_raw).split(",") if x.strip()]

os.makedirs("data/raw", exist_ok=True)
pd.DataFrame({"patient_id": ids}).to_csv("data/raw/patient_ids.csv", index=False)
print(f"✅ Saved {len(ids)} patient IDs to data/raw/patient_ids.csv")