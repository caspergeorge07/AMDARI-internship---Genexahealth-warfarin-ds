import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

BASE_URL = os.getenv("BASE_URL").rstrip("/")
TOKEN = os.getenv("ACCESS_TOKEN")

if not BASE_URL or not TOKEN:
    raise SystemExit("Missing BASE_URL or ACCESS_TOKEN in .env")

HEADERS = {"Authorization": f"Bearer {TOKEN}"}

OUT = Path("data/raw/genomics.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

LIMIT = 1000   # swagger says max 1000
offset = 0
all_rows = []

while True:
    url = f"{BASE_URL}/genomics/"
    params = {"limit": LIMIT, "offset": offset}
    r = requests.get(url, headers=HEADERS, params=params, timeout=60)

    if r.status_code == 401:
        raise SystemExit("401 Unauthorized: token expired. Update ACCESS_TOKEN in .env and rerun.")
    if r.status_code != 200:
        raise SystemExit(f"Failed ({r.status_code}): {r.text}")

    data = r.json()

    # if API returns a list of records
    if not data:
        break

    if isinstance(data, dict) and "data" in data:
        data = data["data"]

    all_rows.extend(data)
    print(f"Fetched {len(data)} records at offset={offset}")

    if len(data) < LIMIT:
        break

    offset += LIMIT
    time.sleep(0.2)  # gentle pacing

df = pd.DataFrame(all_rows)
df.to_csv(OUT, index=False)
print(f"✅ Saved {OUT} | rows={df.shape[0]} cols={df.shape[1]}")