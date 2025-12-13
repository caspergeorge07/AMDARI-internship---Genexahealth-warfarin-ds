import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
import sys

load_dotenv()

BASE_URL = os.getenv("BASE_URL").rstrip("/")
TOKEN = os.getenv("ACCESS_TOKEN")

if not BASE_URL or not TOKEN:
    raise SystemExit("Missing BASE_URL or ACCESS_TOKEN in .env")

HEADERS = {"Authorization": f"Bearer {TOKEN}"}

def fetch_paginated(endpoint: str, outfile: str, limit: int = 1000, sleep_s: float = 0.15):
    endpoint = endpoint.strip("/")
    outpath = Path(outfile)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    offset = 0
    all_rows = []

    while True:
        url = f"{BASE_URL}/{endpoint}/"
        params = {"limit": limit, "offset": offset}
        r = requests.get(url, headers=HEADERS, params=params, timeout=60)

        if r.status_code == 401:
            raise SystemExit("401 Unauthorized: token expired. Update ACCESS_TOKEN in .env and rerun.")
        if r.status_code != 200:
            raise SystemExit(f"Failed ({r.status_code}) on {endpoint}: {r.text}")

        data = r.json()

        # handle possible shapes
        if isinstance(data, dict) and "data" in data:
            data = data["data"]

        if not data:
            break

        all_rows.extend(data)
        print(f"Fetched {len(data)} records at offset={offset}")

        if len(data) < limit:
            break

        offset += limit
        time.sleep(sleep_s)

    df = pd.DataFrame(all_rows)
    df.to_csv(outpath, index=False)
    print(f"✅ Saved {outpath} | rows={df.shape[0]} cols={df.shape[1]}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/fetch_table.py <endpoint> <outfile>")
        print("Example: python scripts/fetch_table.py clinical data/raw/clinical.csv")
        raise SystemExit(1)

    fetch_paginated(sys.argv[1], sys.argv[2])