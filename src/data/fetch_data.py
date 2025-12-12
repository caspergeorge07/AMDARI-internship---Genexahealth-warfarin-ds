import os
import json
from pathlib import Path
import requests

BASE_URL = "https://genexahealth.onrender.com/api/v1/"
USERNAME = "clinician"
PASSWORD = "clinical123"

OUT_RAW = Path("data/raw")
OUT_CURATED = Path("data/curated")

OUT_RAW.mkdir(parents=True, exist_ok=True)
OUT_CURATED.mkdir(parents=True, exist_ok=True)

# 
# Define endpoints mapping
ENDPOINTS = {
    "genomics": ("genomics", OUT_RAW / "genomics.json"),
    "clinical": ("clinical", OUT_RAW / "clinical.json"),
    "lifestyle": ("lifestyle", OUT_RAW / "lifestyle.json"),
    "outcomes": ("outcomes", OUT_RAW / "outcomes.json"),
    "patient_ids": ("patient_ids", OUT_RAW / "patient_ids.json"),
}



def fetch(endpoint: str):
    url = BASE_URL.rstrip('/') + '/' + endpoint.lstrip('/')
    r = requests.get(url, auth=(USERNAME, PASSWORD), timeout=60)
    r.raise_for_status()
    # Try to infer file type
    ctype = r.headers.get('content-type', '').lower()
    if 'application/json' in ctype:
        return r.json(), 'json'
    return r.content, 'bin'

def save_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')

def save_bytes(path: Path, b: bytes):
    path.write_bytes(b)

def main():
    if not ENDPOINTS:
        raise SystemExit('Update ENDPOINTS after checking /docs (add endpoint paths + output files).')
    for name, (endpoint, outpath) in ENDPOINTS.items():
        print(f'Fetching {name}: {endpoint}')
        data, kind = fetch(endpoint)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        if kind == 'json':
            save_json(outpath, data)
        else:
            save_bytes(outpath, data)
        print(f'Saved -> {outpath}')

if __name__ == '__main__':
    main()
