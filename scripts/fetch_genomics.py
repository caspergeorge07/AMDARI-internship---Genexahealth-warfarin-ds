import requests
import pandas as pd
from pathlib import Path

# ---------------------------
# API CONFIG
# ---------------------------
BASE_URL = "https://genexahealth.onrender.com/api/v1"
ENDPOINT = "/genomics"

USERNAME = "clinician"
PASSWORD = "clinical123"

# ---------------------------
# OUTPUT PATH
# ---------------------------
OUTPUT_DIR = Path("../data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "genomics.csv"

# ---------------------------
# FETCH DATA
# ---------------------------
def fetch_genomics_data():
    response = requests.get(
        f"{BASE_URL}{ENDPOINT}",
        auth=(USERNAME, PASSWORD)
    )

    if response.status_code != 200:
        raise Exception(f"API call failed: {response.status_code} - {response.text}")

    return response.json()

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    data = fetch_genomics_data()

    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ Genomics data saved to {OUTPUT_FILE}")
    print(f"📊 Rows: {df.shape[0]}, Columns: {df.shape[1]}")