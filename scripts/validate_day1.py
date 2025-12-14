import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
files = ["patient_ids.csv", "genomics.csv", "clinical.csv", "lifestyle.csv", "outcomes.csv"]

print("=== Day 1 Validation ===")
for f in files:
    p = RAW / f
    df = pd.read_csv(p)
    print(f"{f}: shape={df.shape}")

# Schema check: patient id column presence (case-insensitive)
def has_pid(df):
    cols = [c.lower() for c in df.columns]
    return ("patient_id" in cols) or ("patientid" in cols) or ("patient_id".replace("_","") in cols) or ("id" in cols)

for f in files:
    df = pd.read_csv(RAW / f, nrows=5)
    print(f"{f}: patient id column ok? {has_pid(df)}")

print("✅ Validation complete")