import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load CSVs
patients = pd.read_csv(RAW_DIR / "patient_ids.csv")
genomics = pd.read_csv(RAW_DIR / "genomics.csv")
clinical = pd.read_csv(RAW_DIR / "clinical.csv")
lifestyle = pd.read_csv(RAW_DIR / "lifestyle.csv")
outcomes = pd.read_csv(RAW_DIR / "outcomes.csv")

def standardise_patient_id(df: pd.DataFrame) -> pd.DataFrame:
    # normalise column names for matching
    col_map = {c.lower(): c for c in df.columns}

    if "patient_id" in col_map:
        df = df.rename(columns={col_map["patient_id"]: "patient_id"})
    elif "id" in col_map:
        df = df.rename(columns={col_map["id"]: "patient_id"})
    else:
        raise ValueError(
            f"No patient_id/id column found in columns: {list(df.columns)}"
        )

    df["patient_id"] = df["patient_id"].astype(str)
    return df

patients = standardise_patient_id(patients)
genomics = standardise_patient_id(genomics)
clinical = standardise_patient_id(clinical)
lifestyle = standardise_patient_id(lifestyle)
outcomes = standardise_patient_id(outcomes)

# Merge: start from patient list to keep all patient_ids
merged = patients.merge(genomics, on="patient_id", how="left", suffixes=("", "_gen"))
merged = merged.merge(clinical, on="patient_id", how="left", suffixes=("", "_clin"))
merged = merged.merge(lifestyle, on="patient_id", how="left", suffixes=("", "_life"))
merged = merged.merge(outcomes, on="patient_id", how="left", suffixes=("", "_out"))

# Save merged dataset
out_file = OUT_DIR / "merged_patient_dataset.csv"
merged.to_csv(out_file, index=False)

print(f"✅ Saved: {out_file}")
print(f"📊 Final shape: {merged.shape}")

# Quick QA checks
print("🔎 Unique patient_ids:", merged["patient_id"].nunique())