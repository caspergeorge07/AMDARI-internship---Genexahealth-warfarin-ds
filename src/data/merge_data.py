import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
CURATED_DIR = Path("data/curated")
OUTPUT_PATH = CURATED_DIR / "merged_dataset.csv"

def load_data():
    """Load JSON and CSV files from RAW_DIR into a dict of DataFrames."""
    dataframes = {}
    if not RAW_DIR.exists():
        print(f"Raw directory {RAW_DIR} does not exist")
        return dataframes
    for file in RAW_DIR.iterdir():
        if file.suffix.lower() == ".json":
            try:
                df = pd.read_json(file)
            except ValueError:
                continue
        elif file.suffix.lower() == ".csv":
            df = pd.read_csv(file)
        else:
            continue
        dataframes[file.stem] = df
    return dataframes

def merge_data(dfs):
    """Merge multiple DataFrames on a common key (e.g., patient_id)."""
    if not dfs:
        return pd.DataFrame()
    from functools import reduce
    # Take the list of dataframes and reduce with outer join on 'patient_id'
    dfs_list = list(dfs.values())
    merged = reduce(lambda left, right: pd.merge(left, right, on="patient_id", how="outer"), dfs_list)
    return merged

def main():
    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    dfs = load_data()
    merged_df = merge_data(dfs)
    if merged_df.empty:
        print("No data to merge")
        return
    merged_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved merged dataset to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
