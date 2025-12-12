from pathlib import Path
import json
import pandas as pd

DATA_DIRS = [Path("data/raw"), Path("data/curated")]
REPORT_PATH = Path("docs/data/schema_report.md")

def profile_df(df: pd.DataFrame) -> str:
    rows = []
    rows.append(f"- rows: **{len(df):,}**")
    rows.append(f"- cols: **{df.shape[1]}**")
    miss = (df.isna().mean() * 100).sort_values(ascending=False)
    top_miss = miss.head(15)
    rows.append("- top missingness (%):")
    rows.append("")
    rows.append("| column | missing % | dtype |")
    rows.append("|---|---:|---|")
    for c in top_miss.index:
        rows.append(f"| {c} | {top_miss[c]:.2f} | {df[c].dtype} |")
    return "\n".join(rows)

def load_any(path: Path):
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in [".parquet"]:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        # handle list-of-records
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        # handle dict with "data"
        if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
            return pd.DataFrame(obj["data"])
        # last resort
        return pd.json_normalize(obj)
    return None

def main():
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    parts = ["# Schema & Data Quality Report\n"]
    for d in DATA_DIRS:
        if not d.exists():
            continue
        for f in sorted(d.rglob("*")):
            if f.is_dir():
                continue
            df = load_any(f)
            if df is None or df.empty:
                continue
            parts.append(f"\n## {f.as_posix()}\n")
            parts.append(profile_df(df))
    REPORT_PATH.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")

if __name__ == "__main__":
    main()
