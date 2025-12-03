import os
import json
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

ROOT_DIR = os.path.join(os.path.dirname(__file__), '../data/csvs/parquets/')
OUTPUT_INFO_FILE = os.path.join(ROOT_DIR, "normalization_info.json")

def is_valid_parquet_file(file_path: str) -> bool:
    """Check if a parquet file should be normalized."""
    fname = os.path.basename(file_path).lower()
    if not fname.endswith(".parquet"):
        return False
    # skip stationary files (postfix rule)
    if fname.endswith("_stationary.parquet"):
        return False
    return True

def get_all_parquet_files(root_dir: str):
    """Recursively collect valid parquet file paths."""
    parquet_files = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            path = os.path.join(root, f)
            if is_valid_parquet_file(path):
                parquet_files.append(path)
    return parquet_files

def compute_global_min_max(parquet_files):
    """Compute global min/max for all numeric columns."""
    global_min, global_max = {}, {}

    for file in tqdm(parquet_files, desc="Computing global min/max"):
        try:
            table = pq.read_table(file)
            df = table.to_pandas()
            num_df = df.select_dtypes(include="number")

            for col in num_df.columns:
                cmin = num_df[col].min(skipna=True)
                cmax = num_df[col].max(skipna=True)
                if pd.notna(cmin):
                    global_min[col] = min(global_min.get(col, cmin), cmin)
                if pd.notna(cmax):
                    global_max[col] = max(global_max.get(col, cmax), cmax)
        except Exception as e:
            print(f"⚠️ Skipping {file} (error: {e})")

    return global_min, global_max

def normalize_value(x, min_val, max_val):
    if max_val == min_val:
        return 0.0
    return (x - min_val) / (max_val - min_val)

def normalize_parquet_files(parquet_files, global_min, global_max):
    """Normalize numerical columns in all parquet files (in place)."""
    for file in tqdm(parquet_files, desc="Normalizing parquet files"):
        try:
            table = pq.read_table(file)
            df = table.to_pandas()
            num_cols = df.select_dtypes(include="number").columns

            for col in num_cols:
                if col in global_min and col in global_max:
                    df[col] = df[col].apply(normalize_value,
                                            args=(global_min[col], global_max[col]))

            df.to_parquet(file, index=False)
        except Exception as e:
            print(f"⚠️ Failed to normalize {file}: {e}")

def main():
    parquet_files = get_all_parquet_files(ROOT_DIR)
    print(f"Found {len(parquet_files)} parquet files to normalize under {ROOT_DIR}")

    global_min, global_max = compute_global_min_max(parquet_files)
    print(f"Computed normalization stats for {len(global_min)} numerical columns")

    normalize_parquet_files(parquet_files, global_min, global_max)

    normalization_info = {
        "root_directory": ROOT_DIR,
        "num_files": len(parquet_files),
        "columns": {
            col: {"min": float(global_min[col]), "max": float(global_max[col])}
            for col in sorted(global_min.keys())
        }
    }

    with open(OUTPUT_INFO_FILE, "w") as f:
        json.dump(normalization_info, f, indent=4)

    print("\n✅ Normalization completed successfully.")
    print(f"📄 Info saved at: {OUTPUT_INFO_FILE}")

if __name__ == "__main__":
    main()
