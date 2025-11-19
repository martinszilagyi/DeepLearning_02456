import os
import shutil
import pandas as pd

def reorganize_parquet_files():
    root_folder = os.path.join(os.getcwd(), "csvs", "parquets")

    for mmsi_folder in os.listdir(root_folder):
        mmsi_path = os.path.join(root_folder, mmsi_folder)
        if not os.path.isdir(mmsi_path):
            continue
        
        # Collect all parquet files and their original parent folders
        parquet_files = []
        for segment_folder in os.listdir(mmsi_path):
            segment_path = os.path.join(mmsi_path, segment_folder)
            if not os.path.isdir(segment_path):
                continue
            for file in os.listdir(segment_path):
                if file.endswith('.parquet'):
                    parquet_files.append((os.path.join(segment_path, file), segment_path))
        
        # Create new segment folders and move files first
        for i, (parquet_file, _) in enumerate(parquet_files, start=1):
            new_segment_path = os.path.join(mmsi_path, f'segment={i}')
            os.makedirs(new_segment_path, exist_ok=True)
            shutil.move(parquet_file, os.path.join(new_segment_path, os.path.basename(parquet_file)))
        
        # Now remove old segment folders if empty
        for segment_folder in os.listdir(mmsi_path):
            segment_path = os.path.join(mmsi_path, segment_folder)
            if os.path.isdir(segment_path):
                try:
                    os.rmdir(segment_path)  # only removes if empty
                except OSError:
                    pass  # folder not empty, or other error

    print("Reorganization complete.")

def gather_unique_values(root_folder):
    unique_values = {}
    for dirpath, _, filenames in os.walk(root_folder):
        for file in filenames:
            if file.endswith('.parquet'):
                path = os.path.join(dirpath, file)
                df = pd.read_parquet(path)
                if 'Timestamp' in df.columns:
                    df = df.drop(columns=['Timestamp'])
                non_numeric_cols = df.select_dtypes(exclude='number').columns
                for col in non_numeric_cols:
                    if col not in unique_values:
                        unique_values[col] = set()
                    unique_values[col].update(df[col].dropna().unique())
    return unique_values

def create_enum_maps(unique_values):
    enum_maps = {}
    for col, values in unique_values.items():
        sorted_values = sorted(values)  # optional: sort for consistency
        enum_maps[col] = {v: i for i, v in enumerate(sorted_values)}
    return enum_maps

def apply_enumeration(root_folder, enum_maps):
    for dirpath, _, filenames in os.walk(root_folder):
        for file in filenames:
            if file.endswith('.parquet'):
                path = os.path.join(dirpath, file)
                df = pd.read_parquet(path)
                if 'Timestamp' in df.columns:
                    df = df.drop(columns=['Timestamp'])
                for col, mapping in enum_maps.items():
                    if col in df.columns:
                        df[f'{col}_enum'] = df[col].map(mapping).fillna(-1).astype(int)  # -1 for unknowns
                        orig = df.pop(col)
                        df[col] = orig
                df.to_parquet(path)

if __name__ == "__main__":
    reorganize_parquet_files()
    root_folder = os.path.join(os.getcwd(), "csvs", "parquets")
    unique_values = gather_unique_values(root_folder)
    enum_maps = create_enum_maps(unique_values)
    apply_enumeration(root_folder, enum_maps)
