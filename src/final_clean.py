import os
import shutil
import pandas as pd

# Reorganize Parquet Files into Segment Folders (Because more paths can be added to the same MMSI/Segment folder by previous steps)
def reorganize_parquet_files():
    root_folder = os.path.join(os.getcwd(), "csvs", "parquets")

    # Iterate over each MMSI folder
    for mmsi_folder in os.listdir(root_folder):
        mmsi_path = os.path.join(root_folder, mmsi_folder)
        if not os.path.isdir(mmsi_path):
            continue
        
        # Collect all parquet files grouped by original segment folder
        parquet_files = []
        for segment_folder in os.listdir(mmsi_path):
            segment_path = os.path.join(mmsi_path, segment_folder)
            if not os.path.isdir(segment_path):
                continue
            for file in os.listdir(segment_path):
                if file.endswith('.parquet'):
                    parquet_files.append((os.path.join(segment_path, file), segment_path))
        
        # Create segment folders once per MMSI and move files accordingly
        segment_folder_created = set()
        for i, (parquet_file, _) in enumerate(parquet_files, start=1):
            new_segment_path = os.path.join(mmsi_path, f'segment={i}')
            if new_segment_path not in segment_folder_created:
                os.makedirs(new_segment_path, exist_ok=True)
                segment_folder_created.add(new_segment_path)
            shutil.move(parquet_file, os.path.join(new_segment_path, os.path.basename(parquet_file)))
        
        # Remove old segment folders if empty
        for segment_folder in os.listdir(mmsi_path):
            segment_path = os.path.join(mmsi_path, segment_folder)
            if os.path.isdir(segment_path):
                try:
                    os.rmdir(segment_path)  # only removes if empty
                except OSError:
                    pass  # folder not empty or other error

    print("Reorganization complete.")

# Gather unique non-numeric values from all Parquet files (to create enumeration mappings)
def gather_unique_values(root_folder):
    unique_values = {}
    for dirpath, _, filenames in os.walk(root_folder):
        for file in filenames:
            if file.endswith('.parquet'):
                path = os.path.join(dirpath, file)
                try:
                    df = pd.read_parquet(path)
                except Exception as e:
                    print(f"Warning: Failed to read {path}: {e}")
                    continue
                if 'Timestamp' in df.columns:
                    df = df.drop(columns=['Timestamp'])
                # Collect unique non-numeric values
                non_numeric_cols = df.select_dtypes(exclude='number').columns
                for col in non_numeric_cols:
                    if col not in unique_values:
                        unique_values[col] = set()
                    unique_values[col].update(df[col].dropna().unique())
    return unique_values

# Create enumeration mappings for each non-numeric column
def create_enum_maps(unique_values):
    enum_maps = {}
    for col, values in unique_values.items():
        sorted_values = sorted(values)
        enum_maps[col] = {v: i for i, v in enumerate(sorted_values)}
    return enum_maps

# Apply enumeration mappings to all Parquet files (in new cols)
def apply_enumeration(root_folder, enum_maps):
    for dirpath, _, filenames in os.walk(root_folder):
        for file in filenames:
            if file.endswith('.parquet'):
                path = os.path.join(dirpath, file)
                try:
                    df = pd.read_parquet(path)
                except Exception as e:
                    print(f"Warning: Failed to read {path}: {e}")
                    continue
                if 'Timestamp' in df.columns:
                    df = df.drop(columns=['Timestamp'])
                for col, mapping in enum_maps.items():
                    if col in df.columns:
                        df[f'{col}_enum'] = df[col].map(mapping).fillna(-1).astype(int)
                        orig = df.pop(col)
                        df[col] = orig
                try:
                    df.to_parquet(path)
                except Exception as e:
                    print(f"Warning: Failed to write {path}: {e}")

# Main execution
if __name__ == "__main__":
    reorganize_parquet_files()
    root_folder = os.path.join(os.getcwd(), "csvs", "parquets")
    unique_values = gather_unique_values(root_folder)
    enum_maps = create_enum_maps(unique_values)
    apply_enumeration(root_folder, enum_maps)
    