import os
import json
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
import numpy as np

# Configuration
ROOT_DIR = os.path.join(os.path.dirname(__file__), '../csvs/parquets/')
OUTPUT_INFO_FILE = os.path.join(ROOT_DIR, "normalization_info.json")

# Checks if a file is a valid parquet file
def is_valid_parquet_file(file_path: str) -> bool:
    fname = os.path.basename(file_path).lower()
    if not fname.endswith(".parquet"):
        return False
    return True

# Retrieves all parquet files under the root directory
def get_all_parquet_files(root_dir: str):
    parquet_files = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            path = os.path.join(root, f)
            if is_valid_parquet_file(path):
                parquet_files.append(path)
    return parquet_files

# Converts latitude and longitude to meters relative to a reference point (We don't know if this is really necessary; just in case)
def latlon_to_meters(lat, lon, lat0, lon0):
    # Earth radius in meters
    R = 6371000  
    dlat = np.deg2rad(lat - lat0)
    dlon = np.deg2rad(lon - lon0)
    lat0_rad = np.deg2rad(lat0)

    # Haversine formula components
    x = R * dlon * np.cos(lat0_rad)
    y = R * dlat
    return x, y

# Computes global means and standard deviations for numeric columns across ALL parquet files
def compute_global_stats(parquet_files):
    sums = {}
    sumsqs = {}
    counts = {}
    lat_min, lat_max = float('inf'), float('-inf')
    lon_min, lon_max = float('inf'), float('-inf')

    # First pass: collect all lat/lon intervals
    for file in tqdm(parquet_files, desc="Determining lat/lon bounds"):
        try:
            table = pq.read_table(file)
            df = table.to_pandas()
            # get global min/max lat/lon
            lat_min = min(lat_min, df['Latitude'].min())
            lat_max = max(lat_max, df['Latitude'].max())
            lon_min = min(lon_min, df['Longitude'].min())
            lon_max = max(lon_max, df['Longitude'].max())
        except Exception as e:
            print(f"Skipping {file} (error: {e})")

    # Reference point (center of bounding box)
    lat0 = (lat_min + lat_max) / 2
    lon0 = (lon_min + lon_max) / 2

    # Compute new columns and statistics
    for file in tqdm(parquet_files, desc="Computing global stats"):
        try:
            table = pq.read_table(file)
            df = table.to_pandas()

            # Create new columns for lat/lon in meters
            x_m, y_m = latlon_to_meters(df['Latitude'].values, df['Longitude'].values, lat0, lon0)
            df['x_meters'] = x_m
            df['y_meters'] = y_m

            # New columns for velocity and acceleration
            df = df.sort_values(['MMSI', 'Segment', 'Timestamp'])
            df['dt'] = df.groupby(['MMSI', 'Segment'], observed=False)['Timestamp'].diff().dt.total_seconds()
            df['vx'] = df.groupby(['MMSI', 'Segment'], observed=False)['x_meters'].diff() / df['dt']
            df['vy'] = df.groupby(['MMSI', 'Segment'], observed=False)['y_meters'].diff() / df['dt']
            df['ax'] = df.groupby(['MMSI', 'Segment'], observed=False)['vx'].diff() / df['dt']
            df['ay'] = df.groupby(['MMSI', 'Segment'], observed=False)['vy'].diff() / df['dt']

            # Fill NaNs (first rows per segment) with 0 for velocities and accelerations
            df[['vx', 'vy', 'ax', 'ay']] = df[['vx', 'vy', 'ax', 'ay']].fillna(0)

            # Accumulate sums, sumsqs, counts for numeric columns
            num_df = df.select_dtypes(include="number")
            for col in num_df.columns:
                # Drop NaNs for accurate stats
                vals = num_df[col].dropna()
                # Calculate sum, sum of squares, and count for the column
                csum = vals.sum()
                csumsq = (vals**2).sum()
                ccount = len(vals)

                # Accumulate sums, sumsqs, counts for numeric columns
                sums[col] = sums.get(col, 0.0) + csum
                sumsqs[col] = sumsqs.get(col, 0.0) + csumsq
                counts[col] = counts.get(col, 0) + ccount

        except Exception as e:
            print(f"Skipping {file} (error: {e})")

    # Calculate means and standard deviations (on a global scale)
    means = {}
    stds = {}
    for col in sums:
        mean = sums[col] / counts[col]
        var = (sumsqs[col] / counts[col]) - (mean ** 2)
        std = np.sqrt(var) if var > 0 else 1.0
        means[col] = mean
        stds[col] = std

    # return with means, stds for all numeric columns, and reference lat0, lon0
    return means, stds, lat0, lon0

# Normalizes numeric columns with z-score and adds meters, velocity, acceleration
def normalize_parquet_files(parquet_files, means, stds, lat0, lon0):
    for file in tqdm(parquet_files, desc="Normalizing parquet files"):
        try:
            table = pq.read_table(file)
            #print(f"types of cols in file {file}: {table.schema}")
            df = table.to_pandas()

            # Add lat/lon in meters columns (relative to lat0, lon0)
            x_m, y_m = latlon_to_meters(df['Latitude'].values, df['Longitude'].values, lat0, lon0)
            df['x_meters'] = x_m
            df['y_meters'] = y_m

            # Sort by MMSI, Segment, Timestamp to compute velocities/accelerations
            df = df.sort_values(['MMSI', 'Segment', 'Timestamp'])

            # Calculate velocity components (m/s) using diff of meters / dt
            df['dt'] = df.groupby(['MMSI', 'Segment'], observed=False)['Timestamp'].diff().dt.total_seconds()
            df['vx'] = df.groupby(['MMSI', 'Segment'], observed=False)['x_meters'].diff() / df['dt']
            df['vy'] = df.groupby(['MMSI', 'Segment'], observed=False)['y_meters'].diff() / df['dt']

            # Calculate acceleration components (m/s²) as diff of velocity / dt
            df['ax'] = df.groupby(['MMSI', 'Segment'], observed=False)['vx'].diff() / df['dt']
            df['ay'] = df.groupby(['MMSI', 'Segment'], observed=False)['vy'].diff() / df['dt']

            # Fill NaNs (first rows per segment) with 0 for velocities and accelerations
            df[['vx', 'vy', 'ax', 'ay']] = df[['vx', 'vy', 'ax', 'ay']].fillna(0)

            # Normalize all numeric columns except dt
            num_cols = df.select_dtypes(include="number").columns.drop('dt', errors='ignore')
            for col in num_cols:
                if col in means and col in stds:
                    # Check for non-numeric before coercion
                    non_numeric = df[col].apply(lambda x: not pd.api.types.is_number(x)).sum()
                    if non_numeric > 0:
                        print(f"Column {col} in file {file} has {non_numeric} non-numeric entries, coercing to NaN.")
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(means[col])
                    std_val = stds[col] if stds[col] > 0 else 1.0
                    df[col] = (df[col] - means[col]) / std_val

            df = df.drop(columns=['dt'])

            df.to_parquet(file, index=False)
        except Exception as e:
            print(f"Failed to normalize {file}: {e}")

# Main function
def main():
    # Get all parquet files
    parquet_files = get_all_parquet_files(ROOT_DIR)
    print(f"Found {len(parquet_files)} parquet files to normalize under {ROOT_DIR}")

    # Compute global means and standard deviations
    means, stds, lat0, lon0 = compute_global_stats(parquet_files)
    print(f"Computed means/stds for {len(means)} numerical columns")
    print(f"Reference lat/lon for meters conversion: {lat0:.6f}, {lon0:.6f}")
    print(f"all means: {means}")
    print(f"all stds: {stds}")

    # Normalize parquet files
    normalize_parquet_files(parquet_files, means, stds, lat0, lon0)

    # Save normalization info to JSON for future visualization use
    normalization_info = {
        "root_directory": ROOT_DIR,
        "num_files": len(parquet_files),
        "columns": {
            col: {"mean": float(means[col]), "std": float(stds[col])}
            for col in sorted(means.keys())
        },
        "reference_point": {"lat0": lat0, "lon0": lon0}
    }

    # Save to JSON file
    with open(OUTPUT_INFO_FILE, "w") as f:
        json.dump(normalization_info, f, indent=4)

    print("\nZ-score normalization and feature augmentation completed successfully.")
    print(f"Info saved at: {OUTPUT_INFO_FILE}")

# Main execution
if __name__ == "__main__":
    main()
