import pandas as pd
import pyarrow
import pyarrow.parquet
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import os
import shutil
from glob import glob
from pathlib import Path

n = 0

def resample_data(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

    def interpolate_group(g):
        global n
        n = n + 1
        print(f"We are at the {n}th MMSI")
        inserted_rows = []
        for i in range(len(g) - 1):
            row_current = g.iloc[i]
            row_next = g.iloc[i + 1]
            gap = row_next['dt']
            if gap > 60:
                n_intervals = int(np.ceil(gap / 60))
                step = gap / n_intervals

                for j in range(1, n_intervals):
                    fraction = j / n_intervals
                    new_row = row_current.copy()

                    for col in numeric_cols:
                        new_row[col] = row_current[col] + fraction * (row_next[col] - row_current[col])

                    for col in non_numeric_cols:
                        new_row[col] = row_current[col]

                    new_row['Timestamp'] = row_current['Timestamp'] + pd.Timedelta(seconds=step * j)
                    inserted_rows.append(new_row)

        if inserted_rows:
            g_new = pd.DataFrame(inserted_rows)
            g_expanded = pd.concat([g, g_new], ignore_index=True)
            g_expanded = g_expanded.sort_values(by='Timestamp').reset_index(drop=True)
            return g_expanded
        else:
            return g

    # Apply per MMSI + Segment
    df_resampled = (
        df.groupby(['MMSI', 'Segment'], group_keys=False)
          .apply(interpolate_group)
          .sort_values(['MMSI', 'Segment', 'Timestamp'])
          .reset_index(drop=True)
    )
    return df_resampled

def ais_to_parque(file_path, out_path):
    dtypes = {
        "MMSI": "object",
        "SOG": float,
        "COG": float,
        "Longitude": float,
        "Latitude": float,
        "# Timestamp": "object",
        "Type of mobile": "object",
        "Width": float,
        "Length": float,
        "Navigational status": "object",
        "ROT": float,
        "Heading": float,
        "Ship type": "object",
        "Cargo type": "object",
        "Draught": float
    }
    usecols = list(dtypes.keys())
    df = pd.read_csv(file_path, usecols=usecols, dtype=dtypes)

    # Remove errors
    bbox = [60, 0, 50, 20]
    north, west, south, east = bbox
    df = df[(df["Latitude"] <= north) & (df["Latitude"] >= south) & (df["Longitude"] >= west) & (
            df["Longitude"] <= east)]

    df = df[df["Type of mobile"].isin(["Class A", "Class B"])].drop(columns=["Type of mobile"])
    df = df[df["MMSI"].str.len() == 9]  # Adhere to MMSI format
    df = df[df["MMSI"].str[:3].astype(int).between(200, 775)]  # Adhere to MID standard

    df = df.rename(columns={"# Timestamp": "Timestamp"})
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")

    df = df.drop_duplicates(["Timestamp", "MMSI"], keep="first")

    # Group by MMSI and Segment, then fill NaNs for dynamic features using ffill
    # NaN values for features like SOG, COG, ROT, Heading, Draught, Width, Length 
    # should be filled with the last known value within the track segment.
    imputation_cols = ["Latitude", "Longitude", "SOG", "COG", "ROT", "Heading"]

    def track_filter(g):
        len_filt = len(g) > 256  # Min required length of track/segment
        sog_filt = 1 <= g["SOG"].max() <= 50  # Remove stationary tracks/segments
        time_filt = (g["Timestamp"].max() - g["Timestamp"].min()).total_seconds() >= 60 * 60  # Min required timespan
        return len_filt and sog_filt and time_filt

    # Track filtering
    df = df.groupby("MMSI").filter(track_filter)
    df = df.sort_values(['MMSI', 'Timestamp'])

    # Divide track into segments based on timegap
    df['Segment'] = df.groupby('MMSI')['Timestamp'].transform(
        lambda x: (x.diff().dt.total_seconds().fillna(0) >= 15 * 60).cumsum())  # Max allowed timegap

    #transform
    knots_to_ms = 0.514444
    df["SOG"] = knots_to_ms * df["SOG"]

    # Segment filtering
    df = df.groupby(["MMSI", "Segment"]).filter(track_filter)
    df = df.reset_index(drop=True)

    # Calculate Time Difference (dt) in seconds
    df['dt'] = df.groupby(['MMSI', 'Segment'])['Timestamp'].diff().dt.total_seconds().fillna(0)
    num_mmsi = df["MMSI"].nunique()
    print(f"Unique MMSIs: {num_mmsi}")
    df = resample_data(df)

    for col in imputation_cols:
        # Replace zeros with NaN so they can be interpolated
        df[col] = df[col].replace(0, np.nan)
        
        # Interpolate per group
        df[col] = df.groupby(['MMSI', 'Segment'])[col].transform(
            lambda x: x.interpolate().ffill().bfill()
        )
        
        # Replace any remaining NaNs with 0
        df[col] = df[col].fillna(0)

    df["Minute"] = df["Timestamp"].dt.floor("min")          # Keep one measurement from each minute
    df = df.drop_duplicates(subset=["Minute", "MMSI"], keep="first")
    df = df.drop(columns=["Minute"])

    # recalculate the delta time <- ensure it's < 60
    df = df.drop(columns=["dt"])
    df['dt'] = df.groupby(['MMSI', 'Segment'])['Timestamp'].diff().dt.total_seconds().fillna(0)
    
    # Calculate Lat/Lon Change (dLat, dLon)
    df['dLat'] = df.groupby(['MMSI', 'Segment'])['Latitude'].diff().fillna(0)
    df['dLon'] = df.groupby(['MMSI', 'Segment'])['Longitude'].diff().fillna(0)

    # Velocity Components (Requires SOG in m/s)
    # Note: COG is usually 0-360 degrees. Convert to radians for sin/cos.
    df['COG_rad'] = np.deg2rad(df['COG'])
    df['Velocity_N'] = df['SOG'] * np.cos(df['COG_rad']) # North component (Latitude direction)
    df['Velocity_E'] = df['SOG'] * np.sin(df['COG_rad']) # East component (Longitude direction)
    df.drop(columns=['COG_rad'], inplace=True)

    table = pyarrow.Table.from_pandas(df, preserve_index=False)
    pyarrow.parquet.write_to_dataset(
        table,
        root_path=out_path,
        partition_cols=["MMSI",  # "Date",
                        "Segment",  # "Geocell"
                        ]
    )

def extract_stationary_data(file_path):
    stationary_cols = ["Ship type", "Cargo type", "Width", "Length", "Draught"]
    parquet_files = glob(os.path.join(file_path, "**", "*.parquet"), recursive=True)

    if not parquet_files:
        print("No parquet files found.")
        return

    for file in parquet_files:
        #print(f"Processing: {file}")
        try:
            df = pd.read_parquet(file)
        except Exception as e:
            print(f"Failed to read {file}: {e}")
            continue

        ship_info = {}
        for col in stationary_cols:
            if col in df.columns:
                valid_values = df[col][
                    (df[col].notnull()) &
                    (df[col] != 0) &
                    (df[col].astype(str).str.lower() != "undefined")
                ]
                ship_info[col] = valid_values.iloc[0] if not valid_values.empty else None
            else:
                ship_info[col] = 0

        stationary_df = pd.DataFrame([ship_info])
        stationary_file = os.path.splitext(file)[0] + "_stationary.parquet"

        try:
            stationary_df.to_parquet(stationary_file, index=False)
           # print(f"Written: {stationary_file}")
        except Exception as e:
            print(f"Error writing {stationary_file}: {e}")
            continue

        df = df.drop(columns=[c for c in stationary_cols if c in df.columns], errors="ignore")
        try:
            df.to_parquet(file, index=False)
        except Exception as e:
            print(f"Error overwriting {file}: {e}")

def delete_stationary_data(file_path):
    deleted_count = 0

    # Use glob to find matching files recursively
    pattern = os.path.join(file_path, '**', '*stationary*.parquet')
    for parquet_file in glob(pattern, recursive=True):
        os.remove(parquet_file)
        deleted_count += 1

    print(f"\nTotal files deleted: {deleted_count}")

def reorganize_parquet_pairs(root_path):
    root_path = Path(root_path)

    for mmsi_folder in root_path.glob("MMSI=*"):
        if not mmsi_folder.is_dir():
            continue
        
        print(f"Processing {mmsi_folder.name}...")
        
        # Collect all parquet files under this MMSI folder (recursively)
        parquet_files = list(mmsi_folder.rglob("*.parquet"))
        
        # Build a dict of base name -> both files (normal + stationary)
        file_pairs = {}
        for pf in parquet_files:
            name = pf.stem.replace("_stationary", "")
            file_pairs.setdefault(name, []).append(pf)
        
        # Create new Segment folders for each pair
        for i, (basename, files) in enumerate(sorted(file_pairs.items()), start=1):
            segment_folder = mmsi_folder / f"Segment={i}"
            segment_folder.mkdir(exist_ok=True)
            
            for f in files:
                dest = segment_folder / f.name
                # Move the file to the new segment folder
                shutil.move(str(f), dest)
        
        # Clean up: remove old segment folders if they’re empty
        for old_segment in mmsi_folder.glob("Segment=*"):
            if not any(old_segment.iterdir()):
                old_segment.rmdir()
    
    print("✅ Reorganization complete.")

def get_folder_size(path):
    path = Path(path)
    total = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
    print(f"Size: {total / (1024 ** 2):.2f} MB")

def count_parquet_pairs(root_path):
    root_path = Path(root_path)
    total_pairs = 0
    details = {}

    for mmsi_folder in root_path.glob("MMSI=*"):
        if not mmsi_folder.is_dir():
            continue
        
        # Find all parquet files under this MMSI folder (recursively)
        parquet_files = list(mmsi_folder.rglob("*.parquet"))
        
        # Group by base name (ignoring _stationary)
        file_pairs = {}
        for f in parquet_files:
            base = f.stem.replace("_stationary", "")
            file_pairs.setdefault(base, []).append(f)
        
        # Count valid pairs (both normal + stationary exist)
        pair_count = sum(1 for files in file_pairs.values() if len(files) == 2)
        total_pairs += pair_count
        details[mmsi_folder.name] = pair_count

    print("Pair count by MMSI folder:")
    for k, v in details.items():
        print(f"  {k}: {v} pairs")

    print(f"\nTotal parquet pairs across all MMSI folders: {total_pairs}")
    return total_pairs

if __name__ == "__main__":
    ais_folder_path = os.path.join(os.path.dirname(__file__), '../data/csvs/')
    parquet_folder_path = os.path.join(ais_folder_path, 'parquets/')
    os.makedirs(parquet_folder_path, exist_ok=True)

    for filename in os.listdir(ais_folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(ais_folder_path, filename)
            ais_to_parque(file_path, parquet_folder_path)

    delete_stationary_data(parquet_folder_path)
    extract_stationary_data(parquet_folder_path)

    reorganize_parquet_pairs(parquet_folder_path)

    #test MMSI=538009531
