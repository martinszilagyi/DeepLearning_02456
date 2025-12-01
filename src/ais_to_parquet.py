import pandas as pd
import pyarrow
import pyarrow.parquet
import numpy as np
import os
from tqdm import tqdm
tqdm.pandas()
pd.set_option('future.no_silent_downcasting', True)

def filter_outliers(df, distance_threshold=0.2):
    df = df.copy()

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        dist_prev = np.hypot(
            curr['Latitude'] - prev['Latitude'],
            curr['Longitude'] - prev['Longitude']
        )

        if dist_prev > distance_threshold:
            df.at[i, 'Latitude'] = prev['Latitude']
            df.at[i, 'Longitude'] = prev['Longitude']

    return df

def custom_resample(df, max_gap=60, distance_threshold=0.1):
    df = df.copy()
    new_rows = []

    for i in range(len(df) - 1):
        current_row = df.iloc[i]
        next_row = df.iloc[i + 1]

        new_rows.append(current_row)

        gap = next_row['dt']
        if gap > max_gap:
            n_missing = int(np.floor(gap / max_gap))
            for j in range(1, n_missing + 1):
                new_time = current_row['Timestamp'] + pd.Timedelta(seconds=max_gap * j)
                new_row = current_row.copy()
                new_row['Timestamp'] = new_time

                for col in df.columns:
                    if col not in ['Timestamp', 'dt']:
                        new_row[col] = np.nan

                new_rows.append(new_row)

    new_rows.append(df.iloc[-1])
    df_expanded = pd.DataFrame(new_rows)
    df_expanded = df_expanded.sort_values('Timestamp').reset_index(drop=True)

    return df_expanded

def ais_to_parquet(file_path, out_path):
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
    # Read all cols raw
    df = pd.read_csv(file_path, usecols=usecols, dtype=dtypes)

    # Filter by ROI
    bbox = [60, 0, 50, 20]
    north, west, south, east = bbox
    df = df[(df["Latitude"] <= north) & (df["Latitude"] >= south) & (df["Longitude"] >= west) & (df["Longitude"] <= east)]

    # Filter by Type of mobile and MMSI range
    df = df[df["Type of mobile"].isin(["Class A", "Class B"])].drop(columns=["Type of mobile"])
    df = df[df["MMSI"].str.len() == 9]
    df = df[df["MMSI"].str[:3].astype(int).between(200, 775)]

    # Rename and convert Timestamp
    df = df.rename(columns={"# Timestamp": "Timestamp"})
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    df = df.drop_duplicates(["Timestamp", "MMSI", ], keep="first")

    def track_filter(g):
        len_filt = len(g) > 256  # Min required length of track/segment
        sog_filt = 1 <= g["SOG"].max() <= 50  # Remove stationary tracks/segments
        time_filt = (g["Timestamp"].max() - g["Timestamp"].min()).total_seconds() >= 60 * 60  # Min required timespan
        return len_filt and sog_filt and time_filt

    # Keep only vessel tracks (grouped by MMSI) that are long enough, span at least 1 hour,
    # and have realistic speeds, using the custom track_filter() criteria
    df = df.groupby("MMSI").filter(track_filter)
    df = df.sort_values(['MMSI', 'Timestamp'])

    # Split segments by 10-minute gap (600 seconds)
    df['Segment'] = df.groupby('MMSI')['Timestamp'].transform(lambda x: (x.diff().dt.total_seconds().fillna(0) >= 15 * 60).cumsum())

    #convert SOG from knots to m/s
    knots_to_ms = 0.514444
    df["SOG"] = knots_to_ms * df["SOG"]

    # Apply track_filter again on segments
    df = df.groupby(["MMSI", "Segment"]).filter(track_filter)
    df = df.reset_index(drop=True)

    # Calculate time differences
    df['dt'] = df.groupby(['MMSI', 'Segment'])['Timestamp'].diff().dt.total_seconds().fillna(0)

    results = []

    # Resample each segment to have one sample per minute
    for (mmsi, segment), group_df in tqdm(df.groupby(['MMSI', 'Segment']), total=df.groupby(['MMSI', 'Segment']).ngroups):
        group_df = group_df.sort_values('Timestamp').reset_index(drop=True)
        numeric_cols = group_df.select_dtypes(include=[np.number]).columns
        non_numeric_cols = group_df.select_dtypes(exclude=[np.number]).columns

        g_resampled = filter_outliers(group_df, distance_threshold=0.1)
        g_resampled = custom_resample(g_resampled)
        g_resampled[numeric_cols] = g_resampled[numeric_cols].interpolate(method='linear', limit_direction='both')
        g_resampled[non_numeric_cols] = g_resampled[non_numeric_cols].ffill().bfill()

        g_resampled['MMSI'] = mmsi
        g_resampled['Segment'] = segment
        results.append(g_resampled)

    df_resampled = pd.concat(results, ignore_index=True)

    # Impute remaining NaNs in specific columns
    imputation_cols = ["SOG", "COG", "ROT", "Heading"]
    for col in imputation_cols:
        df_resampled[col] = df_resampled.groupby(['MMSI', 'Segment'])[col].transform(lambda x: x.interpolate().ffill().bfill())
        df_resampled[col] = df_resampled[col].fillna(0)

    latlon_cols = ["Latitude", "Longitude"]
    for col in latlon_cols:
        df_resampled[col] = df_resampled.groupby(['MMSI', 'Segment'])[col].transform(lambda x: x.ffill().bfill())

    # Remove duplicates within the same minute
    df_resampled["Minute"] = df_resampled["Timestamp"].dt.floor("min")
    df_resampled = df_resampled.drop_duplicates(subset=["Minute", "MMSI"], keep="first")
    df_resampled = df_resampled.drop(columns=["Minute"])

    # Final feature engineering
    df_resampled = df_resampled.drop(columns=["dt"])
    df_resampled['dt'] = df_resampled.groupby(['MMSI', 'Segment'])['Timestamp'].diff().dt.total_seconds().fillna(0)
    df_resampled['COG_rad'] = np.deg2rad(df_resampled['COG'])
    df_resampled['sin_cog'] = np.sin(df_resampled['COG_rad'])
    df_resampled['cos_cog'] = np.cos(df_resampled['COG_rad'])
    df_resampled = df_resampled.drop(columns=['COG_rad', 'COG'])

    # Write to parquet files
    #df_resampled["Timestamp"] = df_resampled["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S") #FOR DEBUG ONLY!!!
    table = pyarrow.Table.from_pandas(df_resampled, preserve_index=False)
    pyarrow.parquet.write_to_dataset(
        table,
        root_path=out_path,
        partition_cols=["MMSI", "Segment"]
    )

    print(f"Converted {os.path.basename(file_path)} to Parquet format.")

if __name__ == "__main__":
    ais_folder_path = os.path.join(os.path.dirname(__file__), '../csvs/')
    print(f"Processing AIS CSV files from {ais_folder_path}")
    parquet_folder_path = os.path.join(ais_folder_path, 'parquets/')
    print(f"Saving Parquet files to {parquet_folder_path}")
    os.makedirs(parquet_folder_path, exist_ok=True)

    for filename in os.listdir(ais_folder_path):
        print(f"Found file: {filename}")
        if filename.endswith('.csv'):
            file_path = os.path.join(ais_folder_path, filename)
            ais_to_parquet(file_path, parquet_folder_path)

    print("Parquet files created successfully")
# 636092635