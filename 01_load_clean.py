import pandas as pd
import pyarrow
import pyarrow.parquet
import numpy as np
from sklearn.cluster import KMeans
import os
import shutil
from glob import glob
from pathlib import Path
import logging
from collections import defaultdict

# -------- PATH UNICI PER TUTTO IL PROGETTO --------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_DIR = os.path.join(BASE_DIR, "csvs")
PARQUET_DIR = os.path.join(CSV_DIR, "parquets")

# -------------------------
# Logging setup
# -------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

# Global stats across all files
GLOBAL_STATS = defaultdict(int)
# -------------------------
# Utility geometriche / cleaning
# -------------------------
def approx_km(lat1, lon1, lat2, lon2):
    """
    Distanza approssimata in km tra due punti (lat/lon in gradi).
    Sufficiente per i nostri filtri.
    """
    lat1, lon1, lat2, lon2 = map(np.asarray, (lat1, lon1, lat2, lon2))
    mean_lat = np.deg2rad((lat1 + lat2) / 2.0)
    dlat = (lat2 - lat1) * 111.0
    dlon = (lon2 - lon1) * 111.0 * np.cos(mean_lat)
    return np.sqrt(dlat ** 2 + dlon ** 2)


def track_bounding_box_diag_km(df: pd.DataFrame) -> float:
    lat_min, lat_max = df["Latitude"].min(), df["Latitude"].max()
    lon_min, lon_max = df["Longitude"].min(), df["Longitude"].max()
    return float(approx_km(lat_min, lon_min, lat_max, lon_max))


def track_start_end_disp_km(df: pd.DataFrame) -> float:
    lat1, lon1 = df["Latitude"].iloc[0], df["Longitude"].iloc[0]
    lat2, lon2 = df["Latitude"].iloc[-1], df["Longitude"].iloc[-1]
    return float(approx_km(lat1, lon1, lat2, lon2))


def remove_spatial_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rimuove punti singoli molto lontani dal "corpo" della traccia
    (errori grossolani di posizione).
    """
    if len(df) < 5:
        return df

    df = df.copy()
    lat = df["Latitude"].to_numpy()
    lon = df["Longitude"].to_numpy()

    med_lat = np.median(lat)
    med_lon = np.median(lon)

    dist = np.sqrt((lat - med_lat) ** 2 + (lon - med_lon) ** 2)

    med = np.median(dist)
    mad = np.median(np.abs(dist - med))

    # soglia robusta, con minimo "assoluto" per sicurezza (~5 km)
    if mad == 0:
        thr = med + 0.05
    else:
        thr = med + 5 * 1.4826 * mad
        thr = max(thr, 0.05)

    mask = dist <= thr
    return df[mask].reset_index(drop=True)


def is_two_point_shuttle(df: pd.DataFrame) -> bool:
    """
    Rileva tracce tipo traghetto che va avanti e indietro tra 2 punti.

    Logica:
    - KMeans con 2 cluster sulle posizioni (eventuale subsampling se moltissimi punti)
    - Quasi tutti i punti vicini (raggio piccolo) al rispettivo centro
    - Entrambi i cluster ben popolati
    - Distanza tra i due centri non enorme (traversata breve)
    """
    if len(df) < 200:
        return False   # poche osservazioni → non mi sbilancio

    coords = df[["Latitude", "Longitude"]].to_numpy()

    # Subsample se ci sono troppi punti (solo per stimare i centri)
    if len(coords) > 5000:
        idx = np.linspace(0, len(coords) - 1, 5000).astype(int)
        coords_fit = coords[idx]
    else:
        coords_fit = coords

    try:
        km = KMeans(n_clusters=2, n_init=10, random_state=0)
        km.fit(coords_fit)
        centers = km.cluster_centers_
    except Exception:
        return False

    # Assegnazione di TUTTI i punti al centro più vicino
    # (non solo quelli usati per il fit)
    diff0 = coords - centers[0]
    diff1 = coords - centers[1]
    dist0 = np.linalg.norm(diff0, axis=1)
    dist1 = np.linalg.norm(diff1, axis=1)
    labels_full = (dist1 < dist0).astype(int)

    # Distanza tra i due centri in gradi
    d_center_deg = np.linalg.norm(centers[0] - centers[1])

    # raggio massimo attorno al centro (in gradi) ~1.5 km
    radius_tol_deg = 0.015

    # Distanza di ogni punto dal proprio centro
    dist_to_center = np.where(labels_full == 0, dist0, dist1)
    frac_close = np.mean(dist_to_center < radius_tol_deg)

    counts = np.bincount(labels_full, minlength=2)
    frac_cluster = counts / len(coords)

    # soglie euristiche: la maggior parte dei punti vicini ai centri,
    # entrambi i cluster presenti, distanza tra centri non enorme (< 30 km)
    cond = (
        frac_close > 0.9
        and frac_cluster.min() > 0.2
        and d_center_deg * 111.0 < 30.0
    )

    return bool(cond)

def is_bad_track(df: pd.DataFrame) -> bool:
    """
    Decide se una traccia è "spazzatura" da buttare:
    - movimenti quasi tutti locali (porto, ancoraggio)
    - traghetti avanti/indietro tra 2 punti
    - troppo corta dopo la pulizia
    """
    if len(df) < 50:
        return True

    diag_km = track_bounding_box_diag_km(df)
    disp_km = track_start_end_disp_km(df)

    # movimenti molto locali: bounding box piccolissima
    if diag_km < 2.0:
        return True

    # tracce con percorso quasi tutto a zigzag corto:
    # grande estensione, ma start ~ end (giri in porto, pesca locale, ecc.)
    if diag_km > 5.0 and diag_km > 0 and (disp_km / diag_km) < 0.2:
        return True

    # vera e propria navetta a 2 punti
    if is_two_point_shuttle(df):
        return True

    return False


def clean_sequence_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pulisce una singola sequenza (un segmento di traccia) da outlier dinamici
    e da tracce "sporche".

    - Limiti fisici su SOG (in m/s), ROT e Heading
    - Rimozione teleport (salti di posizione impossibili)
    - Rimozione outlier spaziali isolati
    """
    if df.empty:
        return df

    df = df.copy()
    start_len = len(df)
    GLOBAL_STATS["rows_before_clean_sequence"] += start_len

    # per sicurezza assicuriamoci che sia ordinata nel tempo
    if "Timestamp" in df.columns:
        df = df.sort_values("Timestamp").reset_index(drop=True)

    # --- 1. Limiti fisici per SOG (in m/s: già convertito da nodi in ais_to_parque) ---
    if "SOG" in df.columns:
        before = len(df)
        # 0–20 m/s ≈ 0–39 nodi
        df = df[(df["SOG"] >= 0) & (df["SOG"] <= 20)]
        GLOBAL_STATS["rows_removed_SOG_limit"] += before - len(df)

    # --- 2. Limiti fisici per ROT (°/s) ---
    if "ROT" in df.columns:
        before = len(df)
        df = df[df["ROT"].abs() <= 15]
        GLOBAL_STATS["rows_removed_ROT_limit"] += before - len(df)

    # --- 3. Limiti Heading ---
    if "Heading" in df.columns:
        before = len(df)
        df = df[(df["Heading"] >= 0) & (df["Heading"] <= 360)]
        GLOBAL_STATS["rows_removed_heading_limit"] += before - len(df)

    # --- 4. Teleport: salti di posizione impossibili rispetto a dt ---
    required_cols = {"Latitude", "Longitude", "dt"}
    if required_cols.issubset(df.columns):
        lat = df["Latitude"].to_numpy()
        lon = df["Longitude"].to_numpy()
        dt = df["dt"].to_numpy()

        # limite massimo di velocità in gradi/sec (~20 m/s)
        max_deg_per_sec = 0.0002
        max_move = max_deg_per_sec * dt  # spostamento massimo ammesso tra un punto e il precedente

        # spostamento reale tra un punto e il precedente (distanza euclidea in gradi)
        dlat = np.diff(lat)
        dlon = np.diff(lon)
        dpos = np.sqrt(dlat**2 + dlon**2)

        mask = np.ones(len(df), dtype=bool)
        if len(df) > 1:
            mask[1:] = dpos <= max_move[1:]

        before = len(df)
        df = df[mask]
        GLOBAL_STATS["rows_removed_teleport"] += before - len(df)

    df = df.reset_index(drop=True)

    # --- 5. Outlier spaziali isolati (punti random fuori path) ---
    if {"Latitude", "Longitude"}.issubset(df.columns):
        before = len(df)
        df = remove_spatial_outliers(df)
        GLOBAL_STATS["rows_removed_spatial_outliers"] += before - len(df)

    # Se dopo questi filtri è troppo corta, la consideriamo spazzatura
    if len(df) < 10:
        GLOBAL_STATS["sequences_removed_too_short_after_cleaning"] += 1
        GLOBAL_STATS["rows_in_sequences_too_short_after_cleaning"] += len(df)
        return df.iloc[0:0]   # dataframe vuoto

    GLOBAL_STATS["rows_after_clean_sequence"] += len(df)
    return df


# -------------------------
# PRE-CLEANING leggero PRIMA della segmentazione
# per rimuovere punti isolati "di mezzo" che creano segmenti finti
# -------------------------
def preclean_before_segmentation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-clean leggero per MMSI, PRIMA di creare i Segment:
    - rimuove punti isolati nel tempo (dt_prev e dt_next molto grandi)
      e lontani spazialmente dal resto della traccia
    Questo evita segmenti da 1 solo punto causati da outlier in mezzo.
    """
    if df.empty:
        return df

    def _clean_mmsi(g):
        g = g.sort_values("Timestamp").reset_index(drop=True)
        if len(g) < 5:
            return g

        ts = g["Timestamp"]

        # dt precedente e successivo (in secondi)
        dt_prev = ts.diff().dt.total_seconds().fillna(np.inf).to_numpy()
        dt_next = ts.diff(-1).abs().dt.total_seconds().fillna(np.inf).to_numpy()

        # soglia "isolato": gap > 20 minuti sia prima che dopo
        iso_thr = 20 * 60
        iso_mask = (dt_prev > iso_thr) & (dt_next > iso_thr)

        # distanza dal "centro" della traccia
        lat = g["Latitude"].to_numpy()
        lon = g["Longitude"].to_numpy()
        med_lat = np.median(lat)
        med_lon = np.median(lon)
        dist = np.sqrt((lat - med_lat) ** 2 + (lon - med_lon) ** 2)

        med = np.median(dist)
        mad = np.median(np.abs(dist - med))
        if mad == 0:
            thr = med + 0.05
        else:
            thr = med + 5 * 1.4826 * mad
            thr = max(thr, 0.05)

        spatial_far = dist > thr

        to_drop = iso_mask & spatial_far
        GLOBAL_STATS["rows_removed_preclean_isolated"] += int(to_drop.sum())

        return g[~to_drop]


    before = len(df)
    df = df.groupby("MMSI", group_keys=False).apply(_clean_mmsi)
    after = len(df)
    GLOBAL_STATS["rows_removed_preclean_total"] += before - after

    return df


# -------------------------
# Funzioni originali
# -------------------------
def resample_data(df):
    # Escludiamo esplicitamente dt (che viene ricalcolato) e Segment dal set numerico
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in ("dt", "Segment")]

    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    def interpolate_group(g):
        inserted_rows = []
        for i in range(len(g) - 1):
            row_current = g.iloc[i]
            row_next = g.iloc[i + 1]
            gap = row_next["dt"]  # in secondi

            if gap > 60:
                n_intervals = int(np.ceil(gap / 60))
                step = gap / n_intervals

                for j in range(1, n_intervals):
                    fraction = j / n_intervals
                    new_row = row_current.copy()

                    # Interpolo solo i veri numerici utili (SOG, COG, ecc.)
                    for col in numeric_cols:
                        new_row[col] = (
                            row_current[col]
                            + fraction * (row_next[col] - row_current[col])
                        )

                    # Copio MMSI, Segment, Ship type, ecc.
                    for col in non_numeric_cols:
                        new_row[col] = row_current[col]

                    new_row["Timestamp"] = row_current["Timestamp"] + pd.Timedelta(
                        seconds=step * j
                    )
                    inserted_rows.append(new_row)

        if inserted_rows:
            g_new = pd.DataFrame(inserted_rows)
            g_expanded = pd.concat([g, g_new], ignore_index=True)
            g_expanded = g_expanded.sort_values(by="Timestamp").reset_index(drop=True)
            return g_expanded
        else:
            return g

    df_resampled = (
        df.groupby(["MMSI", "Segment"], group_keys=False)
        .apply(interpolate_group)
        .sort_values(["MMSI", "Segment", "Timestamp"])
        .reset_index(drop=True)
    )
    return df_resampled



def ais_to_parque(file_path, out_path):
    file_name = os.path.basename(file_path)
    logger.info(f"Inizio processamento file CSV: {file_name}")

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

    n0 = len(df)
    GLOBAL_STATS["rows_total_raw"] += n0
    GLOBAL_STATS["files_processed"] += 1

    # Remove errors
    bbox = [60, 0, 50, 20]
    north, west, south, east = bbox
    df = df[(df["Latitude"] <= north) & (df["Latitude"] >= south) & (df["Longitude"] >= west) & (
            df["Longitude"] <= east)]
    n1 = len(df)
    GLOBAL_STATS["rows_removed_bbox"] += n0 - n1

    df = df[df["Type of mobile"].isin(["Class A", "Class B"])].drop(columns=["Type of mobile"])
    n2 = len(df)
    GLOBAL_STATS["rows_removed_type_of_mobile"] += n1 - n2

    df = df[df["MMSI"].str.len() == 9]  # Adhere to MMSI format
    n3 = len(df)
    GLOBAL_STATS["rows_removed_mmsi_length"] += n2 - n3

    prefix = df["MMSI"].str[:3]
    mask_numeric = prefix.str.isnumeric()
    df = df[mask_numeric & prefix.astype(int).between(200, 775)]
    n4 = len(df)
    GLOBAL_STATS["rows_removed_mid_invalid"] += n3 - n4

    df = df.rename(columns={"# Timestamp": "Timestamp"})
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["Timestamp"])

    df = df.drop_duplicates(["Timestamp", "MMSI"], keep="first")
    n5 = len(df)
    GLOBAL_STATS["rows_removed_duplicates"] += n4 - n5

    imputation_cols = ["Latitude", "Longitude", "SOG", "COG", "ROT", "Heading"]

    def track_filter_knots(g):
        """
        Filter usato SOLO finché SOG è in nodi.
        """
        len_filt = len(g) > 256
        sog_max = g["SOG"].max()
        # 1–50 nodi: eliminiamo stazionari e casi assurdi
        sog_filt = 1 <= sog_max <= 50
        time_filt = (g["Timestamp"].max() - g["Timestamp"].min()).total_seconds() >= 60 * 60
        return len_filt and sog_filt and time_filt

    # Track filtering per MMSI (SOG in nodi)
    before_mmsi_filter = len(df)
    df = df.groupby("MMSI").filter(track_filter_knots)
    after_mmsi_filter = len(df)
    GLOBAL_STATS["rows_removed_trackfilter_mmsi"] += before_mmsi_filter - after_mmsi_filter

    # Ordino e PRE-CLEAN per rimuovere punti isolati prima di segmentare
    df = df.sort_values(['MMSI', 'Timestamp'])
    before_preclean = len(df)
    df = preclean_before_segmentation(df)
    after_preclean = len(df)
    GLOBAL_STATS["rows_after_preclean"] += after_preclean
    logger.info(
        f"{file_name}: pre-cleaning prima segmentazione, righe {before_preclean} -> {after_preclean}"
    )

    # Divide track in segmenti basati sul timegap
    df['Segment'] = df.groupby('MMSI')['Timestamp'].transform(
        lambda x: (x.diff().dt.total_seconds().fillna(0) >= 15 * 60).cumsum()
    )  # Max allowed timegap

    # Segment filtering (SOG ancora in nodi)
    before_segment_filter = len(df)
    df = df.groupby(["MMSI", "Segment"]).filter(track_filter_knots)
    after_segment_filter = len(df)
    GLOBAL_STATS["rows_removed_trackfilter_segment"] += before_segment_filter - after_segment_filter

    df = df.reset_index(drop=True)

    # ORA converto SOG in m/s (solo dopo aver finito di usare le soglie in nodi)
    knots_to_ms = 0.514444
    df["SOG"] = knots_to_ms * df["SOG"]

    # Calcolo Time Difference (dt) in seconds (primo round)
    df['dt'] = df.groupby(['MMSI', 'Segment'])['Timestamp'].diff().dt.total_seconds().fillna(0)
    num_mmsi = df["MMSI"].nunique()
    logger.info(f"{file_name}: MMSI unici dopo primi filtri: {num_mmsi}")

    # -------------------------
    # cleaning per segmento PRIMA dell'interpolazione
    # -------------------------
    def _clean_group(g):
        GLOBAL_STATS["num_segments_before_cleaning"] += 1
        g_clean = clean_sequence_df(g)
        if g_clean.empty:
            GLOBAL_STATS["num_segments_removed_clean_sequence"] += 1
            GLOBAL_STATS["rows_removed_clean_sequence"] += len(g)
            return g_clean
        if is_bad_track(g_clean):
            GLOBAL_STATS["num_segments_removed_bad_track"] += 1
            GLOBAL_STATS["rows_in_bad_tracks"] += len(g_clean)
            return g_clean.iloc[0:0]
        GLOBAL_STATS["num_segments_after_cleaning"] += 1
        return g_clean

    before_cleaning = len(df)
    df = (
        df.groupby(["MMSI", "Segment"], group_keys=False)
          .apply(_clean_group)
          .reset_index(drop=True)
    )
    after_cleaning = len(df)
    GLOBAL_STATS["rows_removed_cleaning_total"] += before_cleaning - after_cleaning

    if df.empty:
        logger.warning(f"{file_name}: tutti i segmenti scartati dopo il cleaning, nessun parquet scritto.")
        return

    # *** Ricalcolo dt DOPO il cleaning ***
    df['dt'] = df.groupby(['MMSI', 'Segment'])['Timestamp'] \
                 .diff().dt.total_seconds().fillna(0)

    # -------------------------
    # Interpolazione / resampling
    # -------------------------
    df = resample_data(df)

    for col in imputation_cols:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df.groupby(['MMSI', 'Segment'])[col].transform(
            lambda x: x.interpolate().ffill().bfill()
        )
        df[col] = df[col].fillna(0)

    df["Minute"] = df["Timestamp"].dt.floor("min")  # Keep one measurement from each minute
    before_minute_dedup = len(df)
    df = df.drop_duplicates(subset=["Minute", "MMSI"], keep="first")
    after_minute_dedup = len(df)
    GLOBAL_STATS["rows_removed_minute_dedup"] += before_minute_dedup - after_minute_dedup
    df = df.drop(columns=["Minute"])

    # recalculate the delta time <- ensure it's < 60
    df = df.drop(columns=["dt"])
    df['dt'] = df.groupby(['MMSI', 'Segment'])['Timestamp'].diff().dt.total_seconds().fillna(0)

    # Velocity Components (Requires SOG in m/s)
    df['COG_rad'] = np.deg2rad(df['COG'])
    df['Velocity_N'] = df['SOG'] * np.cos(df['COG_rad'])  # North component (Latitude direction)
    df['Velocity_E'] = df['SOG'] * np.sin(df['COG_rad'])  # East component (Longitude direction)
    df.drop(columns=['COG_rad'], inplace=True)

    df = merge_stationary_into_df(df)

    final_rows = len(df)
    GLOBAL_STATS["rows_final"] += final_rows

    logger.info(
        f"{file_name}: righe iniziali={n0}, bbox-> {n1}, tipo_mobile-> {n2}, "
        f"mmsi_len-> {n3}, MID-> {n4}, dedup-> {n5}, dopo trackfilter_mmsi-> {after_mmsi_filter}, "
        f"dopo preclean-> {after_preclean}, dopo trackfilter_segment-> {after_segment_filter}, "
        f"dopo cleaning-> {after_cleaning}, dopo minute_dedup-> {after_minute_dedup}, finali-> {final_rows}"
    )

    table = pyarrow.Table.from_pandas(df, preserve_index=False)
    pyarrow.parquet.write_to_dataset(
        table,
        root_path=out_path,
        partition_cols=["MMSI",
                        "Segment",
                        ]
    )


def extract_stationary_data(file_path):
    stationary_cols = ["Ship type", "Cargo type", "Width", "Length", "Draught"]
    parquet_files = glob(os.path.join(file_path, "**", "*.parquet"), recursive=True)

    if not parquet_files:
        print("No parquet files found.")
        return

    for file in parquet_files:
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

        parquet_files = list(mmsi_folder.rglob("*.parquet"))

        file_pairs = {}
        for pf in parquet_files:
            name = pf.stem.replace("_stationary", "")
            file_pairs.setdefault(name, []).append(pf)

        for i, (basename, files) in enumerate(sorted(file_pairs.items()), start=1):
            segment_folder = mmsi_folder / f"Segment={i}"
            segment_folder.mkdir(exist_ok=True)

            for f in files:
                dest = segment_folder / f.name
                shutil.move(str(f), dest)

        for old_segment in mmsi_folder.glob("Segment=*"):
            if not any(old_segment.iterdir()):
                old_segment.rmdir()


def get_folder_size(path):
    path = Path(path)
    total = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
    print(f"Size: {total / (1024 ** 2):.2f} MB")


def merge_stationary_into_df(df):
    """
    Extract stationary ship properties for each MMSI and
    merge them back into the original dataframe.
    """

    stationary_cols = ["Ship type", "Cargo type", "Width", "Length", "Draught"]

    out = []

    for mmsi, g in df.groupby("MMSI"):
        ship_info = {}

        for col in stationary_cols:
            if col not in g.columns:
                ship_info[col] = None
                continue

            valid = g[col][
                g[col].notnull() &
                (g[col] != 0) &
                (g[col].astype(str).str.lower() != "undefined")
            ]

            ship_info[col] = valid.iloc[0] if not valid.empty else None

        stat_df = pd.DataFrame([ship_info])
        stat_df["MMSI"] = mmsi
        out.append(stat_df)

    stationary_table = pd.concat(out, ignore_index=True)

    df = df.drop(columns=stationary_cols, errors="ignore")
    df = df.merge(stationary_table, on="MMSI", how="left")

    return df


def count_parquet_pairs(root_path):
    root_path = Path(root_path)
    total_pairs = 0
    details = {}

    for mmsi_folder in root_path.glob("MMSI=*"):
        if not mmsi_folder.is_dir():
            continue

        parquet_files = list(mmsi_folder.rglob("*.parquet"))

        file_pairs = {}
        for f in parquet_files:
            base = f.stem.replace("_stationary", "")
            file_pairs.setdefault(base, []).append(f)

        pair_count = sum(1 for files in file_pairs.values() if len(files) == 2)
        total_pairs += pair_count
        details[mmsi_folder.name] = pair_count

    print("Pair count by MMSI folder:")
    for k, v in details.items():
        print(f"  {k}: {v} pairs")

    print(f"\nTotal parquet pairs across all MMSI folders: {total_pairs}")
    return total_pairs


if __name__ == "__main__":
    ais_folder_path = CSV_DIR
    parquet_folder_path = PARQUET_DIR
    os.makedirs(parquet_folder_path, exist_ok=True)

    for filename in os.listdir(ais_folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(ais_folder_path, filename)
            ais_to_parque(file_path, parquet_folder_path)

    logger.info("Parqueting done")

    logger.info("Riepilogo globale dei filtri applicati:")
    for k in sorted(GLOBAL_STATS.keys()):
        logger.info(f"  {k}: {GLOBAL_STATS[k]}")
