import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import warnings
from time import time

warnings.filterwarnings("ignore", category=DeprecationWarning)

# -------- PATH UNICI PER TUTTO IL PROGETTO --------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_DIR = os.path.join(BASE_DIR, "csvs")
PARQUET_DIR = "/zhome/c7/d/220114/DeepLearning_02456/csvs/parquets"

pd.set_option('future.no_silent_downcasting', True)

# =========================
# HYPERPARAMETERS
# =========================
EPOCHS = 120
BATCH_SIZE = 64
LEARNING_RATE = 5e-4

INPUT_RATIO = 0.6
FORECAST_STEPS = 35

ENCODER_HIDDEN_SIZE = 256
LSTM_LAYERS = 3

NUM_DYNAMIC_FEATURES = 10
OUTPUT_SIZE = 2

NUM_STATIC_NUMERIC = 5
# =========================

# =========================
# FIUMI / CANALI (IN GRADI)
# =========================
RIVER_BOXES_DEG = [
    # Limfjorden Ovest
    {
        "name": "Limfjorden_West",
        "lat_min": 56.55,
        "lat_max": 56.78,
        "lon_min": 8.07,
        "lon_max": 8.88,
    },
    # Limfjorden Est
    {
        "name": "Limfjorden_East",
        "lat_min": 56.85,
        "lat_max": 57.10,
        "lon_min": 9.35,
        "lon_max": 10.45,
    },
    # Kiel Canal (a sud della penisola)
    {
        "name": "Kiel_Canal",
        "lat_min": 53.85,  # poco a sud di Brunsbüttel
        "lat_max": 54.35,  # poco a nord di Kiel
        "lon_min": 9.00,   # ingresso ovest sul lato Elba
        "lon_max": 10.20,  # ingresso est a Kiel-Holtenau
    },
]

RIVER_OVERSAMPLE_FACTOR = 4  # quante volte in più vogliamo vederle

# =========================
# NORMALIZATION INFO (parquets)
# =========================
def load_normalization_info_from_parquets():
    """
    Legge normalization_info.json dalla cartella dei parquets
    e restituisce i min/max originali di lat/lon.
    """
    info_path = os.path.join(PARQUET_DIR, "normalization_info.json")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"normalization_info.json non trovato in {info_path}")

    with open(info_path, "r") as f:
        info = json.load(f)

    lat_min = info["columns"]["Latitude"]["min"]
    lat_max = info["columns"]["Latitude"]["max"]
    lon_min = info["columns"]["Longitude"]["min"]
    lon_max = info["columns"]["Longitude"]["max"]

    return lat_min, lat_max, lon_min, lon_max

LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = load_normalization_info_from_parquets()

def to_norm_lat(lat_deg: float) -> float:
    return (lat_deg - LAT_MIN) / (LAT_MAX - LAT_MIN)

def to_norm_lon(lon_deg: float) -> float:
    return (lon_deg - LON_MIN) / (LON_MAX - LON_MIN)


# =========================
# BOX NORMALIZZATI (stesso sistema dei parquets)
# =========================
RIVER_BOXES = []
for box in RIVER_BOXES_DEG:
    RIVER_BOXES.append({
        "name": box["name"],
        "lat_min": to_norm_lat(box["lat_min"]),
        "lat_max": to_norm_lat(box["lat_max"]),
        "lon_min": to_norm_lon(box["lon_min"]),
        "lon_max": to_norm_lon(box["lon_max"]),
    })


def is_river_sequence(df: pd.DataFrame) -> bool:
    """
    Ritorna True se una sequenza è per buona parte dentro
    uno dei box (in coordinate NORMALIZZATE).
    """
    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        return False

    lat = df["Latitude"].values   # già normalizzate
    lon = df["Longitude"].values

    for box in RIVER_BOXES:
        mask = (
            (lat >= box["lat_min"]) & (lat <= box["lat_max"]) &
            (lon >= box["lon_min"]) & (lon <= box["lon_max"])
        )
        # se almeno il 30% dei punti è dentro, lo consideriamo "fiume"
        if mask.mean() > 0.3:
            return True

    return False

num_ship_types = 0
num_cargo_types = 0

class Seq2SeqTrajectoryModel(nn.Module):
    """
    Encoder LSTM → Static Encoder → LSTM Decoder autoregressivo.
    Predice i futuri (lat, lon) step-by-step, imponendo più smoothness.
    """
    def __init__(self,
                 num_dynamic_features=10,
                 cat_static_cardinalities=[None, None],
                 cat_static_emb_dims=[8, 8],
                 num_static_numeric=5,
                 encoder_hidden_size=128,
                 lstm_layers=2,
                 output_size=2,
                 forecast_steps=50):

        super().__init__()

        self.forecast_steps = forecast_steps
        self.output_size = output_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = encoder_hidden_size
        self.lstm_layers = lstm_layers

        # ----- Embedding categorie statiche -----
        clean_cardinalities = []
        for i, card in enumerate(cat_static_cardinalities):
            if card is None:
                raise ValueError(f"cat_static_cardinalities[{i}] è None, controlla num_ship_types / num_cargo_types")
            card_int = int(card)
            if card_int <= 0:
                raise ValueError(f"cat_static_cardinalities[{i}] = {card_int}, deve essere >= 1")
            clean_cardinalities.append(card_int)

        self.embeddings = nn.ModuleList([
            nn.Embedding(card, dim) for card, dim in zip(clean_cardinalities, cat_static_emb_dims)
        ])

        # ----- Normalizzatori -----
        self.norm_enc = nn.LayerNorm(encoder_hidden_size)
        self.norm_static = nn.LayerNorm(64)

        # ----- Encoder LSTM dinamico -----
        self.encoder_lstm = nn.LSTM(
            input_size=num_dynamic_features,
            hidden_size=encoder_hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )

        # ----- Encoder statico -----
        static_emb_dim = sum(cat_static_emb_dims)
        self.fc_static = nn.Sequential(
            nn.Linear(static_emb_dim + num_static_numeric, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # ----- Fused → hidden decoder -----
        self.fusion_to_h0 = nn.Linear(encoder_hidden_size + 64, self.decoder_hidden_size)

        # ----- Decoder LSTM autoregressivo -----
        self.decoder_lstm = nn.LSTM(
            input_size=output_size,              # in ingresso (Lat_rel, Lon_rel) precedente
            hidden_size=self.decoder_hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )

        self.fc_out = nn.Linear(self.decoder_hidden_size, output_size)

    def forward(self, x_dynamic, x_static_cats, x_static_nums):
        batch_size = x_dynamic.size(0)
        device = x_dynamic.device

        # ===== Statiche categoriali =====
        embedded_static = [emb(cat) for emb, cat in zip(self.embeddings, x_static_cats)]
        static_cat_emb = torch.cat(embedded_static, dim=-1)

        # ===== Statiche numeriche + embedding =====
        static_feat = torch.cat([static_cat_emb, x_static_nums], dim=-1)
        static_vec = self.fc_static(static_feat)
        static_vec = self.norm_static(static_vec)

        # ===== Encoder LSTM dinamico =====
        valid_mask = ~torch.isnan(x_dynamic[:, :, 0])
        lengths = valid_mask.sum(dim=1).clamp(min=1)
        x_dyn = torch.nan_to_num(x_dynamic, nan=0.0)

        packed = nn.utils.rnn.pack_padded_sequence(
            x_dyn, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_T, _) = self.encoder_lstm(packed)
        seq_vec = h_T[-1]               # (B, H)
        seq_vec = self.norm_enc(seq_vec)

        # ===== FUSIONE encoder + statico =====
        fused = torch.cat([seq_vec, static_vec], dim=-1)   # (B, H+64)
        h0 = torch.tanh(self.fusion_to_h0(fused))          # (B, H_dec)

        # Hidden iniziale del decoder
        h = h0.unsqueeze(0).repeat(self.lstm_layers, 1, 1)   # (L, B, H)
        c = torch.zeros(self.lstm_layers, batch_size, self.decoder_hidden_size, device=device)

        # ===== Decoder autoregressivo =====
        outputs = []
        # Step iniziale: partiamo da (0,0) in coordinate relative
        y_t = torch.zeros(batch_size, self.output_size, device=device)

        for _ in range(self.forecast_steps):
            dec_input = y_t.unsqueeze(1)               # (B, 1, 2)
            dec_out, (h, c) = self.decoder_lstm(dec_input, (h, c))  # (B, 1, H)
            y_t = self.fc_out(dec_out[:, 0, :])        # (B, 2)
            outputs.append(y_t.unsqueeze(1))           # (B, 1, 2)

        out = torch.cat(outputs, dim=1)               # (B, T, 2)
        return out


class TrajectoryDataset(Dataset):
    def __init__(self, sequences, input_ratio=INPUT_RATIO, forecast_steps=FORECAST_STEPS):
        self.sequences = sequences
        self.input_ratio = input_ratio
        self.forecast_steps = forecast_steps

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # copia per evitare side-effect
        df = self.sequences[idx].reset_index(drop=True).copy()

        # colonne statiche
        static_cat_cols = ['Ship type_enum', 'Cargo type_enum']
        static_num_base_cols = ['Width', 'Length', 'Draught']

        # lunghezza sequenza e indice di split
        seq_len = len(df)
        split_idx = int(seq_len * self.input_ratio)

        # punto di riferimento: ULTIMO punto osservato (split_idx - 1)
        ref_idx = max(split_idx - 1, 0)
        lat0 = float(df.loc[ref_idx, 'Latitude'])
        lon0 = float(df.loc[ref_idx, 'Longitude'])

        # coordinate relative (in gradi, centrate sull'ultimo punto noto)
        df['Lat_rel'] = df['Latitude'] - lat0
        df['Lon_rel'] = df['Longitude'] - lon0

        dynamic_cols = ['Lat_rel', 'Lon_rel', 'ROT', 'SOG', 'COG', 'Heading',
                        'dt', 'Velocity_N', 'Velocity_E', 'Navigational status_enum']

        # LSTM dynamic input (coordinate relative)
        x_dynamic = df.loc[:split_idx - 1, dynamic_cols].to_numpy(dtype='float32')

        # Target multi-step (sempre coordinate relative)
        y_future = df.loc[split_idx:split_idx + self.forecast_steps - 1,
                        ['Lat_rel', 'Lon_rel']]

        # Se la sequenza è troppo corta, paddiamo ripetendo l'ultimo valore relativo
        if len(y_future) < self.forecast_steps:
            last = y_future.iloc[-1].to_numpy(dtype='float32')
            pad = np.tile(last, (self.forecast_steps - len(y_future), 1))
            y_future = np.concatenate([y_future.to_numpy(dtype='float32'), pad], axis=0)
        else:
            y_future = y_future.to_numpy(dtype='float32')

        # Static numeric: Width, Length, Draught, lat0, lon0
        x_static_base = df.loc[0, static_num_base_cols].fillna(0).infer_objects().to_numpy(dtype='float32')
        origin = np.array([lat0, lon0], dtype='float32')
        x_static_nums = np.concatenate([x_static_base, origin], axis=0)  # shape (5,)

        # Static categorical come prima
        x_static_cats = []
        for c in static_cat_cols:
            val = df.loc[0, c]
            if pd.isna(val):
                idx = 0
            else:
                idx = int(val)   # converte da float64 → int
            x_static_cats.append(torch.tensor(idx, dtype=torch.long))


        return (torch.tensor(x_dynamic),
                x_static_cats,
                torch.tensor(x_static_nums)), torch.tensor(y_future)

def collate_fn(batch):
    # batch: list of tuples: ((x_dynamic, x_static_cats, x_static_nums), y_target)
    # pad sequences to max length in batch for LSTM input

    x_dynamic_list, x_static_cats_list, x_static_nums_list, y_target_list = [], [], [], []
    max_len = max(x[0].shape[0] for x, _ in batch)

    for (x_dynamic, x_static_cats, x_static_nums), y_target in batch:
        # pad dynamic
        pad_len = max_len - x_dynamic.shape[0]
        if pad_len > 0:
            pad = torch.full((pad_len, x_dynamic.shape[1]), float('nan'), dtype=torch.float32)
            x_dynamic = torch.cat([x_dynamic, pad], dim=0)

        x_dynamic_list.append(x_dynamic)
        x_static_cats_list.append(x_static_cats)
        x_static_nums_list.append(x_static_nums)
        y_target_list.append(y_target)

    x_dynamic_batch = torch.stack(x_dynamic_list)
    x_static_nums_batch = torch.stack(x_static_nums_list)
    y_target_batch = y_target_list  # list of tensors (variable length)

    # Stack categorical static features separately
    x_static_cats_batch = []
    for i in range(len(x_static_cats_list[0])):
        cat_feat = torch.stack([cats[i] for cats in x_static_cats_list])
        x_static_cats_batch.append(cat_feat)

    return (x_dynamic_batch, x_static_cats_batch, x_static_nums_batch), y_target_batch

def trajectory_loss(y_pred, y_true, x_dynamic, lambda_smooth=0.3, lambda_dir=0.3, K=5):
    # ---- 1. MSE base ----
    mse = torch.mean((y_pred - y_true) ** 2)

    # ---- 2. Smoothness ----
    v = y_pred[:, 1:, :] - y_pred[:, :-1, :]       # (B, T-1, 2)
    a = v[:, 1:, :] - v[:, :-1, :]                 # (B, T-2, 2)
    smoothness = torch.mean(a ** 2)

    # ---- 3. Direction Consistency ----
    past_rel = x_dynamic[:, -K:, :2]               # (B, K, 2) + padding (NaN)
    past_rel = torch.nan_to_num(past_rel, nan=0.0)

    if past_rel.size(1) > 1:
        d_past = past_rel[:, 1:, :] - past_rel[:, :-1, :]   # (B, K-1, 2)
        d_past = d_past.mean(dim=1)                         # (B, 2)
    else:
        d_past = torch.zeros(y_pred.size(0), 2, device=y_pred.device)

    d_pred = y_pred[:, 0, :]                        # (B, 2)
    dir_loss = torch.mean((d_pred - d_past) ** 2)

    total = mse + lambda_smooth * smoothness + lambda_dir * dir_loss
    return total

def train_loop(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    num_batches = len(dataloader)
    start_time = time()

    for batch_idx, ((x_dynamic, x_static_cats, x_static_nums), y_target) in enumerate(dataloader):

        # ==== LOG PROGRESSO TRAINING ====
        if True:  # DEBUG: stampa OGNI batch
            elapsed = time() - start_time
            batches_left = num_batches - (batch_idx + 1)
            eta = (elapsed / (batch_idx + 1)) * batches_left if batch_idx > 0 else 0.0
            print(f"[Train] Batch {batch_idx+1}/{num_batches}  "
                  f"({(batch_idx+1)/num_batches*100:.1f}%)  "
                  f"ETA: {eta/60:.1f} min")
        # ================================

        x_dynamic = x_dynamic.to(device)
        x_static_nums = x_static_nums.to(device)
        x_static_cats = [c.to(device) for c in x_static_cats]
        
        y_target = torch.stack(y_target).to(device)

        optimizer.zero_grad()
        y_pred = model(x_dynamic, x_static_cats, x_static_nums)

        loss = trajectory_loss(
            y_pred,
            y_target,
            x_dynamic,
            lambda_smooth=0.1,
            lambda_dir=0.3,
            K=5
        )



        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    epoch_time = time() - start_time
    print(f"Epoch train loop finished. Time = {epoch_time/60:.1f} min")

    return total_loss / len(dataloader)

def val_loop(model, dataloader, device):
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)

    with torch.no_grad():
        for batch_idx, ((x_dynamic, x_static_cats, x_static_nums), y_target) in enumerate(dataloader):
            if batch_idx % 50 == 0:
                print(f"[Val]   Batch {batch_idx+1}/{num_batches} "
                      f"({(batch_idx+1)/num_batches*100:.1f}%)")

            x_dynamic = x_dynamic.to(device)
            x_static_nums = x_static_nums.to(device)
            x_static_cats = [c.to(device) for c in x_static_cats]
            
            y_target = torch.stack(y_target).to(device)
            y_pred = model(x_dynamic, x_static_cats, x_static_nums)

            loss = trajectory_loss(
                y_pred,
                y_target,
                x_dynamic,
                lambda_smooth=0.3,
                lambda_dir=0.0,
                K=5
            )
            total_loss += loss.item()
    return total_loss / len(dataloader)

def load_sequences(root):
    # 1) Controllo che la cartella esista
    if not os.path.exists(root):
        raise RuntimeError(f"PARQUET_DIR does not exist: {root}")

    global num_cargo_types
    global num_ship_types
    sequences = []

    # 2) Scansione dei parquet
    for mmsi_folder in glob.glob(os.path.join(root, "MMSI=*")):
        for seg_folder in glob.glob(os.path.join(mmsi_folder, "segment=*")):

            files = glob.glob(os.path.join(seg_folder, "*.parquet"))
            if not files:
                continue

            dfs = [pd.read_parquet(f) for f in files]
            df = pd.concat(dfs, ignore_index=True)

            if "timestamp" in df.columns:
                df = df.sort_values("timestamp").reset_index(drop=True)

            # Shift enums by +1 to fix -1 values
            for col in ['Ship type_enum', 'Cargo type_enum', 'Navigational status_enum']:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: x + 1 if x >= 0 else 0).astype('int64')

            sequences.append(df)
        # ------- OVERSAMPLING SEQUENZE DI FIUME -------
    extra_sequences = []
    for df in sequences:
        if is_river_sequence(df):
            for _ in range(RIVER_OVERSAMPLE_FACTOR - 1):
                extra_sequences.append(df.copy())

    print(f"Oversampling fiumi: aggiunte {len(extra_sequences)} sequenze")
    sequences.extend(extra_sequences)
    # ------------------------------------------------

    # 3) Se non ho trovato nulla, mi fermo PRIMA di fare concat
    if len(sequences) == 0:
        raise RuntimeError("No parquet sequences found! Check PARQUET_DIR.")

    # 4) Calcolo le cardinalità per le embedding
    combined_df = pd.concat(sequences, ignore_index=True)
    num_ship_types = int(combined_df['Ship type_enum'].max()) + 1
    num_cargo_types = int(combined_df['Cargo type_enum'].max()) + 1

    print("Max ship_type_enum:", combined_df['Ship type_enum'].max())
    print("Num ship types (embedding size):", num_ship_types)

    print("Max cargo_type_enum:", combined_df['Cargo type_enum'].max())
    print("Num cargo types (embedding size):", num_cargo_types)

    return sequences, num_ship_types, num_cargo_types

def split_data(sequences, val_ratio=0.10, test_ratio=0.10, random_state=1234):
    # test_ratio = 0.10 → 10% test
    train_val, test = train_test_split(sequences, test_size=test_ratio, random_state=random_state)
    
    # val_ratio è inteso come percentuale TOTALE (10% del totale)
    # quindi qui facciamo il rescaling sul resto (90%)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio / (1 - test_ratio),   # 0.10 / 0.90 ≈ 0.111...
        random_state=random_state
    )
    print(f"Data split: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    return train, val, test

def evaluate_and_save_predictions(model, dataloader, device, output_json="predictions.json"):
    model.eval()
    results = []

    with torch.no_grad():
        seq_id = 0
        for (x_dynamic, x_static_cats, x_static_nums), y_target in dataloader:
            x_dynamic = x_dynamic.to(device)       # (batch, seq_len, features)
            x_static_nums = x_static_nums.to(device)
            x_static_cats = [c.to(device) for c in x_static_cats]
            y_target = torch.stack(y_target).to(device)  # (batch, forecast_steps, 2)

            # Inference: nessun teacher forcing
            y_pred = model(x_dynamic, x_static_cats, x_static_nums)

            # Porta tutto su CPU come numpy
            y_pred_np = y_pred.cpu().numpy()          # (B, 50, 2) rel
            y_true_np = y_target.cpu().numpy()        # (B, 50, 2) rel
            x_dynamic_np = x_dynamic.cpu().numpy()    # (B, seq_len, features) rel
            x_static_nums_np = x_static_nums.cpu().numpy()  # (B, 5) -> [Width, Length, Draught, lat0, lon0]

            for i in range(len(y_pred_np)):
                lat0_norm = float(x_static_nums_np[i, 3])
                lon0_norm = float(x_static_nums_np[i, 4])
                offset_norm = np.array([lat0_norm, lon0_norm], dtype=np.float32)

                known_rel = x_dynamic_np[i][:, :2]
                gt_rel = y_true_np[i]
                pred_rel = y_pred_np[i]

                # --- FIX: rimuovi eventuali NaN dal target e dalla predizione ---
                gt_rel = np.nan_to_num(gt_rel, nan=0.0)
                pred_rel = np.nan_to_num(pred_rel, nan=0.0)

                # remove padding
                mask = ~np.isnan(known_rel[:, 0])
                known_rel_valid = known_rel[mask]

                known_abs_norm = known_rel_valid + offset_norm
                gt_abs_norm = gt_rel + offset_norm
                pred_abs_norm = pred_rel + offset_norm

                # ---- ALLINEAMENTO DEL PUNTO DI PARTENZA ----
                # ancora = ultimo punto conosciuto (ultimo blu)
                anchor = known_abs_norm[-1]  # shape (2,)

                # shiftiamo GT e pred in modo che il loro primo punto coincida con anchor
                gt_abs_norm = gt_abs_norm - gt_abs_norm[0] + anchor
                pred_abs_norm = pred_abs_norm - pred_abs_norm[0] + anchor
                # ---------------------------------------------

                # save normalized coords (plot will denormalize!)
                results.append({
                    "sequence_id": seq_id,
                    "known_path": [{"lat": float(p[0]), "lon": float(p[1])} for p in known_abs_norm],
                    "ground_truth": [{"lat": float(p[0]), "lon": float(p[1])} for p in gt_abs_norm],
                    "prediction": [{"lat": float(p[0]), "lon": float(p[1])} for p in pred_abs_norm],
                })
                seq_id += 1

    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved predictions to {output_json}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if torch.cuda.is_available():
        print("CUDA detected. GPU count:", torch.cuda.device_count())
        print("GPU name:", torch.cuda.get_device_name(0))
    else:
        print("CUDA NOT available. Using CPU.")


    # Carica sempre i parquet dalla stessa directory condivisa
    root = PARQUET_DIR
    data, num_ship_types, num_cargo_types = load_sequences(root)
    
    print(">> num_ship_types:", num_ship_types, type(num_ship_types))
    print(">> num_cargo_types:", num_cargo_types, type(num_cargo_types))

    # filtra solo le sequenze troppo corte, senza altro cleaning
    data = [seq for seq in data if len(seq) > 100]

    train_data, val_data, test_data = split_data(
        data,
        val_ratio=0.10,
        test_ratio=0.10
    )

    train_ds = TrajectoryDataset(train_data)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    validation_ds = TrajectoryDataset(val_data)
    validation_dl = DataLoader(validation_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_ds = TrajectoryDataset(test_data)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = Seq2SeqTrajectoryModel(
        num_dynamic_features=NUM_DYNAMIC_FEATURES,
        cat_static_cardinalities=[int(num_ship_types), int(num_cargo_types)],
        cat_static_emb_dims=[8, 8],
        num_static_numeric=NUM_STATIC_NUMERIC,
        encoder_hidden_size=ENCODER_HIDDEN_SIZE,
        lstm_layers=LSTM_LAYERS,
        output_size=OUTPUT_SIZE,
        forecast_steps=FORECAST_STEPS
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    epochs = EPOCHS
    for epoch in range(epochs):
        train_loss = train_loop(model, train_dl, optimizer, device)
        print(f"Epoch {epoch+1}/{epochs} Train Loss: {train_loss:.10f}")

        validation_loss = val_loop(model, validation_dl, device)
        print(f"Epoch {epoch+1}/{epochs} Validation Loss: {validation_loss:.10f}")

    torch.save(model.state_dict(), "ship_trajectory_model.pth")
    print("Model saved to ship_trajectory_model.pth")

    print("Evaluating on test set...")
    evaluate_and_save_predictions(model, test_dl, device, output_json="test_predictions1.json")

if __name__ == "__main__":
    main()

#   bjobs -u all | grep gpua10 | grep PEND | wc -l
#   bjobs -u all | grep gpua40 | grep PEND | wc -l
#   bjobs -u all | grep gpul40s | grep PEND | wc -l
#   bjobs -u all | grep gpua100 | grep PEND | wc -l

#   bsub -q gpua10 < run_sim4.sh
#   bsub -q gpua40 < run_sim4.sh
#   bsub -q gpul40s < run_sim4.sh
#   bsub -q gpua100 < run_sim4.sh

#   bsub -q gpua10 < run_sim4_martin.sh
#   bsub -q gpua40 < run_sim4_martin.sh
#   bsub -q gpul40s < run_sim4_martin.sh
#   bsub -q gpua100 < run_sim4_martin.sh
