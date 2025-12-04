import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import json
import argparse

USE_ONE_PATH = False

parser = argparse.ArgumentParser()
parser.add_argument("--isbsub", action="store_true")

args = parser.parse_args()
if args.isbsub:
    postifx = "_bsub"
else:
    postifx = ""
    
pd.set_option('future.no_silent_downcasting', True)

num_ship_types = 0
num_cargo_types = 0

class Seq2SeqDeltaModel(nn.Module):
    def __init__(self,
                 num_dynamic_features=5,
                 encoder_hidden_size=128,
                 output_size=2,
                 forecast_steps=59):
        super().__init__()

        self.forecast_steps = forecast_steps
        self.output_size = output_size

        self.pre_encoder_dynamic = nn.Sequential(
            nn.Linear(num_dynamic_features, encoder_hidden_size),
            nn.ReLU(),
        )

        self.encoder_lstm = nn.LSTM(
            input_size=encoder_hidden_size,
            hidden_size=encoder_hidden_size,
            num_layers=2,
            batch_first=True
        )

        self.fc_out = nn.Linear(encoder_hidden_size, (forecast_steps + 1) * output_size)

    def forward(self, x_dynamic, last_known_pos):
        batch_size = x_dynamic.size(0)

        lengths = (x_dynamic.abs().sum(dim=-1) != 0).sum(dim=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x_dynamic, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed = nn.utils.rnn.PackedSequence(
            self.pre_encoder_dynamic(packed.data), packed.batch_sizes, packed.sorted_indices, packed.unsorted_indices
        )

        _, (h_T, _) = self.encoder_lstm(packed)
        seq_vec = h_T[-1]  # (batch_size, encoder_hidden_size)

        out = self.fc_out(seq_vec)  # (batch_size, (forecast_steps+1) * output_size)
        out = out.view(batch_size, self.forecast_steps + 1, self.output_size)  # (batch_size, forecast_steps+1, output_size)

        out[:, 0, :] = last_known_pos

        return out[:, 1:, :]

# Static features have been taken out for simplicity
class TrajectoryDataset(Dataset):
    def __init__(self, sequences, forecast_steps=50, split_ratio=0.67): #180: 3 hours at 1 min intervals
        self.sequences = sequences
        self.forecast_steps = forecast_steps
        self.split_ratio = split_ratio

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        df = self.sequences[idx].copy()
        dynamic_cols = ['Latitude', 'Longitude', 'SOG', 'cos_cog', 'sin_cog', 'vx', 'vy', 'Heading']

        # fix hossz: 180
        # 120 known, 60 forecast
        split_idx = 120

        x_dynamic = df.loc[:split_idx-1, dynamic_cols].to_numpy(dtype='float32')
        y_future_df = df.loc[split_idx:, ['Latitude', 'Longitude']].to_numpy(dtype='float32')

        delta_lat = y_future_df[1:, 0] - y_future_df[:-1, 0]
        delta_lon = y_future_df[1:, 1] - y_future_df[:-1, 1]
        y_deltas = np.stack([delta_lat, delta_lon], axis=1).astype('float32')

        y_abs = y_future_df[1:].astype('float32')

        return torch.tensor(x_dynamic), (torch.tensor(y_deltas), torch.tensor(y_abs))

def collate_fn(batch):
    x_dynamic_list = []
    y_deltas_list = []
    y_abs_list = []

    # Find max sequence length in this batch for x_dynamic
    max_len = max(x_dynamic.shape[0] for (x_dynamic), (y_deltas, y_abs) in batch)

    for (x_dynamic), (y_deltas, y_abs) in batch:
        # Ensure tensors
        if not isinstance(x_dynamic, torch.Tensor):
            x_dynamic = torch.tensor(x_dynamic, dtype=torch.float32)
        if not isinstance(y_deltas, torch.Tensor):
            y_deltas = torch.tensor(y_deltas, dtype=torch.float32)
        if not isinstance(y_abs, torch.Tensor):
            y_abs = torch.tensor(y_abs, dtype=torch.float32)

        # Pad x_dynamic along sequence length dimension (dim=0)
        pad_len = max_len - x_dynamic.shape[0]
        if pad_len > 0:
            pad = torch.zeros(pad_len, x_dynamic.shape[1], dtype=torch.float32)
            x_dynamic = torch.cat([x_dynamic, pad], dim=0)

        x_dynamic_list.append(x_dynamic)
        y_deltas_list.append(y_deltas)
        y_abs_list.append(y_abs)

    # Stack into batch tensors
    x_dynamic_batch = torch.stack(x_dynamic_list)  # shape: (batch_size, max_seq_len, features)
    y_deltas_batch = torch.stack(y_deltas_list)    # shape: (batch_size, forecast_steps, 2)
    y_abs_batch = torch.stack(y_abs_list)          # shape: (batch_size, forecast_steps, 2)

    return (x_dynamic_batch), (y_deltas_batch, y_abs_batch)

def train_loop(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for (x_dynamic), (y_deltas, y_abs) in dataloader:
        x_dynamic = x_dynamic.to(device)
        y_abs = y_abs.to(device)
        last_known_pos = x_dynamic[:, -1, :2]  # utolsó ismert abszolút pozíció

        optimizer.zero_grad()
        y_pred_abs = model(x_dynamic, last_known_pos)  # most abszolút pozíciók

        y_abs = y_abs.float()
        y_pred_abs = y_pred_abs.float()
        loss = criterion(y_pred_abs, y_abs)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
    return total_loss

def val_loop(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for (x_dynamic), (y_deltas, y_abs) in dataloader:
            x_dynamic = x_dynamic.to(device)
            y_abs = y_abs.to(device)
            last_known_pos = x_dynamic[:, -1, :2]  # utolsó ismert abszolút pozíció

            y_pred_abs = model(x_dynamic, last_known_pos)

            loss = criterion(y_pred_abs, y_abs)
            total_loss += loss.item()

    return total_loss

def load_sequences(root):
    global num_cargo_types
    global num_ship_types

    sequences = []

    for mmsi_folder in glob.glob(os.path.join(root, "MMSI=*")):
        for seg_folder in glob.glob(os.path.join(mmsi_folder, "segment=*")):
            files = glob.glob(os.path.join(seg_folder, "*.parquet"))
            if not files:
                continue

            dfs = []
            for f in files:
                df_tmp = pd.read_parquet(f)

                # --- FILTER: skip files with fewer than 182 rows ---
                if len(df_tmp) < 182:
                    continue

                dfs.append(df_tmp)

            # If after filtering there are no valid files, skip
            if not dfs:
                continue

            df = pd.concat(dfs, ignore_index=True)

            # optional: sort by timestamp
            if "timestamp" in df.columns:
                df = df.sort_values("timestamp").reset_index(drop=True)

            # Shift enums by +1 to fix -1 values
            for col in ['Ship type_enum', 'Cargo type_enum', 'Navigational status_enum']:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: x + 1 if x >= 0 else 0)

            sequences.append(df)

    combined_df = pd.concat(sequences, ignore_index=True)

    num_ship_types = combined_df['Ship type_enum'].max() + 1
    num_cargo_types = combined_df['Cargo type_enum'].max() + 1

    print("Max ship_type_enum:", combined_df['Ship type_enum'].max())
    print("Num ship types (embedding size):", num_ship_types)
    print("Max cargo_type_enum:", combined_df['Cargo type_enum'].max())
    print("Num cargo types (embedding size):", num_cargo_types)

    return sequences

def create_random_fixed_length_slices(sequences, slice_length=180):
    slices = []
    for df in sequences:
        seq_len = len(df)
        if seq_len < slice_length:
            continue  # skip rövid szekvencia

        max_start = seq_len - slice_length
        start_idx = np.random.randint(0, max_start + 1)
        slice_df = df.iloc[start_idx:start_idx + slice_length].reset_index(drop=True)
        slices.append(slice_df)
    print(f"Created {len(slices)} random fixed-length slices (one per sequence)")
    return slices

def split_data(sequences, val_ratio=0.1, test_ratio=0.01, random_state=1234):
    train_val, test = train_test_split(sequences, test_size=test_ratio, random_state=random_state)
    train, val = train_test_split(train_val, test_size=val_ratio/(1 - test_ratio), random_state=random_state)
    print(f"Data split: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    return train, val, test

def evaluate_and_save_predictions(model, dataloader, device, output_json="test_predictions.json"):
    model.eval()
    results = []

    with torch.no_grad():
        seq_id = 0
        for (x_dynamic), (y_deltas, y_abs) in dataloader:

            x_dynamic = x_dynamic.to(device)
            y_abs = y_abs.to(device)

            last_known_pos = x_dynamic[:, -1, :2]  # az utolsó ismert pozíció (latitude, longitude)

            y_pred_abs = model(x_dynamic, last_known_pos)

            y_pred_np = y_pred_abs.cpu().numpy()
            y_true_np = y_abs.cpu().numpy()
            x_dynamic_np = x_dynamic.cpu().numpy()

            for i in range(len(y_pred_np)):
                known_points = x_dynamic_np[i][:, :2]
                valid_known = known_points[(known_points[:, 0] != 0.0) | (known_points[:, 1] != 0.0)]
                last_known = valid_known[-1] if len(valid_known) > 0 else np.array([0.0, 0.0])

                abs_preds = y_pred_np[i]

                results.append({
                    "sequence_id": seq_id,
                    "known_path": [{"lat": float(p[0]), "lon": float(p[1])} for p in valid_known],
                    "ground_truth": [{"lat": float(p[0]), "lon": float(p[1])} for p in y_true_np[i]],
                    "prediction": [{"lat": float(p[0]), "lon": float(p[1])} for p in abs_preds]
                })
                seq_id += 1

    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved predictions to {output_json}")

def main():

    model = Seq2SeqDeltaModel(
        num_dynamic_features=8,
        encoder_hidden_size=256,
        output_size=2,
        forecast_steps=59
    )
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    root = os.path.join(os.getcwd(), "data")
    data = load_sequences(root)
    data = create_random_fixed_length_slices(data, slice_length=180)
    
    if (USE_ONE_PATH):
        data = [data[0], data[10], data[20], data[30], data[40], data[50], data[60], data[70], data[80], data[90]]
        train_data = data
        val_data = data
        test_data = data
    else:
        train_data, val_data, test_data = split_data(data)

    train_ds = TrajectoryDataset(train_data)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn)
    print(f"Length of some segments in train dataset: {[len(seq) for seq in train_data[:10]]}")
    validation_ds = TrajectoryDataset(val_data)
    validation_dl = DataLoader(validation_ds, batch_size=128, shuffle=False, collate_fn=collate_fn)
    test_ds = TrajectoryDataset(test_data)
    test_dl = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.SmoothL1Loss(beta = 0.1) #beta = 0.1

    prev_val_loss = float('inf')
    no_improve_epochs = 0
    debounce = 1000
    epochs = 10000
    loss_pairs = []
    for epoch in range(epochs):
        train_loss = train_loop(model, train_dl, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} Train Loss: {train_loss:.15f}")

        validation_loss = val_loop(model, validation_dl, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} Validation Loss: {validation_loss:.15f}")

        loss_pairs.append((train_loss, validation_loss))

        if validation_loss < prev_val_loss:
            no_improve_epochs = 0
            prev_val_loss = validation_loss
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= debounce:
                print("Early stopping triggered.")
                print("Epochs completed:", epoch + 1)
                print("Best Validation Loss:", prev_val_loss)
                break

    torch.save(model.state_dict(), "ship_trajectory_model.pth")
    print("Model saved to ship_trajectory_model.pth")

    #write loss pairs to a json for plotting later
    with open("loss_pairs"+postifx+".json", "w") as f:
        json.dump([{"train_loss": tl, "val_loss": vl} for tl, vl in loss_pairs], f, indent=2)

    print("Evaluating on test set...")
    evaluate_and_save_predictions(model, test_dl, device, output_json="test_predictions"+postifx+".json")
    evaluate_and_save_predictions(model, train_dl, device, output_json="train_predictions"+postifx+".json")

if __name__ == "__main__":
    main()
    