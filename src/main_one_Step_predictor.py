import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json

pd.set_option('future.no_silent_downcasting', True)

num_ship_types = 0
num_cargo_types = 0

class ShipTrajectoryModel(nn.Module):
    def __init__(self,
                 num_dynamic_features=10,          # Lat, Lon, ROT, SOG, COG, Heading, dt, Velocity_N, Velocity_E, Navigational status enum
                 cat_static_cardinalities=[None, None],  # ship type enum, cargo type enum sizes (replace None with ints)
                 cat_static_emb_dims=[8, 8],       # embedding sizes for categorical static features
                 num_static_numeric=3,             # width, length, draught
                 lstm_hidden_size=64,
                 lstm_layers=1,
                 fc_hidden_size=64,
                 output_size=2):                   # predict next Lat, Lon
        super().__init__()

        # Embeddings for static categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, dim) for card, dim in zip(cat_static_cardinalities, cat_static_emb_dims)
        ])

        # LSTM for dynamic features
        self.lstm = nn.LSTM(num_dynamic_features, lstm_hidden_size, lstm_layers, batch_first=True)

        # FC for static numeric + embedded categorical
        static_emb_dim = sum(cat_static_emb_dims)
        self.fc_static = nn.Sequential(
            nn.Linear(static_emb_dim + num_static_numeric, fc_hidden_size),
            nn.ReLU(),
            nn.Linear(fc_hidden_size, fc_hidden_size),
            nn.ReLU()
        )

        # Combine LSTM output and static FC output
        self.fc_combined = nn.Sequential(
            nn.Linear(lstm_hidden_size + fc_hidden_size, fc_hidden_size),
            nn.ReLU(),
            nn.Linear(fc_hidden_size, output_size)
        )

    def forward(self, x_dynamic, x_static_cats, x_static_nums):
        # x_dynamic: (batch, seq_len, num_dynamic_features)
        # x_static_cats: list of (batch,) tensors, categorical static inputs
        # x_static_nums: (batch, num_static_numeric) tensor

        embedded = [emb(cat) for emb, cat in zip(self.embeddings, x_static_cats)]
        static_cat_emb = torch.cat(embedded, dim=-1)

        static_feat = torch.cat([static_cat_emb, x_static_nums], dim=-1)
        static_out = self.fc_static(static_feat)

        lstm_out, _ = self.lstm(x_dynamic)
        lstm_last = lstm_out[:, -1, :]

        combined = torch.cat([lstm_last, static_out], dim=-1)
        out = self.fc_combined(combined)
        return out

class TrajectoryDataset(Dataset):
    def __init__(self, sequences, input_ratio=0.5):
        self.sequences = sequences
        self.input_ratio = input_ratio

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        df = self.sequences[idx]

        # Dynamic features for LSTM
        dynamic_cols = ['Latitude', 'Longitude', 'ROT', 'SOG', 'COG', 'Heading', 'dt', 'Velocity_N', 'Velocity_E', 'Navigational status_enum']
        static_num_cols = ['Width', 'Length', 'Draught']
        static_cat_cols = ['Ship type_enum', 'Cargo type_enum']

        seq_len = len(df)
        split_idx = int(seq_len * self.input_ratio)

        # Dynamic input and target (next positions)
        x_dynamic = df.loc[:split_idx-1, dynamic_cols].to_numpy(dtype='float32')
        y_target = df.loc[split_idx:, ['Latitude', 'Longitude']].to_numpy(dtype='float32')

        # Static numerical features - take from first row (assuming static)
        x_static_nums = df.loc[0, static_num_cols].fillna(0).infer_objects().to_numpy(dtype='float32')
        
        # Static categorical features - convert to int64 tensor
        x_static_cats = [torch.tensor(df.loc[0, c], dtype=torch.long) for c in static_cat_cols]

        return (torch.tensor(x_dynamic), x_static_cats, torch.tensor(x_static_nums)), torch.tensor(y_target)

def collate_fn(batch):
    # batch: list of tuples: ((x_dynamic, x_static_cats, x_static_nums), y_target)
    # pad sequences to max length in batch for LSTM input

    x_dynamic_list, x_static_cats_list, x_static_nums_list, y_target_list = [], [], [], []
    max_len = max(x[0].shape[0] for x, _ in batch)

    for (x_dynamic, x_static_cats, x_static_nums), y_target in batch:
        # pad dynamic
        pad_len = max_len - x_dynamic.shape[0]
        if pad_len > 0:
            pad = torch.zeros(pad_len, x_dynamic.shape[1], dtype=torch.float32)
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

def train_loop(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for (x_dynamic, x_static_cats, x_static_nums), y_target in dataloader:
        x_dynamic = x_dynamic.to(device)
        x_static_nums = x_static_nums.to(device)
        x_static_cats = [c.to(device) for c in x_static_cats]
        y_target = [y.to(device) for y in y_target]

        # Check for NaNs/Infs in inputs & targets
        assert not torch.isnan(x_dynamic).any(), "NaN in x_dynamic"
        assert not torch.isnan(x_static_nums).any(), "NaN in x_static_nums"
        for cat_tensor in x_static_cats:
            assert not torch.isnan(cat_tensor).any(), "NaN in x_static_cats"
        # Target is list of tensors; check first element as example
        assert all([not torch.isnan(y).any() for y in y_target]), "NaN in y_target"

        optimizer.zero_grad()
        y_pred = model(x_dynamic, x_static_cats, x_static_nums)  # (batch, output_size)

        # Here y_pred predicts next lat, lon at last step. For multi-step targets, need adjustment.
        # For now, let's predict only next single point (1-step ahead)
        # So target: first point in y_target sequences
        y_true = torch.stack([y[0] for y in y_target])  # (batch, 2)

        for emb, cat in zip(model.embeddings, x_static_cats):
            max_idx = cat.max().item()
            min_idx = cat.min().item()
            assert max_idx < emb.num_embeddings, f"Category index {max_idx} out of range (max {emb.num_embeddings-1})"
            assert min_idx >= 0, f"Category index {min_idx} less than zero"

        loss = criterion(y_pred, y_true)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def val_loop(model, dataloader, criterion, device):
    model.eval()  # set model to evaluation mode
    total_loss = 0
    with torch.no_grad():  # disable gradient calculation
        for (x_dynamic, x_static_cats, x_static_nums), y_target in dataloader:
            x_dynamic = x_dynamic.to(device)
            x_static_nums = x_static_nums.to(device)
            x_static_cats = [c.to(device) for c in x_static_cats]
            y_target = [y.to(device) for y in y_target]

            y_pred = model(x_dynamic, x_static_cats, x_static_nums)

            # same as train: predict first next point from y_target sequence
            y_true = torch.stack([y[0] for y in y_target])

            loss = criterion(y_pred, y_true)
            total_loss += loss.item()

    return total_loss / len(dataloader)

def load_sequences(root):
    global num_cargo_types
    global num_ship_types
    sequences = []

    for mmsi_folder in glob.glob(os.path.join(root, "MMSI=*")):
        for seg_folder in glob.glob(os.path.join(mmsi_folder, "segment=*")):

            files = glob.glob(os.path.join(seg_folder, "*.parquet"))
            if not files:
                continue

            # pandas handles mixed dtypes safely
            dfs = [pd.read_parquet(f) for f in files]
            df = pd.concat(dfs, ignore_index=True)

            # optional
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

def split_data(sequences, val_ratio=0.1, test_ratio=0.01, random_state=1234):
    train_val, test = train_test_split(sequences, test_size=test_ratio, random_state=random_state)
    train, val = train_test_split(train_val, test_size=val_ratio/(1 - test_ratio), random_state=random_state)
    print(f"Data split: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    return train, val, test

def evaluate_and_save_predictions(model, dataloader, device, output_json="predictions.json"):
    model.eval()
    results = []

    with torch.no_grad():
        seq_id = 0
        for (x_dynamic, x_static_cats, x_static_nums), y_target in dataloader:

            x_dynamic = x_dynamic.to(device)
            x_static_nums = x_static_nums.to(device)
            x_static_cats = [c.to(device) for c in x_static_cats]
            y_target = [y.to(device) for y in y_target]

            # forward pass
            y_pred = model(x_dynamic, x_static_cats, x_static_nums)  # (batch, 2)

            # ground truth = first step of multi-step target
            y_true = torch.stack([y[0] for y in y_target])  # (batch, 2)

            # convert to CPU numpy for JSON
            y_pred_np = y_pred.cpu().numpy()
            y_true_np = y_true.cpu().numpy()

            for i in range(len(y_pred_np)):
                results.append({
                    "sequence_id": seq_id,
                    "ground_truth": {
                        "lat": float(y_true_np[i][0]),
                        "lon": float(y_true_np[i][1])
                    },
                    "prediction": {
                        "lat": float(y_pred_np[i][0]),
                        "lon": float(y_pred_np[i][1])
                    }
                })
                seq_id += 1

    # Save JSON
    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved predictions to {output_json}")

def main():
    root = os.path.join(os.getcwd(), "data")
    data = load_sequences(root)
    train_data, val_data, test_data = split_data(data)

    train_ds = TrajectoryDataset(train_data)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    validation_ds = TrajectoryDataset(val_data)
    validation_dl = DataLoader(validation_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_ds = TrajectoryDataset(test_data)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    model = ShipTrajectoryModel(
        num_dynamic_features=10,
        cat_static_cardinalities=[num_ship_types, num_cargo_types],
        cat_static_emb_dims=[8, 8],
        num_static_numeric=3,
        lstm_hidden_size=64,
        fc_hidden_size=64,
        output_size=2
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    epochs = 2
    for epoch in range(epochs):
        train_loss = train_loop(model, train_dl, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} Train Loss: {train_loss:.4f}")

        validation_loss = val_loop(model, validation_dl, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} Validation Loss: {validation_loss:.4f}")

    torch.save(model.state_dict(), "ship_trajectory_model.pth")
    print("Model saved to ship_trajectory_model.pth")

    print("Evaluating on test set...")
    evaluate_and_save_predictions(model, test_dl, device, output_json="test_predictions.json")

    """ 
    LOAD NN:
    model = ShipTrajectoryModel(
        num_dynamic_features=10,
        cat_static_cardinalities=[num_ship_types, num_cargo_types],
        cat_static_emb_dims=[8, 8],
        num_static_numeric=3,
        lstm_hidden_size=64,
        fc_hidden_size=64,
        output_size=2
    )

    model.load_state_dict(torch.load("ship_trajectory_model.pth", map_location="cpu"))
    model.eval()

    OR if using CUDA:model.load_state_dict(torch.load("ship_trajectory_model.pth"))
    model.to(device)
    model.eval() """

if __name__ == "__main__":
    main()