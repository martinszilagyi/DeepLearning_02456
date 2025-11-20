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

class Seq2SeqTrajectoryModel(nn.Module):
    def __init__(self,
                 num_dynamic_features=10,
                 cat_static_cardinalities=[None, None],
                 cat_static_emb_dims=[8, 8],
                 num_static_numeric=3,
                 encoder_hidden_size=64,
                 decoder_hidden_size=64,
                 lstm_layers=1,
                 output_size=2,          # lat, lon per step
                 forecast_steps=50):
        super().__init__()

        self.forecast_steps = forecast_steps

        # Embeddings for static categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, dim) for card, dim in zip(cat_static_cardinalities, cat_static_emb_dims)
        ])

        # Encoder LSTM for dynamic input
        self.encoder_lstm = nn.LSTM(num_dynamic_features, encoder_hidden_size, lstm_layers, batch_first=True)

        # FC for static numeric + embedded categorical
        static_emb_dim = sum(cat_static_emb_dims)
        self.fc_static = nn.Sequential(
            nn.Linear(static_emb_dim + num_static_numeric, decoder_hidden_size),
            nn.ReLU(),
            nn.Linear(decoder_hidden_size, decoder_hidden_size),
            nn.ReLU()
        )

        # Decoder LSTM input size: previous point (2) + static features (decoder_hidden_size)
        self.decoder_lstm = nn.LSTM(input_size=2 + decoder_hidden_size,
                                    hidden_size=decoder_hidden_size,
                                    num_layers=lstm_layers,
                                    batch_first=True)

        # Output layer to predict lat/lon at each decoder step
        self.output_fc = nn.Linear(decoder_hidden_size, output_size)

    def forward(self, x_dynamic, x_static_cats, x_static_nums, y_target=None, teacher_forcing_ratio=0.5):
        """
        x_dynamic: (batch, seq_len, num_dynamic_features)
        x_static_cats: list of (batch,) tensors
        x_static_nums: (batch, num_static_numeric)
        y_target: (batch, forecast_steps, 2) or None during inference
        teacher_forcing_ratio: probability to use ground truth as next input during training
        """

        batch_size = x_dynamic.size(0)

        # Embed static categorical features
        embedded = [emb(cat) for emb, cat in zip(self.embeddings, x_static_cats)]
        static_cat_emb = torch.cat(embedded, dim=-1)

        # Static features combined and projected for decoder init and input
        static_feat = torch.cat([static_cat_emb, x_static_nums], dim=-1)
        static_context = self.fc_static(static_feat)  # (batch, decoder_hidden_size)

        # Encoder
        encoder_outputs, (hidden, cell) = self.encoder_lstm(x_dynamic)  # hidden/cell: (layers, batch, hidden_size)

        # Initialize decoder hidden state by combining encoder hidden and static context
        # For simplicity, use static context to replace decoder initial hidden state for all layers
        decoder_hidden = static_context.unsqueeze(0).repeat(self.decoder_lstm.num_layers, 1, 1)  # (layers, batch, hidden_size)
        decoder_cell = torch.zeros_like(decoder_hidden)  # init cell state as zeros

        # Prepare decoder inputs
        outputs = torch.zeros(batch_size, self.forecast_steps, 2, device=x_dynamic.device)

        # Initial input to decoder: last point of input dynamic sequence (lat, lon)
        # Assuming dynamic features: Latitude and Longitude are first two columns
        decoder_input = x_dynamic[:, -1, :2]  # (batch, 2)

        for t in range(self.forecast_steps):
            # Concatenate decoder input and static context for LSTM input
            decoder_lstm_input = torch.cat([decoder_input, static_context], dim=1).unsqueeze(1)  # (batch, 1, 2 + hidden_size)

            # Run one step of decoder LSTM
            decoder_output, (decoder_hidden, decoder_cell) = self.decoder_lstm(decoder_lstm_input, (decoder_hidden, decoder_cell))

            # Predict next position
            pred = self.output_fc(decoder_output.squeeze(1))  # (batch, 2)

            outputs[:, t, :] = pred

            # Decide next decoder input: teacher forcing or own prediction
            if self.training and y_target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # Use ground truth next point as next input
                decoder_input = y_target[:, t, :]
            else:
                # Use model prediction as next input
                decoder_input = pred

        return outputs

class TrajectoryDataset(Dataset):
    def __init__(self, sequences, input_ratio=0.5, forecast_steps=50):
        self.sequences = sequences
        self.input_ratio = input_ratio
        self.forecast_steps = forecast_steps

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        df = self.sequences[idx]

        dynamic_cols = ['Latitude', 'Longitude', 'ROT', 'SOG', 'COG', 'Heading',
                        'dt', 'Velocity_N', 'Velocity_E', 'Navigational status_enum']
        static_num_cols = ['Width', 'Length', 'Draught']
        static_cat_cols = ['Ship type_enum', 'Cargo type_enum']

        seq_len = len(df)
        split_idx = int(seq_len * self.input_ratio)

        # LSTM dynamic input
        x_dynamic = df.loc[:split_idx-1, dynamic_cols].to_numpy(dtype='float32')

        # Multi-step (50-step) target
        y_future = df.loc[split_idx:split_idx + self.forecast_steps - 1,
                          ['Latitude', 'Longitude']]

        # If too short, pad last known values
        if len(y_future) < self.forecast_steps:
            last = y_future.iloc[-1].to_numpy(dtype='float32')
            pad = np.tile(last, (self.forecast_steps - len(y_future), 1))
            y_future = np.concatenate([y_future.to_numpy(dtype='float32'), pad], axis=0)
        else:
            y_future = y_future.to_numpy(dtype='float32')

        # Static features
        x_static_nums = df.loc[0, static_num_cols].fillna(0).infer_objects().to_numpy(dtype='float32')
        x_static_cats = [torch.tensor(df.loc[0, c], dtype=torch.long) for c in static_cat_cols]

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

def train_loop(model, dataloader, optimizer, criterion, device, teacher_forcing_ratio=0.5):
    model.train()
    total_loss = 0
    for (x_dynamic, x_static_cats, x_static_nums), y_target in dataloader:
        x_dynamic = x_dynamic.to(device)
        x_static_nums = x_static_nums.to(device)
        x_static_cats = [c.to(device) for c in x_static_cats]
        
        # y_target is a list of tensors (variable lengths) → stack and pad if necessary
        # Here assuming all sequences have length = forecast_steps (e.g., 50)
        y_target = torch.stack(y_target).to(device)  # (batch, seq_len, 2)

        # Check for NaNs/Infs
        assert not torch.isnan(x_dynamic).any(), "NaN in x_dynamic"
        assert not torch.isnan(x_static_nums).any(), "NaN in x_static_nums"
        for cat_tensor in x_static_cats:
            assert not torch.isnan(cat_tensor).any(), "NaN in x_static_cats"
        assert not torch.isnan(y_target).any(), "NaN in y_target"

        optimizer.zero_grad()
        y_pred = model(x_dynamic, x_static_cats, x_static_nums, y_target, teacher_forcing_ratio=teacher_forcing_ratio)

        for emb, cat in zip(model.embeddings, x_static_cats):
            max_idx = cat.max().item()
            min_idx = cat.min().item()
            assert max_idx < emb.num_embeddings, f"Category index {max_idx} out of range (max {emb.num_embeddings-1})"
            assert min_idx >= 0, f"Category index {min_idx} less than zero"

        loss = criterion(y_pred, y_target)
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
            
            y_target = torch.stack(y_target).to(device)  # (batch, forecast_steps, 2)

            # During validation, no teacher forcing (pure autoregressive prediction)
            y_pred = model(x_dynamic, x_static_cats, x_static_nums, y_target=None, teacher_forcing_ratio=0.0)

            loss = criterion(y_pred, y_target)

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

            x_dynamic = x_dynamic.to(device)       # (batch, seq_len, features)
            x_static_nums = x_static_nums.to(device)
            x_static_cats = [c.to(device) for c in x_static_cats]
            y_target = torch.stack(y_target).to(device)  # (batch, forecast_steps, 2)

            # Autoregressive inference: no teacher forcing
            y_pred = model(x_dynamic, x_static_cats, x_static_nums, y_target=None, teacher_forcing_ratio=0.0)  # (batch, 50, 2)

            y_pred_np = y_pred.cpu().numpy()
            y_true_np = y_target.cpu().numpy()
            x_dynamic_np = x_dynamic.cpu().numpy()  # (batch, seq_len, features)

            for i in range(len(y_pred_np)):
                gt_points = y_true_np[i]
                pred_points = y_pred_np[i]
                known_points = x_dynamic_np[i][:, :2]  # lat, lon from first two dynamic features

                results.append({
                    "sequence_id": seq_id,
                    "known_path": [{"lat": float(p[0]), "lon": float(p[1])} for p in known_points],
                    "ground_truth": [{"lat": float(p[0]), "lon": float(p[1])} for p in gt_points],
                    "prediction": [{"lat": float(p[0]), "lon": float(p[1])} for p in pred_points]
                })
                seq_id += 1

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

    model = Seq2SeqTrajectoryModel(
        num_dynamic_features=10,
        cat_static_cardinalities=[num_ship_types, num_cargo_types],
        cat_static_emb_dims=[8, 8],
        num_static_numeric=3,
        encoder_hidden_size=64,
        decoder_hidden_size=64,
        lstm_layers=1,
        output_size=2,
        forecast_steps=50
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
    