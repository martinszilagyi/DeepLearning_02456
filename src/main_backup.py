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

class Seq2SeqDeltaModel(nn.Module):
    def __init__(self,
                 num_dynamic_features=10,
                 cat_static_cardinalities=[None, None],
                 cat_static_emb_dims=[8, 8],
                 num_static_numeric=3,
                 encoder_hidden_size=128,
                 decoder_hidden_size=128,
                 lstm_layers=2,
                 output_size=2,
                 forecast_steps=50):
        super().__init__()

        self.forecast_steps = forecast_steps
        self.output_size = output_size

        # Embeddings for static categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, dim) for card, dim in zip(cat_static_cardinalities, cat_static_emb_dims)
        ])

        self.norm_enc = nn.LayerNorm(encoder_hidden_size)
        self.norm_static = nn.LayerNorm(64)

        # Encoder LSTM for dynamic input
        self.encoder_lstm = nn.LSTM(
            input_size=num_dynamic_features,
            hidden_size=encoder_hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )

        # Static feature encoder (FFNN)
        static_emb_dim = sum(cat_static_emb_dims)
        self.fc_static = nn.Sequential(
            nn.Linear(static_emb_dim + num_static_numeric, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Decoder LSTM that generates deltas autoregressively
        self.decoder_lstm = nn.LSTM(
            input_size=output_size,  # feeding previous delta output as input
            hidden_size=decoder_hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )

        # Project decoder hidden state to delta lat/lon
        self.fc_out = nn.Linear(decoder_hidden_size, output_size)

        # Map encoder hidden state + static vector to initialize decoder hidden state
        self.init_hidden = nn.Sequential(
            nn.Linear(encoder_hidden_size + 64, decoder_hidden_size * lstm_layers * 2),
            nn.Tanh()
        )

    def forward(self, x_dynamic, x_static_cats, x_static_nums, y_target=None, teacher_forcing_ratio=0.0):
        batch_size = x_dynamic.size(0)

        # --- Static categorical embedding ---
        embedded_static = [emb(cat) for emb, cat in zip(self.embeddings, x_static_cats)]
        static_cat_emb = torch.cat(embedded_static, dim=-1)

        # Combine static numeric + categorical
        static_feat = torch.cat([static_cat_emb, x_static_nums], dim=-1)
        static_vec = self.fc_static(static_feat)
        static_vec = self.norm_static(static_vec)

        # --- Encode dynamic sequence ---
        lengths = (x_dynamic.abs().sum(dim=-1) != 0).sum(dim=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x_dynamic, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_T, c_T) = self.encoder_lstm(packed)
        seq_vec = h_T[-1]
        seq_vec = self.norm_enc(seq_vec)

        # --- Initialize decoder hidden state ---
        decoder_init = self.init_hidden(torch.cat([seq_vec, static_vec], dim=-1))
        # split decoder_init into h_0 and c_0
        decoder_h_0, decoder_c_0 = torch.split(decoder_init, decoder_init.size(1)//2, dim=1)
        decoder_h_0 = decoder_h_0.reshape(self.decoder_lstm.num_layers, batch_size, -1)
        decoder_c_0 = decoder_c_0.reshape(self.decoder_lstm.num_layers, batch_size, -1)

        # Prepare decoder input: start with zeros (delta lat/lon at t=0)
        decoder_input = torch.zeros(batch_size, 1, self.output_size, device=x_dynamic.device)

        outputs = []
        hidden = (decoder_h_0, decoder_c_0)

        # Autoregressive decoding with optional teacher forcing
        for t in range(self.forecast_steps):
            out, hidden = self.decoder_lstm(decoder_input, hidden)
            delta_pred = self.fc_out(out)  # shape (batch_size, 1, output_size)
            outputs.append(delta_pred)

            if y_target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # Use true delta as next input (teacher forcing)
                decoder_input = y_target[:, t:t+1, :]
            else:
                # Use predicted delta as next input
                decoder_input = delta_pred

        outputs = torch.cat(outputs, dim=1)  # (batch_size, forecast_steps, output_size)
        return outputs

class Seq2SeqTrajectoryModel(nn.Module):
    """
    NEW NN: Encoder LSTM → Static Encoder → MLP Decoder (no autoregressive loop)
    Predicts all 50 future (lat, lon) in one forward pass.
    Fully compatible with your training/validation/eval loops.
    """
    def __init__(self,
                 num_dynamic_features=10,
                 cat_static_cardinalities=[None, None],
                 cat_static_emb_dims=[8, 8],
                 num_static_numeric=3,
                 encoder_hidden_size=128,
                 lstm_layers=2,
                 output_size=2,
                 forecast_steps=50):
        super().__init__()

        self.forecast_steps = forecast_steps
        self.output_size = output_size

        # Embeddings for static categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, dim) for card, dim in zip(cat_static_cardinalities, cat_static_emb_dims)
        ])

        self.norm_enc = nn.LayerNorm(encoder_hidden_size)
        self.norm_static = nn.LayerNorm(64)

        # Encoder LSTM for dynamic input
        self.encoder_lstm = nn.LSTM(
            input_size=num_dynamic_features,
            hidden_size=encoder_hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )

        # Static feature encoder
        static_emb_dim = sum(cat_static_emb_dims)
        self.fc_static = nn.Sequential(
            nn.Linear(static_emb_dim + num_static_numeric, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Final fused encoder → decoder MLP
        self.fc_fusion = nn.Sequential(
            nn.Linear(encoder_hidden_size + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, forecast_steps * output_size)
        )

    def forward(self, x_dynamic, x_static_cats, x_static_nums, y_target=None, teacher_forcing_ratio=0.0):
        """
        NOTE:
        - teacher_forcing_ratio is ignored (we predict in one shot)
        - This keeps full compatibility with your existing pipeline
        """

        batch_size = x_dynamic.size(0)

        # --- Encode static categorical ---
        embedded_static = [emb(cat) for emb, cat in zip(self.embeddings, x_static_cats)]
        static_cat_emb = torch.cat(embedded_static, dim=-1)

        # --- Combine static numeric + categorical ---
        static_feat = torch.cat([static_cat_emb, x_static_nums], dim=-1)
        static_vec = self.fc_static(static_feat)

        # Normalize static vector
        static_vec = self.norm_static(static_vec)

        # --- Encode sequence using packed sequence (handles padding correctly) ---
        lengths = (x_dynamic.abs().sum(dim=-1) != 0).sum(dim=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x_dynamic, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_T, _) = self.encoder_lstm(packed)
        seq_vec = h_T[-1]

        # Normalize sequence encoder output
        seq_vec = self.norm_enc(seq_vec)

        # --- Fuse ---
        fused = torch.cat([seq_vec, static_vec], dim=-1)

        # --- Decode all 50 steps ---
        out = self.fc_fusion(fused)
        out = out.view(batch_size, self.forecast_steps, self.output_size)
        return out

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

        # Get absolute future positions (forecast_steps + 1 for deltas)
        y_future_df = df.loc[split_idx:split_idx + self.forecast_steps, ['Latitude', 'Longitude']]

        # Pad if too short
        if len(y_future_df) < self.forecast_steps + 1:
            last = y_future_df.iloc[-1].to_numpy(dtype='float32')
            pad = np.tile(last, (self.forecast_steps + 1 - len(y_future_df), 1))
            y_future_df = np.concatenate([y_future_df.to_numpy(dtype='float32'), pad], axis=0)
        else:
            y_future_df = y_future_df.to_numpy(dtype='float32')

        # Compute deltas between consecutive positions (forecast_steps steps)
        delta_lat = y_future_df[1:, 0] - y_future_df[:-1, 0]
        delta_lon = y_future_df[1:, 1] - y_future_df[:-1, 1]
        y_deltas = np.stack([delta_lat, delta_lon], axis=1).astype('float32')  # (forecast_steps, 2)

        # Absolute positions to use for loss (forecast_steps, 2)
        y_abs = y_future_df[1:self.forecast_steps + 1].astype('float32')

        # Static features
        x_static_nums = df.loc[0, static_num_cols].fillna(0).infer_objects().to_numpy(dtype='float32')
        x_static_cats = [torch.tensor(df.loc[0, c], dtype=torch.long) for c in static_cat_cols]

        return (torch.tensor(x_dynamic),
                x_static_cats,
                torch.tensor(x_static_nums)), (torch.tensor(y_deltas), torch.tensor(y_abs))

def collate_fn(batch):
    x_dynamic_list, x_static_cats_list, x_static_nums_list = [], [], []
    y_deltas_list, y_abs_list = [], []
    max_len = max(x[0].shape[0] for x, _ in batch)

    for (x_dynamic, x_static_cats, x_static_nums), (y_deltas, y_abs) in batch:
        # pad dynamic
        pad_len = max_len - x_dynamic.shape[0]
        if pad_len > 0:
            pad = torch.zeros(pad_len, x_dynamic.shape[1], dtype=torch.float32)
            x_dynamic = torch.cat([x_dynamic, pad], dim=0)

        x_dynamic_list.append(x_dynamic)
        x_static_cats_list.append(x_static_cats)
        x_static_nums_list.append(x_static_nums)
        y_deltas_list.append(y_deltas)
        y_abs_list.append(y_abs)

    x_dynamic_batch = torch.stack(x_dynamic_list)
    x_static_nums_batch = torch.stack(x_static_nums_list)
    
    # Stack categorical static features separately
    x_static_cats_batch = []
    for i in range(len(x_static_cats_list[0])):
        cat_feat = torch.stack([cats[i] for cats in x_static_cats_list])
        x_static_cats_batch.append(cat_feat)

    y_deltas_batch = torch.stack(y_deltas_list)  # (batch, forecast_steps, 2)
    y_abs_batch = torch.stack(y_abs_list)        # (batch, forecast_steps, 2)

    return (x_dynamic_batch, x_static_cats_batch, x_static_nums_batch), (y_deltas_batch, y_abs_batch)


def train_loop(model, dataloader, optimizer, criterion, device, teacher_forcing_ratio=0.5):
    model.train()
    total_loss = 0
    for (x_dynamic, x_static_cats, x_static_nums), (y_deltas, y_abs) in dataloader:
        x_dynamic = x_dynamic.to(device)
        x_static_nums = x_static_nums.to(device)
        x_static_cats = [c.to(device) for c in x_static_cats]
        y_deltas = y_deltas.to(device)
        y_abs = y_abs.to(device)

        optimizer.zero_grad()
        y_pred_deltas = model(x_dynamic, x_static_cats, x_static_nums, y_deltas, teacher_forcing_ratio=teacher_forcing_ratio)  # (batch, forecast_steps, 2)

        # Get last known lat/lon from input dynamic sequence (last time step)
        last_known_pos = x_dynamic[:, -1, :2]  # (batch, 2)

        # Reconstruct absolute predicted path by cumulative sum of deltas + last known position
        y_pred_abs = torch.cumsum(y_pred_deltas, dim=1) + last_known_pos.unsqueeze(1)  # (batch, forecast_steps, 2)

        loss = criterion(y_pred_abs, y_abs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)



def val_loop(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for (x_dynamic, x_static_cats, x_static_nums), (y_deltas, y_abs) in dataloader:
            x_dynamic = x_dynamic.to(device)
            x_static_nums = x_static_nums.to(device)
            x_static_cats = [c.to(device) for c in x_static_cats]
            y_abs = y_abs.to(device)

            # No teacher forcing at validation
            y_pred_deltas = model(x_dynamic, x_static_cats, x_static_nums, y_target=None, teacher_forcing_ratio=0.0)

            last_known_pos = x_dynamic[:, -1, :2]
            y_pred_abs = torch.cumsum(y_pred_deltas, dim=1) + last_known_pos.unsqueeze(1)

            loss = criterion(y_pred_abs, y_abs)
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

def evaluate_and_save_predictions(model, dataloader, device, output_json="test_predictions.json"):
    model.eval()
    results = []

    with torch.no_grad():
        seq_id = 0
        for (x_dynamic, x_static_cats, x_static_nums), (y_deltas, y_abs) in dataloader:

            x_dynamic = x_dynamic.to(device)
            x_static_nums = x_static_nums.to(device)
            x_static_cats = [c.to(device) for c in x_static_cats]
            y_abs = y_abs.to(device)

            y_pred_deltas = model(x_dynamic, x_static_cats, x_static_nums, y_target=None, teacher_forcing_ratio=0.0)

            y_pred_np = y_pred_deltas.cpu().numpy()
            y_true_np = y_abs.cpu().numpy()
            x_dynamic_np = x_dynamic.cpu().numpy()

            for i in range(len(y_pred_np)):
                known_points = x_dynamic_np[i][:, :2]
                valid_known = known_points[(known_points[:, 0] != 0.0) | (known_points[:, 1] != 0.0)]
                last_known = valid_known[-1] if len(valid_known) > 0 else np.array([0.0, 0.0])

                abs_preds = np.cumsum(y_pred_np[i], axis=0) + last_known

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
    root = os.path.join(os.getcwd(), "data")
    data = load_sequences(root)
    train_data, val_data, test_data = split_data(data)

    train_ds = TrajectoryDataset(train_data)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn)
    validation_ds = TrajectoryDataset(val_data)
    validation_dl = DataLoader(validation_ds, batch_size=128, shuffle=False, collate_fn=collate_fn)
    test_ds = TrajectoryDataset(test_data)
    test_dl = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collate_fn)

    model = Seq2SeqDeltaModel(
        num_dynamic_features=10,
        cat_static_cardinalities=[num_ship_types, num_cargo_types],
        cat_static_emb_dims=[8, 8],
        num_static_numeric=3,
        encoder_hidden_size=256,
        decoder_hidden_size=256,
        lstm_layers=3,
        output_size=2,
        forecast_steps=50
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    criterion = nn.MSELoss()

    prev_val_loss = float('inf')
    no_improve_epochs = 0
    debounce = 20
    epochs = 10
    for epoch in range(epochs):
        train_loss = train_loop(model, train_dl, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} Train Loss: {train_loss:.15f}")

        validation_loss = val_loop(model, validation_dl, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} Validation Loss: {validation_loss:.15f}")

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

    print("Evaluating on test set...")
    evaluate_and_save_predictions(model, test_dl, device, output_json="test_predictions.json")

if __name__ == "__main__":
    main()
    