"""
This script implements a sequence-to-sequence model with attention mechanism
for ship trajectory prediction using PyTorch. It includes data loading,
preprocessing, model definition, training with early stopping, and evaluation.
"""

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
import sys
import select

# Use only a few path for quick testing
USE_ONE_PATH = True

# Argument parser for bsub flag (for jobs on cluster)
parser = argparse.ArgumentParser()
parser.add_argument("--isbsub", action="store_true")

args = parser.parse_args()
if args.isbsub:
    postifx = "_bsub"
    USE_ONE_PATH = False
else:
    postifx = ""

#Suppress pandas future warning for downcasting
pd.set_option('future.no_silent_downcasting', True)

# Number of unique navigational statuses
num_nav_status = 0

# Attention mechanism
class BahdanauAttention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim, attn_dim):
        super().__init__()
        self.W_h = nn.Linear(enc_hidden_dim, attn_dim, bias=False)
        self.W_s = nn.Linear(dec_hidden_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, encoder_outputs, decoder_hidden, mask=None):
        seq_len = encoder_outputs.size(1)
        decoder_hidden_exp = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
        # Score calculation
        score = self.v(torch.tanh(self.W_h(encoder_outputs) + self.W_s(decoder_hidden_exp)))
        score = score.squeeze(-1)
        # Filter scores with mask (If the point is padded, set score to -inf)
        if mask is not None:
            score = score.masked_fill(~mask, float('-inf'))
        attn_weights = torch.softmax(score, dim=1).unsqueeze(-1)
        context = torch.sum(attn_weights * encoder_outputs, dim=1)
        return context, attn_weights.squeeze(-1)

# NN Model
class DeltaTransformerModel(nn.Module):
    def __init__(self,
                 num_dynamic_features=8,
                 num_of_nav_status=10,
                 embedding_dim=32,
                 d_model=128,
                 nhead=8,
                 num_encoder_layers=3,
                 dim_feedforward=256,
                 dropout=0.1,
                 output_size=2,
                 forecast_steps=60):
        super().__init__()

        self.forecast_steps = forecast_steps
        self.output_size = output_size

        # Navigational status embedding
        self.cat_embedding = nn.Embedding(num_of_nav_status, embedding_dim, padding_idx=0)

        # Feature embedding: concat dynamic features + embedded categorical
        self.input_embedding = nn.Linear(num_dynamic_features + embedding_dim, d_model)

        # Positional encoding (sinusoidal)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Decoder MLP to predict delta
        self.decoder = nn.Sequential(
            nn.Linear(d_model + output_size, d_model),
            nn.ReLU(),
            nn.Linear(d_model, output_size)
        )

    def forward(self, x_dynamic, x_emb_nav_status, ground_truth=None, tf=0.8):
        """
        x_dynamic: (batch, seq_len, num_dynamic_features)
        x_emb_nav_status: (batch, seq_len)
        ground_truth: (batch, forecast_steps, output_size)
        """
        device = x_dynamic.device

        # Embed categorical nav status
        emb_nav = self.cat_embedding(x_emb_nav_status.long())  # (B, S, embedding_dim)

        # Combine features
        x_in = torch.cat([x_dynamic, emb_nav], dim=-1)  # (B, S, num_dynamic+embedding_dim)

        # Project to d_model
        x_emb = self.input_embedding(x_in)  # (B, S, d_model)

        # Add positional encoding (expects seq_len, batch, d_model)
        x_emb = self.pos_encoder(x_emb.permute(1,0,2))  # (S, B, d_model)

        # Transformer encoder
        memory = self.transformer_encoder(x_emb)  # (S, B, d_model)

        # Start prediction from last known position delta = 0
        batch_size = x_dynamic.size(0)
        prev_delta = torch.zeros(batch_size, self.output_size, device=device)

        # Last hidden state from transformer (take last time step)
        last_enc = memory[-1]  # (B, d_model)

        outputs = []
        tf_decay = tf / (self.forecast_steps + 200)

        for t in range(self.forecast_steps):
            # Decoder input: concat last enc output + prev delta
            dec_in = torch.cat([last_enc, prev_delta], dim=-1)  # (B, d_model + output_size)
            pred_delta = self.decoder(dec_in)  # (B, output_size)

            outputs.append(pred_delta)

            # Teacher forcing: ground truth delta vagy predikció váltogatása
            if ground_truth is not None and torch.rand(1).item() < tf:
                prev_delta = ground_truth[:, t, :]
            else:
                prev_delta = pred_delta.detach()

            tf = max(0.0, tf - tf_decay)

        outputs = torch.stack(outputs, dim=1)  # (B, forecast_steps, output_size)
        return outputs


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (seq_len, batch, d_model)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# Trajectory dataset type
class TrajectoryDataset(Dataset):
    def __init__(self, sequences, forecast_steps=60):
        self.sequences = sequences
        self.forecast_steps = forecast_steps

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        df = self.sequences[idx].copy()
        dynamic_cols = ['Latitude', 'Longitude', 'SOG', 'cos_cog', 'sin_cog', 'vx', 'vy', 'Heading']
        nav_status_col = 'Navigational status_enum'

        # Fixed split index for 240-length sequences (4 hours of data)
        # We crop all sequences to 240 length during data loading and slice them into fixed parts
        # 3 hours known (180 points), 1 hour forecast (60 points)
        split_idx = 180

        # Dynamic features + navigational status embedding
        x_dynamic = df.loc[:split_idx-1, dynamic_cols].to_numpy(dtype='float32')
        x_emb_nav_status = df.loc[:split_idx-1, nav_status_col].to_numpy(dtype='int64')

        # Absolute future points (only lat/lon)
        y_abs = df.loc[split_idx:split_idx + self.forecast_steps - 1,['Latitude', 'Longitude']].to_numpy(dtype='float32')

        # Data looks like: ( x_dynamic: (300, 8), x_emb_nav_status: (300,), y_abs: (60, 2) )
        return torch.tensor(x_dynamic), torch.tensor(x_emb_nav_status), torch.tensor(y_abs)

def check_for_quit_key():
    # Non-blocking check for 'q' key press to quit
    dr,dw,de = select.select([sys.stdin], [], [], 0)
    if dr:
        c = sys.stdin.read(1)
        if c == 'q':
            return True
    return False

# Collate function for DataLoader to have variable-length sequences in batch (not used with fixed-length slices)
def collate_fn(batch):
    x_dynamic_list = []
    y_abs_list = []
    x_emb_nav_status_list = []

    lengths = [x_dynamic.shape[0] for (x_dynamic, x_emb_nav_status, y_abs) in batch]
    max_len = max(lengths)

    for x_dynamic, x_emb_nav_status, y_abs in batch:

        # Convert to tensors if not already
        if (True): # Here for make the code a bit cleaner
            if not isinstance(x_dynamic, torch.Tensor):
                x_dynamic = torch.tensor(x_dynamic, dtype=torch.float32)
            else:
                x_dynamic = x_dynamic.detach().clone().float()
            if not isinstance(y_abs, torch.Tensor):
                y_abs = torch.tensor(y_abs, dtype=torch.float32)
            else:
                y_abs = y_abs.detach().clone().float()
            if not isinstance(x_emb_nav_status, torch.Tensor):
                x_emb_nav_status = torch.tensor(x_emb_nav_status, dtype=torch.int64)
            else:
                x_emb_nav_status = x_emb_nav_status.detach().clone().long()

        # Padding with NaNs
        pad_len = max_len - x_dynamic.shape[0]
        if pad_len > 0:
            pad = torch.full((pad_len, x_dynamic.shape[1]), float('nan'))
            x_dynamic = torch.cat([x_dynamic, pad], dim=0)
            pad_y = torch.full((pad_len, y_abs.shape[1]), float('nan'))
            y_abs = torch.cat([y_abs, pad_y], dim=0)
            pad_emb = torch.full((pad_len,), 0, dtype=torch.int64)
            x_emb_nav_status = torch.cat([x_emb_nav_status, pad_emb], dim=0)

        # Append to lists
        x_dynamic_list.append(x_dynamic)
        y_abs_list.append(y_abs)
        x_emb_nav_status_list.append(x_emb_nav_status)

    # Stack into batches
    x_dynamic_batch = torch.stack(x_dynamic_list)
    y_abs_batch = torch.stack(y_abs_list)
    x_emb_nav_status_batch = torch.stack(x_emb_nav_status_list)

    # Return batched tensors. Looks like: ( x_dynamic: (batch_size, max_len, 8), x_emb_nav_status: (batch_size, max_len), y_abs: (batch_size, max_len, 2), y_heading: (batch_size, max_len), y_turn_rate: (batch_size, max_len) )
    return x_dynamic_batch, x_emb_nav_status_batch, y_abs_batch

# Training loop
def train_loop(model, dataloader, optimizer, criterion, device):
    # Set model to training mode
    model.train()
    total_loss = 0
    for x_dynamic, x_emb_nav_status, y_abs in dataloader:
        # Move data to device
        x_dynamic = x_dynamic.to(device)
        x_emb_nav_status = x_emb_nav_status.to(device)
        y_abs = y_abs.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Try model forward pass
        y_pred_abs = model(x_dynamic, x_emb_nav_status, y_abs, 1.0)

        # Ensure y_abs is float
        y_abs = y_abs.float()

        # Compute loss
        loss_pos = criterion(y_pred_abs, y_abs)
        loss = loss_pos

        # Backpropagation
        loss.backward()
        
        # Gradient clipping (for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Validation loop
def val_loop(model, dataloader, criterion, device):
    # Set model to evaluation mode
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x_dynamic, x_emb_nav_status, y_abs in dataloader:
            # Move data to device
            x_dynamic = x_dynamic.to(device)
            x_emb_nav_status = x_emb_nav_status.to(device)
            y_abs = y_abs.to(device)

            # Model forward pass
            y_pred_abs = model(x_dynamic, x_emb_nav_status, y_abs, 0.0)
            
            # Ensure y_abs is float
            y_abs = y_abs.float()
            # Compute loss
            loss_pos = criterion(y_pred_abs, y_abs)
            loss = loss_pos

            # Accumulate loss (no need to clip gradients or optimizer step because no backprop)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# Load sequences from data directory
def load_sequences(root):
    global num_nav_status

    sequences = []

    # For each trajectory (segment) in the dataset
    for mmsi_folder in glob.glob(os.path.join(root, "MMSI=*")):
        for seg_folder in glob.glob(os.path.join(mmsi_folder, "segment=*")):
            files = glob.glob(os.path.join(seg_folder, "*.parquet"))
            if not files:
                continue

            # Load and concatenate all parquet files in the segment
            dfs = []
            for f in files:
                df_tmp = pd.read_parquet(f)
                # Skip short files (shorter than 4 hours of data)
                if len(df_tmp) < 242:
                    continue
                dfs.append(df_tmp)

            # If after filtering there are no valid files, skip
            if not dfs:
                continue

            df = pd.concat(dfs, ignore_index=True)

            # Sort by timestamp if available
            if "Timestamp" in df.columns:
                df = df.sort_values("Timestamp").reset_index(drop=True)

            # Shift enums by +1 to fix -1 values (During calculating enums, -1 was used for NaN)
            for col in ['Ship type_enum', 'Cargo type_enum', 'Navigational status_enum']:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: x + 1 if x >= 0 else 0)

            sequences.append(df)

    # Combine all sequences
    combined_df = pd.concat(sequences, ignore_index=True)

    # Determine number of unique navigational statuses for embedding size
    num_nav_status = combined_df['Navigational status_enum'].max() + 1

    # Debug only (print stats)
    print("Max Navigational status_enum:", combined_df['Navigational status_enum'].max())
    print("Num navigational statuses (embedding size):", num_nav_status)

    # Return with filtered sequences that look like ( pd.DataFrame1, pd.DataFrame2, ... )
    return sequences

# Create random fixed-length slices from sequences with a maximum number of slices per sequence
def create_random_slices_with_max(sequences, slice_length=240, max_slices=3, min_start_idx=2):
    slices = []
    
    for df in sequences:
        seq_len = len(df)
        if seq_len < slice_length + min_start_idx:
            # Nem lehet egyetlen slice-ot sem készíteni
            continue
        
        max_start = seq_len - slice_length  # maximum start index
        # A valid start index-ek tartománya: [min_start_idx, max_start]
        possible_starts = np.arange(min_start_idx, max_start + 1)
        
        # Ha kevesebb start index van, mint max_slices, akkor annyit veszünk, amennyi van
        num_slices = min(max_slices, len(possible_starts))
        
        # Véletlenszerűen választunk ki start indexeket ismétlés nélkül
        chosen_starts = np.random.choice(possible_starts, size=num_slices, replace=False)
        
        for start_idx in chosen_starts:
            slice_df = df.iloc[start_idx:start_idx + slice_length].reset_index(drop=True)
            slices.append(slice_df)
    
    print(f"Created {len(slices)} random slices (max {max_slices} per sequence)")
    return slices

# Split data into train, val, test sets
def split_data(sequences, val_ratio=0.20, test_ratio=0.05, random_state=333):
    # First split off the test set
    train_and_val, test = train_test_split(sequences, test_size=test_ratio, random_state=random_state)

    # Then split train and validation sets
    train, val = train_test_split(train_and_val, test_size=val_ratio/(1 - test_ratio), random_state=random_state)

    print(f"Data split: Train={len(train)}, Val={len(val)}, Test={len(test)}")

    # Return with the split datasets
    return train, val, test

# Evaluate on the test set and save predictions to JSON for further visualizations
def evaluate_and_save_predictions(model, dataloader, device, output_json="test_predictions.json"):
    # Evaluate model
    model.eval()
    results = []

    with torch.no_grad():
        seq_id = 0
        for x_dynamic, x_emb_nav_status, y_abs in dataloader:
            # Move data to device
            x_dynamic = x_dynamic.to(device)
            x_emb_nav_status = x_emb_nav_status.to(device)
            y_abs = y_abs.to(device)

            # Model prediction
            y_pred_abs = model(x_dynamic, x_emb_nav_status, y_abs, tf=0.0)

            # Convert to numpy
            y_pred_np = y_pred_abs.cpu().numpy()
            y_true_np = y_abs.cpu().numpy()
            x_dynamic_np = x_dynamic.cpu().numpy()

            # Iterate over the batch
            for i in range(len(y_pred_np)):
                
                # Get absolute predictions
                abs_preds = y_pred_np[i]

                # JSON format:
                # {"sequence_id": int,
                # "known_path": [ {"lat": float, "lon": float}, ... ],
                # "ground_truth": [ {"lat": float, "lon": float}, ... ],
                # "prediction": [ {"lat": float, "lon": float}, ... ]}
                results.append({
                    "sequence_id": seq_id,
                    "known_path": [
                        {"lat": float(p[0]), "lon": float(p[1])}
                        for p in x_dynamic_np[i][:, :2] if not (p[0] == 0.0 and p[1] == 0.0)
                    ],
                    "ground_truth": [
                        {"lat": float(p[0]), "lon": float(p[1])}
                        for p in y_true_np[i]
                    ],
                    "prediction": [
                        {"lat": float(p[0]), "lon": float(p[1])}
                        for p in abs_preds[:, 0:2]
                    ]
                })

                seq_id += 1

    # Save to JSON file
    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved predictions to {output_json}")

# Main function
def main():

    # Load and prepare data
    root = os.path.join(os.getcwd(), "data")
    data = load_sequences(root)
    data = create_random_slices_with_max(data, slice_length=240, max_slices=3, min_start_idx=2)

    # Initialize model
    model = DeltaTransformerModel(
        num_dynamic_features=8,                 # Latitude, Longitude, SOG, cos_cog, sin_cog, vx, vy, Heading
        num_of_nav_status=num_nav_status,       # Number of unique navigational statuses
        embedding_dim=32,                       # Beállíthatod, vagy hagyhatod alapértelmezettként
        d_model=200,                            # Transformer belső dimenziója, az encoder_hidden_size helyett
        nhead=8,                                # Fejek száma az attentionben, tuningolható
        num_encoder_layers=3,                   # Rétegek száma a Transformer encoderben
        dim_feedforward=400,                    # Feedforward layer mérete (erősítés az eredeti 256-hoz képest)
        dropout=0.1,
        output_size=2,                          # Latitude, Longitude delta-ként
        forecast_steps=60                       # Előrejelzési lépések száma (1 óra, ha 1 perces lépések)
    )

    # Check number of trainable parameters
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    # For Debug: use only a few paths
    if (USE_ONE_PATH):
        #random 100 data points
        data = [data[i] for i in np.random.choice(len(data), 100, replace=False)]
        # train = val = test = data (ONLY FOR DEBUG)
        train_data = data
        val_data = data
        test_data = data
    else:
        # Split data into train, validation, and test sets
        train_data, val_data, test_data = split_data(data)

    # Create datasets and dataloaders
    train_ds = TrajectoryDataset(train_data)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn)
    validation_ds = TrajectoryDataset(val_data)
    validation_dl = DataLoader(validation_ds, batch_size=128, shuffle=False, collate_fn=collate_fn)
    test_ds = TrajectoryDataset(test_data)
    test_dl = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collate_fn)

    # Training setup (CUDA or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Transfer model to device
    model.to(device)
    # Initialize optimizer and loss function
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # Huber loss (Better than MSE for path prediction due to MSE preferencing average paths)
    criterion = nn.SmoothL1Loss()

    # Training loop with early stopping
    prev_val_loss = float('inf')
    # Number of epochs with no improvement in validation loss
    no_improve_epochs = 0
    # Number of epochs to wait before early stopping
    debounce = 100
    # Total epochs
    epochs = 5000

    loss_pairs = []
    for epoch in range(epochs):
        # Training step
        train_loss = train_loop(model, train_dl, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} Train Loss: {train_loss:.15f}")

        # Validation step
        validation_loss = val_loop(model, validation_dl, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} Validation Loss: {validation_loss:.15f}")

        # Record losses for plotting later
        loss_pairs.append((train_loss, validation_loss))

        # Early stopping check
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

        if check_for_quit_key():
            print("Quit key 'q' pressed. Stopping training early.")
            break

    # Save the trained model (Not used at the moment)
    torch.save(model.state_dict(), "ship_trajectory_model.pth")
    print("Model saved to ship_trajectory_model.pth")

    # Write loss pairs to JSON for plotting. JSON format: [ {"train_loss": float, "val_loss": float}, ... ]
    with open("loss_pairs"+postifx+".json", "w") as f:
        json.dump([{"train_loss": tl, "val_loss": vl} for tl, vl in loss_pairs], f, indent=2)

    # Evaluate on test set and save predictions to JSON
    evaluate_and_save_predictions(model, test_dl, device, output_json="test_predictions"+postifx+".json")

    # Also evaluate on train set and save predictions to JSON (for overfitting analysis)
    evaluate_and_save_predictions(model, train_dl, device, output_json="train_predictions"+postifx+".json")

if __name__ == "__main__":
    main()
