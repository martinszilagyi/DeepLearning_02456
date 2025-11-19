import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class SegmentData:
    def __init__(self, path_df, stationary_df):
        self.path_df = path_df
        self.stationary_df = stationary_df

class TrajectoryDataset(Dataset):
    def __init__(self, segment_data_list, input_len, pred_len, feature_cols, target_cols):
        self.samples = []
        self.static = []
        self.targets = []
        for segment in segment_data_list:
            seqs, targets = create_sequences(segment.path_df, input_len, pred_len, feature_cols, target_cols)
            static_feat = segment.stationary_df.values.repeat(len(seqs), axis=0)  # repeat for all seqs
            self.samples.append(seqs)
            self.targets.append(targets)
            self.static.append(torch.tensor(static_feat, dtype=torch.float32))
        self.samples = torch.cat(self.samples, dim=0)
        self.targets = torch.cat(self.targets, dim=0)
        self.static = torch.cat(self.static, dim=0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.static[idx], self.targets[idx]

def create_sequences(df, input_len, pred_len, feature_cols, target_cols):
    seqs = []
    targets = []
    total_len = len(df)
    for start in range(total_len - input_len - pred_len + 1):
        seq = df.iloc[start : start + input_len][feature_cols].values
        target = df.iloc[start + input_len : start + input_len + pred_len][target_cols].values
        seqs.append(seq)
        targets.append(target)
    return torch.tensor(np.array(seqs), dtype=torch.float32), torch.tensor(np.array(targets), dtype=torch.float32)

def find_all_segments(root_folder):
    segment_files = []
    for dirpath, _, files in os.walk(root_folder):
        # Find all segment parquet files (exclude stationary files)
        segment_paths = [f for f in files if f.endswith(".parquet") and not f.endswith("_stationary.parquet")]
        for seg_file in segment_paths:
            path_file = os.path.join(dirpath, seg_file)
            stationary_file = os.path.join(dirpath, seg_file.replace(".parquet", "_stationary.parquet"))
            if os.path.exists(stationary_file):
                segment_files.append((path_file, stationary_file))
    return segment_files

def load_segments(segment_files):
    segment_data_list = []
    for path_file, stationary_file in segment_files:
        path_df = pd.read_parquet(path_file)
        stationary_df = pd.read_parquet(stationary_file)
        segment_data_list.append(SegmentData(path_df, stationary_df))
    return segment_data_list

if __name__ == "__main__":
    data_folder = os.path.join(os.getcwd(), "data")
    all_segment_files = find_all_segments(data_folder)
    segment_data_list = load_segments(all_segment_files)

    train_set, temp = train_test_split(segment_data_list, test_size=0.2, random_state=1)       #80% train
    val_set, test_set = train_test_split(temp, test_size=0.5, random_state=1)                  #10% val, 10% test

    input_len = 90      # e.g., use 90 past points
    pred_len = 20       # predict next 20 points

    feature_cols = ['Latitude', 'Longitude', 'ROT', 'SOG', 'COG', 'Heading', 'Navigational status_enc']
    target_cols = ['Latitude', 'Longitude']

    train_dataset = TrajectoryDataset(train_set, input_len, pred_len, feature_cols, target_cols)
    val_dataset = TrajectoryDataset(val_set, input_len, pred_len, feature_cols, target_cols)
    test_dataset = TrajectoryDataset(test_set, input_len, pred_len, feature_cols, target_cols)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    print(f"Train segments: {len(train_set)}, Validation segments: {len(val_set)}, Test segments: {len(test_set)}")
    print("Example train segment data:")
    print(train_set[0].path_df.head())
    print("Example stationary segment data:")
    print(train_set[0].stationary_df.head())

    print("Example validation segment data:")
    print(val_set[0].path_df.head())
    print("Example stationary segment data:")
    print(val_set[0].stationary_df.head())

    print("Example test segment data:")
    print(test_set[0].path_df.head())
    print("Example stationary segment data:")
    print(test_set[0].stationary_df.head())

    print(f"Train samples after sliding window: {len(train_dataset)}")
    print(f"Validation samples after sliding window: {len(val_dataset)}")
    print(f"Test samples after sliding window: {len(test_dataset)}")

    # Print example sample shapes
    sample_seq, sample_static, sample_target = train_dataset[0]
    print(f"Sample input sequence shape: {sample_seq.shape}")      # (input_len, features)
    print(f"Sample static features shape: {sample_static.shape}")   # (static_features,)
    print(f"Sample target shape: {sample_target.shape}")            # (pred_len, target_features)

    print(f"All column names in path data: {train_set[0].path_df.columns.tolist()}  ")

