import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class MaterialDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.FloatTensor(inputs)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def load_and_prepare_data(file_path, input_columns=None, target_columns=None):
    if input_columns is None:
        input_columns = ["strain", "time", "cooling_rate"]
    if target_columns is None:
        target_columns = ["temperature", "stress", "grain_size", "delta_grain"]
    
    data = pd.read_csv(file_path)
    print(f"Loaded {len(data)} rows, {len(data.columns)} columns")
    
    inputs = data[input_columns].values
    targets = data[target_columns].values
    
    return inputs, targets, data

def normalize_data(data):
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)
    return normalized_data, scaler.mean_, scaler.scale_

def create_data_loaders(inputs, targets, batch_size=32, test_size=0.15, val_size=0.15):
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        inputs, targets, test_size=test_size, random_state=42
    )
    
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=42
    )
    
    train_dataset = MaterialDataset(X_train, y_train)
    val_dataset = MaterialDataset(X_val, y_val)
    test_dataset = MaterialDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    data_info = {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test
    }
    
    return train_loader, val_loader, test_loader, data_info
