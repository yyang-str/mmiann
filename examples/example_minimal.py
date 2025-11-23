"""
Minimal example for training MMIANN model

This script shows the simplest possible workflow.
For full options, see main.py
"""

import torch
import numpy as np

from src.data_utils import load_and_prepare_data, normalize_data, create_data_loaders
from src.networks import ImprovedPhysicsInformedNN
from src.mmiann import MMIANNTrainer

# Configuration
DATA_FILE = 'data/example_data_simple.csv'  # Simple synthetic data for quick testing
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 500
BATCH_SIZE = 8
LEARNING_RATE = 0.001
THERMO_WEIGHT = 0.1

print(f"Using device: {DEVICE}")

# Load and prepare data
print(f"Loading {DATA_FILE}")
inputs, targets, df = load_and_prepare_data(
    DATA_FILE,
    input_columns=["strain", "time", "cooling_rate"],
    target_columns=["temperature", "stress", "grain_size", "delta_grain"]
)

# Normalize
inputs_norm, in_mean, in_std = normalize_data(inputs)
targets_norm, out_mean, out_std = normalize_data(targets)

# Save normalization params
np.savez('models/normalization_params.npz',
         input_mean=in_mean, input_std=in_std,
         target_mean=out_mean, target_std=out_std)

# Create data loaders
train_loader, val_loader, test_loader, _ = create_data_loaders(
    inputs_norm, targets_norm, batch_size=BATCH_SIZE
)

# Initialize model
model = ImprovedPhysicsInformedNN(input_dim=3, hidden_dim=64)
trainer = MMIANNTrainer(model, device=DEVICE, thermo_weight=THERMO_WEIGHT)

# Setup optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-5
)

# Train
print("Training...")
training_info = trainer.train(
    train_loader, val_loader, optimizer, scheduler,
    num_epochs=EPOCHS,
    early_stopping_patience=30,
    min_epochs=50,
    model_save_path='models/best_model.pt'
)

print(f"\nBest epoch: {training_info['best_epoch']}")
print(f"Best validation loss: {training_info['best_val_loss']:.6f}")

# Evaluate on test set
print("\nEvaluating...")
y_true, predictions = trainer.evaluate(
    test_loader, in_mean, in_std, out_mean, out_std
)

# Calculate metrics
target_names = ["temperature", "stress", "grain_size", "delta_grain"]
print("\nTest set metrics:")
for i, name in enumerate(target_names):
    if name in predictions:
        true_vals = y_true[:, i]
        pred_vals = predictions[name].flatten()
        
        r2 = np.corrcoef(true_vals, pred_vals)[0, 1]**2
        rmse = np.sqrt(np.mean((true_vals - pred_vals)**2))
        mae = np.mean(np.abs(true_vals - pred_vals))
        
        print(f"{name:15s}: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

# Thermodynamic consistency
print("\nThermodynamic consistency:")
for key in ['stress_consistency_error', 'second_law_error', 'energy_balance_error']:
    if key in predictions:
        print(f"  {key:30s}: {predictions[key]:.6f}")

print("\nDone! Model saved to models/best_model.pt")
