import numpy as np
import torch
import os
from pathlib import Path

class MMIANNTrainer:
    """Trainer for micro-mechanism informed neural networks with thermodynamic constraints"""
    
    def __init__(self, model, device='cpu', thermo_weight=0.1):
        self.model = model.to(device)
        self.device = device
        self.thermo_weight = thermo_weight
        Path("models").mkdir(exist_ok=True)
        
    def train(self, train_loader, val_loader, optimizer, scheduler, 
             num_epochs=200, early_stopping_patience=30, min_epochs=50, 
             model_save_path='models/material_model_best.pt'):
        
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        component_losses = {
            'train': {'temperature': [], 'stress': [], 'grain_size': [], 'delta_grain': [], 'thermodynamic': []},
            'val': {'temperature': [], 'stress': [], 'grain_size': [], 'delta_grain': [], 'thermodynamic': []}
        }
        
        loss_fn = torch.nn.HuberLoss(delta=1.0)
        
        print(f"Training on {self.device}, thermo_weight={self.thermo_weight}")
        
        for epoch in range(num_epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            train_comp = {k: 0.0 for k in component_losses['train'].keys()}
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                
                temp_loss = loss_fn(outputs['temperature'], y_batch[:, 0:1])
                stress_loss = loss_fn(outputs['stress'], y_batch[:, 1:2])
                grain_loss = loss_fn(outputs['grain_size'], y_batch[:, 2:3])
                delta_loss = loss_fn(outputs['delta_grain'], y_batch[:, 3:4])
                
                thermo_loss = 0.0
                if 'stress_consistency_error' in outputs:
                    thermo_loss += outputs['stress_consistency_error']
                if 'second_law_error' in outputs:
                    thermo_loss += outputs['second_law_error'] * 10.0
                if 'energy_balance_error' in outputs:
                    thermo_loss += outputs['energy_balance_error']
                
                # Handle NaN
                for l in [temp_loss, stress_loss, grain_loss, delta_loss, thermo_loss]:
                    if torch.isnan(l):
                        l = torch.tensor(1.0, device=self.device)
                
                loss = temp_loss + stress_loss + 0.8*grain_loss + 0.8*delta_loss + self.thermo_weight*thermo_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_comp['temperature'] += temp_loss.item()
                train_comp['stress'] += stress_loss.item()
                train_comp['grain_size'] += grain_loss.item()
                train_comp['delta_grain'] += delta_loss.item()
                train_comp['thermodynamic'] += thermo_loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            for k in train_comp:
                train_comp[k] /= len(train_loader)
                component_losses['train'][k].append(train_comp[k])
            
            # Validate
            self.model.eval()
            val_loss = 0.0
            val_comp = {k: 0.0 for k in component_losses['val'].keys()}
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    outputs = self.model(X_batch)
                    
                    temp_loss = loss_fn(outputs['temperature'], y_batch[:, 0:1])
                    stress_loss = loss_fn(outputs['stress'], y_batch[:, 1:2])
                    grain_loss = loss_fn(outputs['grain_size'], y_batch[:, 2:3])
                    delta_loss = loss_fn(outputs['delta_grain'], y_batch[:, 3:4])
                    
                    thermo_loss = 0.0
                    if 'stress_consistency_error' in outputs:
                        thermo_loss += outputs['stress_consistency_error']
                    if 'second_law_error' in outputs:
                        thermo_loss += outputs['second_law_error'] * 10.0
                    if 'energy_balance_error' in outputs:
                        thermo_loss += outputs['energy_balance_error']
                    
                    for l in [temp_loss, stress_loss, grain_loss, delta_loss, thermo_loss]:
                        if torch.isnan(l):
                            l = torch.tensor(1.0, device=self.device)
                    
                    loss = temp_loss + stress_loss + 0.8*grain_loss + 0.8*delta_loss + self.thermo_weight*thermo_loss
                    
                    val_loss += loss.item()
                    val_comp['temperature'] += temp_loss.item()
                    val_comp['stress'] += stress_loss.item()
                    val_comp['grain_size'] += grain_loss.item()
                    val_comp['delta_grain'] += delta_loss.item()
                    val_comp['thermodynamic'] += thermo_loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            for k in val_comp:
                val_comp[k] /= len(val_loader)
                component_losses['val'][k].append(val_comp[k])
            
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Train: {train_loss:.4f}, Val: {val_loss:.4f}")
                print(f"  T: {val_comp['temperature']:.4f}, σ: {val_comp['stress']:.4f}, " +
                      f"d: {val_comp['grain_size']:.4f}, Δd: {val_comp['delta_grain']:.4f}, " +
                      f"Thermo: {val_comp['thermodynamic']:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                try:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'component_losses': component_losses
                    }, model_save_path)
                except Exception as e:
                    print(f"Save failed: {e}")
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience and epoch >= min_epochs:
                print(f"Early stop at epoch {epoch+1}")
                break
        
        # Load best
        try:
            if os.path.exists(model_save_path):
                checkpoint = torch.load(model_save_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded best from epoch {checkpoint['epoch'] + 1}")
        except Exception as e:
            print(f"Load failed: {e}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_epoch': best_epoch + 1,
            'best_val_loss': best_val_loss,
            'component_losses': component_losses
        }
    
    def evaluate(self, test_loader, input_mean=None, input_std=None, target_mean=None, target_std=None):
        self.model.eval()
        
        all_y_true = []
        all_predictions = {
            'temperature': [], 'stress': [], 'grain_size': [], 'delta_grain': [],
            'dislocation_density': [], 'precip_fraction': [], 'free_energy': [], 'dissipation': []
        }
        
        thermo_metrics = {'stress_consistency_error': 0.0, 'second_law_error': 0.0, 'energy_balance_error': 0.0}
        thermo_count = 0
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                
                all_y_true.append(y_batch.numpy())
                for key in all_predictions:
                    if key in outputs:
                        all_predictions[key].append(outputs[key].cpu().numpy())
                
                for key in thermo_metrics:
                    if key in outputs:
                        thermo_metrics[key] += outputs[key].cpu().item()
                        thermo_count += 1
        
        all_y_true = np.vstack(all_y_true)
        for key in all_predictions:
            if len(all_predictions[key]) > 0:
                all_predictions[key] = np.vstack(all_predictions[key])
        
        if thermo_count > 0:
            for key in thermo_metrics:
                thermo_metrics[key] /= thermo_count
        
        # Denormalize
        if target_mean is not None and target_std is not None:
            all_y_true_orig = all_y_true * target_std + target_mean
            all_predictions_orig = {}
            
            target_names = ['temperature', 'stress', 'grain_size', 'delta_grain']
            for i, key in enumerate(target_names):
                if key in all_predictions and len(all_predictions[key]) > 0:
                    all_predictions_orig[key] = all_predictions[key] * target_std[i] + target_mean[i]
            
            for key in ['dislocation_density', 'precip_fraction', 'free_energy', 'dissipation']:
                if key in all_predictions and len(all_predictions[key]) > 0:
                    all_predictions_orig[key] = all_predictions[key]
            
            for key, value in thermo_metrics.items():
                all_predictions_orig[key] = value
            
            return all_y_true_orig, all_predictions_orig
        
        for key, value in thermo_metrics.items():
            all_predictions[key] = value
        
        return all_y_true, all_predictions
    
    def predict(self, inputs, input_mean=None, input_std=None, target_mean=None, target_std=None):
        self.model.eval()
        
        if isinstance(inputs, np.ndarray):
            inputs = torch.FloatTensor(inputs)
        
        if input_mean is not None and input_std is not None:
            inputs = (inputs - torch.FloatTensor(input_mean)) / torch.FloatTensor(input_std)
        
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(inputs)
        
        if target_mean is not None and target_std is not None:
            target_names = ['temperature', 'stress', 'grain_size', 'delta_grain']
            for i, key in enumerate(target_names):
                if key in outputs:
                    outputs[key] = outputs[key].cpu() * target_std[i] + target_mean[i]
        
        return outputs
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Saved to {path}")
    
    def load_model(self, path):
        try:
            checkpoint = torch.load(path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            self.model = self.model.to(self.device)
            print(f"Loaded from {path}")
        except Exception as e:
            print(f"Load failed: {e}")
        return self.model
