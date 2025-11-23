import os
import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path

from src.data_utils import load_and_prepare_data, normalize_data, create_data_loaders
from src.networks import ImprovedPhysicsInformedNN
from src.mmiann import MMIANNTrainer
from src.visualization import MaterialVisualizer

def parse_args():
    parser = argparse.ArgumentParser(description='MMIANN training and evaluation')
    parser.add_argument('--data_file', type=str, default='data/your_dataset.csv')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--min_epochs', type=int, default=50)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--model_save_path', type=str, default='models/best_model_thermo.pt')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--results_dir', type=str, default='results_thermo')
    parser.add_argument('--mode', type=str, default='train', 
                      choices=['train', 'evaluate', 'visualize', 'microstructure_viz', 'export_csv'])
    parser.add_argument('--thermo_weight', type=float, default=1)
    return parser.parse_args()

def main():
    args = parse_args()
    Path('models').mkdir(exist_ok=True)
    Path(args.results_dir).mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    input_columns = ["strain", "time", "cooling_rate"]
    target_columns = ["temperature", "stress", "grain_size", "delta_grain"]

    print(f"Loading {args.data_file}")
    inputs, targets, original_df = load_and_prepare_data(args.data_file, input_columns, target_columns)

    inputs_norm, input_mean, input_std = normalize_data(inputs)
    targets_norm, target_mean, target_std = normalize_data(targets)

    np.savez('models/normalization_params.npz', 
             input_mean=input_mean, input_std=input_std,
             target_mean=target_mean, target_std=target_std)

    train_loader, val_loader, test_loader, data_info = create_data_loaders(
        inputs_norm, targets_norm, batch_size=args.batch_size
    )

    model = ImprovedPhysicsInformedNN(input_dim=len(input_columns), hidden_dim=args.hidden_dim)
    trainer = MMIANNTrainer(model, device=device, thermo_weight=args.thermo_weight)
    visualizer = MaterialVisualizer(save_dir=args.results_dir)

    if args.load_model and os.path.exists(args.model_save_path):
        print(f"Loading {args.model_save_path}")
        trainer.load_model(args.model_save_path)
    
    if args.mode == 'train':
        print(f"Training with thermo_weight={args.thermo_weight}")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=15, verbose=True, min_lr=1e-5
        )
        
        training_info = trainer.train(
            train_loader, val_loader, optimizer, scheduler,
            num_epochs=args.epochs, early_stopping_patience=args.patience,
            min_epochs=args.min_epochs, model_save_path=args.model_save_path
        )
        
        visualizer.plot_training_history(
            training_info['train_losses'], 
            training_info['val_losses'],
            filename='training_history.png'
        )
        
        if 'component_losses' in training_info:
            visualizer.plot_training_component_losses(
                training_info,
                filename='training_components.png'
            )
        
        print(f"Best: epoch {training_info['best_epoch']}, val_loss {training_info['best_val_loss']:.6f}")
        
        all_target_names = ["temperature", "stress", "grain_size", "delta_grain", 
                          "dislocation_density", "precip_fraction", "free_energy", "dissipation"]
        
        y_true, predictions = trainer.evaluate(
            test_loader, input_mean, input_std, target_mean, target_std
        )
        
        visualizer.plot_predictions(y_true, predictions, target_columns, filename='prediction_evaluation.png')
        visualizer.plot_error_distributions(y_true, predictions, target_columns, filename='error_distributions.png')

        thermo_keys = ['stress_consistency_error', 'second_law_error', 'energy_balance_error']
        print("\nThermodynamic consistency:")
        for key in thermo_keys:
            if key in predictions:
                print(f"  {key}: {predictions[key]:.6f}")

    elif args.mode == 'evaluate':
        print("Evaluating...")
        all_target_names = ["temperature", "stress", "grain_size", "delta_grain", 
                          "dislocation_density", "precip_fraction", "free_energy", "dissipation"]
        
        y_true, predictions = trainer.evaluate(
            test_loader, input_mean, input_std, target_mean, target_std
        )
        
        visualizer.plot_predictions(y_true, predictions, target_columns, filename='prediction_evaluation.png')
        visualizer.plot_error_distributions(y_true, predictions, target_columns, filename='error_distributions.png')
        
        print("\nMetrics:")
        for i, name in enumerate(target_columns):
            if name in predictions:
                true_vals = y_true[:, i]
                pred_vals = predictions[name].flatten()
                r2 = np.corrcoef(true_vals, pred_vals)[0, 1]**2
                rmse = np.sqrt(np.mean((true_vals - pred_vals)**2))
                mae = np.mean(np.abs(true_vals - pred_vals))
                print(f"{name}: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
        
        thermo_keys = ['stress_consistency_error', 'second_law_error', 'energy_balance_error']
        print("\nThermodynamic consistency:")
        for key in thermo_keys:
            if key in predictions:
                print(f"  {key}: {predictions[key]:.6f}")

    elif args.mode == 'visualize':
        print("Generating visualizations...")
        norm_params = np.load('models/normalization_params.npz')
        input_mean = norm_params['input_mean']
        input_std = norm_params['input_std']
        
        visualizer.visualize_data_distribution(original_df, filename='data_distribution.png')
        visualizer.create_correlation_heatmap(original_df, filename='correlation_heatmap.png')
        
        if hasattr(visualizer, 'plot_parameter_effects'):
            visualizer.plot_parameter_effects(
                model, device=device, input_mean=input_mean, input_std=input_std,
                filename='parameter_effects.png'
            )
        
        if hasattr(visualizer, 'plot_thermodynamic_consistency'):
            visualizer.plot_thermodynamic_consistency(
                model, device=device, input_mean=input_mean, input_std=input_std,
                filename='thermodynamic_consistency.png'
            )
    
    elif args.mode == 'microstructure_viz':
        print("Microstructure visualizations...")
        
        results_df = visualizer.generate_prediction_csv(
            model, device=device, data_file=args.data_file,
            input_columns=input_columns, target_columns=target_columns
        )
        
        if hasattr(visualizer, 'plot_microstructure_evolution'):
            visualizer.plot_microstructure_evolution(results_df)
        
        if hasattr(visualizer, 'plot_microstructure_relationships'):
            visualizer.plot_microstructure_relationships(results_df)
        
        if hasattr(visualizer, 'plot_thermodynamic_analysis'):
            visualizer.plot_thermodynamic_analysis(results_df)
        
    elif args.mode == 'export_csv':
        print("Exporting predictions...")
        
        results_df = visualizer.generate_prediction_csv(
            model, device=device, data_file=args.data_file,
            input_columns=input_columns, target_columns=target_columns
        )
        
        if hasattr(visualizer, 'plot_stress_strain_curves'):
            visualizer.plot_stress_strain_curves(results_df)
        
        if hasattr(visualizer, 'plot_temperature_strain_curves'):
            visualizer.plot_temperature_strain_curves(results_df)
        
        if hasattr(visualizer, 'plot_comprehensive_comparisons'):
            visualizer.plot_comprehensive_comparisons(results_df)
        
        print("Export complete")

    print("Done")

if __name__ == "__main__":
    main()
