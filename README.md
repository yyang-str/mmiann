# MMIANN: Micro-Mechanism Informed Artificial Neural Networks

Physics-informed neural networks for constitutive modeling of aluminum alloys under non-isothermal deformation.

## Overview

This framework combines metallurgical evolution equations (dislocation density, grain growth, precipitation kinetics) with neural networks to predict material behavior during hot forming processes. The model tracks microstructure changes and enforces thermodynamic consistency through penalty-based training.

## Installation

```bash
git clone https://github.com/yyang-str/mmiann.git
cd mmiann
pip install -r requirements.txt
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0
- NumPy >= 1.22
- Pandas >= 1.5
- scikit-learn >= 1.0
- matplotlib >= 3.5

## Quick Start

```python
from src.data_utils import load_and_prepare_data, normalize_data, create_data_loaders
from src.networks import ImprovedPhysicsInformedNN
from src.mmiann import MMIANNTrainer

# Load experimental data
inputs, targets, df = load_and_prepare_data('your_data.csv')
inputs_norm, in_mean, in_std = normalize_data(inputs)
targets_norm, out_mean, out_std = normalize_data(targets)

# Create data loaders
train_loader, val_loader, test_loader, _ = create_data_loaders(
    inputs_norm, targets_norm, batch_size=8
)

# Initialize model
model = ImprovedPhysicsInformedNN(input_dim=3, hidden_dim=64)
trainer = MMIANNTrainer(model, device='cuda', thermo_weight=0.1)

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15)

training_info = trainer.train(
    train_loader, val_loader, optimizer, scheduler,
    num_epochs=500, early_stopping_patience=30
)

# Evaluate
y_true, predictions = trainer.evaluate(
    test_loader, in_mean, in_std, out_mean, out_std
)
```

## Data Format

Required CSV columns:
- strain: Plastic strain (0-0.5)
- time: Time since start (seconds)
- cooling_rate: Temperature decrease rate (C/s)
- temperature: Current temperature (C)
- stress: Flow stress (MPa)
- grain_size: Average grain diameter (micrometers)
- delta_grain: Change in grain size (micrometers)

Minimum 500 datapoints recommended for robust training. Coverage across full parameter ranges is essential.

## Training

Basic training:
```bash
python main.py --mode train --data_file data/your_data.csv --epochs 500
```

With custom parameters:
```bash
python main.py --mode train \
    --data_file data/your_data.csv \
    --batch_size 8 \
    --lr 0.001 \
    --hidden_dim 64 \
    --thermo_weight 0.1 \
    --patience 30 \
    --model_save_path models/best.pt
```

## Evaluation

```bash
python main.py --mode evaluate --load_model --model_save_path models/best.pt
```

## Model Architecture

The framework uses parallel physics-based and neural network pathways:

Physics pathway:
- Dislocation evolution (Kocks-Mecking framework)
- Grain growth with strain inhibition
- Precipitation kinetics (JMAK model)
- Flow stress calculation (Taylor + Hall-Petch + precipitation hardening)

Neural pathway:
- Feature extractor with shared layers
- Eight specialized output branches
- Adaptive blending coefficients (alpha in [0.2, 0.8])

Thermodynamic constraints:
- Stress consistency: sigma approximately equal to d(Psi)/d(epsilon)
- Second law: D = sigma * strain_rate >= 0
- Energy balance: bounded free energy

## Key Parameters

hidden_dim: Network capacity (32/64/128). Larger networks require more training data.

thermo_weight: Thermodynamic penalty strength (0.01-1.0). Higher values enforce physical constraints more strictly but may reduce data fitting accuracy.

Blending coefficients (set in networks.py):
- Temperature: 0.2 (neural-dominated due to complex heating effects)
- Stress: 0.4 (balanced)
- Grain size: 0.4 (balanced)
- Microstructure variables: 0.7 (physics-dominated)

learning_rate: Initial value 0.001. Reduce if training is unstable.

## Physics Equations

Temperature evolution:
```
T = T_ref - cooling_rate * t + strain_heating
```

Dislocation density (Kocks-Mecking):
```
rho = rho_0 + K_gen * strain_rate - K_rec * strain
```

Grain growth:
```
d = d_0 + K_d * exp(-Q_g/RT) * ln(t) / (1 + beta * strain)
```

Flow stress:
```
sigma = sigma_0 + k_1 * strain^n - k_T * T + k_rho * sqrt(rho) + k_d / sqrt(d) + k_X * X
```

All rate constants follow Arrhenius form: K = K_0 * exp(-Q/RT)

## Visualization

Generate training curves and predictions:
```bash
python main.py --mode visualize --load_model
```

Export predictions to CSV:
```bash
python main.py --mode export_csv
```

## Common Issues

NaN losses: Reduce learning rate or verify data normalization

High validation loss: Increase training data or improve parameter space coverage

Poor temperature predictions: Decrease alpha_T to increase neural network contribution

Unphysical predictions: Increase thermo_weight parameter

Slow training: Use GPU acceleration, reduce batch size if memory limited

## Limitations

- Isotropic materials only (texture effects not included)
- Monotonic loading paths (cyclic loading not supported)
- Single phase materials (precipitation as volume fraction)
- FCC crystal structure (dislocation mechanisms specific to face-centered cubic metals)
- Validated for temperature range 200-560 C, strain rate 0.01-10 per second

For other material systems, modify physics equations in networks.py accordingly.

## File Structure

- main.py: Training and evaluation script
- example_minimal.py: Quick start example
- src/networks.py: Neural network architecture and physics equations
- src/mmiann.py: Training loop and optimization
- src/data_utils.py: Data loading and preprocessing
- src/visualization.py: Plotting and analysis tools

## Citation

If you use this software in your research, please cite:

```bibtex
@article{yang2025mmiann,
  title={A constitutive framework for non-isothermal plasticity through 
         micro-mechanism informed artificial neural networks},
  author={Yang, Yo-Lun and Chung, Tsai-Fu and Liao, Chia-Hung and 
          Chen, Liang-Yu and Wu, Hsing-Yu and Marimuthu, Uthayakumar and 
          Veerasimman, Arumugaprabu and Rajendran, Sundarakannan and 
          Shanmugam, Vigneshwaran},
  journal={Engineering Applications of Artificial Intelligence},
  volume={162},
  pages={112465},
  year={2025},
  doi={10.1016/j.engappai.2025.112465}
}
```

## License

MIT License - see LICENSE file for details.

## Contact

Issues: GitHub issue tracker
Questions: y.yang@ntut.edu.tw
Institution: National Taipei University of Technology

## Acknowledgments

This research was funded by the National Science and Technology Council of Taiwan under grant NSTC 112-2221-E-027-118.

## Disclaimer

This is research software. Validation required for specific material systems and processing conditions. Not certified for safety-critical applications.
