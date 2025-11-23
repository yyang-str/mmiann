"""
MMIANN: Micro-Mechanism Informed Artificial Neural Networks

Physics-informed neural networks for constitutive modeling
"""

__version__ = "1.0.0"

from .networks import ImprovedPhysicsInformedNN
from .mmiann import MMIANNTrainer
from .data_utils import load_and_prepare_data, normalize_data, create_data_loaders

__all__ = [
    'ImprovedPhysicsInformedNN',
    'MMIANNTrainer',
    'load_and_prepare_data',
    'normalize_data',
    'create_data_loaders'
]
