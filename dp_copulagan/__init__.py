"""
DP-CopulaGAN: Differentially Private Synthetic Data Generation using Copula and GANs
====================================================================================

A Python package for generating high-quality synthetic tabular data with formal
differential privacy guarantees using Gaussian copula transformation and 
conditional generative adversarial networks.

Author: Comp
License: GPL-3.0
Version: 0.1.0

Examples
--------
>>> from dp_copulagan import DPCopulaGAN
>>> import pandas as pd
>>> 
>>> # Load your data
>>> data = pd.read_csv('your_data.csv')
>>> 
>>> # Train with differential privacy
>>> model = DPCopulaGAN(epsilon=1.0, delta=1e-5, label_col='target')
>>> model.fit(data)
>>> 
>>> # Generate synthetic data
>>> synthetic = model.sample(n_samples=1000)
>>> 
>>> # Evaluate quality
>>> from dp_copulagan.evaluation import evaluate_synthetic
>>> results = evaluate_synthetic(data, synthetic, label_col='target')
"""

__version__ = "0.1.0"
__author__ = "Comp"
__license__ = "GPL-3.0"

# Core imports
from dp_copulagan.models.dp_copulagan import DPCopulaGAN
from dp_copulagan.preprocessing.preprocessor import CopulaPreprocessor
from dp_copulagan.copula.gaussian_copula import ConditionalGaussianCopula
from dp_copulagan.evaluation.evaluator import evaluate_synthetic
from dp_copulagan.utils.config import DPConfig, GANConfig, CopulaConfig

# Convenience imports for advanced users
from dp_copulagan.gan.generator import build_generator
from dp_copulagan.gan.critic import build_critic
from dp_copulagan.dp.dp_sgd import DPOptimizer, clip_gradients, add_dp_noise
from dp_copulagan.metrics.statistical import compute_statistical_metrics
from dp_copulagan.metrics.ml_utility import compute_ml_utility

__all__ = [
    # Main API
    'DPCopulaGAN',
    'CopulaPreprocessor',
    'ConditionalGaussianCopula',
    'evaluate_synthetic',
    
    # Configuration
    'DPConfig',
    'GANConfig',
    'CopulaConfig',
    
    # Advanced
    'build_generator',
    'build_critic',
    'DPOptimizer',
    'clip_gradients',
    'add_dp_noise',
    'compute_statistical_metrics',
    'compute_ml_utility',
]


def get_version():
    """Return the current version of dp_copulagan."""
    return __version__


def print_system_info():
    """Print system information for reproducibility."""
    import sys
    import tensorflow as tf
    import numpy as np
    import pandas as pd
    
    print("="*80)
    print("DP-CopulaGAN System Information")
    print("="*80)
    print(f"Package version:     {__version__}")
    print(f"Python version:      {sys.version.split()[0]}")
    print(f"TensorFlow version:  {tf.__version__}")
    print(f"NumPy version:       {np.__version__}")
    print(f"Pandas version:      {pd.__version__}")
    
    # GPU info
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU available:       Yes ({len(gpus)} device(s))")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}:            {gpu.name}")
    else:
        print(f"GPU available:       No (using CPU)")
    
    print("="*80)
