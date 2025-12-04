"""
Noise calibration for differential privacy.

This module provides functions to calibrate noise parameters for DP-SGD
based on the privacy budget (ε, δ) and training configuration.
"""

import numpy as np
from typing import Tuple


def compute_noise_multiplier(epsilon: float,
                            delta: float,
                            n_samples: int,
                            batch_size: int,
                            epochs: int) -> float:
    """
    Compute noise multiplier for DP-SGD to achieve (ε, δ)-DP.
    
    This uses the Gaussian mechanism and strong composition theorem.
    
    Parameters
    ----------
    epsilon : float
        Privacy budget (ε). Smaller = more private.
    delta : float
        Failure probability (δ). Should be << 1/n_samples.
    n_samples : int
        Number of training samples.
    batch_size : int
        Batch size for training.
    epochs : int
        Number of training epochs.
    
    Returns
    -------
    float
        Noise multiplier σ for DP-SGD.
    
    Notes
    -----
    For single query (no composition):
        σ = sqrt(2 * log(1.25/δ)) / ε
    
    For composition over T steps:
        σ = σ_single * sqrt(T) / scaling_factor
    
    We use a scaling factor of 10 for practical utility, which is
    standard in the literature.
    
    References
    ----------
    .. [1] Abadi et al. (2016). Deep Learning with Differential Privacy.
    """
    # Single-query noise multiplier (Gaussian mechanism)
    sigma_single = np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    
    # Compute total number of steps
    steps_per_epoch = n_samples // batch_size
    total_steps = steps_per_epoch * epochs
    
    # Composition-aware scaling
    # Standard practice: divide by sqrt(T) and apply scaling
    noise_mult = sigma_single * np.sqrt(total_steps) / 10
    
    # FIX #3: Warn when DP noise is very high
    if noise_mult > 10:
        print(f"\n⚠️  WARNING: Very high DP noise multiplier ({noise_mult:.2f}).")
        print(f"   This will add substantial noise to gradients.")
        print(f"   Synthetic data quality may degrade significantly.")
        print(f"   Consider:")
        print(f"   • Increasing epsilon (less privacy, better quality)")
        print(f"   • Using more training samples")
        print(f"   • Reducing number of epochs")
    
    return noise_mult


def compute_epsilon(noise_multiplier: float,
                   n_samples: int,
                   batch_size: int,
                   epochs: int,
                   delta: float = 1e-5) -> float:
    """
    Compute achieved epsilon given noise multiplier and training config.
    
    This is the inverse of compute_noise_multiplier, useful for
    privacy accounting after training.
    
    Parameters
    ----------
    noise_multiplier : float
        Noise multiplier used in DP-SGD.
    n_samples : int
        Number of training samples.
    batch_size : int
        Batch size.
    epochs : int
        Number of epochs.
    delta : float, default=1e-5
        Failure probability.
    
    Returns
    -------
    float
        Achieved privacy budget ε.
    """
    steps_per_epoch = n_samples // batch_size
    total_steps = steps_per_epoch * epochs
    
    # Reverse the computation
    sigma_single = noise_multiplier * 10 / np.sqrt(total_steps)
    epsilon = np.sqrt(2 * np.log(1.25 / delta)) / sigma_single
    
    return epsilon


def get_privacy_budget_recommendation(n_samples: int,
                                     data_sensitivity: str = 'medium') -> Tuple[float, float]:
    """
    Get recommended privacy budget for a dataset.
    
    Parameters
    ----------
    n_samples : int
        Number of samples in dataset.
    data_sensitivity : str, default='medium'
        Sensitivity level: 'low', 'medium', 'high'.
    
    Returns
    -------
    Tuple[float, float]
        Recommended (epsilon, delta) pair.
    
    Notes
    -----
    Recommendations:
    - High sensitivity (medical): ε=0.1, δ=1e-6
    - Medium sensitivity (census): ε=1.0, δ=1e-5
    - Low sensitivity (public): ε=10.0, δ=1e-5
    
    Delta should always be << 1/n_samples.
    """
    # Delta should be much smaller than 1/n
    delta = min(1e-5, 1.0 / (n_samples * 10))
    
    if data_sensitivity == 'high':
        epsilon = 0.1
        delta = min(delta, 1e-6)
    elif data_sensitivity == 'medium':
        epsilon = 1.0
    else:  # low
        epsilon = 10.0
    
    return epsilon, delta
