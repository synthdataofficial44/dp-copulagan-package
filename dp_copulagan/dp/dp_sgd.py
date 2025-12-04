"""
Differentially Private Stochastic Gradient Descent (DP-SGD).

This module implements DP-SGD following Abadi et al. (2016).
It provides gradient clipping and calibrated Gaussian noise addition
to ensure (ε, δ)-differential privacy during training.
"""

import tensorflow as tf
import numpy as np
from typing import List, Optional


def clip_gradients(gradients: List[tf.Tensor], 
                   clip_norm: float = 1.0) -> List[tf.Tensor]:
    """
    Clip gradients to have a maximum L2 norm.
    
    This is the first step of DP-SGD: per-sample gradient clipping.
    Each gradient tensor is scaled down if its norm exceeds clip_norm.
    
    Parameters
    ----------
    gradients : List[tf.Tensor]
        List of gradient tensors from automatic differentiation.
    clip_norm : float, default=1.0
        Maximum allowed L2 norm (clipping threshold C).
    
    Returns
    -------
    List[tf.Tensor]
        Clipped gradients with norm ≤ clip_norm.
    
    Notes
    -----
    For gradient g, we compute:
        g_clipped = g * min(1, C / ||g||)
    
    This ensures sensitivity Δf = C for the gradient computation.
    """
    clipped_grads = []
    
    # Compute global norm across all gradients
    global_norm = tf.sqrt(
        sum([tf.reduce_sum(tf.square(g)) for g in gradients if g is not None])
    )
    
    # Clip each gradient
    for grad in gradients:
        if grad is not None:
            # Scale factor: min(1, C / ||g||)
            scale = tf.minimum(1.0, clip_norm / (global_norm + 1e-8))
            clipped_grads.append(grad * scale)
        else:
            clipped_grads.append(None)
    
    return clipped_grads


def add_dp_noise(gradients: List[tf.Tensor],
                noise_multiplier: float,
                clip_norm: float = 1.0) -> List[tf.Tensor]:
    """
    Add calibrated Gaussian noise to gradients for differential privacy.
    
    This is the second step of DP-SGD: noise addition after clipping.
    Noise scale is σ = noise_multiplier * clip_norm.
    
    Parameters
    ----------
    gradients : List[tf.Tensor]
        Clipped gradient tensors.
    noise_multiplier : float
        Noise scale multiplier (σ/C in the literature).
    clip_norm : float, default=1.0
        Clipping threshold used in previous step.
    
    Returns
    -------
    List[tf.Tensor]
        Noisy gradients for private optimization.
    
    Notes
    -----
    For clipped gradient g_clip, we add noise:
        g_private = g_clip + N(0, σ²I)
    
    where σ = noise_multiplier * clip_norm.
    
    The noise_multiplier is calibrated to achieve (ε, δ)-DP:
        σ = sqrt(2 * log(1.25/δ)) / ε
    """
    if noise_multiplier == 0:
        # No privacy (baseline mode)
        return gradients
    
    noisy_grads = []
    for grad in gradients:
        if grad is not None:
            # Sample Gaussian noise: N(0, σ²)
            noise = tf.random.normal(
                shape=grad.shape,
                mean=0.0,
                stddev=noise_multiplier * clip_norm,
                dtype=grad.dtype
            )
            noisy_grads.append(grad + noise)
        else:
            noisy_grads.append(None)
    
    return noisy_grads


class DPOptimizer:
    """
    Wrapper for TensorFlow optimizers to enable DP-SGD.
    
    This class wraps any Keras optimizer and adds differential privacy
    through gradient clipping and noise addition.
    
    Parameters
    ----------
    optimizer : tf.keras.optimizers.Optimizer
        Base optimizer (e.g., Adam, SGD).
    noise_multiplier : float
        Noise scale for DP (calibrated from ε, δ).
    clip_norm : float, default=1.0
        Gradient clipping threshold.
    
    Attributes
    ----------
    optimizer : Optimizer
        Base optimizer instance.
    noise_multiplier : float
        Noise multiplier.
    clip_norm : float
        Clipping norm.
    
    Examples
    --------
    >>> from dp_copulagan.dp import DPOptimizer
    >>> import tensorflow as tf
    >>> 
    >>> base_opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    >>> dp_opt = DPOptimizer(base_opt, noise_multiplier=1.1, clip_norm=1.0)
    >>> 
    >>> # In training loop:
    >>> with tf.GradientTape() as tape:
    ...     loss = model(x)
    >>> grads = tape.gradient(loss, model.trainable_variables)
    >>> grads = dp_opt.apply_gradients(zip(grads, model.trainable_variables))
    """
    
    def __init__(self,
                 optimizer: tf.keras.optimizers.Optimizer,
                 noise_multiplier: float,
                 clip_norm: float = 1.0):
        self.optimizer = optimizer
        self.noise_multiplier = noise_multiplier
        self.clip_norm = clip_norm
    
    def apply_gradients(self, grads_and_vars):
        """
        Apply DP-SGD to gradients and update variables.
        
        Parameters
        ----------
        grads_and_vars : List[Tuple[Tensor, Variable]]
            List of (gradient, variable) pairs.
        
        Returns
        -------
        Operation
            TensorFlow operation to apply updates.
        """
        # Separate gradients and variables
        grads, variables = zip(*grads_and_vars)
        grads = list(grads)
        
        # Step 1: Clip gradients
        grads = clip_gradients(grads, self.clip_norm)
        
        # Step 2: Add noise
        grads = add_dp_noise(grads, self.noise_multiplier, self.clip_norm)
        
        # Step 3: Apply to base optimizer
        return self.optimizer.apply_gradients(zip(grads, variables))
    
    def __getattr__(self, name):
        """Delegate attribute access to base optimizer."""
        return getattr(self.optimizer, name)
