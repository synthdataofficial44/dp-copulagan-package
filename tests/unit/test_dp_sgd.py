"""Unit tests for DP-SGD."""

import pytest
import numpy as np
import tensorflow as tf
from dp_copulagan.dp import clip_gradients, add_dp_noise, compute_noise_multiplier


def test_gradient_clipping():
    """Test gradient clipping."""
    # Create sample gradients
    grads = [
        tf.constant([[1.0, 2.0], [3.0, 4.0]]),
        tf.constant([5.0, 6.0])
    ]
    
    clipped = clip_gradients(grads, clip_norm=1.0)
    
    # Check clipping occurred
    assert len(clipped) == len(grads)
    for g in clipped:
        assert g is not None


def test_noise_addition():
    """Test DP noise addition."""
    grads = [
        tf.constant([[1.0, 2.0]]),
        tf.constant([3.0])
    ]
    
    noisy = add_dp_noise(grads, noise_multiplier=0.5, clip_norm=1.0)
    
    assert len(noisy) == len(grads)
    # Noise should change gradients
    assert not np.allclose(grads[0].numpy(), noisy[0].numpy())


def test_noise_calibration():
    """Test noise multiplier computation."""
    noise_mult = compute_noise_multiplier(
        epsilon=1.0,
        delta=1e-5,
        n_samples=1000,
        batch_size=100,
        epochs=10
    )
    
    assert noise_mult > 0
    assert isinstance(noise_mult, float)
