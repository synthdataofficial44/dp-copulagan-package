"""
Gradient penalty for WGAN-GP.

This module implements the gradient penalty loss component that enforces
the Lipschitz constraint on the critic.
"""

import tensorflow as tf


def gradient_penalty(critic, real, fake, cond, batch_size, gp_weight=12.0):
    """
    Compute gradient penalty for WGAN-GP.
    
    The gradient penalty enforces the 1-Lipschitz constraint by penalizing
    the critic when gradients deviate from norm 1:
    
        GP = λ * E[(||∇_x D(x)||_2 - 1)^2]
    
    where x is uniformly interpolated between real and fake samples.
    
    Parameters
    ----------
    critic : keras.Model
        Critic network.
    real : tf.Tensor
        Real data batch, shape (batch_size, data_dim).
    fake : tf.Tensor
        Fake data batch, shape (batch_size, data_dim).
    cond : tf.Tensor
        Conditional labels (one-hot), shape (batch_size, n_classes).
    batch_size : int
        Batch size.
    gp_weight : float, default=12.0
        Gradient penalty coefficient (λ).
    
    Returns
    -------
    tf.Tensor
        Gradient penalty loss (scalar).
    
    Notes
    -----
    Standard WGAN-GP uses λ=10, but we use λ=12 for tabular data
    based on empirical results.
    
    References
    ----------
    .. [1] Gulrajani et al. (2017). Improved Training of Wasserstein GANs.
    """
    
    # Random interpolation between real and fake
    alpha = tf.random.uniform([batch_size, 1], 0., 1.)
    interpolated = alpha * real + (1 - alpha) * fake
    
    # Compute gradients of critic w.r.t. interpolated samples
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = critic([interpolated, cond], training=True)
    
    grads = tape.gradient(pred, interpolated)
    
    # Compute gradient norm
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-8)
    
    # Penalty: (||∇||_2 - 1)^2
    penalty = tf.reduce_mean((norm - 1.0) ** 2)
    
    return gp_weight * penalty
