"""
Critic network for DP-CopulaGAN.

This module implements the critic (discriminator) with spectral normalization
for enforcing the Lipschitz constraint in WGAN-GP.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from dp_copulagan.gan.spectral_norm import SpectralNormalization


def build_critic(data_dim: int, n_classes: int):
    """
    Build critic network with spectral normalization.
    
    Architecture (from original research):
    - Input: data + class_label (one-hot)
    - Layer 1: SN-Dense(512) + LeakyReLU(0.2) + Dropout(0.3)
    - Layer 2: SN-Dense(256) + LeakyReLU(0.2) + Dropout(0.3)
    - Layer 3: SN-Dense(128) + LeakyReLU(0.2)
    - Output:  SN-Dense(1) # No activation (Wasserstein)
    
    All Dense layers use spectral normalization to enforce 1-Lipschitz.
    
    Parameters
    ----------
    data_dim : int
        Dimension of input data.
    n_classes : int
        Number of classes for conditional discrimination.
    
    Returns
    -------
    keras.Model
        Critic model.
    
    Examples
    --------
    >>> critic = build_critic(data_dim=28, n_classes=2)
    >>> data = tf.random.normal([32, 28])
    >>> labels = tf.one_hot([0]*16 + [1]*16, depth=2)
    >>> scores = critic([data, labels])
    >>> scores.shape
    TensorShape([32, 1])
    """
    
    data = layers.Input(shape=(data_dim,), name='data_input')
    cond = layers.Input(shape=(n_classes,), name='cond_input')
    
    x = layers.Concatenate()([data, cond])
    
    # Layer 1
    x = SpectralNormalization(layers.Dense(512))(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    # Layer 2
    x = SpectralNormalization(layers.Dense(256))(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    # Layer 3
    x = SpectralNormalization(layers.Dense(128))(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # Output
    output = SpectralNormalization(layers.Dense(1))(x)
    
    return keras.Model([data, cond], output, name='critic')
"""
Critic (unconditional) for DP-CopulaGAN.

This file contains the unconditional critic builder.
Add this function to your existing gan/critic.py file.
"""

def build_critic_unconditional(data_dim: int):
    """
    Build critic for UNCONDITIONAL mode (no class conditioning).
    
    Architecture identical to conditional critic, but without class input.
    Uses spectral normalization on all Dense layers.
    
    Parameters
    ----------
    data_dim : int
        Dimension of input data.
    
    Returns
    -------
    keras.Model
        Unconditional critic.
    
    Examples
    --------
    >>> critic = build_critic_unconditional(data_dim=10)
    >>> data = tf.random.normal([32, 10])
    >>> score = critic(data)
    >>> score.shape
    TensorShape([32, 1])
    """
    from tensorflow import keras
    from tensorflow.keras import layers
    from dp_copulagan.gan.spectral_norm import SpectralNormalization
    
    # Input: data only (no class conditioning)
    data_input = layers.Input(shape=(data_dim,), name='data')
    
    # Same architecture as conditional critic with spectral norm
    x = SpectralNormalization(layers.Dense(512))(data_input)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = SpectralNormalization(layers.Dense(256))(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = SpectralNormalization(layers.Dense(128))(x)
    x = layers.LeakyReLU(0.2)(x)
    
    output = SpectralNormalization(layers.Dense(1, activation='linear'))(x)
    
    model = keras.Model(inputs=data_input, outputs=output, name='critic_unconditional')
    
    return model
