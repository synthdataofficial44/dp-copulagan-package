"""
Generator network for DP-CopulaGAN.

This module implements the generator architecture that produces synthetic
data from latent noise and class labels.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_generator(latent_dim: int, data_dim: int, n_classes: int):
    """
    Build generator network.
    
    Architecture (from original research):
    - Input: noise (latent_dim) + class_label (one-hot)
    - Layer 1: Dense(512) + GELU + LayerNorm + Dropout(0.2)
    - Layer 2: Dense(768) + GELU + LayerNorm + Dropout(0.2)
    - Layer 3: Dense(512) + GELU + LayerNorm + Dropout(0.2)
    - Layer 4: Dense(256) + GELU + LayerNorm
    - Output:  Dense(data_dim) # Linear output
    
    Parameters
    ----------
    latent_dim : int
        Dimension of latent noise vector (default: 192).
    data_dim : int
        Dimension of output data.
    n_classes : int
        Number of classes for conditional generation.
    
    Returns
    -------
    keras.Model
        Generator model.
    
    Examples
    --------
    >>> generator = build_generator(latent_dim=192, data_dim=28, n_classes=2)
    >>> noise = tf.random.normal([32, 192])
    >>> labels = tf.one_hot([0]*16 + [1]*16, depth=2)
    >>> fake_data = generator([noise, labels])
    >>> fake_data.shape
    TensorShape([32, 28])
    """
    
    noise = layers.Input(shape=(latent_dim,), name='noise_input')
    cond = layers.Input(shape=(n_classes,), name='cond_input')
    
    x = layers.Concatenate()([noise, cond])
    
    # Layer 1
    x = layers.Dense(512, activation='gelu')(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # Layer 2
    x = layers.Dense(768, activation='gelu')(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # Layer 3
    x = layers.Dense(512, activation='gelu')(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # Layer 4
    x = layers.Dense(256, activation='gelu')(x)
    x = layers.LayerNormalization()(x)
    
    # Output
    output = layers.Dense(data_dim)(x)
    
    return keras.Model([noise, cond], output, name='generator')
"""
Generator (unconditional) for DP-CopulaGAN.

This file contains the unconditional generator builder.
Add this function to your existing gan/generator.py file.
"""

def build_generator_unconditional(latent_dim: int, data_dim: int):
    """
    Build generator for UNCONDITIONAL mode (no class conditioning).
    
    Architecture identical to conditional generator, but without class input.
    
    Parameters
    ----------
    latent_dim : int
        Dimension of noise vector.
    data_dim : int
        Dimension of output data.
    
    Returns
    -------
    keras.Model
        Unconditional generator.
    
    Examples
    --------
    >>> generator = build_generator_unconditional(latent_dim=192, data_dim=10)
    >>> noise = tf.random.normal([32, 192])
    >>> fake_data = generator(noise)
    >>> fake_data.shape
    TensorShape([32, 10])
    """
    from tensorflow import keras
    from tensorflow.keras import layers
    
    # Input: noise only (no class conditioning)
    noise_input = layers.Input(shape=(latent_dim,), name='noise')
    
    # Same architecture as conditional generator
    x = layers.Dense(512)(noise_input)
    x = layers.Activation('gelu')(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(768)(x)
    x = layers.Activation('gelu')(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(512)(x)
    x = layers.Activation('gelu')(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(256)(x)
    x = layers.Activation('gelu')(x)
    x = layers.LayerNormalization()(x)
    
    output = layers.Dense(data_dim, activation='linear', name='output')(x)
    
    model = keras.Model(inputs=noise_input, outputs=output, name='generator_unconditional')
    
    return model
