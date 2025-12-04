"""
Spectral Normalization layer for enforcing Lipschitz constraint in critic.

This implementation follows Miyato et al. (2018) "Spectral Normalization for GANs".
It constrains the spectral norm of weight matrices to 1, which helps stabilize
GAN training by enforcing a Lipschitz constraint on the critic.
"""

import tensorflow as tf
from tensorflow.keras import layers


class SpectralNormalization(tf.keras.layers.Wrapper):
    """
    Spectral Normalization wrapper for Keras layers.
    
    This layer wraps any Keras layer with a kernel (e.g., Dense, Conv2D)
    and applies spectral normalization to its weights, constraining the
    spectral norm to 1.
    
    Parameters
    ----------
    layer : tf.keras.layers.Layer
        The layer to wrap (must have a 'kernel' attribute).
    iteration : int, default=1
        Number of power iterations for singular value estimation.
    eps : float, default=1e-12
        Small constant for numerical stability.
    
    Examples
    --------
    >>> from dp_copulagan.gan.spectral_norm import SpectralNormalization
    >>> dense_layer = layers.Dense(256)
    >>> sn_dense = SpectralNormalization(dense_layer)
    
    Notes
    -----
    The spectral norm is the largest singular value of the weight matrix.
    This implementation uses power iteration to estimate it efficiently.
    """
    
    def __init__(self, layer, iteration=1, eps=1e-12, **kwargs):
        super(SpectralNormalization, self).__init__(layer, **kwargs)
        self.iteration = iteration
        self.eps = eps

    def build(self, input_shape):
        """Build the wrapped layer and initialize the u vector."""
        self.layer.build(input_shape)
        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()
        
        # Initialize u vector for power iteration
        self.u = self.add_weight(
            shape=(1, self.w_shape[-1]),
            initializer=tf.random_normal_initializer(),
            trainable=False,
            name='sn_u',
            dtype=tf.float32
        )
        super(SpectralNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        """Apply spectral normalization and call the wrapped layer."""
        self._compute_weights()
        return self.layer(inputs)

    def _compute_weights(self):
        """Compute spectral norm and normalize weights."""
        # Reshape weight matrix for 2D operations
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])
        u_hat = self.u
        
        # Power iteration
        for _ in range(self.iteration):
            # v = u^T W / ||u^T W||
            v_ = tf.matmul(u_hat, tf.transpose(w_reshaped))
            v_hat = v_ / (tf.norm(v_) + self.eps)
            
            # u = W v / ||W v||
            u_ = tf.matmul(v_hat, w_reshaped)
            u_hat = u_ / (tf.norm(u_) + self.eps)
        
        # Compute spectral norm: σ = v^T W u
        sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
        
        # Update u for next iteration
        self.u.assign(u_hat)
        
        # Normalize weights: W_sn = W / σ
        self.layer.kernel.assign(self.w / sigma)
