"""
Main DP-CopulaGAN model class with UNCONDITIONAL and CONDITIONAL modes.

UPDATED: Now supports both labeled and unlabeled data with automatic mode detection.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from sklearn.preprocessing import StandardScaler
from typing import Optional
import warnings

from dp_copulagan.copula import ConditionalGaussianCopula
from dp_copulagan.gan import (build_generator, build_critic, gradient_penalty,
                               build_generator_unconditional, build_critic_unconditional)
from dp_copulagan.dp import clip_gradients, add_dp_noise, compute_noise_multiplier
from dp_copulagan.utils import DPConfig, GANConfig, CopulaConfig, set_random_seed, setup_gpu
from dp_copulagan.utils.validation import validate_dataframe


class DPCopulaGAN:
    """
    Differentially Private Copula GAN for synthetic data generation.
    
    Supports BOTH conditional (with labels) and unconditional (without labels) generation.
    
    Parameters
    ----------
    epsilon : float, default=1.0
        Privacy budget (ε). Smaller = more private.
    delta : float, default=1e-5
        Privacy parameter (δ). Should be << 1/n_samples.
    label_col : Optional[str], default=None
        Column name for conditional generation. If None, uses unconditional mode.
    gan_config : Optional[GANConfig], default=None
        GAN architecture configuration.
    dp_config : Optional[DPConfig], default=None
        Differential privacy configuration.
    copula_config : Optional[CopulaConfig], default=None
        Copula transformation configuration.
    random_state : int, default=42
        Random seed for reproducibility.
    
    Examples
    --------
    # Conditional mode (with labels)
    >>> model = DPCopulaGAN(epsilon=1.0, delta=1e-5, label_col='income')
    >>> model.fit(train_data)
    >>> synthetic = model.sample(10000)
    
    # Unconditional mode (no labels)
    >>> model = DPCopulaGAN(epsilon=1.0, delta=1e-5, label_col=None)
    >>> model.fit(sensor_data)
    >>> synthetic = model.sample(10000)
    """
    
    def __init__(self,
                 epsilon: float = 1.0,
                 delta: float = 1e-5,
                 label_col: Optional[str] = None,
                 gan_config: Optional[GANConfig] = None,
                 dp_config: Optional[DPConfig] = None,
                 copula_config: Optional[CopulaConfig] = None,
                 random_state: int = 42):
        
        set_random_seed(random_state)
        setup_gpu()
        
        self.label_col = label_col
        self.unconditional = (label_col is None)
        
        if dp_config is None:
            self.dp_config = DPConfig(epsilon=epsilon, delta=delta)
        else:
            self.dp_config = dp_config
        
        if gan_config is None:
            self.gan_config = GANConfig()
        else:
            self.gan_config = gan_config
        
        if copula_config is None:
            self.copula_config = CopulaConfig()
        else:
            self.copula_config = copula_config
        
        self.copula = ConditionalGaussianCopula(
            bins=self.copula_config.bins,
            eigenvalue_threshold=self.copula_config.eigenvalue_threshold
        )
        
        self.generator = None
        self.critic = None
        self.feature_scaler = None
        self.fitted = False
        self.label_encoder = None
        self.label_decoder = None
        
        if self.unconditional:
            print("ℹ️  No label column provided → Switching to UNCONDITIONAL mode")
            print("   (Generating all data without class conditioning)")
        else:
            print(f"DP-CopulaGAN initialized: {self.dp_config.get_privacy_description()}")
    
    def fit(self, data: pd.DataFrame):
        """
        Fit DP-CopulaGAN on training data.
        
        Supports both conditional and unconditional generation.
        """
        print()
        print("="*80)
        print("TRAINING DP-COPULAGAN" + (" (UNCONDITIONAL MODE)" if self.unconditional else ""))
        print("="*80)
        
        validate_dataframe(data, self.label_col)
        
        # Fit copula
        print()
        print("📊 Fitting copula transformation...")
        self.copula.fit(data, self.label_col)
        print(f"   ✓ Copula fitted on {len(self.copula.numeric_cols)} numeric features")
        
        # Transform to Gaussian latent space
        print()
        print("🔄 Transforming to Gaussian latent space...")
        
        if self.unconditional:
            X = data[self.copula.numeric_cols]
            Z = self.copula.transform_to_normal(X, None)
            X_train = Z.astype(np.float32)
            y_train = None
            self.n_classes = 1
            self.class_probs = np.array([1.0])
        else:
            # Handle categorical labels
            if data[self.label_col].dtype == 'object' or data[self.label_col].dtype.name == 'category':
                print(f"\n⚠️  Warning: Label column '{self.label_col}' is categorical (string/category type).")
                print(f"   Auto-encoding to integers for compatibility.")
                original_labels = data[self.label_col].unique()
                self.label_encoder = {val: i for i, val in enumerate(sorted(original_labels))}
                self.label_decoder = {i: val for val, i in self.label_encoder.items()}
                data = data.copy()
                data[self.label_col] = data[self.label_col].map(self.label_encoder)
                print(f"   Encoded {len(original_labels)} categories to integers 0-{len(original_labels)-1}")
            
            all_Z = []
            all_labels = []
            
            for label in self.copula.label_values:
                class_data = data[data[self.label_col] == label]
                X = class_data[self.copula.numeric_cols]
                Z = self.copula.transform_to_normal(X, label)
                all_Z.append(Z)
                all_labels.extend([label] * len(Z))
            
            X_train = np.vstack(all_Z).astype(np.float32)
            y_train = np.array(all_labels)
            
            if self.label_encoder is None:
                self.label_encoder = {val: i for i, val in enumerate(self.copula.label_values)}
                self.label_decoder = {i: val for val, i in self.label_encoder.items()}
            
            self.n_classes = len(self.copula.label_values)
            y_train = np.array([self.label_encoder[val] for val in y_train])
            
            self.class_probs = np.array([
                self.copula.label_probs[self.label_decoder[i]] 
                for i in range(self.n_classes)
            ])
        
        self.feature_scaler = StandardScaler()
        X_train = self.feature_scaler.fit_transform(X_train)
        
        print(f"   ✓ Transformed shape: {X_train.shape}")
        
        # Build networks
        print()
        print("🏗️  Building generator and critic...")
        
        if self.unconditional:
            self.generator = build_generator_unconditional(
                latent_dim=self.gan_config.latent_dim,
                data_dim=X_train.shape[1]
            )
            self.critic = build_critic_unconditional(
                data_dim=X_train.shape[1]
            )
        else:
            self.generator = build_generator(
                latent_dim=self.gan_config.latent_dim,
                data_dim=X_train.shape[1],
                n_classes=self.n_classes
            )
            self.critic = build_critic(
                data_dim=X_train.shape[1],
                n_classes=self.n_classes
            )
        
        print(f"   ✓ Generator: {self.generator.count_params():,} parameters")
        print(f"   ✓ Critic: {self.critic.count_params():,} parameters")
        
        self.g_optimizer = keras.optimizers.Adam(
            self.gan_config.g_lr, beta_1=0.5, beta_2=0.9
        )
        self.c_optimizer = keras.optimizers.Adam(
            self.gan_config.c_lr, beta_1=0.5, beta_2=0.9
        )
        
        self._train(X_train, y_train)
        
        self.fitted = True
        print()
        print("="*80)
        print("✅ TRAINING COMPLETE")
        print("="*80)
        
        return self
    
    def _train(self, X_real, y_real):
        """Main training loop with DP-SGD (conditional and unconditional)."""
        n_samples = len(X_real)
        
        # Check if dataset is smaller than batch size
        if n_samples < self.gan_config.batch_size:
            print()
            print("="*80)
            print("⚠️  WARNING: Dataset size is smaller than batch size.")
            print("   Differential privacy noise CANNOT be applied.")
            print("   Noise multiplier becomes 0. This run is NON-DP.")
            print("   (Your synthetic data will be high-utility but NOT private.)")
            print("="*80)
        
        noise_multiplier = compute_noise_multiplier(
            epsilon=self.dp_config.epsilon,
            delta=self.dp_config.delta,
            n_samples=n_samples,
            batch_size=self.gan_config.batch_size,
            epochs=self.gan_config.epochs
        )
        
        print()
        print("🔒 DP Configuration:")
        print(f"   ε = {self.dp_config.epsilon}")
        print(f"   δ = {self.dp_config.delta}")
        print(f"   Noise multiplier = {noise_multiplier:.4f}")
        print(f"   Clip norm = {self.dp_config.clip_norm}")
        
        print()
        print("🎓 Training Configuration:")
        print(f"   Epochs = {self.gan_config.epochs}")
        print(f"   Batch size = {self.gan_config.batch_size}")
        print(f"   Mode = {'Unconditional' if self.unconditional else 'Conditional'}")
        
        print()
        print("🚀 Starting training...")
        
        for epoch in range(self.gan_config.epochs):
            # Train critic
            for _ in range(self.gan_config.n_critic):
                if self.unconditional:
                    idx = np.random.choice(n_samples, self.gan_config.batch_size)
                    real_batch = tf.constant(X_real[idx], dtype=tf.float32)
                    noise = tf.random.normal([self.gan_config.batch_size, self.gan_config.latent_dim])
                    
                    with tf.GradientTape() as tape:
                        fake_batch = self.generator(noise, training=True)
                        real_output = self.critic(real_batch, training=True)
                        fake_output = self.critic(fake_batch, training=True)
                        
                        c_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
                        
                        alpha = tf.random.uniform([self.gan_config.batch_size, 1], 0., 1.)
                        interpolated = alpha * real_batch + (1 - alpha) * fake_batch
                        
                        with tf.GradientTape() as gp_tape:
                            gp_tape.watch(interpolated)
                            pred = self.critic(interpolated, training=True)
                        
                        grads = gp_tape.gradient(pred, interpolated)
                        grad_norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1))
                        gp = self.gan_config.gp_weight * tf.reduce_mean(tf.square(grad_norms - 1.0))
                        c_loss += gp
                else:
                    sampled_classes = np.random.choice(
                        self.n_classes, 
                        self.gan_config.batch_size, 
                        p=self.class_probs
                    )
                    
                    real_batch = []
                    for cls in sampled_classes:
                        mask = (y_real == cls)
                        if mask.sum() > 0:
                            idx = np.random.choice(np.where(mask)[0])
                            real_batch.append(X_real[idx])
                        else:
                            idx = np.random.randint(0, n_samples)
                            real_batch.append(X_real[idx])
                    
                    real_batch = tf.constant(np.array(real_batch), dtype=tf.float32)
                    cond = tf.constant(np.eye(self.n_classes)[sampled_classes], dtype=tf.float32)
                    noise = tf.random.normal([self.gan_config.batch_size, self.gan_config.latent_dim])
                    
                    with tf.GradientTape() as tape:
                        fake_batch = self.generator([noise, cond], training=True)
                        real_output = self.critic([real_batch, cond], training=True)
                        fake_output = self.critic([fake_batch, cond], training=True)
                        
                        c_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
                        gp = gradient_penalty(
                            self.critic, real_batch, fake_batch, cond, 
                            self.gan_config.batch_size, self.gan_config.gp_weight
                        )
                        c_loss += gp
                
                c_grads = tape.gradient(c_loss, self.critic.trainable_variables)
                c_grads = clip_gradients(c_grads, self.dp_config.clip_norm)
                
                if self.dp_config.enable_dp:
                    c_grads = add_dp_noise(c_grads, noise_multiplier, self.dp_config.clip_norm)
                
                self.c_optimizer.apply_gradients(zip(c_grads, self.critic.trainable_variables))
            
            # Train generator
            if self.unconditional:
                noise = tf.random.normal([self.gan_config.batch_size, self.gan_config.latent_dim])
                real_samples_idx = np.random.choice(len(X_real), self.gan_config.batch_size)
                real_samples = tf.constant(X_real[real_samples_idx], dtype=tf.float32)
                
                with tf.GradientTape() as tape:
                    fake_batch = self.generator(noise, training=True)
                    fake_output = self.critic(fake_batch, training=True)
                    
                    adv_loss = -tf.reduce_mean(fake_output)
                    
                    mean_loss = tf.reduce_mean(tf.square(
                        tf.reduce_mean(fake_batch, axis=0) - tf.reduce_mean(real_samples, axis=0)
                    ))
                    
                    std_loss = tf.reduce_mean(tf.square(
                        tf.math.reduce_std(fake_batch, axis=0) - tf.math.reduce_std(real_samples, axis=0)
                    ))
                    
                    per_feature_loss = tf.reduce_mean([
                        tf.reduce_mean(tf.abs(fake_batch[:, i] - real_samples[:, i]))
                        for i in range(fake_batch.shape[1])
                    ])
                    
                    real_norm = (real_samples - tf.reduce_mean(real_samples, axis=0)) / (tf.math.reduce_std(real_samples, axis=0) + 1e-8)
                    fake_norm = (fake_batch - tf.reduce_mean(fake_batch, axis=0)) / (tf.math.reduce_std(fake_batch, axis=0) + 1e-8)
                    
                    real_cov = tfp.stats.covariance(real_norm, sample_axis=0, event_axis=-1)
                    fake_cov = tfp.stats.covariance(fake_norm, sample_axis=0, event_axis=-1)
                    corr_loss = tf.reduce_mean(tf.square(real_cov - fake_cov))
                    
                    g_loss = (adv_loss + 0.5 * mean_loss + 0.5 * std_loss + 
                             self.gan_config.marginal_weight * per_feature_loss + 
                             self.gan_config.correlation_weight * corr_loss)
            else:
                sampled_classes = np.random.choice(
                    self.n_classes, 
                    self.gan_config.batch_size, 
                    p=self.class_probs
                )
                cond = tf.constant(np.eye(self.n_classes)[sampled_classes], dtype=tf.float32)
                noise = tf.random.normal([self.gan_config.batch_size, self.gan_config.latent_dim])
                
                real_samples_idx = np.random.choice(len(X_real), self.gan_config.batch_size)
                real_samples = tf.constant(X_real[real_samples_idx], dtype=tf.float32)
                
                with tf.GradientTape() as tape:
                    fake_batch = self.generator([noise, cond], training=True)
                    fake_output = self.critic([fake_batch, cond], training=True)
                    
                    adv_loss = -tf.reduce_mean(fake_output)
                    
                    mean_loss = tf.reduce_mean(tf.square(
                        tf.reduce_mean(fake_batch, axis=0) - tf.reduce_mean(real_samples, axis=0)
                    ))
                    
                    std_loss = tf.reduce_mean(tf.square(
                        tf.math.reduce_std(fake_batch, axis=0) - tf.math.reduce_std(real_samples, axis=0)
                    ))
                    
                    per_feature_loss = tf.reduce_mean([
                        tf.reduce_mean(tf.abs(fake_batch[:, i] - real_samples[:, i]))
                        for i in range(fake_batch.shape[1])
                    ])
                    
                    real_norm = (real_samples - tf.reduce_mean(real_samples, axis=0)) / (tf.math.reduce_std(real_samples, axis=0) + 1e-8)
                    fake_norm = (fake_batch - tf.reduce_mean(fake_batch, axis=0)) / (tf.math.reduce_std(fake_batch, axis=0) + 1e-8)
                    
                    real_cov = tfp.stats.covariance(real_norm, sample_axis=0, event_axis=-1)
                    fake_cov = tfp.stats.covariance(fake_norm, sample_axis=0, event_axis=-1)
                    corr_loss = tf.reduce_mean(tf.square(real_cov - fake_cov))
                    
                    g_loss = (adv_loss + 0.5 * mean_loss + 0.5 * std_loss + 
                             self.gan_config.marginal_weight * per_feature_loss + 
                             self.gan_config.correlation_weight * corr_loss)
            
            g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
            g_grads = clip_gradients(g_grads, self.dp_config.clip_norm)
            
            if self.dp_config.enable_dp:
                g_grads = add_dp_noise(g_grads, noise_multiplier * 0.5, self.dp_config.clip_norm)
            
            self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))
            
            if (epoch + 1) % 100 == 0:
                print(f"   Epoch {epoch+1:4d}/{self.gan_config.epochs} │ G Loss: {float(g_loss):7.4f} │ C Loss: {float(c_loss):7.4f}")
    
    def sample(self, n_samples: int) -> pd.DataFrame:
        """
        Generate synthetic samples.
        
        Works in both conditional and unconditional modes.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before sampling")
        
        if self.unconditional:
            noise = tf.random.normal([n_samples, self.gan_config.latent_dim])
            Z_fake = self.generator.predict(noise, verbose=0)
            
            Z_fake = self.feature_scaler.inverse_transform(Z_fake)
            X_fake = self.copula.inverse_transform(Z_fake, None)
            
            synthetic_df = pd.DataFrame(X_fake, columns=self.copula.numeric_cols)
        else:
            sampled_classes = np.random.choice(self.n_classes, n_samples, p=self.class_probs)
            
            cond = tf.constant(np.eye(self.n_classes)[sampled_classes], dtype=tf.float32)
            noise = tf.random.normal([n_samples, self.gan_config.latent_dim])
            Z_fake = self.generator.predict([noise, cond], verbose=0)
            
            Z_fake = self.feature_scaler.inverse_transform(Z_fake)
            
            X_fake = []
            for i, cls_idx in enumerate(sampled_classes):
                label = self.label_decoder[cls_idx]
                X_sample = self.copula.inverse_transform(Z_fake[i:i+1], label)
                X_fake.append(X_sample[0])
            
            X_fake = np.array(X_fake)
            
            synthetic_df = pd.DataFrame(X_fake, columns=self.copula.numeric_cols)
            labels = [self.label_decoder[cls] for cls in sampled_classes]
            synthetic_df[self.label_col] = labels
        
        # Handle NaNs
        if synthetic_df.isnull().any().any():
            n_nans = synthetic_df.isnull().sum().sum()
            print(f"\n⚠️  Warning: {n_nans} NaN values detected in synthetic data.")
            print(f"   Applying interpolation and forward/backward fill...")
            synthetic_df = synthetic_df.interpolate(method='linear', limit_direction='both')
            synthetic_df = synthetic_df.fillna(method='bfill').fillna(method='ffill')
            remaining_nans = synthetic_df.isnull().sum().sum()
            if remaining_nans > 0:
                print(f"   Filling {remaining_nans} remaining NaNs with column means...")
                synthetic_df = synthetic_df.fillna(synthetic_df.mean())
        
        return synthetic_df
