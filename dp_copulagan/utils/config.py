"""
Configuration classes for DP-CopulaGAN.

This module provides structured configuration classes for managing:
- Differential privacy parameters
- GAN architecture and training hyperparameters
- Copula transformation settings
- Preprocessing options

Classes
-------
DPConfig : Differential privacy configuration
GANConfig : GAN training configuration
CopulaConfig : Copula transformation configuration
PreprocessingConfig : Data preprocessing configuration
"""

from dataclasses import dataclass, field
from typing import Optional, List
import warnings


@dataclass
class DPConfig:
    """
    Configuration for differential privacy parameters.
    
    This class manages the privacy budget and noise calibration for
    differentially private training. It follows the composition theorems
    from differential privacy literature.
    
    Parameters
    ----------
    epsilon : float
        Privacy budget (ε). Smaller values provide stronger privacy.
        Typical values: 0.01 (very private), 0.1, 1.0, 10.0 (less private).
    delta : float, default=1e-5
        Probability of privacy breach (δ). Should be << 1/n_samples.
        Standard choice: 1e-5 or 1e-6.
    enable_dp : bool, default=True
        Whether to enable differential privacy. Set False for non-private baseline.
    clip_norm : float, default=1.0
        Gradient clipping threshold for DP-SGD.
    noise_multiplier : Optional[float], default=None
        Noise scale multiplier. If None, automatically calibrated from epsilon/delta.
    
    Attributes
    ----------
    epsilon : float
        Privacy budget.
    delta : float
        Failure probability.
    enable_dp : bool
        DP enabled flag.
    clip_norm : float
        Gradient clipping norm.
    noise_multiplier : Optional[float]
        Noise multiplier (auto-calibrated if None).
    
    Examples
    --------
    >>> # High privacy setting
    >>> dp_config = DPConfig(epsilon=0.1, delta=1e-5)
    >>> 
    >>> # Moderate privacy
    >>> dp_config = DPConfig(epsilon=1.0, delta=1e-5)
    >>> 
    >>> # No privacy (baseline)
    >>> dp_config = DPConfig(epsilon=float('inf'), enable_dp=False)
    
    Notes
    -----
    The noise_multiplier is automatically calibrated using the Gaussian mechanism:
        σ = sqrt(2 * log(1.25/δ)) / ε
    
    This implements the strong composition theorem for (ε, δ)-differential privacy.
    """
    
    epsilon: float
    delta: float = 1e-5
    enable_dp: bool = True
    clip_norm: float = 1.0
    noise_multiplier: Optional[float] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")
        
        if not (0 < self.delta < 1):
            raise ValueError(f"delta must be in (0, 1), got {self.delta}")
        
        if self.clip_norm <= 0:
            raise ValueError(f"clip_norm must be positive, got {self.clip_norm}")
        
        # Warn if delta is too large
        if self.delta > 1e-3:
            warnings.warn(
                f"delta={self.delta} is quite large. "
                f"Standard practice is delta << 1/n_samples (e.g., 1e-5).",
                UserWarning
            )
        
        # Warn about privacy strength
        if self.epsilon < 0.1:
            print(f"ℹ️  Strong privacy: ε={self.epsilon} (high noise, lower utility)")
        elif self.epsilon > 10:
            warnings.warn(
                f"epsilon={self.epsilon} provides weak privacy guarantees.",
                UserWarning
            )
    
    def get_privacy_description(self) -> str:
        """Return human-readable privacy level description."""
        if not self.enable_dp:
            return "No Privacy (DP disabled)"
        elif self.epsilon < 0.1:
            return f"Very Strong Privacy (ε={self.epsilon})"
        elif self.epsilon < 1.0:
            return f"Strong Privacy (ε={self.epsilon})"
        elif self.epsilon < 5.0:
            return f"Moderate Privacy (ε={self.epsilon})"
        else:
            return f"Weak Privacy (ε={self.epsilon})"


@dataclass
class GANConfig:
    """
    Configuration for GAN architecture and training.
    
    This class manages all hyperparameters for the generator, critic,
    and training process. Default values are optimized based on extensive
    experiments with Adult, EEG, and Credit Card datasets.
    
    Parameters
    ----------
    latent_dim : int, default=192
        Dimension of latent noise vector z.
    epochs : int, default=1000
        Number of training epochs.
    batch_size : int, default=384
        Training batch size.
    n_critic : int, default=5
        Number of critic updates per generator update (WGAN-GP standard).
    g_lr : float, default=1e-4
        Generator learning rate.
    c_lr : float, default=3e-4
        Critic learning rate (typically 3x generator for WGAN).
    gp_weight : float, default=12.0
        Gradient penalty coefficient (λ in WGAN-GP).
    marginal_weight : float, default=2.0
        Weight for marginal distribution matching loss.
    correlation_weight : float, default=2.0
        Weight for correlation preservation loss.
    dropout_rate : float, default=0.2
        Dropout rate in generator (helps prevent overfitting).
    spectral_norm : bool, default=True
        Use spectral normalization in critic (Lipschitz constraint).
    layer_norm : bool, default=True
        Use layer normalization in generator (training stability).
    activation : str, default='gelu'
        Activation function ('gelu', 'relu', 'leakyrelu').
    
    Attributes
    ----------
    All parameters listed above as attributes.
    
    Examples
    --------
    >>> # Default configuration (recommended)
    >>> gan_config = GANConfig()
    >>> 
    >>> # Quick training for testing
    >>> gan_config = GANConfig(epochs=100, batch_size=128)
    >>> 
    >>> # High-capacity model
    >>> gan_config = GANConfig(latent_dim=256, g_lr=5e-5)
    
    Notes
    -----
    Architecture decisions:
    - latent_dim=192: Balances capacity and overfitting
    - n_critic=5: WGAN-GP standard for stable training
    - gp_weight=12.0: Higher than standard (10) for tabular data
    - Spectral normalization: Enforces 1-Lipschitz constraint
    - GELU activation: Smoother gradients than ReLU
    
    These settings are kept consistent with the original research implementation.
    """
    
    # Architecture
    latent_dim: int = 192
    
    # Training
    epochs: int = 1000
    batch_size: int = 384
    n_critic: int = 5
    
    # Optimization
    g_lr: float = 1e-4
    c_lr: float = 3e-4
    
    # Loss weights
    gp_weight: float = 12.0
    marginal_weight: float = 2.0
    correlation_weight: float = 2.0
    
    # Regularization
    dropout_rate: float = 0.2
    spectral_norm: bool = True
    layer_norm: bool = True
    
    # Activation
    activation: str = 'gelu'
    
    # Hidden dimensions (computed internally)
    generator_dims: List[int] = field(default_factory=lambda: [512, 768, 512, 256])
    critic_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    
    def __post_init__(self):
        """Validate configuration."""
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {self.latent_dim}")
        
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        
        if self.n_critic <= 0:
            raise ValueError(f"n_critic must be positive, got {self.n_critic}")
        
        if not (0 <= self.dropout_rate < 1):
            raise ValueError(f"dropout_rate must be in [0, 1), got {self.dropout_rate}")
        
        if self.activation not in ['gelu', 'relu', 'leakyrelu', 'swish']:
            warnings.warn(
                f"activation='{self.activation}' is non-standard. "
                f"Recommended: 'gelu' or 'relu'.",
                UserWarning
            )


@dataclass
class CopulaConfig:
    """
    Configuration for Gaussian copula transformation.
    
    Parameters
    ----------
    bins : int, default=150
        Number of quantile bins for empirical CDF estimation.
    eigenvalue_threshold : float, default=0.4
        Minimum eigenvalue for correlation matrix correction.
    conditional : bool, default=True
        Use class-conditional copula (separate per class).
    
    Attributes
    ----------
    bins : int
        Quantile bins.
    eigenvalue_threshold : float
        Eigenvalue correction threshold.
    conditional : bool
        Conditional copula flag.
    
    Examples
    --------
    >>> copula_config = CopulaConfig(bins=200, eigenvalue_threshold=0.5)
    
    Notes
    -----
    The eigenvalue threshold ensures the correlation matrix is positive definite
    after Gaussian transformation. Standard threshold is 0.4 for numerical stability.
    """
    
    bins: int = 150
    eigenvalue_threshold: float = 0.4
    conditional: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.bins < 10:
            raise ValueError(f"bins must be >= 10, got {self.bins}")
        
        if not (0 < self.eigenvalue_threshold < 1):
            raise ValueError(
                f"eigenvalue_threshold must be in (0, 1), got {self.eigenvalue_threshold}"
            )


@dataclass
class PreprocessingConfig:
    """
    Configuration for data preprocessing.
    
    Parameters
    ----------
    auto_detect_types : bool, default=True
        Automatically detect column types (numeric/categorical).
    handle_missing : str, default='drop'
        Missing value strategy: 'drop', 'mean', 'mode', 'ignore'.
    categorical_encoding : str, default='onehot'
        Encoding for categorical variables: 'onehot', 'ordinal'.
    scale_numeric : bool, default=True
        Standardize numeric features.
    clip_outliers : Optional[float], default=None
        If provided, clip outliers at ±n standard deviations.
    
    Attributes
    ----------
    All parameters listed above.
    
    Examples
    --------
    >>> preprocessing_config = PreprocessingConfig(
    ...     handle_missing='mean',
    ...     clip_outliers=3.0
    ... )
    """
    
    auto_detect_types: bool = True
    handle_missing: str = 'drop'
    categorical_encoding: str = 'onehot'
    scale_numeric: bool = True
    clip_outliers: Optional[float] = None
    
    def __post_init__(self):
        """Validate configuration."""
        valid_missing = ['drop', 'mean', 'mode', 'ignore']
        if self.handle_missing not in valid_missing:
            raise ValueError(
                f"handle_missing must be one of {valid_missing}, "
                f"got '{self.handle_missing}'"
            )
        
        valid_encoding = ['onehot', 'ordinal']
        if self.categorical_encoding not in valid_encoding:
            raise ValueError(
                f"categorical_encoding must be one of {valid_encoding}, "
                f"got '{self.categorical_encoding}'"
            )
        
        if self.clip_outliers is not None and self.clip_outliers <= 0:
            raise ValueError(
                f"clip_outliers must be positive if provided, got {self.clip_outliers}"
            )
