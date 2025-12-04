"""Unit tests for configuration."""

import pytest
from dp_copulagan.utils import DPConfig, GANConfig


def test_dp_config():
    """Test DP configuration."""
    config = DPConfig(epsilon=1.0, delta=1e-5)
    
    assert config.epsilon == 1.0
    assert config.delta == 1e-5
    assert config.clip_norm == 1.0


def test_dp_config_validation():
    """Test DP config validation."""
    with pytest.raises(ValueError):
        DPConfig(epsilon=-1.0)  # Invalid epsilon
    
    with pytest.raises(ValueError):
        DPConfig(epsilon=1.0, delta=1.5)  # Invalid delta


def test_gan_config():
    """Test GAN configuration."""
    config = GANConfig(latent_dim=192, epochs=1000)
    
    assert config.latent_dim == 192
    assert config.epochs == 1000
    assert config.batch_size == 384
