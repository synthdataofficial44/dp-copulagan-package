"""Integration test for full pipeline."""

import pytest
import numpy as np
import pandas as pd
from dp_copulagan import DPCopulaGAN
from dp_copulagan.evaluation import evaluate_synthetic


def test_full_pipeline():
    """Test complete DP-CopulaGAN pipeline."""
    # Create sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'feature1': np.random.randn(500),
        'feature2': np.random.randn(500),
        'feature3': np.random.randn(500),
        'target': np.random.choice([0, 1], 500)
    })
    
    # Train model (quick config for testing)
    from dp_copulagan.utils import GANConfig
    gan_config = GANConfig(epochs=10, batch_size=64)
    
    model = DPCopulaGAN(
        epsilon=10.0,  # High epsilon for quick test
        delta=1e-5,
        label_col='target',
        gan_config=gan_config
    )
    
    model.fit(data)
    
    # Generate synthetic data
    synthetic = model.sample(n_samples=100)
    
    # Check output
    assert len(synthetic) == 100
    assert 'target' in synthetic.columns
    assert synthetic['target'].nunique() == 2
    
    # Evaluate
    results = evaluate_synthetic(
        data, synthetic, label_col='target', verbose=False
    )
    
    assert 'statistical' in results
    assert 'ml_utility' in results
