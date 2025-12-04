"""Unit tests for Gaussian Copula."""

import pytest
import numpy as np
import pandas as pd
from dp_copulagan.copula import ConditionalGaussianCopula


def test_copula_fit():
    """Test copula fitting."""
    data = pd.DataFrame({
        'x1': np.random.randn(1000),
        'x2': np.random.randn(1000),
        'label': np.random.choice([0, 1], 1000)
    })
    
    copula = ConditionalGaussianCopula(bins=50)
    copula.fit(data, label_col='label')
    
    assert copula.fitted == True
    assert len(copula.class_copulas) == 2
    assert copula.n_features == 2


def test_copula_transform():
    """Test copula transformation."""
    data = pd.DataFrame({
        'x1': np.random.randn(1000),
        'x2': np.random.randn(1000),
        'label': np.random.choice([0, 1], 1000)
    })
    
    copula = ConditionalGaussianCopula(bins=50)
    copula.fit(data, label_col='label')
    
    X = data[data['label'] == 0][['x1', 'x2']]
    Z = copula.transform_to_normal(X, label=0)
    
    assert Z.shape == X.shape
    assert abs(np.mean(Z)) < 0.5  # Approximately Gaussian


def test_copula_inverse():
    """Test copula inverse transformation."""
    data = pd.DataFrame({
        'x1': np.random.randn(500),
        'x2': np.random.randn(500),
        'label': np.random.choice([0, 1], 500)
    })
    
    copula = ConditionalGaussianCopula(bins=30)
    copula.fit(data, label_col='label')
    
    X = data[data['label'] == 0][['x1', 'x2']]
    Z = copula.transform_to_normal(X, label=0)
    X_reconstructed = copula.inverse_transform(Z, label=0)
    
    assert X_reconstructed.shape == X.shape
