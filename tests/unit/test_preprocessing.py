"""Unit tests for preprocessing."""

import pytest
import numpy as np
import pandas as pd
from dp_copulagan.preprocessing import detect_column_types, CategoricalEncoder


def test_type_detection():
    """Test automatic type detection."""
    data = pd.DataFrame({
        'numeric': [1.0, 2.0, 3.0],
        'categorical': ['A', 'B', 'C'],
        'binary': [0, 1, 0]
    })
    
    types = detect_column_types(data, categorical_threshold=5)
    
    assert 'numeric' in types['numeric']
    assert 'categorical' in types['categorical']
    assert 'binary' in types['categorical']


def test_categorical_encoding():
    """Test categorical encoding."""
    data = pd.DataFrame({
        'gender': ['M', 'F', 'M', 'F'],
        'education': ['HS', 'BS', 'MS', 'BS']
    })
    
    encoder = CategoricalEncoder(encoding='onehot')
    encoder.fit(data, categorical_cols=['gender', 'education'])
    
    encoded = encoder.transform(data)
    decoded = encoder.inverse_transform(encoded)
    
    assert 'gender_M' in encoded.columns
    assert 'gender' in decoded.columns
    assert list(decoded['gender']) == list(data['gender'])
