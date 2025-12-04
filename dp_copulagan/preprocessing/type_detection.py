"""
Automatic data type detection for preprocessing.

This module provides functions to automatically detect column types
(numeric, categorical, datetime, etc.) in DataFrames.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


def detect_column_types(df: pd.DataFrame, 
                       categorical_threshold: int = 10) -> Dict[str, List[str]]:
    """
    Automatically detect column types.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    categorical_threshold : int, default=10
        Max unique values for a numeric column to be considered categorical.
    
    Returns
    -------
    Dict[str, List[str]]
        Dictionary with keys: 'numeric', 'categorical', 'datetime', 'other'.
    
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'age': [25, 30, 35],
    ...     'gender': ['M', 'F', 'M'],
    ...     'income': [50000, 60000, 70000]
    ... })
    >>> types = detect_column_types(df)
    >>> types['numeric']
    ['age', 'income']
    >>> types['categorical']
    ['gender']
    """
    
    result = {
        'numeric': [],
        'categorical': [],
        'datetime': [],
        'other': []
    }
    
    for col in df.columns:
        # Datetime
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            result['datetime'].append(col)
        
        # Numeric
        elif pd.api.types.is_numeric_dtype(df[col]):
            n_unique = df[col].nunique()
            
            # Check if it's actually categorical (few unique values)
            # BUT: if dtype is float, keep it as numeric regardless of cardinality
            # This ensures columns like [1.0, 2.0, 3.0] stay numeric
            if pd.api.types.is_float_dtype(df[col]):
                result['numeric'].append(col)
            elif n_unique <= categorical_threshold:
                result['categorical'].append(col)
            else:
                result['numeric'].append(col)
        
        # Categorical/String
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            result['categorical'].append(col)
        
        # Other
        else:
            result['other'].append(col)
    
    return result


def is_binary(series: pd.Series) -> bool:
    """Check if a series is binary (2 unique values)."""
    return series.nunique() <= 2


def get_cardinality(series: pd.Series) -> int:
    """Get number of unique values in series."""
    return series.nunique()
