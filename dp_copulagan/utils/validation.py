"""
Data validation utilities for DP-CopulaGAN.

This module provides functions to validate user inputs and ensure
data quality before training.
"""

import pandas as pd
import numpy as np
from typing import Optional, List


def validate_dataframe(df: pd.DataFrame, 
                       label_col: Optional[str] = None,
                       min_samples: int = 100) -> None:
    """
    Validate input DataFrame for DP-CopulaGAN.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    label_col : Optional[str]
        Label column name (if conditional).
    min_samples : int
        Minimum required samples.
    
    Raises
    ------
    ValueError
        If validation fails.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df)}")
    
    if len(df) < min_samples:
        raise ValueError(
            f"Insufficient data: need at least {min_samples} samples, "
            f"got {len(df)}"
        )
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    # Check for all-null columns
    null_cols = df.columns[df.isnull().all()].tolist()
    if null_cols:
        raise ValueError(f"Columns with all null values: {null_cols}")
    
    # Check label column
    if label_col is not None:
        validate_label_column(df, label_col)


def validate_label_column(df: pd.DataFrame, label_col: str) -> None:
    """
    Validate label column for conditional generation.
    
    Automatically encodes categorical (string) labels to integers.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    label_col : str
        Label column name.
    
    Raises
    ------
    ValueError
        If label column is invalid.
    """
    if label_col not in df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Check for null labels
    if df[label_col].isnull().any():
        raise ValueError(
            f"Label column '{label_col}' contains null values. "
            f"Please remove or impute them first."
        )
    
    # FIX #1: Auto-encode categorical labels to integers
    if df[label_col].dtype == 'object' or df[label_col].dtype.name == 'category':
        print(f"\n⚠️  Warning: Label column '{label_col}' is categorical (string/category type).")
        print(f"   Auto-encoding to integers for compatibility with sklearn models.")
        original_values = df[label_col].unique()
        df[label_col] = df[label_col].astype('category').cat.codes
        print(f"   Encoded {len(original_values)} categories to integers 0-{len(original_values)-1}")
        print(f"   Original values: {list(original_values)[:5]}{'...' if len(original_values) > 5 else ''}")
    
    # Check class distribution
    class_counts = df[label_col].value_counts()
    if len(class_counts) < 2:
        raise ValueError(
            f"Label column must have at least 2 classes, found {len(class_counts)}"
        )
    
    min_class_count = class_counts.min()
    if min_class_count < 10:
        raise ValueError(
            f"Smallest class has only {min_class_count} samples. "
            f"Each class needs at least 10 samples."
        )


def check_numeric_columns(df: pd.DataFrame, 
                         cols: List[str],
                         allow_infinite: bool = False) -> None:
    """
    Validate numeric columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    cols : List[str]
        Column names to check.
    allow_infinite : bool
        Whether to allow infinite values.
    
    Raises
    ------
    ValueError
        If columns contain invalid numeric values.
    """
    for col in cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found")
        
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' is not numeric")
        
        if not allow_infinite:
            if np.isinf(df[col]).any():
                raise ValueError(f"Column '{col}' contains infinite values")
