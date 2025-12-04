"""
Main preprocessing pipeline for DP-CopulaGAN.

This module handles all data preprocessing including type detection,
categorical encoding, and numerical scaling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Optional, List, Dict
import warnings

from dp_copulagan.preprocessing.type_detection import detect_column_types
from dp_copulagan.preprocessing.categorical_encoder import CategoricalEncoder


class CopulaPreprocessor:
    """
    Complete preprocessing pipeline for DP-CopulaGAN.
    
    Handles:
    - Automatic type detection
    - Categorical encoding (one-hot or ordinal)
    - Numerical scaling (optional)
    - Missing value handling
    
    Parameters
    ----------
    categorical_encoding : str, default='onehot'
        Encoding method for categorical variables.
    scale_numeric : bool, default=False
        Whether to standardize numeric features (not needed for copula).
    handle_missing : str, default='drop'
        How to handle missing values: 'drop', 'mean', 'mode', 'ignore'.
    
    Examples
    --------
    >>> preprocessor = CopulaPreprocessor()
    >>> preprocessor.fit(train_data, label_col='target')
    >>> processed_data = preprocessor.transform(train_data)
    >>> original_data = preprocessor.inverse_transform(processed_data)
    """
    
    def __init__(self,
                 categorical_encoding: str = 'onehot',
                 scale_numeric: bool = False,
                 handle_missing: str = 'drop'):
        
        self.categorical_encoding = categorical_encoding
        self.scale_numeric = scale_numeric
        self.handle_missing = handle_missing
        
        self.categorical_encoder = None
        self.numeric_scaler = None
        self.column_types = None
        self.fitted = False
    
    def fit(self, df: pd.DataFrame, label_col: Optional[str] = None):
        """
        Fit preprocessor on training data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Training DataFrame.
        label_col : Optional[str]
            Label column to exclude from preprocessing.
        """
        df_work = df.copy()
        
        # Separate label if provided
        if label_col and label_col in df_work.columns:
            label_data = df_work[label_col]
            df_work = df_work.drop(columns=[label_col])
        else:
            label_data = None
        
        # Detect column types
        self.column_types = detect_column_types(df_work)
        
        # Fit categorical encoder
        if self.column_types['categorical']:
            self.categorical_encoder = CategoricalEncoder(encoding=self.categorical_encoding)
            self.categorical_encoder.fit(df_work, self.column_types['categorical'])
        
        # Fit numeric scaler
        if self.scale_numeric and self.column_types['numeric']:
            self.numeric_scaler = StandardScaler()
            self.numeric_scaler.fit(df_work[self.column_types['numeric']])
        
        self.label_col = label_col
        self.fitted = True
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Transformed DataFrame.
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted first")
        
        df_transformed = df.copy()
        
        # Separate label
        if self.label_col and self.label_col in df_transformed.columns:
            label_data = df_transformed[self.label_col]
            df_transformed = df_transformed.drop(columns=[self.label_col])
        else:
            label_data = None
        
        # Encode categoricals
        if self.categorical_encoder:
            df_transformed = self.categorical_encoder.transform(df_transformed)
        
        # Scale numerics
        if self.numeric_scaler:
            numeric_cols = [col for col in self.column_types['numeric'] 
                          if col in df_transformed.columns]
            if numeric_cols:
                df_transformed[numeric_cols] = self.numeric_scaler.transform(
                    df_transformed[numeric_cols]
                )
        
        # Add label back
        if label_data is not None:
            df_transformed[self.label_col] = label_data.values
        
        return df_transformed
    
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform to original space.
        
        Parameters
        ----------
        df : pd.DataFrame
            Transformed DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame in original space.
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted first")
        
        df_original = df.copy()
        
        # Separate label
        if self.label_col and self.label_col in df_original.columns:
            label_data = df_original[self.label_col]
            df_original = df_original.drop(columns=[self.label_col])
        else:
            label_data = None
        
        # Inverse scale numerics
        if self.numeric_scaler:
            numeric_cols = [col for col in self.column_types['numeric'] 
                          if col in df_original.columns]
            if numeric_cols:
                df_original[numeric_cols] = self.numeric_scaler.inverse_transform(
                    df_original[numeric_cols]
                )
        
        # Decode categoricals
        if self.categorical_encoder:
            df_original = self.categorical_encoder.inverse_transform(df_original)
        
        # Add label back
        if label_data is not None:
            df_original[self.label_col] = label_data.values
        
        return df_original
