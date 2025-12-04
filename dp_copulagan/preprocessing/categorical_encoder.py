"""
Categorical encoding for preprocessing.

This module provides encoding transformers for categorical variables.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class CategoricalEncoder:
    """
    Encode categorical variables.
    
    Supports one-hot encoding and ordinal encoding.
    
    Parameters
    ----------
    encoding : str, default='onehot'
        Encoding method: 'onehot' or 'ordinal'.
    
    Attributes
    ----------
    encoding_maps : Dict
        Mapping of column names to encoding dictionaries.
    
    Examples
    --------
    >>> encoder = CategoricalEncoder(encoding='onehot')
    >>> encoder.fit(df, categorical_cols=['gender', 'education'])
    >>> encoded_df = encoder.transform(df)
    >>> original_df = encoder.inverse_transform(encoded_df)
    """
    
    def __init__(self, encoding: str = 'onehot'):
        if encoding not in ['onehot', 'ordinal']:
            raise ValueError(f"encoding must be 'onehot' or 'ordinal', got '{encoding}'")
        
        self.encoding = encoding
        self.encoding_maps = {}
        self.column_names = {}
        self.fitted = False
    
    def fit(self, df: pd.DataFrame, categorical_cols: List[str]):
        """
        Fit encoder on categorical columns.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        categorical_cols : List[str]
            List of categorical column names.
        """
        for col in categorical_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")
            
            unique_vals = df[col].dropna().unique()
            
            if self.encoding == 'onehot':
                # Create mapping: value -> column index
                self.encoding_maps[col] = {val: i for i, val in enumerate(unique_vals)}
                self.column_names[col] = [f"{col}_{val}" for val in unique_vals]
            
            else:  # ordinal
                # Create mapping: value -> integer
                self.encoding_maps[col] = {val: i for i, val in enumerate(unique_vals)}
                self.column_names[col] = [col]
        
        self.fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform categorical columns.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with encoded columns.
        """
        if not self.fitted:
            raise ValueError("Encoder must be fitted first")
        
        df_transformed = df.copy()
        
        for col in self.encoding_maps.keys():
            if col not in df_transformed.columns:
                continue
            
            if self.encoding == 'onehot':
                # One-hot encoding
                for val, idx in self.encoding_maps[col].items():
                    col_name = self.column_names[col][idx]
                    df_transformed[col_name] = (df_transformed[col] == val).astype(float)
                
                # Drop original column
                df_transformed = df_transformed.drop(columns=[col])
            
            else:  # ordinal
                # Ordinal encoding
                df_transformed[col] = df_transformed[col].map(self.encoding_maps[col])
                df_transformed[col] = df_transformed[col].fillna(-1).astype(int)
        
        return df_transformed
    
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform encoded columns back to original.
        
        Parameters
        ----------
        df : pd.DataFrame
            Encoded DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with original categorical columns.
        """
        if not self.fitted:
            raise ValueError("Encoder must be fitted first")
        
        df_original = df.copy()
        
        for col, encoding_map in self.encoding_maps.items():
            reverse_map = {v: k for k, v in encoding_map.items()}
            
            if self.encoding == 'onehot':
                # Decode one-hot
                encoded_cols = self.column_names[col]
                
                # Find which encoded column has max value
                encoded_data = df_original[encoded_cols].values
                max_indices = np.argmax(encoded_data, axis=1)
                
                # Map back to original values
                df_original[col] = [reverse_map[idx] for idx in max_indices]
                
                # Drop encoded columns
                df_original = df_original.drop(columns=encoded_cols)
            
            else:  # ordinal
                # Decode ordinal
                df_original[col] = df_original[col].map(reverse_map)
        
        return df_original
