"""
Enhanced Conditional Gaussian Copula for DP-CopulaGAN.

This module implements the core copula transformation that maps
tabular data to a multivariate Gaussian latent space while preserving
correlation structure.

UPDATED: Now supports unconditional mode (label_col=None).
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import cholesky
from typing import Dict, List, Optional
import warnings


class ConditionalGaussianCopula:
    """
    Class-conditional Gaussian copula transformation.
    
    Supports both conditional (with labels) and unconditional (without labels) modes.
    
    Parameters
    ----------
    bins : int, default=150
        Number of quantile bins for empirical CDF estimation.
    eigenvalue_threshold : float, default=0.4
        Minimum eigenvalue for correlation matrix correction.
    
    Examples
    --------
    # Conditional mode
    >>> copula = ConditionalGaussianCopula(bins=150)
    >>> copula.fit(data, label_col='target')
    >>> Z = copula.transform_to_normal(X, label=0)
    
    # Unconditional mode
    >>> copula = ConditionalGaussianCopula(bins=150)
    >>> copula.fit(data, label_col=None)
    >>> Z = copula.transform_to_normal(X, label=None)
    """
    
    def __init__(self, bins: int = 150, eigenvalue_threshold: float = 0.4):
        self.bins = bins
        self.eigenvalue_threshold = eigenvalue_threshold
        self.copulas = {}
        self.fitted = False
        self.unconditional = False
        self.n_features = None  # Will be set during fit
    
    @property
    def class_copulas(self):
        """
        Alias for self.copulas for backward compatibility with tests.
        
        Returns
        -------
        dict
            Dictionary mapping class labels to copula parameters.
        """
        return self.copulas
        
    def fit(self, data: pd.DataFrame, label_col: Optional[str] = None):
        """
        Fit copula transformation.
        
        Supports both conditional (with label) and unconditional (without label) modes.
        
        Parameters
        ----------
        data : pd.DataFrame
            Training data.
        label_col : Optional[str]
            Label column for conditional mode. If None, fits unconditionally.
        """
        # Identify numeric columns
        if label_col:
            # CONDITIONAL MODE
            self.unconditional = False
            self.numeric_cols = [c for c in data.columns if c != label_col and pd.api.types.is_numeric_dtype(data[c])]
            self.label_values = sorted(data[label_col].unique())
            self.label_probs = {val: (data[label_col] == val).mean() for val in self.label_values}
        else:
            # UNCONDITIONAL MODE
            self.unconditional = True
            self.numeric_cols = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]
            self.label_values = [None]  # Single "class" representing all data
            self.label_probs = {None: 1.0}
        
        if len(self.numeric_cols) == 0:
            raise ValueError("No numeric columns found in data")
        
        # Set n_features
        self.n_features = len(self.numeric_cols)
        
        # Fit per-class copulas
        self.copulas = {}
        for label in self.label_values:
            if label is None:
                # Unconditional: use all data
                class_data = data[self.numeric_cols]
            else:
                # Conditional: filter by class
                class_data = data[data[label_col] == label][self.numeric_cols]
            
            copula = self._fit_single_copula(class_data)
            self.copulas[label] = copula
        
        self.fitted = True
    
    def _fit_single_copula(self, data: pd.DataFrame) -> Dict:
        """
        Fit a single copula on class data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Class-specific data.
        
        Returns
        -------
        Dict
            Copula parameters including marginals and correlation matrix.
        """
        n_samples = len(data)
        n_features = len(self.numeric_cols)
        
        # 1. Fit marginal distributions
        marginals = {}
        for col in self.numeric_cols:
            values = data[col].values
            
            # Store quantiles for empirical CDF
            quantiles = np.linspace(0, 1, self.bins)
            quantile_values = np.quantile(values, quantiles)
            
            marginals[col] = {
                'sorted_values': np.sort(values),
                'quantiles': quantiles,
                'quantile_values': quantile_values,
                'mean': float(np.mean(values)),
                'std': float(np.std(values) + 1e-8),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        
        # 2. Transform to uniform via empirical CDF
        U = np.zeros((n_samples, n_features))
        for i, col in enumerate(self.numeric_cols):
            values = data[col].values
            sorted_vals = marginals[col]['sorted_values']
            # Empirical CDF
            ranks = np.searchsorted(sorted_vals, values, side='right')
            U[:, i] = (ranks + 0.5) / (len(sorted_vals) + 1)  # Add small offset
        
        # 3. Transform to Gaussian via inverse normal CDF
        Z = stats.norm.ppf(np.clip(U, 1e-6, 1-1e-6))
        
        # 4. Estimate correlation matrix
        corr = np.corrcoef(Z.T)
        
        # 5. Ensure positive definite
        eigvals, eigvecs = np.linalg.eigh(corr)
        eigvals = np.maximum(eigvals, self.eigenvalue_threshold)
        corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        # Normalize to correlation matrix
        d = np.sqrt(np.diag(corr))
        corr = corr / d[:, None] / d[None, :]
        
        # 6. Cholesky decomposition for sampling
        try:
            L = cholesky(corr, lower=True)
        except np.linalg.LinAlgError:
            warnings.warn("Cholesky decomposition failed, using identity")
            L = np.eye(n_features)
        
        return {
            'marginals': marginals,
            'correlation': corr,
            'cholesky': L
        }
    
    def transform_to_normal(self, X: pd.DataFrame, label) -> np.ndarray:
        """
        Transform data to Gaussian latent space.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data with numeric columns.
        label : any
            Class label. Use None for unconditional mode.
        
        Returns
        -------
        np.ndarray
            Transformed data in Gaussian space.
        """
        if not self.fitted:
            raise ValueError("Copula must be fitted before transform")
        
        copula = self.copulas[label]
        marginals = copula['marginals']
        
        n_samples = len(X)
        n_features = len(self.numeric_cols)
        Z = np.zeros((n_samples, n_features))
        
        for i, col in enumerate(self.numeric_cols):
            values = X[col].values
            quantile_values = marginals[col]['quantile_values']
            quantiles = marginals[col]['quantiles']
            
            # Interpolate to get uniform values
            U = np.interp(values, quantile_values, quantiles)
            U = np.clip(U, 1e-6, 1-1e-6)
            
            # Transform to Gaussian
            Z[:, i] = stats.norm.ppf(U)
        
        return Z
    
    def inverse_transform(self, Z: np.ndarray, label) -> np.ndarray:
        """
        Transform from Gaussian latent space back to original space.
        
        Parameters
        ----------
        Z : np.ndarray
            Latent Gaussian samples.
        label : any
            Class label. Use None for unconditional mode.
        
        Returns
        -------
        np.ndarray
            Reconstructed data in original space.
        """
        if not self.fitted:
            raise ValueError("Copula must be fitted before inverse transform")
        
        copula = self.copulas[label]
        marginals = copula['marginals']
        
        n_samples = Z.shape[0]
        n_features = len(self.numeric_cols)
        X = np.zeros((n_samples, n_features))
        
        for i, col in enumerate(self.numeric_cols):
            # Transform to uniform
            U = stats.norm.cdf(Z[:, i])
            U = np.clip(U, 0, 1)
            
            # Inverse empirical CDF
            quantile_values = marginals[col]['quantile_values']
            quantiles = marginals[col]['quantiles']
            
            values = np.interp(U, quantiles, quantile_values)
            
            # Clip to observed range
            values = np.clip(values, marginals[col]['min'], marginals[col]['max'])
            
            X[:, i] = values
        
        return X
    
    def sample(self, n_samples: int, label) -> np.ndarray:
        """
        Sample from the copula distribution.
        
        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        label : any
            Class label. Use None for unconditional mode.
        
        Returns
        -------
        np.ndarray
            Sampled data in original space.
        """
        if not self.fitted:
            raise ValueError("Copula must be fitted before sampling")
        
        copula = self.copulas[label]
        L = copula['cholesky']
        
        # Sample from standard normal
        Z_independent = np.random.randn(n_samples, len(self.numeric_cols))
        
        # Apply correlation via Cholesky
        Z_correlated = Z_independent @ L.T
        
        # Inverse transform to original space
        X = self.inverse_transform(Z_correlated, label)
        
        return X
