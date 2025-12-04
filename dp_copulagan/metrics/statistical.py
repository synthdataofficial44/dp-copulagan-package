"""
Statistical similarity metrics for synthetic data evaluation.

This module computes various statistical measures to assess how well
synthetic data matches the real data distribution.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from typing import Dict


def compute_jsd(real: pd.DataFrame, synthetic: pd.DataFrame, bins: int = 50) -> float:
    """
    Compute Jensen-Shannon Divergence between marginal distributions.
    
    Parameters
    ----------
    real : pd.DataFrame
        Real data.
    synthetic : pd.DataFrame
        Synthetic data.
    bins : int, default=50
        Number of bins for histograms.
    
    Returns
    -------
    float
        Mean JSD across all numeric columns.
    """
    numeric_cols = real.select_dtypes(include=[np.number]).columns
    jsds = []
    
    for col in numeric_cols:
        # Create histograms
        min_val = min(real[col].min(), synthetic[col].min())
        max_val = max(real[col].max(), synthetic[col].max())
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        real_hist, _ = np.histogram(real[col], bins=bin_edges, density=True)
        synth_hist, _ = np.histogram(synthetic[col], bins=bin_edges, density=True)
        
        # Normalize
        real_hist = real_hist / (real_hist.sum() + 1e-10)
        synth_hist = synth_hist / (synth_hist.sum() + 1e-10)
        
        # Compute JSD
        jsd = jensenshannon(real_hist, synth_hist)
        jsds.append(jsd)
    
    return np.mean(jsds)


def compute_wasserstein(real: pd.DataFrame, synthetic: pd.DataFrame) -> float:
    """
    Compute Wasserstein distance between distributions.
    
    Parameters
    ----------
    real : pd.DataFrame
        Real data.
    synthetic : pd.DataFrame
        Synthetic data.
    
    Returns
    -------
    float
        Mean Wasserstein distance across numeric columns.
    """
    numeric_cols = real.select_dtypes(include=[np.number]).columns
    distances = []
    
    for col in numeric_cols:
        dist = stats.wasserstein_distance(real[col], synthetic[col])
        distances.append(dist)
    
    return np.mean(distances)


def compute_correlation_rmse(real: pd.DataFrame, synthetic: pd.DataFrame) -> float:
    """
    Compute RMSE between correlation matrices.
    
    Parameters
    ----------
    real : pd.DataFrame
        Real data.
    synthetic : pd.DataFrame
        Synthetic data.
    
    Returns
    -------
    float
        RMSE between correlation matrices.
    """
    numeric_cols = real.select_dtypes(include=[np.number]).columns
    
    real_corr = real[numeric_cols].corr().values
    synth_corr = synthetic[numeric_cols].corr().values
    
    rmse = np.sqrt(np.mean((real_corr - synth_corr) ** 2))
    
    return rmse


def compute_statistical_metrics(real: pd.DataFrame, 
                                synthetic: pd.DataFrame) -> Dict[str, float]:
    """
    Compute all statistical similarity metrics.
    
    Parameters
    ----------
    real : pd.DataFrame
        Real data.
    synthetic : pd.DataFrame
        Synthetic data.
    
    Returns
    -------
    Dict[str, float]
        Dictionary of metric names and values.
    
    Examples
    --------
    >>> metrics = compute_statistical_metrics(real_data, synthetic_data)
    >>> print(f"JSD: {metrics['jsd']:.4f}")
    >>> print(f"Wasserstein: {metrics['wasserstein']:.4f}")
    """
    return {
        'jsd': compute_jsd(real, synthetic),
        'wasserstein': compute_wasserstein(real, synthetic),
        'correlation_rmse': compute_correlation_rmse(real, synthetic),
    }
