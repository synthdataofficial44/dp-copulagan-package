"""
Unsupervised ML Utility Evaluation for Unconditional Mode.

This module provides evaluation metrics for synthetic data when no label column
is available. It measures how well the synthetic data preserves the structure
of the real data using clustering, dimensionality reduction, and distributional
similarity.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from scipy.spatial.distance import cosine
from typing import Dict, Tuple
import warnings


def compute_cluster_ari(real_data: pd.DataFrame, 
                       synthetic_data: pd.DataFrame,
                       n_clusters: int = 5,
                       random_state: int = 42) -> float:
    """
    Compute Adjusted Rand Index between clusters formed on real vs synthetic data.
    
    Higher ARI (closer to 1) indicates that the clustering structure is preserved.
    
    Parameters
    ----------
    real_data : pd.DataFrame
        Real data (numeric columns only).
    synthetic_data : pd.DataFrame
        Synthetic data (numeric columns only).
    n_clusters : int, default=5
        Number of clusters for K-Means.
    random_state : int, default=42
        Random seed for reproducibility.
    
    Returns
    -------
    float
        Adjusted Rand Index in [0, 1]. Higher is better.
    """
    # Get numeric columns
    numeric_cols = real_data.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        warnings.warn("No numeric columns found. Returning ARI=0")
        return 0.0
    
    # Ensure same columns
    common_cols = list(set(numeric_cols) & set(synthetic_data.columns))
    if len(common_cols) == 0:
        warnings.warn("No common numeric columns. Returning ARI=0")
        return 0.0
    
    real = real_data[common_cols].dropna()
    synthetic = synthetic_data[common_cols].dropna()
    
    if len(real) < n_clusters or len(synthetic) < n_clusters:
        warnings.warn(f"Not enough samples for {n_clusters} clusters. Returning ARI=0")
        return 0.0
    
    # Standardize
    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real)
    synthetic_scaled = scaler.transform(synthetic)
    
    # Fit K-Means on real data
    kmeans_real = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels_real = kmeans_real.fit_predict(real_scaled)
    
    # Fit K-Means on synthetic data
    kmeans_synthetic = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels_synthetic = kmeans_synthetic.fit_predict(synthetic_scaled)
    
    # Use minimum sample size for fair comparison
    min_size = min(len(labels_real), len(labels_synthetic))
    
    # Compute ARI between the two clusterings
    ari = adjusted_rand_score(labels_real[:min_size], labels_synthetic[:min_size])
    
    # Normalize to [0, 1] (ARI is in [-1, 1], but typically positive)
    ari_normalized = max(0.0, min(1.0, (ari + 1) / 2))
    
    return ari_normalized


def compute_pca_similarity(real_data: pd.DataFrame,
                          synthetic_data: pd.DataFrame,
                          n_components: int = None) -> float:
    """
    Compute PCA geometry similarity between real and synthetic data.
    
    Measures:
    1. Cosine similarity between principal eigenvectors
    2. Distance between explained variance ratios
    
    Parameters
    ----------
    real_data : pd.DataFrame
        Real data (numeric columns only).
    synthetic_data : pd.DataFrame
        Synthetic data (numeric columns only).
    n_components : int, optional
        Number of PCA components. If None, uses min(n_features, 10).
    
    Returns
    -------
    float
        PCA similarity score in [0, 1]. Higher is better.
    """
    # Get numeric columns
    numeric_cols = real_data.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        warnings.warn("No numeric columns found. Returning PCA similarity=0")
        return 0.0
    
    # Ensure same columns
    common_cols = list(set(numeric_cols) & set(synthetic_data.columns))
    if len(common_cols) == 0:
        warnings.warn("No common numeric columns. Returning PCA similarity=0")
        return 0.0
    
    real = real_data[common_cols].dropna()
    synthetic = synthetic_data[common_cols].dropna()
    
    if len(real) < 2 or len(synthetic) < 2:
        warnings.warn("Not enough samples for PCA. Returning similarity=0")
        return 0.0
    
    # Determine number of components
    if n_components is None:
        n_components = min(len(common_cols), 10, len(real) - 1, len(synthetic) - 1)
    
    if n_components < 1:
        return 0.0
    
    # Standardize
    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real)
    synthetic_scaled = scaler.transform(synthetic)
    
    # Fit PCA on real data
    pca_real = PCA(n_components=n_components)
    pca_real.fit(real_scaled)
    
    # Fit PCA on synthetic data
    pca_synthetic = PCA(n_components=n_components)
    pca_synthetic.fit(synthetic_scaled)
    
    # 1. Cosine similarity between first principal components
    pc1_real = pca_real.components_[0]
    pc1_synthetic = pca_synthetic.components_[0]
    
    # Handle sign ambiguity (PCA components can flip signs)
    cos_sim = 1 - cosine(pc1_real, pc1_synthetic)
    cos_sim = abs(cos_sim)  # Take absolute value to handle sign flip
    
    # 2. Distance between explained variance ratios
    var_real = pca_real.explained_variance_ratio_
    var_synthetic = pca_synthetic.explained_variance_ratio_
    
    # L2 distance, normalized
    var_distance = np.linalg.norm(var_real - var_synthetic)
    var_similarity = 1 / (1 + var_distance)
    
    # Combine both measures
    pca_similarity = 0.6 * cos_sim + 0.4 * var_similarity
    
    return float(np.clip(pca_similarity, 0.0, 1.0))


def compute_mmd_rbf(real_data: pd.DataFrame,
                   synthetic_data: pd.DataFrame,
                   gamma: float = 1.0,
                   sample_size: int = 1000) -> float:
    """
    Compute Maximum Mean Discrepancy with RBF kernel.
    
    Lower MMD indicates better distributional similarity.
    
    Parameters
    ----------
    real_data : pd.DataFrame
        Real data (numeric columns only).
    synthetic_data : pd.DataFrame
        Synthetic data (numeric columns only).
    gamma : float, default=1.0
        RBF kernel bandwidth parameter.
    sample_size : int, default=1000
        Maximum samples to use (for computational efficiency).
    
    Returns
    -------
    float
        MMD distance (lower is better). Returns normalized score in [0, 1].
    """
    # Get numeric columns
    numeric_cols = real_data.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        warnings.warn("No numeric columns found. Returning MMD=1.0")
        return 1.0
    
    # Ensure same columns
    common_cols = list(set(numeric_cols) & set(synthetic_data.columns))
    if len(common_cols) == 0:
        warnings.warn("No common numeric columns. Returning MMD=1.0")
        return 1.0
    
    real = real_data[common_cols].dropna().values
    synthetic = synthetic_data[common_cols].dropna().values
    
    if len(real) < 10 or len(synthetic) < 10:
        warnings.warn("Not enough samples for MMD. Returning MMD=1.0")
        return 1.0
    
    # Sample for efficiency
    if len(real) > sample_size:
        indices = np.random.choice(len(real), sample_size, replace=False)
        real = real[indices]
    
    if len(synthetic) > sample_size:
        indices = np.random.choice(len(synthetic), sample_size, replace=False)
        synthetic = synthetic[indices]
    
    # Standardize
    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real)
    synthetic_scaled = scaler.transform(synthetic)
    
    # RBF kernel
    def rbf_kernel(X, Y, gamma):
        """Compute RBF kernel matrix between X and Y."""
        XX = np.sum(X**2, axis=1).reshape(-1, 1)
        YY = np.sum(Y**2, axis=1).reshape(1, -1)
        XY = np.dot(X, Y.T)
        distances = XX + YY - 2 * XY
        return np.exp(-gamma * distances)
    
    # Compute kernel matrices
    K_xx = rbf_kernel(real_scaled, real_scaled, gamma)
    K_yy = rbf_kernel(synthetic_scaled, synthetic_scaled, gamma)
    K_xy = rbf_kernel(real_scaled, synthetic_scaled, gamma)
    
    # MMD^2 = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]
    n = len(real_scaled)
    m = len(synthetic_scaled)
    
    mmd_squared = (np.sum(K_xx) - np.trace(K_xx)) / (n * (n - 1)) \
                - 2 * np.sum(K_xy) / (n * m) \
                + (np.sum(K_yy) - np.trace(K_yy)) / (m * (m - 1))
    
    mmd = np.sqrt(max(0, mmd_squared))
    
    # Normalize to [0, 1] range (empirically, MMD is typically < 0.5)
    mmd_normalized = min(1.0, mmd / 0.5)
    
    return float(mmd_normalized)


def evaluate_unsupervised_utility(real_data: pd.DataFrame,
                                 synthetic_data: pd.DataFrame,
                                 n_clusters: int = 5,
                                 verbose: bool = True) -> Dict[str, float]:
    """
    Evaluate unsupervised ML utility for unconditional synthetic data.
    
    Computes:
    - Cluster ARI (Adjusted Rand Index)
    - PCA Geometry Similarity
    - MMD (Maximum Mean Discrepancy)
    - Combined Utility Score
    
    Parameters
    ----------
    real_data : pd.DataFrame
        Real data.
    synthetic_data : pd.DataFrame
        Synthetic data.
    n_clusters : int, default=5
        Number of clusters for K-Means.
    verbose : bool, default=True
        Whether to print results.
    
    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - 'cluster_ari': Adjusted Rand Index [0, 1]
        - 'pca_similarity': PCA geometry similarity [0, 1]
        - 'mmd_distance': MMD distance [0, 1] (lower is better)
        - 'unsupervised_utility': Combined score [0, 1] (higher is better)
    """
    if verbose:
        print()
        print("="*80)
        print("UNSUPERVISED ML UTILITY (UNCONDITIONAL MODE)")
        print("="*80)
    
    # Compute individual metrics
    ari = compute_cluster_ari(real_data, synthetic_data, n_clusters=n_clusters)
    pca_sim = compute_pca_similarity(real_data, synthetic_data)
    mmd = compute_mmd_rbf(real_data, synthetic_data)
    
    # Combined utility score
    # Higher is better, so we use (1 - mmd) for MMD
    utility = 0.4 * ari + 0.3 * pca_sim + 0.3 * (1 - mmd)
    
    if verbose:
        print()
        print("="*80)
        print("PER-MODEL UNSUPERVISED ML UTILITY")
        print("="*80)
        print(f"KMeans ARI:                 {ari:.4f}")
        print(f"PCA Similarity Score:       {pca_sim:.4f}")
        print(f"MMD Distance:               {mmd:.4f}")
        print()
        print("-" * 80)
        print(f"Final Unsupervised Utility: {utility:.4f}")
        print("-" * 80)
    
    return {
        'cluster_ari': float(ari),
        'pca_similarity': float(pca_sim),
        'mmd_distance': float(mmd),
        'unsupervised_utility': float(utility)
    }
