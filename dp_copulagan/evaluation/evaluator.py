"""
Main evaluation pipeline for DP-CopulaGAN.

This module combines statistical and ML utility metrics into a single
evaluation function.
"""

import pandas as pd
from typing import Dict, Optional
import json
import numpy as np

from dp_copulagan.metrics import (
    compute_statistical_metrics, 
    compute_ml_utility,
    compute_multiclass_ml_utility
)
from dp_copulagan.utils.unsupervised_eval import evaluate_unsupervised_utility


def evaluate_synthetic(real: pd.DataFrame,
                      synthetic: pd.DataFrame,
                      label_col: Optional[str] = None,
                      test_size: float = 0.2,
                      verbose: bool = True) -> Dict:
    """
    Comprehensive evaluation of synthetic data quality.
    
    Computes both statistical similarity metrics and ML utility (TSTR).
    
    Parameters
    ----------
    real : pd.DataFrame
        Real data.
    synthetic : pd.DataFrame
        Synthetic data.
    label_col : Optional[str]
        Label column for ML utility evaluation. If None, skips ML evaluation.
    test_size : float, default=0.2
        Test set fraction for ML evaluation.
    verbose : bool, default=True
        Whether to print results.
    
    Returns
    -------
    Dict
        Dictionary containing:
        - 'statistical': Statistical similarity metrics
        - 'ml_utility': ML utility scores (if label_col provided)
        - 'summary': Summary statistics
    
    Examples
    --------
    >>> results = evaluate_synthetic(real_data, synthetic_data, label_col='income')
    >>> print(f"JSD: {results['statistical']['jsd']:.4f}")
    >>> print(f"Mean AuROC: {results['summary']['mean_auroc']:.4f}")
    """
    
    results = {}
    
    # Statistical metrics
    if verbose:
        print()
        print("="*80)
        print("EVALUATING SYNTHETIC DATA QUALITY")
        print("="*80)
        print()
        print("📊 Computing statistical metrics...")
    
    statistical = compute_statistical_metrics(real, synthetic)
    results['statistical'] = statistical
    
    if verbose:
        print(f"   Jensen-Shannon Divergence:  {statistical['jsd']:.4f}")
        print(f"   Wasserstein Distance:       {statistical['wasserstein']:.4f}")
        print(f"   Correlation RMSE:           {statistical['correlation_rmse']:.4f}")
    
    # ML utility (conditional mode) OR Unsupervised utility (unconditional mode)
    if label_col is not None:
        # CONDITIONAL MODE: Determine if binary or multi-class
        n_classes = real[label_col].nunique()
        
        if n_classes == 2:
            # BINARY CLASSIFICATION: Use existing AUROC evaluation (unchanged)
            if verbose:
                print()
                print("🎯 Computing ML utility (TSTR - Binary Classification)...")
            
            ml_scores = compute_ml_utility(real, synthetic, label_col, test_size)
            results['ml_utility'] = ml_scores
            
            if verbose:
                print()
                print("="*80)
                print("PER-MODEL SUPERVISED ML UTILITY (TSTR)")
                print("="*80)
                for clf_name, score in ml_scores.items():
                    print(f"\n{clf_name}:")
                    print(f"   AUC: {score:.4f}")
            
            # Summary
            mean_auroc = sum(ml_scores.values()) / len(ml_scores)
            results['summary'] = {
                'mean_auroc': mean_auroc,
                'min_auroc': min(ml_scores.values()),
                'max_auroc': max(ml_scores.values()),
            }
            
            if verbose:
                print()
                print("-" * 80)
                print(f"MEAN ({len(ml_scores)} models):")
                print(f"   Mean AuROC: {mean_auroc:.4f}")
                print("-" * 80)
        
        else:
            # MULTI-CLASS CLASSIFICATION: Use new multi-class metrics
            if verbose:
                print()
                print(f"🎯 Computing ML utility (TSTR - Multi-Class, n={n_classes})...")
            
            ml_scores = compute_multiclass_ml_utility(real, synthetic, label_col, test_size)
            results['ml_utility_multiclass'] = ml_scores
            
            if verbose:
                print()
                print("="*80)
                print("PER-MODEL SUPERVISED ML UTILITY (TSTR)")
                print("="*80)
                
                # Print per-model detailed scores
                for clf_name, scores in ml_scores.items():
                    print(f"\n{clf_name}:")
                    print(f"   Macro F1:           {scores['macro_f1']:.4f}")
                    print(f"   Balanced Accuracy:  {scores['balanced_accuracy']:.4f}")
                    if scores['macro_auc'] is not None:
                        print(f"   Macro AUC (OVR):    {scores['macro_auc']:.4f}")
                    else:
                        print(f"   Macro AUC (OVR):    N/A")
                
                # Aggregate metrics
                macro_f1_scores = [s['macro_f1'] for s in ml_scores.values() if s['macro_f1'] is not None]
                balanced_acc_scores = [s['balanced_accuracy'] for s in ml_scores.values() if s['balanced_accuracy'] is not None]
                macro_auc_scores = [s['macro_auc'] for s in ml_scores.values() if s['macro_auc'] is not None]
                
                mean_f1 = np.mean(macro_f1_scores) if macro_f1_scores else 0.0
                mean_balanced_acc = np.mean(balanced_acc_scores) if balanced_acc_scores else 0.0
                mean_auc = np.mean(macro_auc_scores) if macro_auc_scores else None
                
                print()
                print("-" * 80)
                print(f"MEAN ({len(ml_scores)} models):")
                print(f"   Macro F1:           {mean_f1:.4f}")
                print(f"   Balanced Accuracy:  {mean_balanced_acc:.4f}")
                if mean_auc is not None:
                    print(f"   Macro AUC (OVR):    {mean_auc:.4f}")
                else:
                    print(f"   Macro AUC (OVR):    N/A")
                print("-" * 80)
            
            # Summary for multi-class
            macro_f1_scores = [s['macro_f1'] for s in ml_scores.values() if s['macro_f1'] is not None]
            balanced_acc_scores = [s['balanced_accuracy'] for s in ml_scores.values() if s['balanced_accuracy'] is not None]
            macro_auc_scores = [s['macro_auc'] for s in ml_scores.values() if s['macro_auc'] is not None]
            
            results['summary'] = {
                'mean_macro_f1': float(np.mean(macro_f1_scores)) if macro_f1_scores else 0.0,
                'mean_balanced_accuracy': float(np.mean(balanced_acc_scores)) if balanced_acc_scores else 0.0,
                'mean_macro_auc': float(np.mean(macro_auc_scores)) if macro_auc_scores else None,
                'n_classes': int(n_classes),
            }
    else:
        # UNCONDITIONAL MODE: Unsupervised evaluation
        unsupervised_metrics = evaluate_unsupervised_utility(
            real, synthetic, n_clusters=5, verbose=verbose
        )
        results['unsupervised_utility'] = unsupervised_metrics
        
        # Summary for unconditional mode
        results['summary'] = {
            'unsupervised_utility': unsupervised_metrics['unsupervised_utility'],
            'cluster_ari': unsupervised_metrics['cluster_ari'],
            'pca_similarity': unsupervised_metrics['pca_similarity'],
            'mmd_distance': unsupervised_metrics['mmd_distance'],
        }
    
    if verbose:
        print()
        print("="*80)
        print("✅ EVALUATION COMPLETE")
        print("="*80)
    
    return results


def save_evaluation_results(results: Dict, filepath: str):
    """
    Save evaluation results to JSON file.
    
    Parameters
    ----------
    results : Dict
        Results from evaluate_synthetic().
    filepath : str
        Output file path.
    """
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to {filepath}")


def load_evaluation_results(filepath: str) -> Dict:
    """
    Load evaluation results from JSON file.
    
    Parameters
    ----------
    filepath : str
        Input file path.
    
    Returns
    -------
    Dict
        Evaluation results.
    """
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    return results
