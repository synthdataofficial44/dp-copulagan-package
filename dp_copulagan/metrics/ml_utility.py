"""
Machine learning utility metrics for synthetic data evaluation.

This module implements TSTR (Train on Synthetic, Test on Real) evaluation
using 12 classifiers as in PATE-GAN paper.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              BaggingClassifier, GradientBoostingClassifier)
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
from sklearn.preprocessing import label_binarize
from typing import Dict
import warnings

warnings.filterwarnings('ignore')


def get_classifiers():
    """
    Get the 12 classifiers used in PATE-GAN evaluation.
    
    Returns
    -------
    Dict[str, classifier]
        Dictionary of classifier names and instances.
    """
    return {
        'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1),
        'Random Forests': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gaussian Naive Bayes': GaussianNB(),
        'Bernoulli Naive Bayes': BernoulliNB(),
        'Linear SVM': LinearSVC(max_iter=2000, random_state=42, dual=False),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'LDA': LinearDiscriminantAnalysis(),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
        'Bagging': BaggingClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'GBM': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'MLP': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, 
                                verbosity=0, use_label_encoder=False, eval_metric='logloss')
    }


def compute_ml_utility(real: pd.DataFrame,
                      synthetic: pd.DataFrame,
                      label_col: str,
                      test_size: float = 0.2) -> Dict[str, float]:
    """
    Compute ML utility via TSTR (Train on Synthetic, Test on Real).
    
    Parameters
    ----------
    real : pd.DataFrame
        Real data.
    synthetic : pd.DataFrame
        Synthetic data.
    label_col : str
        Name of label column.
    test_size : float, default=0.2
        Fraction of real data to use for testing.
    
    Returns
    -------
    Dict[str, float]
        Dictionary of classifier names and AuROC scores.
    
    Examples
    --------
    >>> scores = compute_ml_utility(real_data, synthetic_data, label_col='income')
    >>> print(f"Mean AuROC: {np.mean(list(scores.values())):.4f}")
    """
    # Split real data
    X_real = real.drop(label_col, axis=1)
    y_real = real[label_col]
    
    _, X_test, _, y_test = train_test_split(
        X_real, y_real, test_size=test_size, random_state=42, stratify=y_real
    )
    
    # Synthetic training data
    X_synth = synthetic.drop(label_col, axis=1)
    y_synth = synthetic[label_col]
    
    # Ensure same columns
    common_cols = list(set(X_test.columns) & set(X_synth.columns))
    X_synth = X_synth[common_cols]
    X_test = X_test[common_cols]
    
    # Evaluate classifiers
    results = {}
    classifiers = get_classifiers()
    
    for name, clf in classifiers.items():
        try:
            # Train on synthetic
            clf.fit(X_synth, y_synth)
            
            # Predict on real test set
            if hasattr(clf, 'predict_proba'):
                y_pred_proba = clf.predict_proba(X_test)[:, 1]
            elif hasattr(clf, 'decision_function'):
                y_pred_proba = clf.decision_function(X_test)
            else:
                continue
            
            # Compute AuROC
            auroc = roc_auc_score(y_test, y_pred_proba)
            results[name] = auroc
            
        except Exception as e:
            print(f"Warning: {name} failed - {e}")
            results[name] = 0.0
    
    return results


def compute_multiclass_ml_utility(real: pd.DataFrame,
                                  synthetic: pd.DataFrame,
                                  label_col: str,
                                  test_size: float = 0.2) -> Dict[str, float]:
    """
    Compute multi-class ML utility via TSTR (Train on Synthetic, Test on Real).
    
    Uses metrics appropriate for multi-class classification:
    - Macro F1 Score
    - Balanced Accuracy
    - Macro AUC (One-vs-Rest)
    
    Parameters
    ----------
    real : pd.DataFrame
        Real data.
    synthetic : pd.DataFrame
        Synthetic data.
    label_col : str
        Name of label column.
    test_size : float, default=0.2
        Fraction of real data to use for testing.
    
    Returns
    -------
    Dict[str, Dict[str, float]]
        Nested dictionary with classifier names and their scores:
        {
            'Logistic Regression': {
                'macro_f1': 0.85,
                'balanced_accuracy': 0.87,
                'macro_auc': 0.90
            },
            ...
        }
    
    Examples
    --------
    >>> scores = compute_multiclass_ml_utility(real_data, synthetic_data, label_col='species')
    >>> print(f"Mean F1: {np.mean([s['macro_f1'] for s in scores.values()]):.4f}")
    """
    # Split real data
    X_real = real.drop(label_col, axis=1)
    y_real = real[label_col]
    
    _, X_test, _, y_test = train_test_split(
        X_real, y_real, test_size=test_size, random_state=42, stratify=y_real
    )
    
    # Synthetic training data
    X_synth = synthetic.drop(label_col, axis=1)
    y_synth = synthetic[label_col]
    
    # Ensure same columns
    common_cols = list(set(X_test.columns) & set(X_synth.columns))
    X_synth = X_synth[common_cols]
    X_test = X_test[common_cols]
    
    # Get unique classes for OVR encoding
    classes = np.unique(y_real)
    n_classes = len(classes)
    
    # Evaluate classifiers
    results = {}
    classifiers = get_classifiers()
    
    for name, clf in classifiers.items():
        try:
            # Train on synthetic
            clf.fit(X_synth, y_synth)
            
            # Predict on real test set
            y_pred = clf.predict(X_test)
            
            # Compute Macro F1
            macro_f1 = f1_score(y_test, y_pred, average='macro')
            
            # Compute Balanced Accuracy
            balanced_acc = balanced_accuracy_score(y_test, y_pred)
            
            # Compute Macro AUC (One-vs-Rest)
            macro_auc = None
            try:
                if hasattr(clf, 'predict_proba'):
                    y_pred_proba = clf.predict_proba(X_test)
                    # Binarize labels for OVR
                    y_test_bin = label_binarize(y_test, classes=classes)
                    if n_classes == 2:
                        y_test_bin = np.hstack([1 - y_test_bin, y_test_bin])
                    # Compute macro AUC
                    macro_auc = roc_auc_score(y_test_bin, y_pred_proba, 
                                             multi_class='ovr', average='macro')
                elif hasattr(clf, 'decision_function'):
                    y_scores = clf.decision_function(X_test)
                    # Ensure 2D array
                    if y_scores.ndim == 1:
                        # Binary case - already handled above, skip
                        pass
                    else:
                        # Binarize labels
                        y_test_bin = label_binarize(y_test, classes=classes)
                        if n_classes == 2:
                            y_test_bin = np.hstack([1 - y_test_bin, y_test_bin])
                        # Compute macro AUC
                        macro_auc = roc_auc_score(y_test_bin, y_scores,
                                                 multi_class='ovr', average='macro')
            except Exception as e:
                # If AUC computation fails, just skip it
                pass
            
            results[name] = {
                'macro_f1': float(macro_f1),
                'balanced_accuracy': float(balanced_acc),
                'macro_auc': float(macro_auc) if macro_auc is not None else None
            }
            
        except Exception as e:
            # If classifier fails completely, skip it
            print(f"   ⚠ {name} failed - {str(e)[:50]}")
            continue
    
    return results

