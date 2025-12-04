"""Evaluation module for DP-CopulaGAN."""

from dp_copulagan.evaluation.evaluator import (
    evaluate_synthetic,
    save_evaluation_results,
    load_evaluation_results
)

__all__ = [
    'evaluate_synthetic',
    'save_evaluation_results',
    'load_evaluation_results',
]
