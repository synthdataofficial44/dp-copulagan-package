#!/usr/bin/env python3
"""
Multi-Class Evaluation Example for DP-CopulaGAN

Demonstrates automatic multi-class ML utility evaluation for n_classes > 2.
"""

from dp_copulagan import DPCopulaGAN, GANConfig
from dp_copulagan.evaluation import evaluate_synthetic
import pandas as pd
import numpy as np

print("="*80)
print("DP-COPULAGAN: MULTI-CLASS EVALUATION EXAMPLE")
print("="*80)
print()

# ============================================================================
# Create Multi-Class Dataset (Iris-style)
# ============================================================================

print("📊 Creating multi-class dataset (3 classes)...")
np.random.seed(42)

n_per_class = 200

# Class 0: Setosa-like
X0 = np.random.randn(n_per_class, 4) * 0.3 + np.array([5.0, 3.4, 1.5, 0.2])

# Class 1: Versicolor-like
X1 = np.random.randn(n_per_class, 4) * 0.5 + np.array([5.9, 2.8, 4.2, 1.3])

# Class 2: Virginica-like
X2 = np.random.randn(n_per_class, 4) * 0.6 + np.array([6.5, 3.0, 5.5, 2.0])

# Combine
X = np.vstack([X0, X1, X2])
y = np.array([0]*n_per_class + [1]*n_per_class + [2]*n_per_class)

# Create DataFrame with string labels
label_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
data = pd.DataFrame(X, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
data['species'] = [label_map[label] for label in y]

print(f"   ✓ Created {len(data)} samples")
print(f"   ✓ Features: {list(data.columns[:-1])}")
print(f"   ✓ Classes: {data['species'].unique()}")
print(f"   ✓ Class distribution: {data['species'].value_counts().to_dict()}")
print()

print("Data sample:")
print(data.head(10))
print()

# ============================================================================
# Train in Conditional Mode (Multi-Class)
# ============================================================================

print("="*80)
print("Training DP-CopulaGAN in CONDITIONAL mode (multi-class)...")
print("="*80)
print()

config = GANConfig(epochs=500, batch_size=64)

model = DPCopulaGAN(
    epsilon=1.0,
    delta=1e-5,
    label_col='species',  # ← Multi-class conditional mode
    gan_config=config,
    random_state=42
)

model.fit(data)

# ============================================================================
# Generate Synthetic Data
# ============================================================================

print()
print("="*80)
print("Generating synthetic data...")
print("="*80)
print()

synthetic = model.sample(n_samples=600)

print(f"✓ Generated {len(synthetic)} synthetic samples")
print()
print("Synthetic data sample:")
print(synthetic.head(10))
print()
print("Synthetic class distribution:")
print(synthetic['species'].value_counts())
print()

# ============================================================================
# Evaluate Using Multi-Class ML Utility
# ============================================================================

print("="*80)
print("EVALUATING WITH MULTI-CLASS ML UTILITY")
print("="*80)
print()
print("ℹ️  Detected n_classes = 3 → Using multi-class metrics")
print()

# Automatic multi-class evaluation
results = evaluate_synthetic(
    real=data,
    synthetic=synthetic,
    label_col='species',  # ← Triggers multi-class evaluation
    verbose=True
)

print()
print("="*80)
print("Detailed Results:")
print("="*80)
print()

print("📊 Statistical Metrics:")
print(f"   Jensen-Shannon Divergence: {results['statistical']['jsd']:.4f}")
print(f"   Wasserstein Distance:      {results['statistical']['wasserstein']:.4f}")
print(f"   Correlation RMSE:          {results['statistical']['correlation_rmse']:.4f}")
print()

print("🎯 Multi-Class ML Utility:")
print(f"   Mean Macro F1:             {results['summary']['mean_macro_f1']:.4f}")
print(f"   Mean Balanced Accuracy:    {results['summary']['mean_balanced_accuracy']:.4f}")
if results['summary']['mean_macro_auc'] is not None:
    print(f"   Mean Macro AUC (OVR):      {results['summary']['mean_macro_auc']:.4f}")
else:
    print(f"   Mean Macro AUC (OVR):      N/A")
print(f"   Number of Classes:         {results['summary']['n_classes']}")
print()

# ============================================================================
# Individual Classifier Results
# ============================================================================

print("="*80)
print("Individual Classifier Results:")
print("="*80)
print()

ml_scores = results['ml_utility_multiclass']

print(f"{'Classifier':<30} {'Macro F1':<12} {'Bal. Acc.':<12} {'Macro AUC':<12}")
print("-" * 68)

for clf_name, scores in ml_scores.items():
    f1 = scores['macro_f1']
    bal_acc = scores['balanced_accuracy']
    auc = scores['macro_auc']
    
    auc_str = f"{auc:.4f}" if auc is not None else "N/A"
    print(f"{clf_name:<30} {f1:<12.4f} {bal_acc:<12.4f} {auc_str:<12}")

print()

# ============================================================================
# Interpretation Guide
# ============================================================================

print("="*80)
print("📖 INTERPRETATION GUIDE")
print("="*80)
print()

mean_f1 = results['summary']['mean_macro_f1']
mean_bal_acc = results['summary']['mean_balanced_accuracy']

print("Multi-Class Utility Scores:")
print()

print(f"Macro F1 Score: {mean_f1:.4f}")
if mean_f1 >= 0.8:
    print("  → EXCELLENT: Synthetic data very useful for multi-class classification")
elif mean_f1 >= 0.6:
    print("  → GOOD: Synthetic data captures most class-specific patterns")
elif mean_f1 >= 0.4:
    print("  → FAIR: Synthetic data captures some class information")
else:
    print("  → POOR: Consider increasing epsilon or training longer")
print()

print(f"Balanced Accuracy: {mean_bal_acc:.4f}")
if mean_bal_acc >= 0.8:
    print("  → EXCELLENT: Well-balanced across all classes")
elif mean_bal_acc >= 0.6:
    print("  → GOOD: Reasonably balanced performance")
elif mean_bal_acc >= 0.4:
    print("  → FAIR: Some class imbalance in predictions")
else:
    print("  → POOR: Significant class imbalance issues")
print()

# ============================================================================
# Comparison with Binary Case
# ============================================================================

print("="*80)
print("📝 Comparison: Binary vs Multi-Class Evaluation")
print("="*80)
print()

print("Binary Classification (n_classes = 2):")
print("  • Metrics: Binary AUROC for each classifier")
print("  • Output: Mean/Min/Max AUROC")
print("  • Use case: Income prediction, fraud detection, churn")
print()

print("Multi-Class Classification (n_classes > 2):")
print("  • Metrics: Macro F1, Balanced Accuracy, Macro AUC (OVR)")
print("  • Output: Aggregated multi-class scores")
print("  • Use case: Species classification, disease diagnosis, sentiment")
print()

print("✅ Automatic Detection:")
print("  → Package automatically selects appropriate metrics")
print("  → No manual configuration needed")
print("  → Optimal evaluation for your data")
print()

# ============================================================================
# Save Results
# ============================================================================

synthetic.to_csv('synthetic_multiclass_data.csv', index=False)
print(f"✓ Synthetic data saved to: synthetic_multiclass_data.csv")
print()

print("="*80)
print("✅ MULTI-CLASS EVALUATION DEMO COMPLETE!")
print("="*80)
print()

print("Key takeaways:")
print("  • Multi-class evaluation automatic for n_classes > 2")
print("  • Three complementary metrics: F1, Balanced Acc, Macro AUC")
print("  • Binary mode unchanged (backward compatible)")
print("  • No architecture modifications needed")
print()
