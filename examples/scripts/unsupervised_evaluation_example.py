#!/usr/bin/env python3
"""
Unsupervised Evaluation Example for DP-CopulaGAN

Demonstrates the new unsupervised ML utility metrics for unconditional mode.
"""

from dp_copulagan import DPCopulaGAN, GANConfig
from dp_copulagan.evaluation import evaluate_synthetic
from dp_copulagan.utils import evaluate_unsupervised_utility
import pandas as pd
import numpy as np

print("="*80)
print("DP-COPULAGAN: UNSUPERVISED EVALUATION EXAMPLE")
print("="*80)
print()

# ============================================================================
# Create Unlabeled Dataset (Sensor Data)
# ============================================================================

print("📊 Creating unlabeled sensor dataset...")
np.random.seed(42)

n_samples = 2000

# Create sensor data with correlations
temperature = np.random.normal(25, 5, n_samples)
humidity = 60 + 0.3 * temperature + np.random.normal(0, 10, n_samples)
pressure = 1013 - 0.2 * temperature + np.random.normal(0, 5, n_samples)
wind_speed = np.random.gamma(2, 3, n_samples)

data = pd.DataFrame({
    'temperature': temperature,
    'humidity': humidity,
    'pressure': pressure,
    'wind_speed': wind_speed
})

print(f"   ✓ Created {len(data)} sensor readings")
print(f"   ✓ Features: {list(data.columns)}")
print(f"   ✓ NO label column (perfect for unconditional mode)")
print()

print("Data sample:")
print(data.head(10))
print()
print("Data statistics:")
print(data.describe())
print()

# ============================================================================
# Train in Unconditional Mode
# ============================================================================

print("="*80)
print("Training DP-CopulaGAN in UNCONDITIONAL mode...")
print("="*80)
print()

config = GANConfig(epochs=500, batch_size=64)

model = DPCopulaGAN(
    epsilon=1.0,
    delta=1e-5,
    label_col=None,  # ← Unconditional mode
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

synthetic = model.sample(n_samples=1000)

print(f"✓ Generated {len(synthetic)} synthetic samples")
print()
print("Synthetic data sample:")
print(synthetic.head(10))
print()

# ============================================================================
# Evaluate Using Unsupervised ML Utility
# ============================================================================

print("="*80)
print("EVALUATING WITH UNSUPERVISED ML UTILITY")
print("="*80)
print()

# Method 1: Using the main evaluation function
results = evaluate_synthetic(
    real=data,
    synthetic=synthetic,
    label_col=None,  # ← Triggers unsupervised evaluation
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

print("🎯 Unsupervised ML Utility:")
print(f"   Cluster ARI:               {results['unsupervised_utility']['cluster_ari']:.4f}")
print(f"   PCA Similarity:            {results['unsupervised_utility']['pca_similarity']:.4f}")
print(f"   MMD Distance:              {results['unsupervised_utility']['mmd_distance']:.4f}")
print(f"   Combined Utility Score:    {results['summary']['unsupervised_utility']:.4f}")
print()

# ============================================================================
# Interpretation Guide
# ============================================================================

print("="*80)
print("📖 INTERPRETATION GUIDE")
print("="*80)
print()

utility_score = results['summary']['unsupervised_utility']

print("Unsupervised Utility Score:", f"{utility_score:.4f}")
print()

if utility_score >= 0.8:
    quality = "EXCELLENT"
    desc = "Synthetic data preserves structure very well"
elif utility_score >= 0.6:
    quality = "GOOD"
    desc = "Synthetic data preserves most structural properties"
elif utility_score >= 0.4:
    quality = "FAIR"
    desc = "Synthetic data captures some structural properties"
else:
    quality = "POOR"
    desc = "Consider increasing epsilon or training longer"

print(f"Quality: {quality}")
print(f"→ {desc}")
print()

# Individual metric interpretation
ari = results['unsupervised_utility']['cluster_ari']
pca = results['unsupervised_utility']['pca_similarity']
mmd = results['unsupervised_utility']['mmd_distance']

print("Component Scores:")
print(f"  • Cluster ARI ({ari:.4f}): {'✓ Good' if ari > 0.5 else '⚠ Could be better'}")
print(f"  • PCA Similarity ({pca:.4f}): {'✓ Good' if pca > 0.7 else '⚠ Could be better'}")
print(f"  • MMD Distance ({mmd:.4f}): {'✓ Good' if mmd < 0.3 else '⚠ Could be better'}")
print()

# ============================================================================
# Method 2: Direct Computation
# ============================================================================

print("="*80)
print("Alternative: Direct Metric Computation")
print("="*80)
print()

from dp_copulagan.utils import (
    compute_cluster_ari,
    compute_pca_similarity,
    compute_mmd_rbf
)

ari_direct = compute_cluster_ari(data, synthetic, n_clusters=5)
pca_direct = compute_pca_similarity(data, synthetic)
mmd_direct = compute_mmd_rbf(data, synthetic)

print(f"Cluster ARI (direct):     {ari_direct:.4f}")
print(f"PCA Similarity (direct):  {pca_direct:.4f}")
print(f"MMD Distance (direct):    {mmd_direct:.4f}")
print()

# ============================================================================
# Save Results
# ============================================================================

synthetic.to_csv('synthetic_sensor_data_evaluated.csv', index=False)
print(f"✓ Synthetic data saved to: synthetic_sensor_data_evaluated.csv")
print()

print("="*80)
print("✅ UNSUPERVISED EVALUATION DEMO COMPLETE!")
print("="*80)
print()

print("Key takeaways:")
print("  • Unconditional mode works without labels")
print("  • Unsupervised metrics evaluate structural preservation")
print("  • Three complementary metrics: clustering, PCA, MMD")
print("  • Combined score gives overall quality assessment")
print("  • Perfect for sensor data, logs, time series features")
print()
