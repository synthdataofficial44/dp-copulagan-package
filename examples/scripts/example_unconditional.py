#!/usr/bin/env python3
"""
Unconditional Mode Example for DP-CopulaGAN

This example demonstrates unconditional generation for datasets
without a target column (e.g., sensor data, logs, time series).
"""

from dp_copulagan import DPCopulaGAN, GANConfig
import pandas as pd
import numpy as np

print("="*80)
print("DP-COPULAGAN: UNCONDITIONAL MODE EXAMPLE")
print("="*80)
print()

# ============================================================================
# SCENARIO: Sensor Data Without Labels
# ============================================================================

print("📊 Creating unlabeled sensor data...")
np.random.seed(42)

# Simulate sensor readings (temperature, humidity, pressure)
n_samples = 1000
data = pd.DataFrame({
    'temperature': np.random.normal(25, 5, n_samples),  # °C
    'humidity': np.random.normal(60, 15, n_samples),    # %
    'pressure': np.random.normal(1013, 10, n_samples),  # hPa
    'wind_speed': np.random.gamma(2, 3, n_samples)      # m/s
})

# Add some correlations
data['temperature'] = data['temperature'] + 0.3 * data['humidity'] / 10
data['pressure'] = data['pressure'] - 0.2 * data['temperature']

print(f"   ✓ Created {len(data)} sensor readings")
print(f"   ✓ Features: {list(data.columns)}")
print(f"   ✓ NO label column (unconditional mode)")
print()

print("Sample data:")
print(data.head(10))
print()
print("Data statistics:")
print(data.describe())
print()

# ============================================================================
# TRAIN IN UNCONDITIONAL MODE
# ============================================================================

print("="*80)
print("Training DP-CopulaGAN in UNCONDITIONAL mode...")
print("="*80)
print()

# Quick config for demo
config = GANConfig(
    epochs=100,      # Use 1000 for production
    batch_size=64,
    latent_dim=192
)

# Initialize WITHOUT label column → activates unconditional mode
model = DPCopulaGAN(
    epsilon=1.0,
    delta=1e-5,
    label_col=None,  # ← This activates unconditional mode
    gan_config=config,
    random_state=42
)

# You should see: "ℹ️ No label column provided → Switching to UNCONDITIONAL mode"

# Train model
model.fit(data)

# ============================================================================
# GENERATE SYNTHETIC DATA
# ============================================================================

print()
print("="*80)
print("Generating synthetic sensor data...")
print("="*80)
print()

# Generate synthetic samples
synthetic = model.sample(n_samples=500)

print(f"✓ Generated {len(synthetic)} synthetic samples")
print()

print("Synthetic data sample:")
print(synthetic.head(10))
print()

print("Synthetic statistics:")
print(synthetic.describe())
print()

# ============================================================================
# COMPARE DISTRIBUTIONS
# ============================================================================

print("="*80)
print("Comparing real vs synthetic distributions...")
print("="*80)
print()

for col in data.columns:
    real_mean = data[col].mean()
    synth_mean = synthetic[col].mean()
    real_std = data[col].std()
    synth_std = synthetic[col].std()
    
    print(f"{col:15s} │ Real: μ={real_mean:7.2f}, σ={real_std:6.2f} │ Synth: μ={synth_mean:7.2f}, σ={synth_std:6.2f}")

print()

# ============================================================================
# EVALUATE QUALITY (Optional)
# ============================================================================

print("="*80)
print("Evaluating synthetic data quality...")
print("="*80)
print()

from dp_copulagan.evaluation import evaluate_synthetic

# Note: ML utility evaluation requires a label column
# For unconditional mode, we can only compute statistical metrics
from dp_copulagan.metrics import compute_statistical_metrics

stats = compute_statistical_metrics(data, synthetic)

print(f"📊 Statistical Similarity:")
print(f"   Jensen-Shannon Divergence: {stats['jsd']:.4f}")
print(f"   Wasserstein Distance:      {stats['wasserstein']:.4f}")
print(f"   Correlation RMSE:          {stats['correlation_rmse']:.4f}")
print()

# ============================================================================
# SAVE RESULTS
# ============================================================================

synthetic.to_csv('synthetic_sensor_data.csv', index=False)
print(f"✓ Synthetic data saved to: synthetic_sensor_data.csv")
print()

print("="*80)
print("✅ UNCONDITIONAL MODE DEMO COMPLETE!")
print("="*80)
print()

print("Key observations:")
print("  • Model trained WITHOUT a label column")
print("  • Synthetic data has SAME columns as input (no label added)")
print("  • Distributions and correlations preserved")
print("  • Differential privacy guarantees maintained (ε=1.0)")
print()

print("Use cases for unconditional mode:")
print("  ✓ Sensor/IoT data without classes")
print("  ✓ Time series feature tables")
print("  ✓ Financial transactions (no target)")
print("  ✓ Network traffic logs")
print("  ✓ Any dataset without a natural label column")
print()
