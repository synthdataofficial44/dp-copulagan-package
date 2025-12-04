#!/usr/bin/env python3
"""
Quickstart Example for DP-CopulaGAN

This is the simplest possible example showing the basic workflow,
including automatic handling of categorical labels.
"""

from dp_copulagan import DPCopulaGAN
from dp_copulagan.utils import GANConfig
import pandas as pd
import numpy as np

print("="*80)
print("DP-COPULAGAN QUICKSTART EXAMPLE")
print("="*80)
print()

# Create sample data with CATEGORICAL LABELS (demonstrates automatic encoding)
np.random.seed(42)
print("📊 Creating sample data with categorical labels...")
data = pd.DataFrame({
    'feature1': np.random.randn(1000),
    'feature2': np.random.randn(1000),
    'feature3': np.random.randn(1000),
    'target': np.random.choice(['Low', 'High'], 1000)  # String labels!
})

print(f"   ✓ Created {len(data)} samples")
print(f"   ✓ Features: feature1, feature2, feature3")
print(f"   ✓ Target: {data['target'].unique()} (categorical)")
print()

print("Original data sample:")
print(data.head())
print()

# Initialize and train
print("="*80)
print("Training DP-CopulaGAN...")
print("="*80)
print()

# Use quick config for demo
config = GANConfig(epochs=10, batch_size=32)

model = DPCopulaGAN(
    epsilon=10.0,  # High epsilon for quick demo (low privacy, better quality)
    delta=1e-5,
    label_col='target',
    gan_config=config,
    random_state=42
)

# NOTE: You will see a warning about categorical labels being auto-encoded
# This is EXPECTED and CORRECT behavior
model.fit(data)

# Generate synthetic data
print()
print("="*80)
print("Generating synthetic data...")
print("="*80)
print()

# NOTE: You may see a warning about NaN values with higher DP noise
# This is EXPECTED and automatically handled
synthetic = model.sample(n_samples=100)

print(f"✓ Generated {len(synthetic)} synthetic samples")
print()

print("Synthetic data sample:")
print(synthetic.head())
print()

# Evaluate
print("="*80)
print("Evaluating quality...")
print("="*80)
print()

from dp_copulagan.evaluation import evaluate_synthetic
results = evaluate_synthetic(data, synthetic, label_col='target')

print()
print("="*80)
print("✅ QUICKSTART COMPLETE!")
print("="*80)
print()
print("Key observations:")
print("  • Categorical labels ('Low'/'High') were automatically encoded")
print("  • Synthetic data was generated successfully")
print("  • Quality metrics computed (AuROC depends on epsilon)")
print()
print("For more examples, see:")
print("  • examples/scripts/run_adult.py")
print("  • LIMITATIONS.md (dataset requirements)")
print("  • README.md (full documentation)")
print()
