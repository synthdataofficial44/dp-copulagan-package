#!/usr/bin/env python3
"""
Example: DP-CopulaGAN on Adult Income Dataset

This script demonstrates the complete workflow:
1. Load Adult dataset
2. Train DP-CopulaGAN with different privacy budgets
3. Generate synthetic data
4. Evaluate quality
"""

from dp_copulagan import DPCopulaGAN
from dp_copulagan.evaluation import evaluate_synthetic
import pandas as pd
import numpy as np

# Set random seed
np.random.seed(42)

print("="*80)
print("DP-CopulaGAN: Adult Income Dataset Example")
print("="*80)

# Load Adult dataset
print("
📁 Loading Adult dataset...")
# Download from: https://archive.ics.uci.edu/ml/datasets/adult
# For this example, we'll create sample data

# Sample data creation (replace with actual Adult dataset)
n_samples = 5000
data = pd.DataFrame({
    'age': np.random.randint(18, 80, n_samples),
    'education_num': np.random.randint(1, 16, n_samples),
    'hours_per_week': np.random.randint(20, 80, n_samples),
    'income': np.random.choice([0, 1], n_samples, p=[0.75, 0.25])
})

print(f"✓ Dataset shape: {data.shape}")
print(f"✓ Class distribution:\n{data['income'].value_counts()}")

# Train models with different privacy budgets
privacy_budgets = [0.1, 1.0, 10.0]

for epsilon in privacy_budgets:
    print(f"
{'='*80}")
    print(f"Training with ε={epsilon}")
    print("="*80)
    
    # Initialize model
    model = DPCopulaGAN(
        epsilon=epsilon,
        delta=1e-5,
        label_col='income'
    )
    
    # Train
    model.fit(data)
    
    # Generate synthetic data
    print(f"
🎲 Generating {len(data)} synthetic samples...")
    synthetic = model.sample(n_samples=len(data))
    
    # Evaluate
    results = evaluate_synthetic(
        real=data,
        synthetic=synthetic,
        label_col='income'
    )
    
    # Save synthetic data
    synthetic.to_csv(f'adult_synthetic_eps{epsilon}.csv', index=False)
    print(f"\n✓ Saved synthetic data to adult_synthetic_eps{epsilon}.csv")

print("
" + "="*80)
print("✅ EXAMPLE COMPLETE")
print("="*80)
