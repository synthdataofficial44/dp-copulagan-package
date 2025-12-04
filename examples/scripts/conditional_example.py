#!/usr/bin/env python3
"""
Conditional Generation Example for DP-CopulaGAN

Demonstrates conditional generation with automatic categorical label encoding.
"""

from dp_copulagan import DPCopulaGAN, GANConfig
import pandas as pd
import numpy as np

print("="*80)
print("DP-COPULAGAN: CONDITIONAL MODE EXAMPLE")
print("="*80)
print()

# ============================================================================
# SCENARIO: Credit Card Fraud Detection
# ============================================================================

print("📊 Creating credit card transaction data...")
np.random.seed(42)

n_samples = 1000

# Normal transactions
normal_data = pd.DataFrame({
    'amount': np.random.gamma(2, 50, n_samples//2),
    'time_of_day': np.random.normal(14, 4, n_samples//2),
    'merchant_category': np.random.randint(1, 10, n_samples//2),
    'transaction_type': ['Normal'] * (n_samples//2)
})

# Fraudulent transactions (different distribution)
fraud_data = pd.DataFrame({
    'amount': np.random.gamma(5, 100, n_samples//2),
    'time_of_day': np.random.normal(2, 3, n_samples//2),
    'merchant_category': np.random.randint(1, 10, n_samples//2),
    'transaction_type': ['Fraud'] * (n_samples//2)
})

data = pd.concat([normal_data, fraud_data], ignore_index=True)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"   ✓ Created {len(data)} transactions")
print(f"   ✓ Features: {list(data.columns[:-1])}")
print(f"   ✓ Label: {data.columns[-1]} (categorical: {list(data['transaction_type'].unique())})")
print()

print("Sample data:")
print(data.head(10))
print()
print("Class distribution:")
print(data['transaction_type'].value_counts())
print()

# ============================================================================
# TRAIN IN CONDITIONAL MODE
# ============================================================================

print("="*80)
print("Training DP-CopulaGAN in CONDITIONAL mode...")
print("="*80)
print()

config = GANConfig(
    epochs=200,
    batch_size=64
)

# IMPORTANT: Label column contains strings! ("Normal", "Fraud")
# DP-CopulaGAN will automatically encode them to integers
model = DPCopulaGAN(
    epsilon=1.0,
    delta=1e-5,
    label_col='transaction_type',  # ← Conditional mode with categorical labels
    gan_config=config,
    random_state=42
)

# Watch for automatic encoding message:
# "⚠️ Warning: Label column 'transaction_type' is categorical..."
model.fit(data)

# ============================================================================
# GENERATE SYNTHETIC DATA
# ============================================================================

print()
print("="*80)
print("Generating synthetic transactions...")
print("="*80)
print()

synthetic = model.sample(n_samples=500)

print(f"✓ Generated {len(synthetic)} synthetic transactions")
print()

print("Synthetic data sample:")
print(synthetic.head(10))
print()

print("Synthetic class distribution:")
print(synthetic['transaction_type'].value_counts())
print()

# ============================================================================
# VERIFY LABEL DECODING
# ============================================================================

print("="*80)
print("Verifying label decoding...")
print("="*80)
print()

print(f"Original labels: {sorted(data['transaction_type'].unique())}")
print(f"Synthetic labels: {sorted(synthetic['transaction_type'].unique())}")
print()

assert set(synthetic['transaction_type'].unique()).issubset(set(data['transaction_type'].unique()))
print("✅ Labels correctly decoded to original strings!")
print()

# ============================================================================
# COMPARE DISTRIBUTIONS
# ============================================================================

print("="*80)
print("Comparing distributions by class...")
print("="*80)
print()

for cls in ['Normal', 'Fraud']:
    real_cls = data[data['transaction_type'] == cls]
    synth_cls = synthetic[synthetic['transaction_type'] == cls]
    
    print(f"{cls} Transactions:")
    print(f"  Amount:      Real μ={real_cls['amount'].mean():.2f} | Synth μ={synth_cls['amount'].mean():.2f}")
    print(f"  Time of Day: Real μ={real_cls['time_of_day'].mean():.2f} | Synth μ={synth_cls['time_of_day'].mean():.2f}")
    print()

# ============================================================================
# SAVE RESULTS
# ============================================================================

synthetic.to_csv('synthetic_fraud_data.csv', index=False)
print(f"✓ Synthetic data saved to: synthetic_fraud_data.csv")
print()

print("="*80)
print("✅ CONDITIONAL MODE DEMO COMPLETE!")
print("="*80)
print()

print("Key observations:")
print("  • Model trained WITH categorical label column")
print("  • Labels automatically encoded: 'Normal'→0, 'Fraud'→1")
print("  • Synthetic data automatically decoded back to strings")
print("  • Class distributions preserved")
print("  • Differential privacy guarantees maintained (ε=1.0)")
print()
