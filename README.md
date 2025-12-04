# DP-CopulaGAN: Differentially Private Copula-GAN

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.12+](https://img.shields.io/badge/tensorflow-2.12+-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready implementation of Differentially Private Copula-GAN for synthetic tabular data generation, combining differential privacy guarantees with high-quality data synthesis.

## ✨ New Features (v0.1.0)

- **🆕 Unconditional Mode**: Generate synthetic data without requiring a label column
- **🆕 Automatic Label Encoding**: Handles categorical labels (strings) automatically
- **🆕 Robust Type Detection**: Works with any dataset structure
- **✅ Backward Compatible**: All existing code continues to work

---

## 🚀 Quick Start

### Installation

```bash
pip install dp_copulagan_package.tar.gz
```

Or for development:

```bash
git clone <repository>
cd dp_copulagan_package
pip install -e .
```

### Conditional Generation (With Labels)

```python
from dp_copulagan import DPCopulaGAN
import pandas as pd

# Load data with labels
data = pd.read_csv('census_data.csv')

# Train model
model = DPCopulaGAN(
    epsilon=1.0,
    delta=1e-5,
    label_col='income'  # ← Conditional mode
)

model.fit(data)

# Generate synthetic data
synthetic = model.sample(n_samples=10000)
```

### Unconditional Generation (No Labels) 🆕

```python
from dp_copulagan import DPCopulaGAN
import pandas as pd

# Load data WITHOUT labels (e.g., sensor data)
data = pd.read_csv('sensor_readings.csv')

# Train in unconditional mode
model = DPCopulaGAN(
    epsilon=1.0,
    delta=1e-5,
    label_col=None  # ← Unconditional mode
)

model.fit(data)

# Generate synthetic data (no label column in output)
synthetic = model.sample(n_samples=10000)
```

---

## 📖 Features

### Core Capabilities

- ✅ **(ε, δ)-Differential Privacy**: Formal privacy guarantees via DP-SGD
- ✅ **Conditional & Unconditional Generation**: Works with or without labels
- ✅ **Automatic Label Encoding**: Handles categorical labels seamlessly
- ✅ **Gaussian Copula Transform**: Preserves correlations and marginal distributions
- ✅ **Spectral Normalization**: Stable GAN training
- ✅ **Comprehensive Evaluation**: Statistical metrics + ML utility

### What's New in v0.2.0

#### 1. Unconditional Mode

Generate synthetic data from unlabeled datasets:

```python
# Works with sensor data, logs, time series, etc.
model = DPCopulaGAN(epsilon=1.0, label_col=None)
model.fit(unlabeled_data)
synthetic = model.sample(5000)
```

**Use cases:**
- Sensor/IoT data
- Time series features
- Transaction logs
- Scientific measurements
- Any dataset without a target column

#### 2. Automatic Label Encoding 🆕

Handles categorical labels automatically:

```python
# Labels can be strings!
data['category'] = ['Low', 'High', 'Medium', ...]  # Or: ['Düşük', 'Yüksek']

model = DPCopulaGAN(epsilon=1.0, label_col='category')
model.fit(data)  # Automatically encodes to 0, 1, 2

synthetic = model.sample(1000)
print(synthetic['category'].unique())  # ['Low', 'High', 'Medium'] ← Decoded!
```

No manual preprocessing required!

#### 3. Robust Input Handling

- ✅ Handles mixed data types automatically
- ✅ Detects and preserves numeric columns
- ✅ Auto-fills NaN values from DP noise
- ✅ Clear warnings for edge cases

---

## 🎯 When to Use What Mode

| Scenario | Mode | Example |
|----------|------|---------|
| Classification dataset | **Conditional** | `label_col='income'` |
| Unlabeled sensor data | **Unconditional** | `label_col=None` |
| Time series features | **Unconditional** | `label_col=None` |
| Fraud detection data | **Conditional** | `label_col='is_fraud'` |
| Network traffic logs | **Unconditional** | `label_col=None` |

---

## 📊 Architecture

```
Input Data
    ↓
Gaussian Copula Transform (per-class or global)
    ↓
DP-SGD GAN Training
  ├─ Generator: Noise → Latent Samples
  ├─ Critic: WGAN-GP with Spectral Norm
  └─ Privacy: Gradient Clipping + Gaussian Noise
    ↓
Inverse Copula Transform
    ↓
Synthetic Data (with DP guarantees)
```

---

## 🔒 Privacy Guarantees

DP-CopulaGAN provides formal **(ε, δ)-differential privacy** via DP-SGD:

```python
model = DPCopulaGAN(
    epsilon=1.0,   # Privacy budget (lower = more private)
    delta=1e-5     # Privacy parameter (typically 1/n²)
)
```

**Privacy Levels:**
- **ε = 10**: Weak privacy, high utility
- **ε = 1**: Moderate privacy (recommended)
- **ε = 0.1**: Strong privacy, lower utility

See [LIMITATIONS.md](LIMITATIONS.md) for privacy-utility tradeoffs.

---

## 📦 Examples

### Basic Usage

```python
from dp_copulagan import DPCopulaGAN

# Conditional mode
model = DPCopulaGAN(epsilon=1.0, label_col='target')
model.fit(train_data)
synthetic = model.sample(10000)

# Unconditional mode
model = DPCopulaGAN(epsilon=1.0, label_col=None)
model.fit(sensor_data)
synthetic = model.sample(10000)
```

### Custom Configuration

```python
from dp_copulagan import DPCopulaGAN, GANConfig, DPConfig

# Custom GAN config
gan_config = GANConfig(
    epochs=1000,
    batch_size=64,
    latent_dim=192
)

# Custom DP config
dp_config = DPConfig(
    epsilon=1.0,
    delta=1e-5,
    clip_norm=1.0
)

model = DPCopulaGAN(
    label_col='income',
    gan_config=gan_config,
    dp_config=dp_config
)

model.fit(data)
synthetic = model.sample(10000)
```

### Evaluation

#### Conditional Mode (With Labels)

##### Binary Classification (n_classes = 2)

```python
from dp_copulagan.evaluation import evaluate_synthetic

metrics = evaluate_synthetic(
    real_data=train_data,
    synthetic_data=synthetic,
    label_col='income'  # Binary: 'high' or 'low'
)

print(f"JSD: {metrics['statistical']['jsd']:.4f}")
print(f"Wasserstein: {metrics['statistical']['wasserstein']:.4f}")
print(f"Mean AuROC: {metrics['summary']['mean_auroc']:.4f}")
```

##### Multi-Class Classification (n_classes > 2) 🆕

When the label column contains more than two classes, the package automatically uses **multi-class supervised evaluation** instead of binary AUROC:

```python
from dp_copulagan.evaluation import evaluate_synthetic

metrics = evaluate_synthetic(
    real_data=iris_data,
    synthetic_data=synthetic,
    label_col='species'  # Multi-class: 'setosa', 'versicolor', 'virginica'
)

# Access multi-class metrics
print(f"Macro F1: {metrics['summary']['mean_macro_f1']:.4f}")
print(f"Balanced Accuracy: {metrics['summary']['mean_balanced_accuracy']:.4f}")
print(f"Macro AUC: {metrics['summary']['mean_macro_auc']:.4f}")
```

**Multi-Class Metrics:**
- **Macro F1 Score**: F1 score averaged across all classes
- **Balanced Accuracy**: Accuracy normalized by class support
- **Macro AUC (One-vs-Rest)**: AUC computed in OVR fashion and macro-averaged

This ensures that synthetic data utility is evaluated correctly for multi-class classification problems without altering the model architecture.

#### Unconditional Mode (No Labels) 🆕

When no label column is provided, the package automatically computes **unsupervised ML utility** using structure-preserving metrics:

```python
from dp_copulagan.evaluation import evaluate_synthetic

metrics = evaluate_synthetic(
    real_data=sensor_data,
    synthetic_data=synthetic,
    label_col=None  # Unsupervised evaluation
)

# Access unsupervised metrics
print(f"Cluster ARI: {metrics['unsupervised_utility']['cluster_ari']:.4f}")
print(f"PCA Similarity: {metrics['unsupervised_utility']['pca_similarity']:.4f}")
print(f"MMD Distance: {metrics['unsupervised_utility']['mmd_distance']:.4f}")
print(f"Overall Utility: {metrics['summary']['unsupervised_utility']:.4f}")
```

**Unsupervised Metrics:**
- **Cluster ARI**: Adjusted Rand Index between K-Means clusterings (k=5)
- **PCA Similarity**: Cosine similarity of principal components + explained variance
- **MMD Distance**: Maximum Mean Discrepancy with RBF kernel
- **Combined Score**: Weighted average (40% ARI + 30% PCA + 30% (1-MMD))

This allows quality evaluation even when no target variable exists!

---

### Using Unsupervised Metrics Directly

```python
from dp_copulagan.utils import (
    compute_cluster_ari,
    compute_pca_similarity,
    compute_mmd_rbf,
    evaluate_unsupervised_utility
)

# Compute individual metrics
ari = compute_cluster_ari(real_data, synthetic_data, n_clusters=5)
pca_sim = compute_pca_similarity(real_data, synthetic_data)
mmd = compute_mmd_rbf(real_data, synthetic_data)

# Or get all at once
results = evaluate_unsupervised_utility(real_data, synthetic_data, verbose=True)
```

---

## 📚 Documentation

- **[LIMITATIONS.md](LIMITATIONS.md)** - Known limitations and troubleshooting
- **[examples/](examples/)** - Complete working examples
  - `quickstart.py` - Basic usage
  - `conditional_example.py` - Conditional generation
  - `unconditional_example.py` - Unconditional generation 🆕
- **[tests/](tests/)** - Comprehensive test suite

---

## 🔧 Requirements

- Python 3.8+
- TensorFlow 2.12+
- NumPy, Pandas, SciPy
- scikit-learn
- tensorflow-probability
- xgboost (for evaluation)

---

## 🎓 Citation

If you use DP-CopulaGAN in your research, please cite:

```bibtex
@software{dp_copulagan,
  title = {DP-CopulaGAN: Differentially Private Copula-GAN},
  author = {[Your Name]},
  year = {2024},
  url = {[repository URL]}
}
```

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

---

## 🆘 Support

- **Issues**: Report bugs via GitHub Issues
- **Documentation**: See [LIMITATIONS.md](LIMITATIONS.md) for common issues
- **Examples**: Check [examples/](examples/) for working code

---

## 🎯 Roadmap

- [x] Unconditional generation mode
- [x] Automatic label encoding
- [ ] Categorical feature support
- [ ] Time series mode
- [ ] Multi-table synthesis
- [ ] Privacy accounting integration

---

## ⚠️ Important Notes

### Privacy

- DP guarantees apply to the training process
- Small ε values may degrade synthetic data quality
- Always validate synthetic data before use
- See [LIMITATIONS.md](LIMITATIONS.md) for privacy-utility tradeoffs

### Performance

- Training time depends on dataset size and privacy budget
- Larger ε → faster convergence
- Smaller ε → more training epochs needed
- GPU recommended for large datasets (>10,000 samples)

### Data Requirements

- **Minimum samples**: 100+ per class (conditional) or 500+ total (unconditional)
- **Numeric features**: At least 3 continuous columns required
- **Label encoding**: Categorical labels auto-encoded
- **NaN handling**: Automatic interpolation and imputation

---

## 🏆 Comparison with Existing Methods

| Method | Conditional | Unconditional | Differential Privacy | Copula Transform |
|--------|-------------|---------------|---------------------|------------------|
| CTGAN | ✅ | ✅ | ❌ | ❌ |
| TVAE | ✅ | ✅ | ❌ | ❌ |
| DP-GAN | ✅ | ❌ | ✅ | ❌ |
| **DP-CopulaGAN** | ✅ | ✅ | ✅ | ✅ |

---

**DP-CopulaGAN**: Privacy-preserving synthetic data generation for any tabular dataset. 🚀
