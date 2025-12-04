# Installation Guide

## Requirements

- Python 3.8 or higher
- pip (Python package manager)
- (Optional) GPU with CUDA for faster training

## Installation

### Option 1: Install from Source (Current Method)

```bash
# Clone or navigate to the package directory
cd dp_copulagan_package

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"

# Or install with all extras
pip install -e ".[dev,docs]"
```

### Option 2: Install from PyPI (After Publishing)

```bash
pip install dp-copulagan
```

## Verify Installation

```python
import dp_copulagan

# Print version
print(dp_copulagan.get_version())

# Print system info
dp_copulagan.print_system_info()
```

## Quick Test

```python
from dp_copulagan import DPCopulaGAN
import pandas as pd
import numpy as np

# Create sample data
data = pd.DataFrame({
    'x1': np.random.randn(100),
    'x2': np.random.randn(100),
    'label': np.random.choice([0, 1], 100)
})

# Train model (quick test)
from dp_copulagan.utils import GANConfig
config = GANConfig(epochs=10, batch_size=32)

model = DPCopulaGAN(epsilon=10.0, gan_config=config, label_col='label')
model.fit(data)

# Generate
synthetic = model.sample(10)
print("✓ Installation successful!")
print(synthetic.head())
```

## Troubleshooting

### GPU Issues

If you have GPU but it's not detected:

```python
from dp_copulagan.utils import setup_gpu
gpu_available = setup_gpu(memory_growth=True)
print(f"GPU available: {gpu_available}")
```

### TensorFlow Issues

If you encounter TensorFlow errors:

```bash
# Update TensorFlow
pip install --upgrade tensorflow

# For GPU support
pip install tensorflow[and-cuda]
```

### Missing Dependencies

If you get import errors:

```bash
# Install all required packages
pip install numpy pandas scipy scikit-learn tensorflow tensorflow-probability xgboost

# Or reinstall the package
pip install -e ".[dev]" --force-reinstall
```

## Running Tests

```bash
# Install pytest
pip install pytest pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=dp_copulagan --cov-report=html

# Run specific test
pytest tests/unit/test_copula.py
```

## Running Examples

```bash
# Navigate to examples
cd examples/scripts

# Run quickstart
python quickstart.py

# Run Adult dataset example
python run_adult.py
```

## Building Documentation

```bash
cd docs
pip install sphinx sphinx-rtd-theme
make html

# View documentation
open build/html/index.html
```

## Uninstallation

```bash
pip uninstall dp-copulagan
```

---

For issues, please visit: https://github.com/yourname/dp-copulagan/issues
