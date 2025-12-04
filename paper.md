---
title: 'DP-CopulaGAN: A Differentially Private Synthetic Data Generator Using Copula and Generative Adversarial Networks'
tags:
  - Python
  - differential privacy
  - synthetic data
  - copula
  - generative adversarial networks
  - official statistics
authors:
  - name: Emre Çırak
    orcid: "0009-0005-9463-7153"
    affiliation: "1, 2"
affiliations:
 - name: Department of Statistics, Gazi University, Türkiye
   index: 1
 - name: Turkish Statistical Institute (TURKSTAT), Türkiye
   index: 2
date: 2 December 2025
bibliography: paper.bib
---

# Summary

Synthetic data generation has become essential for sharing statistical information while protecting individual privacy. `DP-CopulaGAN` is a Python package that combines Gaussian copula transformations with Wasserstein Generative Adversarial Networks (WGAN-GP) under formal differential privacy guarantees via differentially private stochastic gradient descent (DP-SGD). This approach addresses a critical gap in synthetic data generation: producing high-utility, privacy-preserving synthetic tabular data that preserves complex statistical dependencies.

The package is designed for practitioners in official statistics, machine learning research, and data privacy who need to generate synthetic microdata with provable privacy guarantees. Unlike existing synthetic data tools, `DP-CopulaGAN` provides formal ($\varepsilon, \delta$)-differential privacy while maintaining correlation structures through copula-based transformations. This makes it particularly suitable for publishing synthetic versions of sensitive survey data, medical records, and other confidential datasets where both utility and privacy are paramount.

`DP-CopulaGAN` supports both conditional generation (for labeled datasets) and unconditional generation (for unlabeled data), with automatic handling of multi-class classification problems and comprehensive evaluation metrics including statistical similarity measures and machine learning utility tests. The package has been successfully applied to real-world datasets from the Turkish Labour Force Survey, demonstrating its practical applicability in official statistics production.

# Statement of Need

Statistical agencies, research institutions, and private organizations increasingly face the challenge of sharing valuable data while protecting individual privacy. Traditional disclosure control methods such as aggregation, suppression, or perturbation often result in significant utility loss [@hundepool2012statistical]. Synthetic data generation offers an alternative approach, but existing tools lack either formal privacy guarantees or the ability to preserve complex dependency structures in tabular data.

The primary users of `DP-CopulaGAN` are: (1) official statistics agencies seeking to release privacy-preserving microdata [@drechsler2011empirical; @reiter2005using]; (2) machine learning researchers requiring private training data; (3) healthcare organizations sharing patient data for research [@chen2021synthetic]; and (4) privacy practitioners implementing differential privacy in production systems. These users require tools that combine strong privacy guarantees with high data utility, particularly for tabular data with complex correlation structures.

Existing synthetic data packages present significant limitations. The Synthetic Data Vault (SDV) [@patki2016synthetic] offers tools like CTGAN [@xu2019modeling] and TVAE for generating synthetic tabular data, and includes a CopulaGAN model that uses Gaussian copulas. However, these methods lack formal privacy guarantees, making them unsuitable for sensitive data release. Conversely, differentially private generative models like DP-WGAN [@xie2018differentially] and DP-CGAN provide privacy guarantees but do not incorporate copula transformations, resulting in poor preservation of marginal distributions and correlations, especially for mixed-type tabular data.

`DP-CopulaGAN` fills this gap by combining three critical components: (1) Gaussian copula transformations [@nelsen2006introduction] for preserving marginal distributions and dependency structures, (2) WGAN-GP architecture [@gulrajani2017improved] for stable training and high-quality generation, and (3) DP-SGD [@abadi2016deep] for formal differential privacy guarantees. This combination enables the generation of synthetic data that maintains statistical fidelity while providing mathematically rigorous privacy protection. The package advances the state of the field by offering the first open-source implementation that integrates all three components in a production-ready, easy-to-use Python package specifically designed for tabular data.

# Software Description

## Key Features

`DP-CopulaGAN` provides several distinctive features for privacy-preserving synthetic data generation:

- **Gaussian Copula Transformation**: Transforms arbitrary marginal distributions to multivariate normal space, preserving correlation structures while enabling stable GAN training [@nelsen2006introduction; @li2017mmd].

- **Differentially Private Training**: Implements DP-SGD [@abadi2016deep] with gradient clipping and Gaussian noise addition, providing formal ($\varepsilon, \delta$)-differential privacy guarantees with transparent privacy accounting.

- **WGAN-GP Architecture**: Uses Wasserstein distance with gradient penalty [@gulrajani2017improved] for stable adversarial training, avoiding mode collapse and ensuring diverse synthetic samples.

- **Dual Generation Modes**: Supports conditional generation for labeled datasets and unconditional generation for unlabeled data, with automatic detection and handling of binary and multi-class classification problems.

- **Comprehensive Evaluation**: Includes statistical similarity metrics (Jensen-Shannon divergence, Wasserstein distance, correlation RMSE) and machine learning utility evaluation via Train-on-Synthetic-Test-on-Real (TSTR) methodology with 12 classifiers [@yale2020generation].

- **Production-Ready Design**: Offers automatic categorical encoding, robust type detection, NaN handling, and clear warnings for privacy-critical scenarios (e.g., dataset size smaller than batch size).

## Architecture Overview

The software architecture follows a modular design with five core components:

1. **Copula Module**: Implements Gaussian copula transformations for both conditional (per-class) and unconditional modes, transforming data to and from Gaussian latent space.

2. **GAN Module**: Contains generator and critic networks with spectral normalization [@miyato2018spectral], supporting both conditional and unconditional architectures.

3. **Differential Privacy Module**: Implements gradient clipping, Gaussian noise injection, and privacy budget tracking following the moments accountant method [@abadi2016deep].

4. **Evaluation Module**: Provides statistical and machine learning utility metrics, with automatic selection of appropriate metrics based on data characteristics (binary, multi-class, or unlabeled).

5. **Preprocessing Module**: Handles type detection, categorical encoding/decoding, and missing value management.

## Dependencies

The package requires Python 3.8+ and depends on:

- TensorFlow 2.12+ for deep learning and automatic differentiation
- NumPy for numerical computation
- SciPy for statistical functions and copula transformations  
- scikit-learn for evaluation metrics and preprocessing
- pandas for data manipulation
- XGBoost for ML utility evaluation

## Example Usage

```python
from dp_copulagan import DPCopulaGAN
from dp_copulagan.evaluation import evaluate_synthetic
import pandas as pd

# Load sensitive data
data = pd.read_csv('sensitive_survey.csv')

# Initialize with privacy budget
model = DPCopulaGAN(
    epsilon=1.0,        # Privacy parameter
    delta=1e-5,         # Privacy parameter
    label_col='income'  # Conditional generation
)

# Train with differential privacy
model.fit(data)

# Generate synthetic data
synthetic = model.sample(n_samples=10000)

# Evaluate quality and utility
results = evaluate_synthetic(
    real=data,
    synthetic=synthetic,
    label_col='income'
)

print(f"Mean AuROC: {results['summary']['mean_auroc']:.3f}")
```

## Software Quality

The package includes:

- Comprehensive unit and integration tests with pytest
- Type hints throughout the codebase
- Extensive documentation with examples
- Continuous integration for automated testing
- Clear error messages and user warnings
- Example scripts for common use cases

# State of the Field

Synthetic data generation for privacy has evolved through several approaches. Early methods focused on parametric modeling [@reiter2005using] and multiple imputation [@drechsler2011empirical], but these struggle with high-dimensional data and complex dependencies. More recent deep learning approaches have shown promise but present distinct tradeoffs.

The Synthetic Data Vault (SDV) [@patki2016synthetic] popularized neural network-based synthetic data generation with models like CTGAN [@xu2019modeling], TVAE, and CopulaGAN. CTGAN uses mode-specific normalization and conditional generators to handle mixed-type tabular data, while TVAE employs variational autoencoders. The SDV's CopulaGAN combines Gaussian copulas with GANs, similar to our approach, but crucially lacks any privacy guarantees. These methods generate high-utility synthetic data but provide no protection against privacy attacks such as membership inference [@shokri2017membership] or attribute disclosure.

Differentially private generative models address privacy concerns but typically sacrifice utility. DP-WGAN [@xie2018differentially] and DP-CGAN apply differential privacy to conditional GANs but do not use copula transformations, resulting in poor marginal distribution preservation. PATE-GAN [@jordon2018pate] uses the Private Aggregation of Teacher Ensembles framework but requires multiple teacher models and significant computational resources. DP-VAE models [@chen2018differentially] provide privacy guarantees but often produce lower-quality samples than GANs, especially for tabular data.

Recent work has explored specialized techniques for tabular data. TableGAN [@park2018data] and TGAN [@xu2018synthesizing] focus on table structure but lack privacy. DP-CTGAN [@torfi2022differentially] attempts to add differential privacy to CTGAN but does not incorporate copula transformations, limiting its ability to preserve complex dependencies. DPGAN [@frigerio2019differentially] provides privacy but uses simple noise addition rather than copula-based transformations.

`DP-CopulaGAN` advances beyond these approaches by uniquely combining:

1. **Formal Privacy**: Provides ($\varepsilon, \delta$)-differential privacy via DP-SGD with transparent privacy accounting, unlike SDV tools.

2. **Preserved Dependencies**: Uses Gaussian copula transformations to maintain correlation structures and marginal distributions, unlike DP-WGAN/DP-CGAN.

3. **Tabular Data Optimization**: Specifically designed for mixed-type tabular data common in official statistics, with per-class copulas for conditional generation.

4. **Comprehensive Evaluation**: Includes both statistical metrics and ML utility evaluation with automatic mode selection (binary, multi-class, unlabeled).

5. **Production Readiness**: Offers automatic type detection, categorical encoding, robust error handling, and clear documentation.

The package has been validated on real-world datasets and demonstrates that copula transformations significantly improve synthetic data quality under differential privacy compared to direct DP-GAN approaches. This makes it particularly valuable for official statistics agencies and research organizations that must balance data utility with strict privacy requirements.



# References
