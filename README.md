# CRDA - Counterfactual Residual Data Augmentation

[![PyPI version](https://badge.fury.io/py/crda.svg)](https://badge.fury.io/py/crda)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A novel data augmentation methodology that improves regression model performance by generating informed synthetic training examples through residual-guided feature perturbation.**

---

## Installation

### From PyPI (Recommended)

```bash
pip install crda
```

### From Source

```bash
git clone https://github.com/mhmohebbi/CRDA-package.git
cd CRDA-package
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

---

## Quick Start

### Basic Usage

```python
from crda import CRDA, Config
from xgboost import XGBRegressor

# Configure the experiment
config = Config(
    dataset="path/to/your/data.csv",
    dataset_name="my_dataset",
    random_seed=42,
    verbose=True
)

# Create CRDA instance
crda = CRDA(config)

# Run the augmentation pipeline with your model
results = crda.run(XGBRegressor())

# View results
print(f"Original Score: {results['score'].values[0]:.4f}")
print(f"Augmented Score: {results['aug_score'].values[0]:.4f}")
print(f"Improvement: {results['delta_score'].values[0]:.2f}%")
```

### Using a DataFrame

```python
import pandas as pd
from crda import CRDA, Config
from sklearn.ensemble import RandomForestRegressor

# Load your data
df = pd.read_csv("data.csv")

# Configure with DataFrame directly
config = Config(
    dataset=df,
    dataset_name="my_data",
    verbose=True
)

# Run with any sklearn-compatible regressor
crda = CRDA(config)
results = crda.run(RandomForestRegressor(n_estimators=100))
```

---

## Usage Modes

CRDA supports multiple usage patterns to fit different workflows:

### 1. Standard Mode (Dataset from Config)

The simplest way to use CRDA - provide the dataset in the config:

```python
config = Config(dataset="data.csv", dataset_name="my_data")
crda = CRDA(config)
results = crda.run(XGBRegressor())
```

### 2. Pre-Split Data Mode

Provide your own train/test splits:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

config = Config(dataset_name="my_data", skip_preprocess=True)
crda = CRDA(config)
results = crda.run(XGBRegressor(), X_train, y_train, X_test, y_test)
```

### 3. Pre-Trained Model Mode

Use an already-trained model - CRDA will skip training and continue from residual calculation:

```python
# Train your model externally with custom hyperparameters
my_model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05)
my_model.fit(X_train, y_train)

# CRDA uses the pre-trained model (skips preprocessing and training)
config = Config(dataset_name="my_data")
crda = CRDA(config)
results = crda.run(my_model, X_train, y_train, X_test, y_test, pretrained=True)
```

### 4. Get Augmented Data Only

Just retrieve the augmented data without running the full evaluation pipeline:

```python
config = Config(dataset="data.csv", dataset_name="my_data")
crda = CRDA(config)

# Get only the augmented samples
aug_X, aug_y = crda.get_augmented_data(XGBRegressor())

# Or get combined (original + augmented) data
combined_X, combined_y, aug_X, aug_y = crda.get_augmented_data(
    XGBRegressor(), return_combined=True
)
```

### 5. With Categorical Feature Indices

When providing pre-split numpy arrays with one-hot encoded categorical features:

```python
# Columns 0-2 are one-hot encoded "color", columns 3-5 are "size"
cat_indices = {
    "color": [0, 1, 2],
    "size": [3, 4, 5]
}

config = Config(dataset_name="my_data")
crda = CRDA(config)
results = crda.run(
    XGBRegressor(),
    X_train, y_train, X_test, y_test,
    cat_indices=cat_indices
)
```

> **Note**: If `cat_indices` is not provided when using pre-split data, all features are treated as continuous.

### 6. Reuse CRDA with Different Models

The same CRDA instance can be reused with different models:

```python
config = Config(dataset="data.csv", dataset_name="my_data")
crda = CRDA(config)

# Try XGBoost
results_xgb = crda.run(XGBRegressor())

# Try RandomForest with same config
results_rf = crda.run(RandomForestRegressor())
```

### 7. With Hyperparameter Tuning

```python
config = Config(
    dataset="data.csv",
    dataset_name="my_data",
    crda_param_tune=True,  # Enable Optuna-based tuning
    random_seed=42,
    save_params=True,
    save_models=True,
)

crda = CRDA(config)
results = crda.run(MLPRegressor(hidden_layer_sizes=(100, 50)))
```

---

## Dataset Format

CRDA expects tabular data where:
- **All columns except the last** are features (numerical or categorical)
- **The last column** is the target variable (must be numerical/continuous)
- Supported formats: CSV, Excel (.xlsx), JSON, pickle (.pkl), or pandas DataFrame

```csv
feature1,feature2,feature3,target
1.2,3.4,5.6,10.5
2.1,4.3,6.5,12.3
...
```

The dataset is automatically preprocessed (unless `skip_preprocess=True`):
- Duplicate rows are removed
- Missing values are dropped
- Categorical features are one-hot encoded
- Continuous features are standardized (mean=0, std=1)
- Target variable is normalized to [0, 1]

---

## Configuration Options

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | str \| DataFrame | None | Path to data file or pandas DataFrame. Optional if providing splits to `run()`. |
| `dataset_name` | str | "my_dataset" | Name identifier for the experiment. |
| `skip_preprocess` | bool | False | Skip preprocessing - use data as-is. |
| `evaluation_metric` | str | "mse" | Metric for evaluation: "mse", "rmse", or "r2" |
| `random_seed` | int | 0 | Random seed for reproducibility |
| `test_size` | float | 0.2 | Proportion of data for testing |

### Augmentation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `aug_data_size_factor` | float | 1.0 | Multiplier for augmented data size |
| `max_n_features_to_perturb` | int | 5 | Maximum features to perturb |
| `max_perturb_percent` | float | 0.1 | Maximum perturbation (+10%) |
| `min_perturb_percent` | float | -0.1 | Minimum perturbation (-10%) |
| `crda_param_tune` | bool | False | Enable Optuna hyperparameter tuning for CRDA params |

### Statistical Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `indep_test_threshold` | float | 0.05 | P-value threshold for independence test |
| `p_wilcoxon_threshold` | float | 0.05 | Significance threshold for Wilcoxon test |
| `ignore_filter` | bool | False | Proceed even if CRDA seems to produce bad results with the augmented data. |

### Output Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `results_dir` | str | "./runs" | Directory for experiment results |
| `save_models` | bool | False | Save trained model artifacts |
| `save_params` | bool | False | Save optimized parameters |
| `verbose` | bool | False | Print logs to console |
| `log_file` | str | None | Path for log file output |

---

## API Reference

### CRDA

The main class for running counterfactual residual data augmentation.

```python
from crda import CRDA, Config

# Initialize with config only (model passed to methods)
crda = CRDA(config)

# Run full pipeline
results = crda.run(model)

# Or just get augmented data
aug_X, aug_y = crda.get_augmented_data(model)
```

**Constructor:**
```python
CRDA(config: Config)
```

**Methods:**

#### `run()`

```python
def run(
    model,                          # sklearn-compatible regressor
    X_train: np.ndarray = None,     # Optional: training features
    y_train: np.ndarray = None,     # Optional: training targets
    X_test: np.ndarray = None,      # Optional: test features
    y_test: np.ndarray = None,      # Optional: test targets
    pretrained: bool = False,       # Is model already trained?
    cat_indices: dict = None        # Categorical column indices
) -> pd.DataFrame | None
```

Execute the full CRDA pipeline.

#### `get_augmented_data()`

```python
def get_augmented_data(
    model,                          # sklearn-compatible regressor
    X_train: np.ndarray = None,     # Optional: training features
    y_train: np.ndarray = None,     # Optional: training targets
    X_test: np.ndarray = None,      # Optional: test features
    y_test: np.ndarray = None,      # Optional: test targets
    pretrained: bool = False,       # Is model already trained?
    cat_indices: dict = None,       # Categorical column indices
    return_combined: bool = False   # Include original data in return?
) -> tuple[np.ndarray, np.ndarray]
```

Generate augmented data without running full evaluation.

### Config

Configuration management for experiments.

```python
from crda import Config

# Full config
config = Config(
    dataset="data.csv",
    dataset_name="example",
    evaluation_metric="mse",
    random_seed=42
)

# Minimal config (for pre-split data)
config = Config(evaluation_metric="rmse", random_seed=42)
# dataset_name defaults to "my_dataset"
```

**Methods:**
- `to_dict()` - Convert config to dictionary
- `from_dict(d)` - Create config from dictionary

### AbstractDataset

Dataset handling and preprocessing.

```python
from crda import AbstractDataset

# From file or DataFrame
dataset = AbstractDataset("name", "data.csv", seed=42)
X, y = dataset.preprocess()
X_train, X_test, y_train, y_test = dataset.split()

# From pre-split numpy arrays
dataset = AbstractDataset.from_splits(
    name="my_data",
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    cat_indices={"color": [0, 1, 2]}  # optional
)
```

### BaselineRegressor

Wrapper for sklearn-compatible regressors.

```python
from crda import BaselineRegressor
from xgboost import XGBRegressor

regressor = BaselineRegressor(XGBRegressor())
regressor.train(X_train, y_train)
predictions = regressor.predict(X_test)
mse = regressor.evaluate(X_test, y_test, metric="mse")
```

---

## Results

The `run()` method returns a pandas DataFrame with:

| Column | Description |
|--------|-------------|
| `dataset` | Dataset name |
| `seed` | Random seed used |
| `score` | Baseline model test score (MSE, RMSE, or R² based on config) |
| `aug_score` | Augmented model test score |
| `delta_score` | Percent improvement (positive = better) |
| `p_wilcoxon` | Statistical significance p-value |
| `should_proceed` | Whether augmentation was beneficial |
| `features_perturbed` | Names of perturbed features |

---

## Method Overview

**CRDA (Counterfactual Residual Data Augmentation)** improves regression models through:

1. **Residual Analysis**: Compute prediction residuals from baseline model
2. **Causal Filtering**: Identify features uncorrelated with residuals and conditionally independent of target
3. **Selective Perturbation**: Perturb filtered features to create interventional data
4. **Counterfactual Targets**: Generate targets using residual patterns
5. **Augmented Training**: Train new model on combined original + augmented data

### Key Innovation

Unlike traditional augmentation that blindly perturbs features, CRDA uses causal reasoning to select features that can be safely modified without corrupting the underlying data generating process (keeping residuals invariant).

---

## Differences from the Paper

This package is a reference implementation of CRDA but it diverges from the ICML 2026 paper (Mohebbi et al., 2026) in two principled ways. The core algorithm — residual reuse `y'_i = ĝ(x'_i) + ẑ_i` and the Wilcoxon signed-rank safety gate — is identical.

### 1. Different Independence Filter

The paper uses a two-stage screen over `{X, Y, Z}`:

1. PC algorithm to remove features with a direct edge to the residual `Z`
2. Pearson correlation check vs. `Z`

The package **replaces** this with a *per-feature marginal independence test* against the residual:

- **Continuous features:** distance correlation (`hyppo.independence.Dcorr`) with 1000 permutations
- **Categorical features:** mutual information (`sklearn.feature_selection.mutual_info_regression`) with a 1000-permutation null distribution

Features are ranked by p-value, most independent first. The acceptance threshold is controlled by `Config.indep_test_threshold` (default `0.05`).

**Why:**

- Distance correlation captures non-linear dependence that Pearson misses.
- A direct per-feature test avoids the PC algorithm's computational cost and its brittleness under hidden confounders — a failure mode the paper itself flags in its Limitations section.
- A clean p-value gives a principled way to rank candidate perturbable features.

### 2. Type-Aware Perturbation

The paper's intervention is purely multiplicative scaling, `x'_P = x_P · (1 + δ)`, which is only defined for continuous numeric features. The package branches on feature type:

- **Continuous features:** the paper's `(1 + δ)` rule, with `δ ∼ Uniform[min_perturb_percent, max_perturb_percent]`
- **One-hot-encoded categorical features:** uniform category resampling — draw a different category uniformly at random, with collision handling so the value is guaranteed to change

**Why:** many real tabular datasets contain categorical columns, and the paper's scaling rule is undefined on one-hot vectors. Making the perturbation type-aware lets CRDA cover broader tabular data without falling back to ad-hoc encoding tricks.

### What's Identical

- The additive-noise decomposition `Y = g(X) + Z` and the residual-invariance assumption `Pr(Z | X_P, X_R) = Pr(Z | X_R)`
- The residual-reuse construction `y'_i = ĝ(x'_i) + ẑ_i`
- The Wilcoxon signed-rank safety gate that decides whether to commit the augmentation

For the exact paper pipeline and experiments, see the [research repository](https://github.com/mhmohebbi/CRDA).

---

## Supported Models

CRDA works with any sklearn-compatible regressor:

- **Tree-based**: XGBoost, LightGBM, CatBoost, RandomForest, GradientBoosting
- **Neural Networks**: MLPRegressor, PyTorch models (with sklearn wrapper)
- **Linear Models**: Ridge, Lasso, ElasticNet, LinearRegression
- **Others**: SVR, KNeighborsRegressor, etc.

---

## Requirements

- Python >= 3.8
- numpy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- scipy >= 1.10.0
- torch >= 2.0.0
- optuna >= 3.0.0
- joblib >= 1.3.0
- hyppo >= 0.4.0

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Citation

If you use CRDA in your research, please cite:

```bibtex
@software{crda2026,
  author = {Mohebbi, Hossein},
  title = {CRDA: Counterfactual Residual Data Augmentation for Regression},
  year = {2026},
  url = {https://github.com/mhmohebbi/CRDA-package}
}
```
