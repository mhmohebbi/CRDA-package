"""
CRDA - Causal Residual Data Augmentation for Regression

A novel data augmentation methodology that improves regression model performance
by generating informed synthetic training examples through residual-guided
feature perturbation.

This package implements the CRDA algorithm which:
1. Trains a baseline regression model and computes prediction residuals
2. Identifies features that are statistically independent of residuals
3. Performs causal interventions on selected features
4. Generates counterfactual target values using residual patterns
5. Trains an improved model on the combined original and augmented data

Example:
    Basic usage with XGBoost::

        from crda import CRDA, Config
        from xgboost import XGBRegressor

        config = Config(
            dataset="path/to/data.csv",
            dataset_name="my_dataset"
        )
        crda = CRDA(config)
        results = crda.run(XGBRegressor())

    With pre-split data::

        config = Config(dataset_name="my_data", skip_preprocess=True)
        crda = CRDA(config)
        results = crda.run(XGBRegressor(), X_train, y_train, X_test, y_test)

    With a pre-trained model::

        my_model = XGBRegressor().fit(X_train, y_train)
        config = Config(dataset_name="my_data")
        crda = CRDA(config)
        results = crda.run(my_model, X_train, y_train, X_test, y_test, pretrained=True)

    Get augmented data only::

        config = Config(dataset="data.csv", dataset_name="my_data")
        crda = CRDA(config)
        aug_X, aug_y = crda.get_augmented_data(XGBRegressor())

Attributes:
    __version__: Package version string.
"""

__version__ = "1.0.2"

from crda.crda import CRDA
from crda.utils.config import Config
from crda.dataset import AbstractDataset
from crda.baseline import BaselineRegressor
from crda.filter import Filter

__all__ = [
    "__version__",
    "CRDA",
    "Config",
    "AbstractDataset",
    "BaselineRegressor",
    "Filter",
]
