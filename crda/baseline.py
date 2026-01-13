"""
Baseline regression model wrapper.

This module provides a standardized wrapper for sklearn-compatible regression
models, enabling consistent model creation, training, evaluation, and persistence.

The BaselineRegressor class wraps any sklearn-compatible regressor and provides
a unified interface for the CRDA augmentation pipeline.

Example:
    Using with XGBoost::

        from crda.baseline import BaselineRegressor
        from xgboost import XGBRegressor

        regressor = BaselineRegressor(XGBRegressor())
        regressor.train(X_train, y_train)
        predictions = regressor.predict(X_test)
        mse = regressor.evaluate(X_test, y_test, metric="mse")
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Callable
import json
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin, clone
from joblib import dump, load


class BaselineRegressor(BaseEstimator, RegressorMixin):
    """Wrapper that provides a standardized interface for sklearn-compatible regressors.

    This class wraps an arbitrary sklearn-compatible regression model,
    providing consistent methods for training, prediction, evaluation,
    and model persistence across different underlying implementations.

    Args:
        model: A sklearn-compatible regressor instance. Must implement
            fit, predict, get_params, and set_params methods.

    Attributes:
        model: The underlying regression model instance.

    Example:
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> regressor = BaselineRegressor(RandomForestRegressor(n_estimators=100))
        >>> regressor.train(X_train, y_train)
        >>> predictions = regressor.predict(X_test)
    """

    def __init__(self, model: Any) -> None:
        """Initialize with a sklearn-compatible regression model instance.

        Args:
            model: A sklearn-compatible regressor instance. Must have fit,
                predict, get_params, and set_params methods.
        """
        self.model = model

    def clone(self, **override_params: Any) -> "BaselineRegressor":
        """Create a new unfitted BaselineRegressor with the same or modified parameters.

        Uses sklearn.base.clone() to create a fresh copy of the underlying model,
        then optionally applies parameter overrides.

        Args:
            **override_params: Parameters to override in the cloned model.
                These are passed directly to the model's set_params method.

        Returns:
            A new BaselineRegressor instance with an unfitted cloned model.

        Example:
            >>> original = BaselineRegressor(XGBRegressor(n_estimators=100))
            >>> cloned = original.clone(n_estimators=200)
        """
        new_model = clone(self.model)
        if override_params:
            new_model.set_params(**override_params)
        return BaselineRegressor(new_model)

    def train(self, X: np.ndarray, y: np.ndarray) -> "BaselineRegressor":
        """Train the regression model on the provided data.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target values of shape (n_samples,).

        Returns:
            Self, to allow for method chaining.
        """
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for the input data.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predicted values of shape (n_samples,).
        """
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray, *, metric: str = "mse") -> float:
        """Evaluate model performance on the provided data.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: True target values of shape (n_samples,).
            metric: Evaluation metric to use. Supported options:
                - "mse": Mean Squared Error (lower is better)
                - "rmse": Root Mean Squared Error (lower is better)
                - "r2": R-squared coefficient (higher is better)

        Returns:
            Scalar performance score.

        Raises:
            ValueError: If an unsupported metric is specified.
        """
        preds = self.predict(X)
        evaluation_function = self._get_evaluation_function(metric)
        return float(evaluation_function(y, preds))

    def _get_evaluation_function(self, metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
        """Get the evaluation function for the specified metric.

        Args:
            metric: Name of the evaluation metric.

        Returns:
            A callable that takes (y_true, y_pred) and returns a score.

        Raises:
            ValueError: If the metric is not supported.
        """
        metric = metric.lower()
        if metric == "rmse":
            return lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))
        elif metric == "mse":
            return mean_squared_error
        elif metric == "r2":
            return r2_score
        else:
            raise ValueError(
                f"Unsupported metric '{metric}'. Supported metrics: 'mse', 'rmse', 'r2'."
            )

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Return the parameters of the underlying model.

        Args:
            deep: If True, returns parameters for this estimator and
                contained subobjects that are estimators.

        Returns:
            Dictionary of parameter names mapped to their values.
        """
        return self.model.get_params(deep=deep)

    def set_params(self, **params: Any) -> "BaselineRegressor":
        """Set the parameters of the underlying model.

        Args:
            **params: Parameters to set on the underlying model.

        Returns:
            Self, to allow for method chaining.
        """
        self.model.set_params(**params)
        return self

    def __repr__(self) -> str:
        """Return a string representation of the BaselineRegressor.

        Returns:
            String representation showing class name and the underlying model.
        """
        return f"{self.__class__.__name__}(model={self.model!r})"

    def save(self, path: str) -> None:
        """Save the model and its parameters to files.

        Creates two files:
        - {path}.pkl: The serialized model object
        - {path}.params: JSON file with model parameters

        Args:
            path: Base path for saving (without extension).
        """
        dump(self, path + ".pkl")
        with open(path + ".params", "w") as f:
            json.dump(self.get_params(), f)

    @classmethod
    def load(cls, path: str) -> "BaselineRegressor":
        """Load a model from files.

        Args:
            path: Base path to load from (without extension).

        Returns:
            A BaselineRegressor instance with the loaded model.
        """
        model = load(path + ".pkl")
        with open(path + ".params", "r") as f:
            model.set_params(**json.load(f))
        return model

    def reset(self) -> "BaselineRegressor":
        """Reset the model to its original unfitted state.

        Returns:
            A new BaselineRegressor with an unfitted cloned model.
        """
        return self.clone()


__all__: Tuple[str, ...] = ("BaselineRegressor",)
