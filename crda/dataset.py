"""
Dataset handling and preprocessing module.

This module provides the AbstractDataset class for loading, preprocessing,
and managing datasets used in the CRDA data augmentation pipeline.

The AbstractDataset handles:
- Loading data from various file formats (CSV, Excel, JSON, pickle)
- Automatic cleaning (duplicate removal, missing value handling)
- Feature preprocessing (standardization, one-hot encoding)
- Train/test splitting
- Residual computation for the CRDA algorithm
- Direct creation from pre-split numpy arrays

Example:
    Loading from a CSV file::

        from crda.dataset import AbstractDataset

        dataset = AbstractDataset("my_data", "path/to/data.csv", seed=42)
        X, y = dataset.preprocess()
        X_train, X_test, y_train, y_test = dataset.split(test_size=0.2)

    Using a pandas DataFrame::

        import pandas as pd
        from crda.dataset import AbstractDataset

        df = pd.read_csv("data.csv")
        dataset = AbstractDataset("my_data", df, seed=42)
        X, y = dataset.preprocess()

    Creating from pre-split numpy arrays::

        dataset = AbstractDataset.from_splits(
            name="my_data",
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            cat_indices={"color": [0, 1, 2]}  # optional
        )
"""

from __future__ import annotations

from torch.utils.data import Dataset
from crda.baseline import BaselineRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from typing import Optional, Union, Dict, List


class AbstractDataset(Dataset):
    """Dataset container with preprocessing and splitting capabilities.

    This class handles loading, preprocessing, and managing datasets for the
    CRDA algorithm. It supports multiple input formats and automatically
    handles data cleaning, feature encoding, and normalization.

    Can be created in two ways:
    1. From a file path or DataFrame (standard constructor)
    2. From pre-split numpy arrays (using from_splits() factory method)

    Args:
        name: A descriptive name identifier for the dataset.
        dataset: Either a file path (str) to a data file or a pandas DataFrame.
            Supported file formats: CSV, Excel (.xlsx), JSON, pickle (.pkl).
        seed: Random seed for reproducible train/test splits. Defaults to None.

    Attributes:
        name: Dataset name identifier.
        seed: Random seed for reproducibility.
        df: The loaded pandas DataFrame (None if created from splits).
        X: Feature columns as DataFrame (with generic names if from splits).
        y: Target column as DataFrame.
        X_preprocessed: Preprocessed feature matrix (after calling preprocess).
        y_preprocessed: Normalized target array (after calling preprocess).
        X_train: Training features (after calling split or from_splits).
        X_test: Test features (after calling split or from_splits).
        y_train: Training targets (after calling split or from_splits).
        y_test: Test targets (after calling split or from_splits).
        all_residuals: Residuals for full dataset (after add_residuals).
        train_residuals: Training set residuals (after add_residuals).
        test_residuals: Test set residuals (after add_residuals).
        cat_cols_dict: Mapping of categorical column names to one-hot columns.
        cat_cols_index_dict: Mapping of categorical columns to their indices.
        all_cat_col_indices: List of all categorical column indices.

    Raises:
        ValueError: If an unsupported file type is provided.
        AssertionError: If the target variable is not numerical.

    Example:
        Standard usage::

            >>> dataset = AbstractDataset("housing", "housing.csv", seed=42)
            >>> X, y = dataset.preprocess()
            >>> X_train, X_test, y_train, y_test = dataset.split(test_size=0.2)

        From pre-split arrays::

            >>> dataset = AbstractDataset.from_splits(
            ...     "my_data", X_train, y_train, X_test, y_test
            ... )
    """

    def __init__(
        self,
        name: str,
        dataset: Union[str, pd.DataFrame],
        seed: Optional[int] = None
    ):
        """Initialize the AbstractDataset.

        Args:
            name: Name identifier for the dataset.
            dataset: Path to a data file (CSV, Excel, JSON, or pickle) or a
                pandas DataFrame containing the data. The last column is
                assumed to be the target variable.
            seed: Random seed for reproducible sampling and train/test splits.
        """
        self.name = name
        self.seed = seed

        if isinstance(dataset, str):
            # Load from file based on extension
            if dataset.endswith(".csv"):
                self.df = pd.read_csv(dataset)
            elif dataset.endswith(".xlsx"):
                self.df = pd.read_excel(dataset)
            elif dataset.endswith(".json"):
                self.df = pd.read_json(dataset)
            elif dataset.endswith(".pkl"):
                self.df = pd.read_pickle(dataset)
            else:
                raise ValueError(f"Unsupported file type: {dataset}")
        else:
            self.df = dataset

        # Clean the data
        self.df = self.df.drop_duplicates()
        self.df = self.df.dropna()
        self.df = self.df.reset_index(drop=True)

        # Split features and target (last column is target)
        self.X = self.df.iloc[:, :-1]
        self.y = self.df.iloc[:, -1:]
        assert self.y.select_dtypes(include=['int64', 'float64']).shape[1] == 1, (
            "Target variable is not numerical"
        )

        # Initialize preprocessing attributes
        self.X_preprocessed = None
        self.y_preprocessed = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.all_residuals = None
        self.train_residuals = None
        self.test_residuals = None

        # Categorical column tracking
        self.cat_cols_dict = {}
        self.cat_cols_index_dict = {}
        self.all_cat_col_indices = []

    def _drop_bad_cols(self) -> None:
        """Remove columns that are not useful for regression.

        Drops:
        - Highly unique categorical columns (>95% unique values)
        - Zero-variance numerical columns (constants)
        """
        categorical_cols = self.X.select_dtypes(exclude=['number']).columns
        numerical_cols = self.X.select_dtypes(include=['number']).columns

        # Remove highly unique categorical columns
        if len(categorical_cols) > 0:
            unique_ratio = self.X[categorical_cols].nunique() / len(self.X)
            drop_cats = unique_ratio[unique_ratio > 0.95].index
            self.X = self.X.drop(columns=drop_cats)

        # Remove constant numerical columns
        if len(numerical_cols) > 0:
            var_selector = VarianceThreshold(threshold=0.0).fit(self.X[numerical_cols])
            drop_nums = numerical_cols[~var_selector.get_support()]
            self.X = self.X.drop(columns=drop_nums)

    def _one_hot_encode(self) -> None:
        """One-hot encode categorical features in the dataset.

        Updates cat_cols_dict, cat_cols_index_dict, and all_cat_col_indices
        to track the mapping between original categorical columns and their
        one-hot encoded representations.
        """
        categorical_cols = self.X.select_dtypes(exclude=['number']).columns

        encoder = OneHotEncoder(sparse_output=False)
        encoded_features = encoder.fit_transform(self.X[categorical_cols])
        encoded_df = pd.DataFrame(
            encoded_features,
            columns=encoder.get_feature_names_out(categorical_cols)
        )

        for col in categorical_cols:
            self.cat_cols_dict[col] = encoded_df.columns[
                encoded_df.columns.str.startswith(col)
            ].tolist()

        for col in self.cat_cols_dict:
            cat_col_indices = [
                encoded_df.columns.get_loc(col_name)
                for col_name in self.cat_cols_dict[col]
            ]
            self.all_cat_col_indices.extend(cat_col_indices)
            self.cat_cols_index_dict[col] = cat_col_indices

        self.X = self.X.drop(columns=categorical_cols)
        self.X = pd.concat([encoded_df, self.X], axis=1)

    def preprocess(self, skip: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Preprocess the dataset for model training.

        When skip=False (default), performs the following steps:
        1. Removes bad columns (high-cardinality categorical, zero-variance)
        2. One-hot encodes remaining categorical features
        3. Standardizes continuous features (mean=0, std=1)
        4. Normalizes target variable to [0, 1] range

        When skip=True, bypasses all preprocessing:
        - Converts data to numpy arrays without transformation
        - Assumes data is already cleaned, encoded, and normalized
        - Categorical column tracking is cleared (all features treated as continuous)

        One-hot encoded categorical features are kept as binary [0, 1] values
        and are not standardized.

        Args:
            skip: If True, skip all preprocessing steps and use data as-is.
                Use this when providing already preprocessed data.
                Defaults to False.

        Returns:
            Tuple containing:
                - X_preprocessed: Preprocessed feature matrix as numpy array.
                - y_preprocessed: Target variable as numpy array.

        Raises:
            AssertionError: If no columns remain after dropping bad columns
                (only when skip=False).
        """
        if skip:
            # Skip all preprocessing - use data as-is
            if hasattr(self.X, 'values'):
                self.X_preprocessed = self.X.values.astype(np.float64)
            else:
                self.X_preprocessed = np.asarray(self.X, dtype=np.float64)

            if hasattr(self.y, 'values'):
                self.y_preprocessed = self.y.values.ravel().astype(np.float64)
            else:
                self.y_preprocessed = np.asarray(self.y, dtype=np.float64).ravel()

            # Clear categorical tracking - all features treated as continuous
            self.cat_cols_dict = {}
            self.cat_cols_index_dict = {}
            self.all_cat_col_indices = []

            return self.X_preprocessed, self.y_preprocessed

        # Standard preprocessing path
        # Drop bad columns
        self._drop_bad_cols()
        assert len(self.X.columns) > 0, "No columns left after dropping bad columns"

        # One-hot encode categorical columns
        self._one_hot_encode()

        # Standardize only continuous features
        self.scaler_X = StandardScaler()

        # Get indices of continuous (non-categorical) features
        all_indices = set(range(self.X.shape[1]))
        continuous_indices = list(all_indices - set(self.all_cat_col_indices))

        # Convert to numpy for indexing
        X_array = self.X.values

        # Create preprocessed array (start with original values)
        self.X_preprocessed = X_array.copy()

        # Standardize only continuous features
        if len(continuous_indices) > 0:
            self.X_preprocessed[:, continuous_indices] = self.scaler_X.fit_transform(
                X_array[:, continuous_indices]
            )

        # Normalize the target variable
        self.normalizer_y = MinMaxScaler()
        self.y_preprocessed = self.normalizer_y.fit_transform(self.y).ravel()

        return self.X_preprocessed, self.y_preprocessed

    def split(
        self,
        test_size: float = 0.2,
        X: np.ndarray = None,
        y: np.ndarray = None,
        seed: int = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split the dataset into training and testing sets.

        Args:
            test_size: Proportion of the dataset for the test split.
                Defaults to 0.2 (20%).
            X: Feature matrix to split. If None, uses preprocessed features.
            y: Target variable to split. If None, uses preprocessed target.
            seed: Random seed for the split. If None, uses the dataset's seed.

        Returns:
            Tuple containing:
                - X_train: Training feature matrix.
                - X_test: Testing feature matrix.
                - y_train: Training target array.
                - y_test: Testing target array.

        Raises:
            AssertionError: If data hasn't been preprocessed and no X, y provided.
        """
        if X is None and y is None:
            assert self.y_preprocessed is not None and self.X_preprocessed is not None, (
                "Preprocess the data first."
            )
            X = self.X_preprocessed
            y = self.y_preprocessed

        if seed is None:
            seed = self.seed

        self.X_train, self.X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, shuffle=True
        )
        self.y_train = y_train.ravel()
        self.y_test = y_test.ravel()
        return self.X_train, self.X_test, self.y_train, self.y_test

    def add_residuals(
        self,
        model: BaselineRegressor
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate and store residuals from a baseline regressor model.

        Residuals are computed as actual values minus predicted values.
        Calculates residuals for the full dataset, training set, and test set.

        Args:
            model: A fitted BaselineRegressor that implements a predict() method.

        Returns:
            Tuple containing:
                - all_residuals: Residuals for the entire preprocessed dataset.
                - train_residuals: Residuals for the training set.
                - test_residuals: Residuals for the test set.
        """
        self.all_residuals = self.y_preprocessed - model.predict(self.X_preprocessed)

        self.train_residuals = self.y_train - model.predict(self.X_train)
        self.test_residuals = self.y_test - model.predict(self.X_test)

        return self.all_residuals, self.train_residuals, self.test_residuals

    @classmethod
    def from_splits(
        cls,
        name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        cat_indices: Optional[Dict[str, List[int]]] = None,
        seed: Optional[int] = None
    ) -> "AbstractDataset":
        """Create a dataset directly from pre-split numpy arrays.

        Use this factory method when you have pre-split, pre-processed data
        and want to use CRDA without loading from a file. This is useful when:
        - You have your own preprocessing pipeline
        - You want to use a specific train/test split
        - You're using CRDA with a pre-trained model

        Feature names are auto-generated as "feature_0", "feature_1", etc.
        If cat_indices is not provided, all features are treated as continuous.

        Args:
            name: Dataset name identifier for logging and results.
            X_train: Training features array of shape (n_train, n_features).
            y_train: Training targets array of shape (n_train,).
            X_test: Test features array of shape (n_test, n_features).
            y_test: Test targets array of shape (n_test,).
            cat_indices: Optional dict mapping categorical feature names to their
                one-hot encoded column indices. Format:
                {"cat_name": [idx1, idx2, ...]} where indices are column positions.
                If None, all features are treated as continuous.
            seed: Random seed for reproducibility (stored but not used since
                splits are already provided).

        Returns:
            AbstractDataset instance with splits already set and ready for use.

        Example:
            Basic usage::

                dataset = AbstractDataset.from_splits(
                    "my_data",
                    X_train, y_train,
                    X_test, y_test
                )

            With categorical indices::

                # Columns 0-2 are one-hot encoded "color"
                dataset = AbstractDataset.from_splits(
                    "my_data",
                    X_train, y_train,
                    X_test, y_test,
                    cat_indices={"color": [0, 1, 2]}
                )
        """
        # Create instance without calling __init__
        instance = cls.__new__(cls)

        # Set basic attributes
        instance.name = name
        instance.seed = seed
        instance.df = None  # No DataFrame when using splits directly

        # Convert and store splits
        instance.X_train = np.asarray(X_train, dtype=np.float64)
        instance.y_train = np.asarray(y_train, dtype=np.float64).ravel()
        instance.X_test = np.asarray(X_test, dtype=np.float64)
        instance.y_test = np.asarray(y_test, dtype=np.float64).ravel()

        # Combine for X_preprocessed/y_preprocessed (needed for all_residuals)
        instance.X_preprocessed = np.vstack([instance.X_train, instance.X_test])
        instance.y_preprocessed = np.concatenate([instance.y_train, instance.y_test])

        # Generate generic feature names for Filter compatibility
        n_features = instance.X_train.shape[1]
        feature_names = [f"feature_{i}" for i in range(n_features)]
        instance.X = pd.DataFrame(instance.X_preprocessed, columns=feature_names)
        instance.y = pd.DataFrame(instance.y_preprocessed, columns=["target"])

        # Handle categorical indices
        instance.cat_cols_dict = {}
        instance.cat_cols_index_dict = {}
        instance.all_cat_col_indices = []

        if cat_indices is not None:
            for cat_name, indices in cat_indices.items():
                col_names = [f"feature_{i}" for i in indices]
                instance.cat_cols_dict[cat_name] = col_names
                instance.cat_cols_index_dict[cat_name] = list(indices)
                instance.all_cat_col_indices.extend(indices)

        # Initialize residuals as None
        instance.all_residuals = None
        instance.train_residuals = None
        instance.test_residuals = None

        return instance

    def set_splits(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        cat_indices: Optional[Dict[str, List[int]]] = None
    ) -> None:
        """Directly set train/test splits, bypassing the split() method.

        Use this when you have pre-split data and want to inject it directly
        into an existing dataset instance. Also updates X_preprocessed and
        y_preprocessed by combining the splits.

        If cat_indices is provided, it will override any existing categorical
        column tracking. If not provided and the dataset was loaded from a file,
        existing categorical tracking is preserved.

        Args:
            X_train: Training features array of shape (n_train, n_features).
            y_train: Training targets array of shape (n_train,).
            X_test: Test features array of shape (n_test, n_features).
            y_test: Test targets array of shape (n_test,).
            cat_indices: Optional dict mapping categorical feature names to their
                one-hot encoded column indices. If provided, overrides existing
                categorical tracking. If None, existing tracking is preserved
                (or empty if no categoricals were detected).

        Example:
            >>> dataset = AbstractDataset("my_data", df)
            >>> dataset.set_splits(X_train, y_train, X_test, y_test)
        """
        # Convert and store splits
        self.X_train = np.asarray(X_train, dtype=np.float64)
        self.y_train = np.asarray(y_train, dtype=np.float64).ravel()
        self.X_test = np.asarray(X_test, dtype=np.float64)
        self.y_test = np.asarray(y_test, dtype=np.float64).ravel()

        # Combine for X_preprocessed/y_preprocessed
        self.X_preprocessed = np.vstack([self.X_train, self.X_test])
        self.y_preprocessed = np.concatenate([self.y_train, self.y_test])

        # Update X DataFrame with generic feature names if needed
        n_features = self.X_train.shape[1]
        if self.X is None or not hasattr(self.X, 'columns') or len(self.X.columns) != n_features:
            feature_names = [f"feature_{i}" for i in range(n_features)]
            self.X = pd.DataFrame(self.X_preprocessed, columns=feature_names)
            self.y = pd.DataFrame(self.y_preprocessed, columns=["target"])

        # Handle categorical indices if provided
        if cat_indices is not None:
            self.cat_cols_dict = {}
            self.cat_cols_index_dict = {}
            self.all_cat_col_indices = []

            for cat_name, indices in cat_indices.items():
                col_names = [f"feature_{i}" for i in indices]
                self.cat_cols_dict[cat_name] = col_names
                self.cat_cols_index_dict[cat_name] = list(indices)
                self.all_cat_col_indices.extend(indices)

        # Reset residuals since data changed
        self.all_residuals = None
        self.train_residuals = None
        self.test_residuals = None
