"""
Causal Residual Data Augmentation (CRDA) implementation.

This module contains the main CRDA class that implements the causal residual
data augmentation algorithm for improving regression model performance.

The CRDA algorithm works by:
1. Training a baseline regression model on the original data
2. Computing prediction residuals to understand model errors
3. Filtering features based on their independence from residuals
4. Performing causal interventions on selected features
5. Generating counterfactual target values using residual patterns
6. Training an improved model on the combined original and augmented data

Example:
    Basic usage::

        from crda import CRDA, Config
        from xgboost import XGBRegressor

        config = Config(
            dataset="path/to/data.csv",
            dataset_name="my_dataset",
            verbose=True
        )
        crda = CRDA(config)
        results = crda.run(XGBRegressor())

    With pre-split data::

        config = Config(dataset_name="my_data", skip_preprocess=True)
        crda = CRDA(config)
        results = crda.run(XGBRegressor(), X_train, y_train, X_test, y_test)

    With pre-trained model::

        # Train your model externally
        my_model = XGBRegressor(n_estimators=200).fit(X_train, y_train)

        config = Config(dataset_name="my_data")
        crda = CRDA(config)
        results = crda.run(my_model, X_train, y_train, X_test, y_test, pretrained=True)

    Get augmented data only::

        config = Config(dataset="data.csv", dataset_name="my_data")
        crda = CRDA(config)
        aug_X, aug_y = crda.get_augmented_data(XGBRegressor())
"""

from __future__ import annotations

import os
import pandas as pd
import numpy as np
import random
from typing import Optional, Dict, List, Any, Union
from sklearn.model_selection import KFold, train_test_split
import torch
import json
import math
import optuna
from crda.baseline import BaselineRegressor
from crda.dataset import AbstractDataset
from crda.utils.config import Config
from crda.utils.logger import Logger
from crda.filter import Filter
from joblib import parallel_backend, Parallel, delayed
from joblib.externals.loky import get_reusable_executor
from scipy.stats import wilcoxon


class CRDA:
    """Causal Residual Data Augmentation for regression tasks.

    CRDA improves regression model performance by generating synthetic training
    data through causal interventions on features and counterfactual target
    generation. The method identifies features that are independent of model
    residuals and perturbs them to create augmented samples.

    The CRDA class acts as a reusable processor - you initialize it once with
    a Config, then call run() or get_augmented_data() with different models.

    Args:
        config: Configuration object containing experiment parameters such as
            dataset path, augmentation settings, and evaluation options.

    Attributes:
        config: The configuration object for this experiment.
        model: The regression model (set when run() is called).
        baseline: BaselineRegressor wrapper (set when run() is called).
        logger: Logger instance for experiment logging.
        dataset: The dataset (set when run() is called).
        aug_X_train: Augmented training features (set after augmentation).
        aug_y_train: Augmented training targets (set after augmentation).
        results_df: DataFrame containing experiment results.

    Example:
        Standard usage::

            >>> from crda import CRDA, Config
            >>> from sklearn.ensemble import GradientBoostingRegressor
            >>> config = Config(dataset="data.csv", dataset_name="example")
            >>> crda = CRDA(config)
            >>> results = crda.run(GradientBoostingRegressor())

        With pre-trained model::

            >>> my_model = XGBRegressor().fit(X_train, y_train)
            >>> results = crda.run(my_model, X_train, y_train, X_test, y_test, pretrained=True)

        Reuse with different models::

            >>> results_xgb = crda.run(XGBRegressor())
            >>> results_rf = crda.run(RandomForestRegressor())
    """

    def __init__(self, config: Config):
        """Initialize the CRDA experiment.

        Sets up logging and experiment directories. The model and data are
        provided later when calling run() or get_augmented_data().

        Args:
            config: Configuration object with experiment parameters.
        """
        self.config = config

        # Model and baseline are set when run() is called
        self.model = None
        self.baseline = None

        self.logger = Logger(
            log_to_console=self.config.verbose,
            log_file=self.config.log_file,
            log_to_file=False if self.config.log_file is None else True
        )

        self.aug_X_train = None
        self.aug_y_train = None
        self.results_df = None
        self.dataset = None  # Set when run() is called

        # Seed RNGs
        self.seeding(self.config.random_seed)

        # Save the config (will be updated when run() is called with more details)
        self._save_config()

        self.logger.info(f"CRDA initialized with config:\n{self.config}")

    def _save_config(self) -> None:
        """Save the current configuration to the experiment directory."""
        config_file = os.path.join(self.config.experiment_dir, "config.json")
        # Convert config to dict, handling non-serializable items
        config_dict = {}
        for k, v in self.config.to_dict().items():
            if isinstance(v, pd.DataFrame):
                config_dict[k] = f"<DataFrame: {v.shape}>"
            elif hasattr(v, '__class__') and not isinstance(v, (str, int, float, bool, list, dict, type(None))):
                config_dict[k] = f"<{type(v).__name__}>"
            else:
                config_dict[k] = v
        with open(config_file, "w") as f:
            json.dump(config_dict, f, indent=2)

    def _intervention(
        self,
        X_train: np.ndarray,
        features_to_perturb: list,
        aug_data_size_factor: float,
        min_perturb_percent: float,
        max_perturb_percent: float
    ) -> np.ndarray:
        """Generate augmented features by intervening on selected features.

        For each feature to perturb, generates new samples by applying
        interventions:
        - Continuous features (integer index): Random multiplicative
          perturbation within the specified min/max percent range.
        - Categorical features (list of indices for one-hot columns):
          Randomly changes the category to a different one.

        Args:
            X_train: Original training feature matrix of shape
                (n_samples, n_features).
            features_to_perturb: List where each entry is either an integer
                (continuous feature index) or a list of integers (one-hot
                encoded categorical feature indices).
            aug_data_size_factor: Factor controlling augmented sample count.
                1.0 means same number as original data.
            min_perturb_percent: Minimum relative change for continuous
                features (e.g., -0.2 for -20%).
            max_perturb_percent: Maximum relative change for continuous
                features (e.g., 0.2 for +20%).

        Returns:
            Augmented feature matrix of shape (aug_data_len, n_features).
        """
        aug_data_len = int(aug_data_size_factor * X_train.shape[0])
        n = int(math.ceil(aug_data_size_factor))

        perturbations = np.random.uniform(
            min_perturb_percent, max_perturb_percent, X_train.shape[0] * n
        )

        new_X_train = np.tile(X_train, (n, 1))

        for feature in features_to_perturb:
            if isinstance(feature, list):
                # Categorical feature (one-hot encoded)
                K = len(feature)  # Number of categories
                N = new_X_train.shape[0]  # Number of samples
                ohe_slice = new_X_train[:, feature]  # (N, K)

                current_labels = np.argmax(ohe_slice, axis=1)
                new_labels = np.random.randint(0, K, size=N)
                collisions = current_labels == new_labels

                # Ensure the label is actually perturbed
                new_labels[collisions] = (new_labels[collisions] + 1) % K
                new_ohe_slice = np.eye(K, dtype=new_X_train.dtype)[new_labels]
                new_X_train[:, feature] = new_ohe_slice
            else:
                # Continuous feature
                new_X_train[:, feature] = (
                    new_X_train[:, feature] * (1 + perturbations)
                )

        aug_X_train = new_X_train[:aug_data_len]
        return aug_X_train

    def _counterfactuals(
        self,
        X_train: np.ndarray,
        train_residuals: np.ndarray,
        baseline: BaselineRegressor,
        aug_X_train: np.ndarray,
        aug_data_size_factor: float
    ) -> np.ndarray:
        """Generate counterfactual target values for augmented features.

        Creates target values for augmented data by using the baseline model's
        predictions plus the corresponding residuals, simulating what the
        targets would be under the interventions.

        Args:
            X_train: Original training feature matrix.
            train_residuals: Residuals from baseline model on training data.
            baseline: Trained baseline model for predictions.
            aug_X_train: Augmented feature matrix.
            aug_data_size_factor: Factor determining augmented data size.

        Returns:
            Counterfactual target values for augmented data.
        """
        aug_data_len = int(aug_data_size_factor * X_train.shape[0])
        n = int(math.ceil(aug_data_size_factor))
        z_train = np.tile(train_residuals.reshape(-1, 1), (n, 1))
        z_train = z_train[:aug_data_len]

        aug_y_train = baseline.predict(aug_X_train) + z_train.ravel()
        return aug_y_train

    def _get_combined_aug_training_set(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        aug_X_train: np.ndarray,
        aug_y_train: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Combine original and augmented training data.

        Creates a unified training dataset by concatenating original and
        augmented data.

        Args:
            X_train: Original training feature matrix.
            y_train: Original training target vector.
            aug_X_train: Augmented training feature matrix.
            aug_y_train: Augmented training target vector.

        Returns:
            Tuple containing:
                - combined_X_train: Combined feature matrix.
                - combined_y_train: Combined target vector.
        """
        combined_X_train = np.concatenate([X_train, aug_X_train])
        combined_y_train = np.concatenate([y_train, aug_y_train])

        return combined_X_train, combined_y_train

    def _data_augmentation(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        train_residuals: np.ndarray,
        baseline: BaselineRegressor,
        best_features_to_perturb: list,
        **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform the complete data augmentation pipeline.

        Orchestrates the full data augmentation process: selects features to
        perturb, creates interventional data, and generates counterfactual
        targets.

        Args:
            X_train: Original training feature matrix.
            y_train: Original training target vector.
            train_residuals: Residuals from baseline model.
            baseline: Trained baseline model.
            best_features_to_perturb: Ranked list of feature indices to perturb.
            **kwargs: Additional parameters overriding config defaults:
                - max_n_features_to_perturb: Max features to perturb.
                - aug_data_size_factor: Augmentation size multiplier.
                - min_perturb_percent: Min perturbation percentage.
                - max_perturb_percent: Max perturbation percentage.

        Returns:
            Tuple containing:
                - aug_X_train: Augmented feature matrix.
                - aug_y_train: Augmented target vector.
        """
        max_n_features_to_perturb = kwargs.get(
            "max_n_features_to_perturb", self.config.max_n_features_to_perturb
        )
        aug_data_size_factor = kwargs.get(
            "aug_data_size_factor", self.config.aug_data_size_factor
        )
        min_perturb_percent = kwargs.get(
            "min_perturb_percent", self.config.min_perturb_percent
        )
        max_perturb_percent = kwargs.get(
            "max_perturb_percent", self.config.max_perturb_percent
        )

        perturbed_features = best_features_to_perturb[:max_n_features_to_perturb]

        aug_X_train = self._intervention(
            X_train, perturbed_features, aug_data_size_factor,
            min_perturb_percent, max_perturb_percent
        )
        aug_y_train = self._counterfactuals(
            X_train, train_residuals, baseline, aug_X_train, aug_data_size_factor
        )
        assert aug_X_train.shape[0] == aug_y_train.shape[0], (
            "Augmented X and Y train have different lengths"
        )

        self.aug_X_train = aug_X_train
        self.aug_y_train = aug_y_train
        return self.aug_X_train, self.aug_y_train

    def _tune_augmentation_params(
        self,
        dataset: AbstractDataset,
        baseline: BaselineRegressor,
        best_features_to_perturb: list,
        n_trials: int = 30,
        seed: int = None,
    ) -> dict:
        """Optimize augmentation hyperparameters using Optuna.

        Uses Bayesian optimization (TPE) with Hyperband pruning to find optimal
        augmentation parameters that optimize the validation score (minimizes for
        MSE/RMSE, maximizes for R²).

        Args:
            dataset: Dataset for training and validation.
            baseline: Baseline model to use for augmentation.
            best_features_to_perturb: Features identified for perturbation.
            n_trials: Number of optimization trials. Defaults to 30.
            seed: Random seed for reproducible optimization.

        Returns:
            Dictionary of best augmentation parameters found:
                - max_n_features_to_perturb: Optimal number of features.
                - aug_data_size_factor: Optimal augmentation size.
                - min_perturb_percent: Optimal min perturbation.
                - max_perturb_percent: Optimal max perturbation.
        """
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial):
            # Sample candidate augmentation parameters
            aug_params = dict(
                max_n_features_to_perturb=trial.suggest_int(
                    "max_n_features_to_perturb", 1, 5
                ),
                aug_data_size_factor=trial.suggest_float(
                    "aug_data_size_factor", 0.5, 1.5, step=0.25
                ),
                max_perturb_percent=trial.suggest_float(
                    "max_perturb_percent", 0.1, 1.0, step=0.1
                ),
            )
            aug_params["min_perturb_percent"] = -aug_params["max_perturb_percent"]

            # 80/20 train/validation split
            Xtr, Xval, ytr, yval, residuals_tr, residuals_val = train_test_split(
                dataset.X_train,
                dataset.y_train,
                dataset.train_residuals,
                test_size=0.2,
                random_state=trial.number
            )

            # Augment the training part only
            X_aug, y_aug = self._data_augmentation(
                Xtr, ytr, residuals_tr,
                baseline,
                best_features_to_perturb,
                **aug_params,
            )
            combined_X_train, combined_y_train = self._get_combined_aug_training_set(
                Xtr, ytr, X_aug, y_aug
            )

            # Clone and train on augmented data
            override_params = {}
            if self._model_supports_random_state():
                override_params["random_state"] = trial.number
            if self._model_supports_early_stopping():
                override_params["early_stopping_rounds"] = None
            new_baseline = baseline.clone(**override_params)

            self._train_aug_data(
                dataset, new_baseline,
                combined_X_train, combined_y_train
            )

            # Validate
            score = new_baseline.evaluate(Xval, yval, metric=self.config.evaluation_metric)
            return score

        # Determine optimization direction based on metric
        # MSE/RMSE: lower is better → minimize
        # R²: higher is better → maximize
        direction = "maximize" if self.config.evaluation_metric == "r2" else "minimize"

        # Create study with TPE + Hyperband
        study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=optuna.pruners.HyperbandPruner(),
        )
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=1,
            show_progress_bar=True,
        )
        best_aug = study.best_params
        # Recover symmetric perturbation bounds
        best_aug["min_perturb_percent"] = -best_aug["max_perturb_percent"]
        return best_aug

    def _cv_evaluation(
        self,
        dataset: AbstractDataset,
        baseline: BaselineRegressor,
        best_features_to_perturb: list[int],
        best_aug_params: dict,
        seed: int,
    ) -> tuple[float, list]:
        """Evaluate augmentation effectiveness using cross-validation.

        Performs 10-fold cross-validation to compare original vs augmented
        training and tests for statistical significance using Wilcoxon
        signed-rank test.

        Args:
            dataset: Dataset for evaluation.
            baseline: Baseline regression model.
            best_features_to_perturb: Features to perturb during augmentation.
            best_aug_params: Optimized augmentation parameters.
            seed: Random seed for cross-validation splits.

        Returns:
            Tuple containing:
                - p_wilcoxon: P-value from Wilcoxon signed-rank test.
                - aug_models: List of trained models from CV folds.
        """
        cv = KFold(n_splits=10, shuffle=True, random_state=seed)
        splits = list(cv.split(dataset.X_train))

        def evaluate_fold(train_idx, val_idx, fold_seed):
            """Evaluate a single fold for both unaugmented and augmented models."""
            # Seed each worker process for reproducibility
            self.seeding(fold_seed)

            # Split data for this fold
            X_train_fold = dataset.X_train[train_idx]
            y_train_fold = dataset.y_train[train_idx]
            X_val_fold = dataset.X_train[val_idx]
            y_val_fold = dataset.y_train[val_idx]
            residuals_fold = dataset.train_residuals[train_idx]

            # Train and evaluate unaugmented model
            override_params = {}
            if self._model_supports_random_state():
                override_params["random_state"] = fold_seed

            model_unaug = baseline.clone(**override_params)
            model_unaug.train(X_train_fold, y_train_fold)
            pred_unaug = model_unaug.predict(X_val_fold)
            evaluation_function = model_unaug._get_evaluation_function(
                self.config.evaluation_metric
            )
            score_unaug = evaluation_function(y_val_fold, pred_unaug)

            # Augment only the training fold
            aug_X_train, aug_y_train = self._data_augmentation(
                X_train_fold, y_train_fold, residuals_fold, baseline,
                best_features_to_perturb, **best_aug_params
            )
            combined_X_train, combined_y_train = self._get_combined_aug_training_set(
                X_train_fold, y_train_fold, aug_X_train, aug_y_train
            )

            # Train and evaluate augmented model
            model_aug = baseline.clone(**override_params)
            model_aug.train(combined_X_train, combined_y_train)
            pred_aug = model_aug.predict(X_val_fold)
            score_aug = evaluation_function(y_val_fold, pred_aug)

            return score_unaug, score_aug, model_aug.model

        # Generate deterministic seeds for each fold
        fold_seeds = [seed + i for i in range(len(splits))]

        # Run all folds in parallel
        with parallel_backend("loky", inner_max_num_threads=1):
            results = Parallel(n_jobs=-1)(
                delayed(evaluate_fold)(train_idx, val_idx, fold_seed)
                for (train_idx, val_idx), fold_seed in zip(splits, fold_seeds)
            )

        scores_unaug = np.array([r[0] for r in results])
        scores_aug = np.array([r[1] for r in results])
        aug_models = [r[2] for r in results]

        # Test direction depends on metric:
        # MSE/RMSE: lower is better → test if aug < unaug (alternative="less")
        # R²: higher is better → test if aug > unaug (alternative="greater")
        alternative = "greater" if self.config.evaluation_metric == "r2" else "less"
        stat, p = wilcoxon(
            np.array(scores_aug) - np.array(scores_unaug),
            alternative=alternative,
            zero_method="wilcox"
        )
        return p, aug_models

    def _train_baseline(
        self,
        dataset: AbstractDataset,
        baseline: BaselineRegressor
    ) -> None:
        """Train the baseline regression model on the training data.

        Trains the baseline model and saves it if configured to do so.

        Args:
            dataset: Dataset containing training data.
            baseline: Baseline regression model to train.
        """
        model_name = type(baseline.model).__name__
        self.logger.info(f"Training baseline model: {model_name}")
        baseline.train(dataset.X_train, dataset.y_train)

        # Save the model if configured
        if self.config.save_models:
            model_dir = os.path.join(self.config.experiment_dir, "models")
            os.makedirs(model_dir, exist_ok=True)
            baseline.save(os.path.join(model_dir, f"baseline_{dataset.name}"))
            self.logger.info(f"Saved baseline model")

    def _train_aug_data(
        self,
        dataset: AbstractDataset,
        baseline: BaselineRegressor,
        combined_X_train: np.ndarray,
        combined_y_train: np.ndarray
    ) -> None:
        """Train a new model on the augmented dataset.

        Trains the baseline model on the combined original and augmented
        training data and saves the model if configured to do so.

        Args:
            dataset: Dataset for naming and organization.
            baseline: Baseline regression model to train.
            combined_X_train: Combined original and augmented feature data.
            combined_y_train: Combined original and augmented target data.
        """
        model_name = type(baseline.model).__name__
        self.logger.info(f"Training new {model_name} model on augmented data\n")
        baseline.train(combined_X_train, combined_y_train)

        # Save the model if configured
        if self.config.save_models:
            model_dir = os.path.join(self.config.experiment_dir, "models")
            os.makedirs(model_dir, exist_ok=True)
            baseline.save(os.path.join(model_dir, f"aug_model_{dataset.name}"))
            self.logger.info(f"Saved newly trained model")

    def _get_selected_features_names(
        self,
        features_to_perturb: list,
        filter: Filter
    ) -> list[str]:
        """Get human-readable names for selected features.

        Args:
            features_to_perturb: List of feature indices or index lists.
            filter: Filter instance with feature name mappings.

        Returns:
            List of feature names corresponding to the indices.
        """
        names = []
        for feature in features_to_perturb:
            if isinstance(feature, int):
                names.append(filter.feature_index_to_name[feature])
            else:
                cat_name = filter.feature_index_to_name[feature[0]]
                names.append(cat_name.split("_")[0])
        return names

    def _model_supports_random_state(self, model=None) -> bool:
        """Check if the underlying model supports random_state parameter.

        Args:
            model: Model to check. If None, uses self.model.

        Returns:
            True if the model has a random_state parameter.
        """
        model = model or self.model
        if model is None:
            return False
        return "random_state" in model.get_params()

    def _model_supports_early_stopping(self, model=None) -> bool:
        """Check if the underlying model supports early_stopping_rounds.

        Args:
            model: Model to check. If None, uses self.model.

        Returns:
            True if the model has an early_stopping_rounds parameter.
        """
        model = model or self.model
        if model is None:
            return False
        return "early_stopping_rounds" in model.get_params()

    def _run_crda_internal(
        self,
        dataset: AbstractDataset,
        seed: int,
        pretrained: bool = False,
        splits_provided: bool = False
    ) -> dict | None:
        """Run the complete CRDA pipeline for one dataset and seed.

        Executes the full experimental pipeline including baseline training,
        feature filtering, augmentation parameter optimization, cross-validation
        evaluation, and final testing.

        Args:
            dataset: Dataset to run experiment on.
            seed: Random seed for reproducible results.
            pretrained: If True, skip training (model is already fitted).
            splits_provided: If True, splits are already set on dataset.

        Returns:
            Dictionary containing experiment results including:
                - dataset: Dataset name
                - seed: Random seed used
                - score: Baseline model test score
                - aug_score: Augmented model test score
                - delta_score: Percent improvement (positive = better)
                - p_wilcoxon: Statistical significance p-value
                - should_proceed: Whether augmentation was beneficial
                - features_perturbed: Names of perturbed features
            Returns None if experiment cannot proceed.
        """
        # Get baseline - either use pretrained or clone for fresh training
        if pretrained:
            # Use the model as-is (already trained)
            baseline = self.baseline
            self.logger.info("Using pre-trained model")
        else:
            # Clone for fresh training
            baseline = self.baseline.clone()
            if self._model_supports_random_state(baseline.model):
                baseline.set_params(random_state=seed)

        # Handle data preprocessing and splitting
        if splits_provided:
            # Splits already set via from_splits() - data is ready
            X_train, X_test = dataset.X_train, dataset.X_test
            y_train, y_test = dataset.y_train, dataset.y_test
        else:
            # Need to preprocess and split
            X, y = dataset.preprocess(skip=self.config.skip_preprocess)
            X_train, X_test, y_train, y_test = dataset.split(
                test_size=self.config.test_size, seed=seed
            )

        # Train baseline model (skip if pretrained)
        if not pretrained:
            self._train_baseline(dataset, baseline)
            self.logger.info("Done training baseline model")
        else:
            self.logger.info("Skipping training (using pre-trained model)")

        # Calculate evaluation metrics for the baseline
        score = baseline.evaluate(X_test, y_test, metric=self.config.evaluation_metric)

        # Calculate residuals
        all_residuals, train_residuals, test_residuals = dataset.add_residuals(baseline)
        self.logger.info("Done calculating residuals\n")

        # Filter for features to perturb
        filter_obj = Filter(dataset, self.config, self.logger)
        self.logger.info("Filtering for features to perturb")
        best_features_to_perturb = filter_obj.run_checks()

        should_proceed = True
        if best_features_to_perturb is None:
            self.logger.warning(f"No candidate features found for {dataset.name}.")
            return None

        self.logger.info("Done filtering.\n")

        # Data augmentation step
        if self.config.crda_param_tune:
            best_aug_params = self._tune_augmentation_params(
                dataset, baseline, best_features_to_perturb, seed=seed
            )
        else:
            best_aug_params = {
                "max_n_features_to_perturb": self.config.max_n_features_to_perturb,
                "aug_data_size_factor": self.config.aug_data_size_factor,
                "min_perturb_percent": self.config.min_perturb_percent,
                "max_perturb_percent": self.config.max_perturb_percent,
            }

        self.logger.info(f"Best augmentation parameters:\n{best_aug_params}\n")
        perturbed_features = best_features_to_perturb[
            :best_aug_params["max_n_features_to_perturb"]
        ]
        self.logger.info(
            f"Selected features to perturb by index: {perturbed_features}\n"
        )
        self.logger.info(
            f"Selected features to perturb by name: "
            f"{self._get_selected_features_names(perturbed_features, filter_obj)}\n"
        )

        # Cross-validation evaluation
        p_wilcoxon, aug_estimators = self._cv_evaluation(
            dataset, baseline, best_features_to_perturb, best_aug_params, seed=seed
        )
        get_reusable_executor().shutdown(wait=True, kill_workers=True)
        self.logger.info("Done cross-validation evaluation")
        self.logger.info(
            f"The p-value from the Wilcoxon signed-rank test is: {p_wilcoxon}\n"
        )

        if p_wilcoxon >= self.config.p_wilcoxon_threshold:
            if self.config.ignore_filter:
                self.logger.warning(
                    f"No significant improvement in score after augmentation for "
                    f"{dataset.name}. Ignoring filter and proceeding anyways."
                )
                should_proceed = False
            else:
                self.logger.warning(
                    f"No significant improvement in score after augmentation for "
                    f"{dataset.name}."
                )
                return None

        preds = np.column_stack([est.predict(X_test) for est in aug_estimators])
        ensemble_pred = preds.mean(axis=1)
        evaluation_function = baseline._get_evaluation_function(
            self.config.evaluation_metric
        )
        aug_score = evaluation_function(y_test, ensemble_pred)

        # Normalize delta so positive always means improvement
        # R²: higher is better → (aug - baseline) is improvement
        # MSE/RMSE: lower is better → (baseline - aug) is improvement
        if self.config.evaluation_metric == "r2":
            delta_score = 100.0 * (aug_score - score) / abs(score)
        else:
            delta_score = 100.0 * (score - aug_score) / abs(score)

        self.logger.info("Evaluation metrics for the experiment:")
        self.logger.info(f"Original Score: {score}, Augmented Score: {aug_score}\n\n")

        if self.config.save_params:
            params_dir = os.path.join(self.config.experiment_dir, "params")
            os.makedirs(params_dir, exist_ok=True)
            params_file = os.path.join(
                params_dir, f"{dataset.name}_params_seed_{seed}.json"
            )
            with open(params_file, "w") as f:
                json.dump({
                    "best_aug_params": best_aug_params,
                    "model_params": baseline.get_params(),
                }, f)
            self.logger.info(
                f"Best augmentation parameters and model parameters saved to "
                f"{params_file}"
            )

        result = {
            "dataset": dataset.name,
            "seed": seed,
            "score": score,
            "aug_score": aug_score,
            "delta_score": delta_score,
            "p_wilcoxon": p_wilcoxon,
            "should_proceed": should_proceed,
            "features_perturbed": self._get_selected_features_names(
                perturbed_features, filter_obj
            ),
        }

        return result

    def seeding(self, seed: int = None) -> None:
        """Seed all random number generators for reproducibility.

        Sets seeds for Python's random module, NumPy, PyTorch, and CUDA
        operations to ensure reproducible results across multiple runs.

        Args:
            seed: Random seed value. If None, generates a random seed.
        """
        if seed is None:
            seed = random.randint(0, 1000000)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def run(
        self,
        model,
        X_train: np.ndarray = None,
        y_train: np.ndarray = None,
        X_test: np.ndarray = None,
        y_test: np.ndarray = None,
        pretrained: bool = False,
        cat_indices: Optional[Dict[str, List[int]]] = None
    ) -> pd.DataFrame | None:
        """Run the CRDA experiment.

        This method supports three usage modes:

        1. **Standard mode** (model only):
           - Uses dataset from config
           - Preprocesses data (unless config.skip_preprocess=True)
           - Performs train/test split
           - Trains the model on training data
           - Runs full CRDA pipeline

        2. **Pre-split mode** (model + splits, pretrained=False):
           - Uses provided train/test splits
           - Preprocessing controlled by config.skip_preprocess
           - Trains the model on provided training data
           - Runs full CRDA pipeline

        3. **Pre-trained mode** (model + splits, pretrained=True):
           - Uses provided train/test splits as-is (no preprocessing)
           - Skips model training (model must already be fitted)
           - Calculates residuals and continues CRDA pipeline

        Args:
            model: A sklearn-compatible regressor instance. Must implement fit(),
                predict(), get_params(), and set_params() methods.
                If pretrained=True, the model must already be fitted.
            X_train: Training features. Required if pretrained=True.
            y_train: Training targets. Required if pretrained=True.
            X_test: Test features. Required if pretrained=True.
            y_test: Test targets. Required if pretrained=True.
            pretrained: If True, the model is assumed to be already trained.
                When True:
                - All split arrays (X_train, y_train, X_test, y_test) are required
                - No preprocessing is performed on the data
                - Model training is skipped; residuals calculated directly
            cat_indices: Optional dict mapping categorical feature names to their
                column indices in the data. Used for proper intervention handling
                when providing pre-split numpy arrays. Format:
                {"cat_feature_name": [idx1, idx2, ...]} where indices are the
                one-hot encoded column positions.
                If None, all features are treated as continuous.

        Returns:
            DataFrame containing experiment results with columns:
                - dataset: Dataset name
                - seed: Random seed used
                - score: Baseline model test score
                - aug_score: Augmented model test score
                - delta_score: Percent improvement (positive = better)
                - p_wilcoxon: Statistical significance p-value
                - should_proceed: Whether augmentation was beneficial
                - features_perturbed: Names of perturbed features
            Returns None if the experiment could not produce results.

        Raises:
            ValueError: If pretrained=True but split arrays are not provided.
            ValueError: If only some split arrays are provided.
            ValueError: If no dataset and no splits are provided.

        Example:
            Standard mode::

                crda = CRDA(config)
                results = crda.run(XGBRegressor())

            Pre-split mode::

                crda = CRDA(config)
                results = crda.run(XGBRegressor(), X_train, y_train, X_test, y_test)

            Pre-trained mode::

                my_model = XGBRegressor().fit(X_train, y_train)
                crda = CRDA(config)
                results = crda.run(my_model, X_train, y_train, X_test, y_test, pretrained=True)

            With categorical indices::

                cat_indices = {"color": [0, 1, 2], "size": [3, 4, 5]}
                results = crda.run(model, X_train, y_train, X_test, y_test, cat_indices=cat_indices)
        """
        # Validation
        splits_provided = [X_train, y_train, X_test, y_test]
        has_all_splits = all(x is not None for x in splits_provided)
        has_no_splits = all(x is None for x in splits_provided)

        if pretrained:
            if not has_all_splits:
                raise ValueError(
                    "When pretrained=True, all split arrays (X_train, y_train, "
                    "X_test, y_test) must be provided."
                )
        else:
            # Either all or none must be provided
            if not has_all_splits and not has_no_splits:
                raise ValueError(
                    "Either provide all split arrays (X_train, y_train, X_test, y_test) "
                    "or none of them."
                )
            # If no splits provided, must have dataset in config
            if has_no_splits and self.config.dataset is None:
                raise ValueError(
                    "No dataset provided in config and no split arrays provided. "
                    "Either set config.dataset or provide X_train, y_train, X_test, y_test."
                )

        # Store model and create baseline wrapper
        self.model = model
        self.baseline = BaselineRegressor(model)

        # Seed RNGs
        self.seeding(self.config.random_seed)

        # Setup dataset based on mode
        if has_all_splits:
            # Use provided splits
            self.logger.info("Using provided train/test splits")
            self.dataset = AbstractDataset.from_splits(
                name=self.config.dataset_name,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                cat_indices=cat_indices,
                seed=self.config.random_seed
            )
        else:
            # Load from config
            self.logger.info(f"Loading dataset from config: {self.config.dataset_name}")
            self.dataset = AbstractDataset(
                self.config.dataset_name,
                self.config.dataset,
                seed=self.config.random_seed
            )
            if self.dataset.df is not None:
                self.logger.info("Loaded dataset:")
                self.logger.info(self.dataset.df.head())

        # Run the pipeline
        self.logger.info(f"Running experiment on dataset: {self.dataset.name}")

        result = self._run_crda_internal(
            self.dataset,
            self.config.random_seed,
            pretrained=pretrained,
            splits_provided=has_all_splits
        )

        if result is None:
            self.logger.error(
                f"No results were possible for {self.dataset.name} on seed "
                f"{self.config.random_seed}. Please check the data and the logs "
                f"for more information and possibly change the config."
            )
            return None

        # Save results
        self.results_df = pd.DataFrame([result])
        results_file = os.path.join(self.config.experiment_dir, "results.csv")
        self.results_df.to_csv(results_file, index=False)
        self.logger.info(f"All results saved to {results_file}")

        return self.results_df

    def get_augmented_data(
        self,
        model,
        X_train: np.ndarray = None,
        y_train: np.ndarray = None,
        X_test: np.ndarray = None,
        y_test: np.ndarray = None,
        pretrained: bool = False,
        cat_indices: Optional[Dict[str, List[int]]] = None,
        return_combined: bool = False
    ) -> Union[
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ]:
        """Generate augmented data without running the full evaluation pipeline.

        This method runs the augmentation pipeline but skips cross-validation
        evaluation and statistical testing. Use this when you just need the
        augmented training data for your own workflow.

        Pipeline executed:

        1. Preprocess data (unless pretrained=True or config.skip_preprocess=True)
        2. Split data (unless splits provided)
        3. Train model (unless pretrained=True)
        4. Calculate residuals
        5. Filter features for independence
        6. Generate augmented data via intervention + counterfactuals

        Pipeline skipped:

        - Cross-validation evaluation
        - Wilcoxon statistical test
        - Final model scoring

        Args:
            model: A sklearn-compatible regressor instance.
                If pretrained=True, the model must already be fitted.
            X_train: Training features. Required if pretrained=True.
            y_train: Training targets. Required if pretrained=True.
            X_test: Test features. Optional, used for baseline evaluation.
            y_test: Test targets. Optional, used for baseline evaluation.
            pretrained: If True, the model is assumed to be already trained.
                When True:
                - X_train and y_train are required
                - No preprocessing is performed on the data
                - Model training is skipped
            cat_indices: Optional dict mapping categorical feature names to their
                column indices. See run() for format details.
                If None, all features are treated as continuous.
            return_combined: If True, also returns original data combined with augmented.

        Returns:
            If return_combined is False:
                Tuple of (aug_X, aug_y) - the augmented data only.
            If return_combined is True:
                Tuple of (combined_X, combined_y, aug_X, aug_y) where combined
                includes both original training data and augmented data.

        Raises:
            ValueError: If pretrained=True but X_train/y_train not provided.
            ValueError: If no dataset in config and no X_train/y_train provided.

        Example:
            Standard mode::

                crda = CRDA(config)
                aug_X, aug_y = crda.get_augmented_data(XGBRegressor())

            Get combined dataset::

                combined_X, combined_y, aug_X, aug_y = crda.get_augmented_data(
                    XGBRegressor(), return_combined=True
                )

            With pre-trained model::

                my_model = XGBRegressor().fit(X_train, y_train)
                aug_X, aug_y = crda.get_augmented_data(
                    my_model, X_train, y_train, pretrained=True
                )

            With categorical indices::

                cat_indices = {"color": [0, 1, 2]}
                aug_X, aug_y = crda.get_augmented_data(
                    model, X_train, y_train, cat_indices=cat_indices
                )
        """
        # Validation
        if pretrained:
            if X_train is None or y_train is None:
                raise ValueError(
                    "When pretrained=True, X_train and y_train must be provided."
                )

        # Check if we have data
        has_train_data = X_train is not None and y_train is not None
        if not has_train_data and self.config.dataset is None:
            raise ValueError(
                "No dataset provided in config and no X_train/y_train provided. "
                "Either set config.dataset or provide training data."
            )

        # Store model and create baseline wrapper
        self.model = model
        self.baseline = BaselineRegressor(model)

        # Seed RNGs
        self.seeding(self.config.random_seed)

        # Setup dataset based on mode
        if has_train_data:
            # Use provided data
            self.logger.info("Using provided training data")
            # If test data not provided, use training data for test too (just for Filter)
            _X_test = X_test if X_test is not None else X_train
            _y_test = y_test if y_test is not None else y_train

            self.dataset = AbstractDataset.from_splits(
                name=self.config.dataset_name,
                X_train=X_train,
                y_train=y_train,
                X_test=_X_test,
                y_test=_y_test,
                cat_indices=cat_indices,
                seed=self.config.random_seed
            )
        else:
            # Load from config
            self.logger.info(f"Loading dataset from config: {self.config.dataset_name}")
            self.dataset = AbstractDataset(
                self.config.dataset_name,
                self.config.dataset,
                seed=self.config.random_seed
            )
            # Preprocess
            self.dataset.preprocess(skip=self.config.skip_preprocess)
            # Split
            self.dataset.split(test_size=self.config.test_size, seed=self.config.random_seed)

        # Get baseline - either use pretrained or train new
        if pretrained:
            baseline = self.baseline
            self.logger.info("Using pre-trained model")
        else:
            baseline = self.baseline.clone()
            if self._model_supports_random_state(baseline.model):
                baseline.set_params(random_state=self.config.random_seed)
            # Train baseline
            self._train_baseline(self.dataset, baseline)
            self.logger.info("Done training baseline model")

        # Calculate residuals
        self.dataset.add_residuals(baseline)
        self.logger.info("Done calculating residuals\n")

        # Filter for features to perturb
        filter_obj = Filter(self.dataset, self.config, self.logger)
        self.logger.info("Filtering for features to perturb")
        best_features_to_perturb = filter_obj.run_checks()

        if best_features_to_perturb is None:
            raise ValueError(
                f"No candidate features found for {self.dataset.name}. "
            )

        self.logger.info("Done filtering.\n")

        # Get augmentation parameters
        best_aug_params = {
            "max_n_features_to_perturb": self.config.max_n_features_to_perturb,
            "aug_data_size_factor": self.config.aug_data_size_factor,
            "min_perturb_percent": self.config.min_perturb_percent,
            "max_perturb_percent": self.config.max_perturb_percent,
        }

        # Generate augmented data
        aug_X, aug_y = self._data_augmentation(
            self.dataset.X_train,
            self.dataset.y_train,
            self.dataset.train_residuals,
            baseline,
            best_features_to_perturb,
            **best_aug_params
        )

        self.logger.info(f"Generated {aug_X.shape[0]} augmented samples")

        if return_combined:
            combined_X, combined_y = self._get_combined_aug_training_set(
                self.dataset.X_train,
                self.dataset.y_train,
                aug_X,
                aug_y
            )
            return combined_X, combined_y, aug_X, aug_y
        else:
            return aug_X, aug_y
