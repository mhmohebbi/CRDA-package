"""
Configuration management for CRDA experiments.

This module provides the Config class for managing experiment parameters,
including dataset settings, augmentation hyperparameters, evaluation metrics,
and output options.

The Config class automatically generates unique experiment names and creates
the necessary directory structure for storing results.

Example:
    Basic configuration::

        from crda.utils.config import Config

        config = Config(
            dataset="path/to/data.csv",
            dataset_name="my_dataset",
            random_seed=42,
            verbose=True
        )

    With augmentation tuning::

        config = Config(
            dataset=df,
            dataset_name="my_data",
            crda_param_tune=True,
            aug_data_size_factor=1.5,
            max_n_features_to_perturb=3
        )

    Minimal config (when providing splits directly to run())::

        config = Config(evaluation_metric="rmse", random_seed=42)
        # dataset not needed when providing splits to run()
"""

import os
from typing import Union, Dict, Any
import time
import pandas as pd


class Config:
    """Configuration class for CRDA experiments.

    Manages all experiment parameters and automatically generates unique
    experiment names and directory structures for result storage.

    Args:
        dataset: Path to a data file (CSV, Excel, JSON, pickle) or a pandas
            DataFrame. The last column is assumed to be the target variable.
            Optional if providing splits directly to run() method.
        dataset_name: Human-readable name for the dataset. Used in logging,
            result file naming, and experiment directory naming.
            Defaults to "my_dataset".
        skip_preprocess: If True, skip all preprocessing steps (cleaning,
            encoding, standardization) and use data as-is. Use this when
            providing already preprocessed data. Defaults to False.
        evaluation_metric: Metric for model evaluation. Options: "mse" (Mean
            Squared Error), "rmse" (Root MSE), "r2" (R-squared). Defaults to "mse".
        aug_data_size_factor: Multiplier for augmented data size relative to
            original. 1.0 generates same number of augmented samples as original.
            Defaults to 1.0.
        max_n_features_to_perturb: Maximum number of features to perturb during
            augmentation. Defaults to 5.
        max_perturb_percent: Maximum percentage for feature perturbation
            (e.g., 0.1 means +10%). Defaults to 0.1.
        min_perturb_percent: Minimum percentage for feature perturbation
            (e.g., -0.1 means -10%). Defaults to -0.1.
        test_size: Proportion of data for test split. Defaults to 0.2.
        random_seed: Random seed for reproducibility. Defaults to 0.
        p_wilcoxon_threshold: Significance threshold for Wilcoxon test.
            Defaults to 0.05.
        indep_test_threshold: Significance threshold for feature-residual
            independence test. Defaults to 0.05.
        crda_param_tune: If True, uses Optuna to tune augmentation parameters.
            Defaults to False.
        ignore_filter: If True, proceeds with augmentation even if CRDA seems to harm. Defaults to False.
        results_dir: Directory for storing experiment results. Defaults to "./runs".
        save_models: If True, saves trained model artifacts. Defaults to False.
        save_params: If True, saves optimized parameters. Defaults to False.
        verbose: If True, prints logs to console. Defaults to False.
        log_file: Path for log file output. If None, no file logging.
            Defaults to None.
        **kwargs: Additional parameters stored as instance attributes.

    Attributes:
        timestamp: Timestamp when config was created (format: YYYYMMDD-HHMMSS).
        experiment_name: Unique experiment identifier ({dataset_name}_{timestamp}).
        experiment_dir: Full path to experiment output directory.
        skip_preprocess: Whether to skip preprocessing.

    Example:
        >>> config = Config(
        ...     dataset="housing.csv",
        ...     dataset_name="housing",
        ...     random_seed=42,
        ...     crda_param_tune=True
        ... )
        >>> print(config.experiment_dir)
        ./runs/housing_20260112-143052
    """

    def __init__(
        self,
        # Dataset parameters
        dataset: Union[str, pd.DataFrame] = None,
        dataset_name: str = "my_dataset",
        skip_preprocess: bool = False,

        # Evaluation parameters
        evaluation_metric: str = "mse",

        # Data augmentation parameters
        aug_data_size_factor: float = 1.0,
        max_n_features_to_perturb: int = 5,
        max_perturb_percent: float = 0.1,
        min_perturb_percent: float = -0.1,

        # Experiment parameters
        test_size: float = 0.2,
        random_seed: int = 0,
        p_wilcoxon_threshold: float = 0.05,
        indep_test_threshold: float = 0.05,
        crda_param_tune: bool = False,
        ignore_filter: bool = False,

        # Output parameters
        results_dir: str = "./runs",
        save_models: bool = False,
        save_params: bool = False,
        verbose: bool = False,
        log_file: str = None,

        # Additional parameters
        **kwargs: Any
    ):
        """Initialize the configuration with experiment parameters.

        Args:
            dataset: Path to data file or pandas DataFrame. Optional if providing
                splits directly to run() method.
            dataset_name: Human-readable name for the dataset. Defaults to "my_dataset" if not provided.
            skip_preprocess: If True, skip preprocessing and use data as-is.
            evaluation_metric: Metric for evaluation ("mse", "rmse", "r2").
            aug_data_size_factor: Multiplier for augmented data size.
            max_n_features_to_perturb: Maximum features to perturb.
            max_perturb_percent: Maximum perturbation percentage.
            min_perturb_percent: Minimum perturbation percentage.
            test_size: Test split proportion.
            random_seed: Random seed for reproducibility.
            p_wilcoxon_threshold: Wilcoxon test threshold.
            indep_test_threshold: Independence test threshold.
            crda_param_tune: Enable Optuna parameter tuning.
            ignore_filter: Proceed even if CRDA seems to harm.
            results_dir: Output directory for results.
            save_models: Save trained models.
            save_params: Save optimized parameters.
            verbose: Enable console logging.
            log_file: Path for file logging.
            **kwargs: Additional parameters as attributes.
        """
        # Dataset parameters
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.skip_preprocess = skip_preprocess

        # Generate unique experiment identifier
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.experiment_name = f"{self.dataset_name}_{self.timestamp}"

        # Evaluation parameters
        self.evaluation_metric = evaluation_metric

        # Data augmentation parameters
        self.aug_data_size_factor = aug_data_size_factor
        self.max_n_features_to_perturb = max_n_features_to_perturb
        self.max_perturb_percent = max_perturb_percent
        self.min_perturb_percent = min_perturb_percent

        # Experiment parameters
        self.test_size = test_size
        self.random_seed = random_seed
        self.p_wilcoxon_threshold = p_wilcoxon_threshold
        self.indep_test_threshold = indep_test_threshold
        self.crda_param_tune = crda_param_tune
        self.ignore_filter = ignore_filter

        # Output parameters
        self.results_dir = results_dir
        self.save_models = save_models
        self.save_params = save_params
        self.verbose = verbose
        self.log_file = log_file

        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)

        # Create experiment directory
        self.experiment_dir = self.get_experiment_dir()

        # Store additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_experiment_dir(self) -> str:
        """Get the directory path for this experiment's results.

        Creates the directory if it doesn't exist.

        Returns:
            Full path to the experiment directory.
        """
        exp_dir = os.path.join(self.results_dir, f"{self.experiment_name}")
        os.makedirs(exp_dir, exist_ok=True)
        return exp_dir

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary.

        Useful for serialization and saving configuration to JSON.

        Returns:
            Dictionary with all configuration parameters.
        """
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create a Config instance from a dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters.

        Returns:
            A new Config instance with the specified parameters.
        """
        config = cls(**config_dict)
        return config

    def __str__(self) -> str:
        """Return a human-readable string representation.

        Returns:
            Formatted string showing experiment name and all parameters.
        """
        return f"Experiment Config (Name: {self.experiment_name})\n" + "\n".join(
            f"  {k}: {v}" for k, v in self.__dict__.items() if k != 'experiment_name'
        )
