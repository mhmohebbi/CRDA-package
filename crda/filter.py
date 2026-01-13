"""
Feature filtering for causal data augmentation.

This module implements statistical tests to identify features that are
independent of model residuals, which can be safely perturbed for data
augmentation without corrupting the underlying data generating process.

The Filter class uses:
- Distance correlation (Dcorr) for continuous features
- Mutual information with permutation testing for categorical features

Features that pass the independence test are ranked by their p-values,
with higher p-values indicating stronger independence from residuals.

Example:
    Basic usage::

        from crda.filter import Filter
        from crda.dataset import AbstractDataset
        from crda.utils.config import Config
        from crda.utils.logger import Logger

        dataset = AbstractDataset("data", "data.csv")
        dataset.preprocess()
        dataset.split()
        # ... compute residuals ...

        filter = Filter(dataset, config, logger)
        candidate_features = filter.run_checks()
"""

from sklearn.feature_selection import mutual_info_regression
from hyppo.independence import Dcorr
import numpy as np
import pandas as pd
from crda.utils.logger import Logger
from crda.dataset import AbstractDataset
from crda.utils.config import Config


class Filter:
    """Feature filter for identifying perturbable features in CRDA.

    Identifies features that are statistically independent of model residuals
    using appropriate tests for continuous and categorical features.

    The filter works by testing each feature's independence from the residuals:
    - Features independent of residuals can be safely perturbed
    - The causal reasoning ensures perturbations don't corrupt the data
      generating process

    Args:
        dataset: Preprocessed AbstractDataset containing features, target,
            and computed residuals.
        config: Configuration object with experiment parameters including
            the independence test threshold.
        logger: Logger instance for recording test results and warnings.

    Attributes:
        dataset: The dataset being filtered.
        config: Configuration parameters.
        logger: Logger for output.
        target: Training target values.
        features: Feature matrix with residuals appended as last column.
        feature_names: List of feature column names.
        feature_indices: Mapping from feature names to indices.
        feature_index_to_name: Reverse mapping from indices to names.
        df: DataFrame containing features, residuals (Z), and target.
        cat_feature_names: Names of categorical features.
        numerical_feature_names: Names of numerical features.

    Example:
        >>> filter = Filter(dataset, config, logger)
        >>> features_to_perturb = filter.run_checks()
        >>> print(f"Found {len(features_to_perturb)} perturbable features")
    """

    def __init__(
        self,
        dataset: AbstractDataset,
        config: Config,
        logger: Logger
    ):
        """Initialize the Filter for feature selection.

        Args:
            dataset: Preprocessed dataset containing features, target, and
                residuals. Must have train_residuals computed.
            config: Configuration object containing experiment parameters
                and thresholds (particularly indep_test_threshold).
            logger: Logger instance for recording information and warnings.
        """
        self.dataset = dataset
        self.config = config
        self.logger = logger

        self.target = dataset.y_train
        self.features = np.hstack((
            dataset.X_train,
            dataset.train_residuals.reshape(-1, 1)
        ))

        self.feature_names = list(dataset.X.columns)
        self.feature_indices = {}
        for i in range(len(self.feature_names)):
            self.feature_indices[self.feature_names[i]] = i

        self.feature_index_to_name = {v: k for k, v in self.feature_indices.items()}

        self.df = pd.DataFrame(self.features, columns=self.feature_names + ['Z'])
        self.df['target'] = self.target

        self.cat_feature_names = [
            self.feature_names[idx] for idx in self.dataset.all_cat_col_indices
        ]
        self.numerical_feature_names = [
            col for col in self.feature_names if col not in self.cat_feature_names
        ]

    def _test_independence(
        self,
        X: np.ndarray,
        Z: np.ndarray,
        feature_type: str = 'continuous'
    ) -> tuple[bool, float]:
        """Test statistical independence between a feature and residuals.

        Uses different statistical tests depending on feature type:
        - Continuous features: Distance correlation (Dcorr) with permutation test
        - Categorical features: Mutual information with permutation test

        Args:
            X: Feature values array of shape (n_samples,).
            Z: Residual values array of shape (n_samples,).
            feature_type: Type of feature being tested. Either 'continuous'
                for numerical features or 'categorical' for encoded features.

        Returns:
            Tuple containing:
                - is_independent: True if the feature is statistically
                  independent of residuals (p-value >= threshold).
                - p_value: The p-value from the independence test. Higher
                  values indicate stronger evidence of independence.
        """
        X = X.reshape(-1, 1)

        if feature_type == 'continuous':
            # Use distance correlation (non-linear dependence detection)
            stat, p_value = Dcorr().test(X, Z, reps=1000)
            return p_value >= self.config.indep_test_threshold, p_value
        else:
            # Use mutual information with permutation testing for categorical
            observed_mi = mutual_info_regression(X, Z, discrete_features=[True])[0]

            n_permutations = 1000
            null_mi_scores = []
            Z_shuffled = np.copy(Z)

            for _ in range(n_permutations):
                np.random.shuffle(Z_shuffled)
                mi = mutual_info_regression(X, Z_shuffled, discrete_features=[True])[0]
                null_mi_scores.append(mi)

            p_value = (
                (np.sum(np.array(null_mi_scores) >= observed_mi) + 1)
                / (n_permutations + 1)
            )

            return p_value >= self.config.indep_test_threshold, p_value

    def run_checks(self) -> list[int] | None:
        """Identify features that are statistically independent of residuals.

        Tests independence for both continuous and categorical features:
        - Continuous features: Uses distance correlation test with 1000 permutations
        - Categorical features: Uses mutual information with permutation testing

        Features that pass the independence test (p-value >= threshold) are
        ranked by their p-values, with higher p-values indicating stronger
        independence and thus safer candidates for perturbation.

        Returns:
            List of feature indices ranked by independence (most independent
            first). For categorical features, returns a list of indices for
            the one-hot encoded columns. Returns None if no features are
            independent of the residuals.
        """
        candidates = []
        numerical_df = self.df.drop(columns=self.cat_feature_names)

        # Test continuous features
        for feature in self.numerical_feature_names:
            is_independent, p_value = self._test_independence(
                numerical_df[feature].values,
                numerical_df['Z'].values,
                feature_type='continuous'
            )
            self.logger.info(
                f"Continuous feature {feature} is "
                f"{'NOT ' if not is_independent else ''}independent of Z, "
                f"p-value: {p_value}"
            )
            if is_independent:
                candidates.append({
                    "feature_name": feature,
                    "p_value": p_value,
                    "type": "continuous"
                })

        # Test categorical features
        for cat_feature_name, col_names in self.dataset.cat_cols_dict.items():
            cat_feature_df = self.df[col_names]
            one_hot_col = cat_feature_df.values
            categorical_values = np.argmax(one_hot_col, axis=1)
            is_independent, p_value = self._test_independence(
                categorical_values,
                self.df['Z'].values,
                feature_type='categorical'
            )
            self.logger.info(
                f"Categorical feature {cat_feature_name} is "
                f"{'NOT ' if not is_independent else ''}independent of Z, "
                f"p-value: {p_value}"
            )
            if is_independent:
                candidates.append({
                    "feature_name": cat_feature_name,
                    "p_value": p_value,
                    "type": "categorical"
                })

        if len(candidates) == 0:
            self.logger.warning("No features are independent of Z.")
            return None

        # Sort by p-value descending (most independent first)
        candidates_sorted = sorted(
            candidates, key=lambda d: d["p_value"], reverse=True
        )
        self.logger.info(
            f"Ranked candidates by independence: "
            f"{[c['feature_name'] for c in candidates_sorted]}"
        )

        return [
            self.dataset.cat_cols_index_dict[c['feature_name']]
            if c['type'] == 'categorical'
            else self.feature_indices[c['feature_name']]
            for c in candidates_sorted
        ]
