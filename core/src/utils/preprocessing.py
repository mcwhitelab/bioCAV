"""Preprocessing utilities for embeddings (standardization)"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def create_scaler_and_pca(
    X_train: np.ndarray,
) -> Tuple[StandardScaler, dict]:
    """
    Fit a StandardScaler on training data.

    Args:
        X_train: Training data (n_samples, n_features)

    Returns:
        Tuple of (scaler, metadata_dict)
    """
    scaler = StandardScaler()
    scaler.fit_transform(X_train)

    metadata = {
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_std': scaler.scale_.tolist(),
        'input_dim': X_train.shape[1],
        'output_dim': X_train.shape[1],
    }

    return scaler, metadata


def preprocess_embeddings(
    X: np.ndarray,
    scaler: StandardScaler,
) -> np.ndarray:
    """Apply scaler to embeddings."""
    return scaler.transform(X)


def validate_preprocessing_compatibility(
    X: np.ndarray,
    scaler: StandardScaler,
) -> None:
    """Raise ValueError if X's feature count doesn't match the scaler."""
    expected_dim = len(scaler.mean_)
    actual_dim = X.shape[1]
    if actual_dim != expected_dim:
        raise ValueError(
            f"Dimension mismatch: scaler expects {expected_dim} features, "
            f"data has {actual_dim}."
        )

