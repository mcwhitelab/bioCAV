"""
Evaluation module with threshold registry.
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Any
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    roc_auc_score, average_precision_score
)
import scipy.stats as stats

from .utils.visualization import (
    plot_roc_pr_curves,
    plot_random_cav_comparison,
    plot_threshold_selection,
)
from .utils.preprocessing import preprocess_embeddings

logger = logging.getLogger(__name__)


def load_cav_artifacts(
    cav_dir: str,
    version: str = "v1"
) -> Dict:
    """Load all CAV artifacts from a directory."""
    import joblib

    cav_path = Path(cav_dir)

    concept_cav = np.load(cav_path / f"concept_{version}.npy")

    random_cavs = []
    for random_file in sorted(cav_path.glob(f"random_*_{version}.npy")):
        random_cavs.append(np.load(random_file))

    scaler = joblib.load(cav_path / f"scaler_{version}.pkl")

    with open(cav_path / f"report_{version}.json", 'r') as f:
        report = json.load(f)

    logger.info(f"Loaded CAV artifacts: concept + {len(random_cavs)} random CAVs")

    return {
        'concept_cav': concept_cav,
        'random_cavs': random_cavs,
        'scaler': scaler,
        'report': report,
    }


def compute_projections(
    embeddings: np.ndarray,
    cav: np.ndarray,
    scaler: Any,
) -> np.ndarray:
    """Compute CAV projections for embeddings."""
    X_preprocessed = preprocess_embeddings(embeddings, scaler)
    return X_preprocessed @ cav


def evaluate_projection_performance(
    y_true: np.ndarray,
    y_scores: np.ndarray
) -> Dict:
    """
    Compute classification metrics from projection scores.

    Args:
        y_true: True labels
        y_scores: Projection scores

    Returns:
        Dictionary of metrics
    """
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)
    auroc = auc(fpr, tpr)

    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_scores)
    auprc = auc(recall, precision)

    return {
        'auroc': float(auroc),
        'auprc': float(auprc),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'thresholds_roc': thresholds_roc.tolist(),
        'thresholds_pr': thresholds_pr.tolist()
    }


def select_optimal_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    method: str = "f1_max"
) -> Tuple[float, Dict]:
    """
    Select optimal threshold based on method.

    Args:
        y_true: True labels
        y_scores: Prediction scores
        method: "f1_max" | "precision_at_recall_90" | "fpr_0.05"

    Returns:
        Tuple of (threshold, metadata_dict)
    """
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_scores)
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)

    if method == "f1_max":
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (
            precision[:-1] + recall[:-1] + 1e-10
        )
        best_idx = np.argmax(f1_scores)
        threshold = thresholds_pr[best_idx]
        metadata = {
            'method': 'f1_max',
            'f1_score': float(f1_scores[best_idx]),
            'precision': float(precision[best_idx]),
            'recall': float(recall[best_idx])
        }

    elif method == "precision_at_recall_90":
        valid_idx = recall[:-1] >= 0.90
        if valid_idx.any():
            idx = np.where(valid_idx)[0][0]
            threshold = thresholds_pr[idx]
        else:
            f1_scores = 2 * (precision[:-1] * recall[:-1]) / (
                precision[:-1] + recall[:-1] + 1e-10
            )
            idx = np.argmax(f1_scores)
            threshold = thresholds_pr[idx]
            logger.warning("Could not achieve recall >= 0.90, using max F1")
        metadata = {
            'method': 'precision_at_recall_90',
            'precision': float(precision[idx]),
            'recall': float(recall[idx])
        }

    elif method == "fpr_0.05":
        valid_idx = fpr <= 0.05
        if valid_idx.any():
            idx = np.where(valid_idx)[0][-1]
            threshold = thresholds_roc[idx]
        else:
            idx = 0
            threshold = thresholds_roc[idx]
            logger.warning("Could not achieve FPR <= 0.05, using strictest threshold")
        metadata = {
            'method': 'fpr_0.05',
            'fpr': float(fpr[idx]),
            'tpr': float(tpr[idx])
        }

    else:
        raise ValueError(f"Unknown threshold method: {method}")

    logger.info(f"Selected threshold: {threshold:.4f} (method: {method})")
    return float(threshold), metadata


def compare_with_random_cavs(
    concept_auroc: float,
    random_aurocs,
    alpha: float = 0.05
) -> Dict:
    """
    Statistical comparison of concept CAV vs random CAVs.

    Args:
        concept_auroc: AUROC of concept CAV
        random_aurocs: AUROCs of random CAVs
        alpha: Significance level

    Returns:
        Dictionary of comparison statistics
    """
    random_mean = np.mean(random_aurocs)
    random_std = np.std(random_aurocs)
    z_score = (concept_auroc - random_mean) / (random_std + 1e-10)
    p_value = 1 - stats.norm.cdf(z_score)
    threshold_95 = np.percentile(random_aurocs, 95)
    is_significant = concept_auroc > threshold_95

    return {
        'concept_auroc': float(concept_auroc),
        'random_auroc_mean': float(random_mean),
        'random_auroc_std': float(random_std),
        'random_auroc_min': float(np.min(random_aurocs)),
        'random_auroc_max': float(np.max(random_aurocs)),
        'z_score': float(z_score),
        'p_value': float(p_value),
        'threshold_95': float(threshold_95),
        'is_significant': bool(is_significant),
        'alpha': alpha
    }


def evaluate_cav(
    embed_dir: str,
    cav_dir: str,
    output_dir: str,
    artifact_version: str = "v1",
    threshold_method: str = "f1_max"
) -> Dict:
    """
    Evaluate a trained CAV against its training embeddings.

    Args:
        embed_dir: Directory with pos.npy and neg.npy
        cav_dir: CAV artifacts directory
        output_dir: Directory for evaluation outputs
        artifact_version: CAV version string
        threshold_method: Threshold selection method

    Returns:
        Dictionary of evaluation metrics
    """
    from .train_cav import load_embeddings

    X, y = load_embeddings(embed_dir)
    artifacts = load_cav_artifacts(cav_dir, artifact_version)

    concept_projections = compute_projections(
        X,
        artifacts['concept_cav'],
        artifacts['scaler'],
    )

    metrics = evaluate_projection_performance(y, concept_projections)

    threshold, threshold_meta = select_optimal_threshold(
        y, concept_projections, method=threshold_method
    )
    metrics['threshold'] = threshold
    metrics['threshold_metadata'] = threshold_meta

    random_aurocs = [
        roc_auc_score(y, compute_projections(X, rc, artifacts['scaler']))
        for rc in artifacts['random_cavs']
    ]

    if random_aurocs:
        metrics['random_cav_comparison'] = compare_with_random_cavs(
            metrics['auroc'], random_aurocs
        )

    logger.info(
        f"Performance: AUROC={metrics['auroc']:.3f}, "
        f"AUPRC={metrics['auprc']:.3f}, Threshold={threshold:.4f}"
    )

    return metrics


def save_evaluation_results(metrics: Dict, output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_file = output_path / "eval.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Saved evaluation metrics: {metrics_file}")


def save_threshold_registry(metrics: Dict, output_path: str) -> None:
    """
    Save threshold for use in detection.

    Args:
        metrics: Evaluation metrics dict (from evaluate_cav)
        output_path: Path to save thresholds.json
    """
    registry = {
        'threshold': metrics['threshold'],
        'method': metrics['threshold_metadata']['method'],
        'auroc': metrics['auroc'],
        'auprc': metrics['auprc']
    }
    registry.update(metrics['threshold_metadata'])

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(registry, f, indent=2)

    logger.info(f"Saved threshold registry: {output_file}")
