"""
Detection module: apply trained CAVs to pre-computed embeddings.

Two scan modes:
  - sliding_window_scan: for motifs / binding sites (span-based)
  - per_position_scan:   for PTMs (single-position scoring)
  - sequence_scan:       for whole-sequence embeddings (single score per sample)
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from .evaluate import load_cav_artifacts, compute_projections

logger = logging.getLogger(__name__)


def sliding_window_scan(
    aa_embeddings: np.ndarray,
    cav_artifacts: Dict,
    window_size: int,
    stride: int = 1,
    seq_len: Optional[int] = None
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Scan token-level embeddings with a sliding window and compute CAV projections.

    Args:
        aa_embeddings: Token embeddings (seq_len, hidden_dim). If padded, pass seq_len.
        cav_artifacts: CAV artifacts dict from load_cav_artifacts()
        window_size: Number of positions per window
        stride: Stride between windows
        seq_len: Actual sequence length if aa_embeddings is padded (default: full array)

    Returns:
        Tuple of (projection_scores array, list of (start, end) spans)
    """
    if seq_len is None:
        seq_len = aa_embeddings.shape[0]

    projection_scores = []
    window_spans = []

    for start in range(0, seq_len - window_size + 1, stride):
        end = start + window_size
        pooled = aa_embeddings[start:end].mean(axis=0, keepdims=True)  # (1, hidden_dim)
        score = compute_projections(
            pooled,
            cav_artifacts['concept_cav'],
            cav_artifacts['scaler'],
        )[0]
        projection_scores.append(score)
        window_spans.append((start, end))

    return np.array(projection_scores), window_spans


def per_position_scan(
    aa_embeddings: np.ndarray,
    cav_artifacts: Dict,
    seq_len: Optional[int] = None
) -> np.ndarray:
    """
    Score each individual token position against the CAV.
    For PTM detection or any single-residue scoring task.

    Args:
        aa_embeddings: Token embeddings (seq_len, hidden_dim). If padded, pass seq_len.
        cav_artifacts: CAV artifacts dict from load_cav_artifacts()
        seq_len: Actual sequence length if aa_embeddings is padded

    Returns:
        Projection scores array of shape (seq_len,)
    """
    if seq_len is None:
        seq_len = aa_embeddings.shape[0]

    tokens = aa_embeddings[:seq_len]  # (seq_len, hidden_dim)
    scores = compute_projections(
        tokens,
        cav_artifacts['concept_cav'],
        cav_artifacts['scaler'],
    )
    return scores


def sequence_scan(
    sequence_embeddings: np.ndarray,
    cav_artifacts: Dict
) -> np.ndarray:
    """
    Score whole-sequence (or cell-level) embeddings against the CAV.

    Args:
        sequence_embeddings: Shape (n_samples, hidden_dim)
        cav_artifacts: CAV artifacts dict from load_cav_artifacts()

    Returns:
        Projection scores array of shape (n_samples,)
    """
    return compute_projections(
        sequence_embeddings,
        cav_artifacts['concept_cav'],
        cav_artifacts['scaler'],
    )


def load_threshold(threshold_path: str) -> Dict:
    """Load threshold registry saved by save_threshold_registry()."""
    with open(threshold_path, 'r') as f:
        return json.load(f)
