"""
Attribution module: score each token position (residue / gene) against a trained CAV.

Given token-level embeddings (aa_embeddings) for one or more samples and a trained
CAV, compute a scalar attribution score per position by projecting each token's
embedding onto the CAV direction (after applying the saved scaler/PCA).

Higher scores indicate positions whose embedding most strongly aligns with the
concept direction.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

from .utils.preprocessing import preprocess_embeddings

logger = logging.getLogger(__name__)


def compute_token_attributions(
    aa_embeddings: np.ndarray,
    cav_artifacts: Dict,
    seq_len: Optional[int] = None
) -> np.ndarray:
    """
    Project each token position onto the CAV direction.

    Args:
        aa_embeddings: Token embeddings for one sample, shape (max_seq_len, hidden_dim).
                       If padded, pass seq_len to restrict scoring to real tokens.
        cav_artifacts: Dict from evaluate.load_cav_artifacts(), containing
                       'concept_cav' and 'scaler'.
        seq_len:       Number of real (non-padded) tokens. Defaults to full array.

    Returns:
        1-D array of shape (seq_len,) with one projection score per position.
    """
    if seq_len is None:
        seq_len = aa_embeddings.shape[0]

    tokens = aa_embeddings[:seq_len]  # (seq_len, hidden_dim)

    preprocessed = preprocess_embeddings(
        tokens,
        cav_artifacts['scaler'],
    )

    scores = preprocessed @ cav_artifacts['concept_cav']
    return scores


def compute_batch_attributions(
    aa_embeddings: np.ndarray,
    sample_ids: List[str],
    cav_artifacts: Dict,
    seq_lens: Optional[List[int]] = None
) -> List[Tuple[str, np.ndarray]]:
    """
    Compute per-position attributions for a batch of samples.

    Args:
        aa_embeddings: Shape (n_samples, max_seq_len, hidden_dim).
        sample_ids:    List of sample identifiers (length n_samples).
        cav_artifacts: Dict from evaluate.load_cav_artifacts().
        seq_lens:      Optional list of actual sequence lengths per sample.
                       If None, uses the full max_seq_len for all samples.

    Returns:
        List of (sample_id, scores_array) tuples.
    """
    results = []
    for i, sid in enumerate(sample_ids):
        sl = seq_lens[i] if seq_lens is not None else None
        scores = compute_token_attributions(aa_embeddings[i], cav_artifacts, seq_len=sl)
        results.append((sid, scores))
    return results
