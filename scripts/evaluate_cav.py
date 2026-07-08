#!/usr/bin/env python3
"""
Evaluate a trained CAV on held-out positive and negative test sets.

Loads sequence_embeddings from pos/neg pkl files, projects them onto the
trained CAV direction, and reports AUROC, AUPRC, and an optimal threshold.

Use this after train_cav_from_embeddings.py to get unbiased classification
metrics on a test set the CAV never saw during training.

Examples
--------
python evaluate_cav.py \
    --pos temporal_positive_test.fasta.pkl \
    --neg temporal_negative_test.fasta.pkl \
    --cav-dir ./cavs/GO_0005887/ \
    --out ./cavs/GO_0005887/test_eval.json
"""

import sys
import json
import argparse
import logging
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from src.evaluate import load_cav_artifacts, compute_projections, \
    evaluate_projection_performance, select_optimal_threshold
from src.utils.data_loader import load_sequence_embeddings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_vectors(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix == '.npy':
        arr = np.load(p)
        logger.info(f"Loaded {arr.shape} from {p}")
        return arr
    arr, ids = load_sequence_embeddings(path)
    logger.info(f"Loaded {arr.shape} from {p} ({len(ids)} samples)")
    return arr


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained CAV on a held-out test set"
    )
    parser.add_argument('--pos', required=True,
                        help='Positive test embeddings (.pkl or .npy)')
    parser.add_argument('--neg', required=True,
                        help='Negative test embeddings (.pkl or .npy)')
    parser.add_argument('--cav-dir', required=True,
                        help='Directory containing trained CAV artifacts')
    parser.add_argument('--out',
                        help='Optional path to save metrics JSON')
    parser.add_argument('--version', default='v1',
                        help='CAV artifact version (default: v1)')
    parser.add_argument('--threshold-method', default='f1_max',
                        choices=['f1_max', 'precision_at_recall_90', 'fpr_0.05'],
                        help='Threshold selection method (default: f1_max)')
    args = parser.parse_args()

    pos = load_vectors(args.pos)
    neg = load_vectors(args.neg)

    X = np.vstack([pos, neg])
    y = np.hstack([np.ones(len(pos)), np.zeros(len(neg))])

    artifacts = load_cav_artifacts(args.cav_dir, version=args.version)

    scores = compute_projections(
        X,
        artifacts['concept_cav'],
        artifacts['scaler'],
    )

    metrics = evaluate_projection_performance(y, scores)
    threshold, threshold_meta = select_optimal_threshold(
        y, scores, method=args.threshold_method
    )
    metrics['threshold'] = threshold
    metrics['threshold_metadata'] = threshold_meta
    metrics['n_pos'] = int(len(pos))
    metrics['n_neg'] = int(len(neg))

    logger.info(
        f"Test AUROC: {metrics['auroc']:.3f}  "
        f"AUPRC: {metrics['auprc']:.3f}  "
        f"Threshold ({args.threshold_method}): {threshold:.4f}"
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Strip non-serialisable curve arrays before saving
        save_metrics = {k: v for k, v in metrics.items()
                        if k not in ('fpr', 'tpr', 'precision', 'recall',
                                     'thresholds_roc', 'thresholds_pr')}
        with open(out_path, 'w') as f:
            json.dump(save_metrics, f, indent=2)
        logger.info(f"Saved metrics to {out_path}")


if __name__ == '__main__':
    main()
