#!/usr/bin/env python3
"""
Score unlabeled samples against one or more trained CAVs.

Loads sequence_embeddings from a pkl, projects each sample onto each
CAV direction (scaler -> PCA -> dot product), and writes a TSV with
one row per sample and one score column per CAV.

Usage
-----
python specific_scripts/score_samples_against_cavs.py \
    --pkl   embeddings/scrnaseq/tumor_GSE226870.pkl \
    --cavs  cavs/C1_menstrual cavs/C1_prolif cavs/C1_secretory \
    --names menstrual proliferative secretory \
    --out   results/tumor_cav_scores.tsv
"""

import sys
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Archived: not called by any live pipeline. Also predates the scaler-only
# simplification of compute_projections()/load_cav_artifacts() -- the
# artifacts['pca'] call below is broken (that key no longer exists) and
# was not fixed, since this script isn't in use.
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "core"))

from src.evaluate import load_cav_artifacts, compute_projections
from src.utils.data_loader import load_sequence_embeddings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def softmax(X):
    """Row-wise softmax so scores across CAVs sum to 1 per sample."""
    e = np.exp(X - X.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser(
        description="Project samples onto trained CAV directions and output scores.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--pkl', required=True,
                        help='Embedding pkl (sequence_embeddings for all samples).')
    parser.add_argument('--cavs', nargs='+', required=True,
                        help='One or more CAV artifact directories.')
    parser.add_argument('--names', nargs='+',
                        help='Column names for each CAV (same order as --cavs). '
                             'Defaults to the directory name of each --cavs entry.')
    parser.add_argument('--out', required=True,
                        help='Output TSV path.')
    parser.add_argument('--version', default='v1',
                        help='CAV artifact version suffix (default: v1).')
    parser.add_argument('--softmax', action='store_true',
                        help='Apply softmax across CAV scores so they sum to 1 '
                             'per sample. Useful for comparing relative phase '
                             'alignment when CAV scales differ.')
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Validate names
    # ------------------------------------------------------------------ #
    if args.names:
        if len(args.names) != len(args.cavs):
            raise ValueError(
                f"--names has {len(args.names)} entries but --cavs has "
                f"{len(args.cavs)}. They must match."
            )
        names = args.names
    else:
        names = [Path(c).name for c in args.cavs]

    # ------------------------------------------------------------------ #
    # Load embeddings
    # ------------------------------------------------------------------ #
    logger.info(f"Loading embeddings from {args.pkl}")
    seq_embs, sample_ids = load_sequence_embeddings(args.pkl)
    logger.info(f"  {len(sample_ids)} samples, hidden_dim={seq_embs.shape[1]}")

    # ------------------------------------------------------------------ #
    # Score each CAV
    # ------------------------------------------------------------------ #
    score_cols = {}
    for name, cav_dir in zip(names, args.cavs):
        logger.info(f"Scoring CAV: {name} ({cav_dir})")
        artifacts = load_cav_artifacts(cav_dir, version=args.version)
        scores = compute_projections(
            seq_embs,
            artifacts['concept_cav'],
            artifacts['scaler'],
            artifacts['pca'],
        )
        score_cols[name] = scores.astype(float)
        logger.info(
            f"  {name}: min={scores.min():.3f}, "
            f"mean={scores.mean():.3f}, max={scores.max():.3f}"
        )

    # ------------------------------------------------------------------ #
    # Build dataframe
    # ------------------------------------------------------------------ #
    df = pd.DataFrame({'sample_id': sample_ids})
    score_matrix = np.column_stack(list(score_cols.values()))

    if args.softmax:
        logger.info("Applying softmax across CAV scores")
        score_matrix = softmax(score_matrix)

    for i, name in enumerate(names):
        df[name] = score_matrix[:, i]

    # Add predicted phase (highest-scoring CAV)
    df['predicted_phase'] = [names[i] for i in score_matrix.argmax(axis=1)]

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep='\t', index=False, float_format='%.4f')
    logger.info(f"Saved scores: {out_path}")

    # Print summary
    print("\n--- CAV scores per sample ---")
    print(df.to_string(index=False))

    print("\n--- Predicted phase counts ---")
    print(df['predicted_phase'].value_counts().to_string())


if __name__ == '__main__':
    main()
