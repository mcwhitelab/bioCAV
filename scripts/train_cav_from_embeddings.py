#!/usr/bin/env python3
"""
Train a CAV from positive and negative embeddings.

Accepts either:
  - .pkl files (sequence_embeddings loaded directly; .seqnames auto-discovered)
  - .npy files (pre-pooled vectors, e.g. from prepare_embeddings.py for span-based cases)

Examples
--------
# From pkl files (sequence-level, no spans needed):
python train_cav_from_embeddings.py \
    --pos temporal_positive_train.fasta.pkl \
    --neg temporal_negative_train.fasta.pkl \
    --out ./cavs/GO_0005887/

# From npy files (span-pooled, produced by prepare_embeddings.py):
python train_cav_from_embeddings.py \
    --pos ./embeddings/my_concept/pos.npy \
    --neg ./embeddings/my_concept/neg.npy \
    --out ./cavs/my_concept/

# With holdout evaluation:
python train_cav_from_embeddings.py \
    --pos temporal_positive_train.fasta.pkl \
    --neg temporal_negative_train.fasta.pkl \
    --out ./cavs/GO_0005887/ \
    --holdout 0.2
"""

import sys
import argparse
import logging
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from src.train_cav import train_cav
from src.utils.data_loader import load_sequence_embeddings


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_vectors(path: str) -> np.ndarray:
    """Load embeddings from either a .pkl or .npy file."""
    p = Path(path)
    if p.suffix == '.npy':
        arr = np.load(p)
        logger.info(f"Loaded {arr.shape} from {p}")
        return arr
    else:
        # Assume pkl with sequence_embeddings
        arr, ids = load_sequence_embeddings(path)
        logger.info(f"Loaded {arr.shape} from {p} ({len(ids)} samples)")
        return arr


def main():
    parser = argparse.ArgumentParser(
        description="Train a CAV from positive and negative embeddings"
    )
    parser.add_argument('--pos', required=True,
                        help='Positive embeddings (.pkl or .npy)')
    parser.add_argument('--neg', required=True,
                        help='Negative embeddings (.pkl or .npy)')
    parser.add_argument('--out', required=True,
                        help='Output directory for CAV artifacts')
    parser.add_argument('--holdout', type=float, default=0.0,
                        help='Fraction to hold out for evaluation (default: 0)')
    parser.add_argument('--C', type=float, default=1.0,
                        help='Logistic regression regularization strength (default: 1.0)')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='Cross-validation folds (default: 5); use 0 to skip CV entirely (e.g. single positive example)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--version', default='v1',
                        help='Artifact version string (default: v1)')
    parser.add_argument('--scaler-pkl', default=None,
                        help='Path to a pre-fit StandardScaler (joblib pkl). '
                             'If provided, skips fitting a new scaler so all CAVs '
                             'share the same coordinate system.')
    args = parser.parse_args()

    scaler = None
    if args.scaler_pkl:
        import joblib
        scaler = joblib.load(args.scaler_pkl)
        logger.info(f"Loaded pre-fit scaler from {args.scaler_pkl}")

    pos = load_vectors(args.pos)
    neg = load_vectors(args.neg)

    # Write pos.npy / neg.npy under --out so train_cav() can find them
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    np.save(out_path / 'pos.npy', pos)
    np.save(out_path / 'neg.npy', neg)

    config = {
        'regularization_C': args.C,
        'cv_folds': args.cv_folds,
        'random_seed': args.seed,
    }

    train_cav(
        embed_dir=str(out_path),
        output_dir=str(out_path),
        config=config,
        artifact_version=args.version,
        holdout_fraction=args.holdout,
        scaler=scaler,
    )


if __name__ == '__main__':
    main()
