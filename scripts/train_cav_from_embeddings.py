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

# In a top-k PCA subspace of the reference population (see fit_global_pca.py):
python train_cav_from_embeddings.py \
    --pos ./embeddings/my_concept/pos.npy \
    --neg reference_population/neg_10000.pkl \
    --out ./cavs/my_concept_pca128/ \
    --cv-folds 0 \
    --pca-pkl reference_population/global_pca_128.pkl
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
    parser.add_argument('--C', type=float, default=None,
                        help='Logistic regression inverse regularization strength '
                             '(default: 1/n_train, i.e. ~1e-4 with the shared 10k background). '
                             'sklearn sums the loss over samples, so a fixed C means different '
                             'regularization at different n; set it explicitly only on purpose.')
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
    parser.add_argument('--pca-pkl', default=None,
                        help='Scaler+PCA bundle from fit_global_pca.py. The CAV is trained on '
                             'the top-k reference PCs and folded back to the full space, so the '
                             'saved concept vector is scored like any other CAV (with the '
                             "bundle's scaler). If --scaler-pkl is also given it must match "
                             "the bundle's scaler.")
    args = parser.parse_args()

    import joblib
    scaler = None
    if args.scaler_pkl:
        scaler = joblib.load(args.scaler_pkl)
        logger.info(f"Loaded pre-fit scaler from {args.scaler_pkl}")

    pca = None
    if args.pca_pkl:
        bundle = joblib.load(args.pca_pkl)
        pca = bundle['pca']
        if scaler is None:
            scaler = bundle['scaler']
        elif not (np.allclose(scaler.mean_, bundle['scaler'].mean_)
                  and np.allclose(scaler.scale_, bundle['scaler'].scale_)):
            parser.error(f"--scaler-pkl does not match the scaler inside --pca-pkl; "
                         f"the PCA was fit in a different coordinate system.")
        logger.info(f"Loaded scaler+PCA({pca.n_components_}) bundle from {args.pca_pkl}")

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
        'pca_pkl': args.pca_pkl,
    }

    train_cav(
        embed_dir=str(out_path),
        output_dir=str(out_path),
        config=config,
        artifact_version=args.version,
        holdout_fraction=args.holdout,
        scaler=scaler,
        pca=pca,
    )


if __name__ == '__main__':
    main()
