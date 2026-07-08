#!/usr/bin/env python3
"""
Fit a global StandardScaler + PCA on a reference embedding pkl.

Pass the resulting pkl to train_cav_from_embeddings.py --pca-pkl so that
all CAVs share the same coordinate system and are directly comparable.

Usage
-----
python scripts/fit_global_pca.py \\
    --pkl  reference_population/neg_10000.pkl \\
    --out  reference_population/global_pca_v1.pkl
"""

import sys
import argparse
import logging
import joblib
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
from src.utils.data_loader import load_sequence_embeddings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Fit a global scaler+PCA on a reference embedding pkl.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--pkl', required=True,
                        help='Reference embeddings pkl (sequence_embeddings key).')
    parser.add_argument('--out', required=True,
                        help='Output path for the scaler+PCA bundle pkl.')
    parser.add_argument('--pca-dim', type=int, default=128,
                        help='Number of PCA components (default: 128).')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42).')
    args = parser.parse_args()

    logger.info(f"Loading embeddings from {args.pkl}")
    embs, ids = load_sequence_embeddings(args.pkl)
    logger.info(f"  {len(ids):,} sequences, hidden_dim={embs.shape[1]}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embs)

    actual_dim = min(args.pca_dim, embs.shape[0], embs.shape[1])
    pca = PCA(n_components=actual_dim, random_state=args.seed)
    pca.fit(X_scaled)
    var_explained = pca.explained_variance_ratio_.sum()
    logger.info(f"PCA({actual_dim}) explains {var_explained:.1%} of variance")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "pca": pca}, out_path)
    logger.info(f"Saved: {out_path}")

    print(f"\nPass to train_cav_from_embeddings.py with:")
    print(f"  --pca-pkl {out_path}")


if __name__ == '__main__':
    main()
