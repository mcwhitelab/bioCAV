#!/usr/bin/env python3
"""
Fit a StandardScaler on a reference negative embedding pkl and save it.

The saved scaler can then be shared across all CAV trainings (--scaler-pkl)
so that every CAV lives in the same coordinate system, enabling fast batched
scoring at inference time.

Usage
-----
python scripts/fit_reference_scaler.py \
    --neg  reference_population/neg_10000.pkl \
    --out  reference_population/scaler_v1.pkl
"""

import sys
import argparse
import logging
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from src.utils.data_loader import load_sequence_embeddings
from src.utils.preprocessing import create_scaler_and_pca

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Fit a StandardScaler on a reference negative pkl.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--neg', required=True,
                        help='Reference negative embeddings pkl.')
    parser.add_argument('--out', required=True,
                        help='Output path for the fitted scaler (.pkl).')
    args = parser.parse_args()

    embs, ids = load_sequence_embeddings(args.neg)
    logger.info(f"Loaded {len(ids)} reference embeddings, dim={embs.shape[1]}")

    scaler, meta = create_scaler_and_pca(embs)
    logger.info(f"Fitted StandardScaler: mean range [{embs.mean(axis=0).min():.4f}, "
                f"{embs.mean(axis=0).max():.4f}]")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, out_path)
    logger.info(f"Saved scaler to {out_path}")


if __name__ == '__main__':
    main()
