#!/usr/bin/env python3
"""
Prepare pooled embedding vectors from a .pkl file for CAV training.

Reads a .pkl + .pkl.info embedding file and a spans file, pools each sample's
embeddings according to its span, and saves the result as a .npy file.

Run this separately for positive and negative sets, then pass the resulting
.npy files to train_cav_from_embeddings.py.

Spans file format (tab-separated, no header, '#' lines are comments):
    accession                       # whole-sequence
    accession  TAB  pos             # single position (e.g. PTM)
    accession  TAB  start  TAB  end # window [start, end)

Examples
--------
# Motif positives (window spans):
python prepare_embeddings.py \\
    --pkl pos_embeddings.pkl \\
    --info pos_embeddings.pkl.info \\
    --spans pos_spans.txt \\
    --out embeddings/my_concept/pos.npy

# PTM negatives (single positions):
python prepare_embeddings.py \\
    --pkl neg_embeddings.pkl \\
    --info neg_embeddings.pkl.info \\
    --spans neg_spans.txt \\
    --out embeddings/my_concept/neg.npy

# Whole-sequence (no spans file needed):
python prepare_embeddings.py \\
    --pkl pos_embeddings.pkl \\
    --info pos_embeddings.pkl.info \\
    --out embeddings/my_concept/pos.npy
"""

import sys
import argparse
import logging
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from src.utils.data_loader import load_embeddings_pkl, load_spans, build_embedding_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Pool embeddings from .pkl to .npy for CAV training"
    )
    parser.add_argument('--pkl', required=True,
                        help='Path to .pkl embedding file')
    parser.add_argument('--info', required=True,
                        help='Path to .pkl.info file (one ID per line)')
    parser.add_argument('--spans',
                        help='Path to spans file (optional; absent = whole-sequence)')
    parser.add_argument('--out', required=True,
                        help='Output path for pooled .npy file (e.g. pos.npy)')
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    embeddings, sample_ids = load_embeddings_pkl(args.pkl, args.info)

    spans = None
    if args.spans:
        spans = load_spans(args.spans)
    else:
        logger.info("No --spans provided: using whole-sequence mode for all samples")

    vectors = build_embedding_matrix(embeddings, sample_ids, spans)

    np.save(out_path, vectors)
    logger.info(f"Saved {vectors.shape} to {out_path}")


if __name__ == '__main__':
    main()
