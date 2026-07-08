#!/usr/bin/env python3
"""
Build an hnswlib nearest-neighbor index from protein sequence embeddings.

Loads sequence_embeddings from a pkl file and builds an hnswlib index for
fast approximate nearest-neighbor search (used instead of FAISS).

Output files
------------
- protein_embeddings.npy  : (N, D) float32 embeddings
- protein_ids.pkl         : list of N sequence IDs (same row order)
- protein_index.bin       : hnswlib index for similarity search

Usage
-----
python specific_scripts/build_protein_index.py \\
    --pkl  embeddings/proteins.pkl \\
    --out  protein_index/

# Query the index later:
#   import hnswlib, pickle, numpy as np
#   index = hnswlib.Index(space='l2', dim=1152)
#   index.load_index("protein_index/protein_index.bin")
#   ids = pickle.load(open("protein_index/protein_ids.pkl", "rb"))
#   labels, distances = index.knn_query(query_vec, k=10)
#   hits = [ids[i] for i in labels[0]]
"""

import sys
import argparse
import logging
import pickle
import numpy as np
import hnswlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
from src.utils.data_loader import load_sequence_embeddings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_protein_index(
    pkl_path: str,
    output_dir: str,
    ef_construction: int = 200,
    M: int = 16,
    ef_search: int = 200,
) -> dict:
    """
    Build an hnswlib index from protein sequence embeddings.

    Parameters
    ----------
    pkl_path : str
        Path to pkl file containing sequence_embeddings.
    output_dir : str
        Directory to save index files.
    ef_construction : int
        hnswlib ef_construction parameter (accuracy vs. build speed).
    M : int
        hnswlib M parameter (connections per node; higher = more accurate).
    ef_search : int
        hnswlib ef parameter set for query time.

    Returns
    -------
    dict
        Summary with file paths and stats.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Step 1: Load embeddings
    # ------------------------------------------------------------------ #
    logger.info(f"Loading embeddings from {pkl_path}")
    embs, protein_ids = load_sequence_embeddings(pkl_path)
    logger.info(f"  {len(protein_ids):,} proteins, hidden_dim={embs.shape[1]}")

    embs_f32 = embs.astype('float32')

    # ------------------------------------------------------------------ #
    # Step 2: Build hnswlib index
    # ------------------------------------------------------------------ #
    logger.info("Building hnswlib index (L2 space)...")
    dim = embs_f32.shape[1]
    n = embs_f32.shape[0]

    index = hnswlib.Index(space='l2', dim=dim)
    index.init_index(max_elements=n, ef_construction=ef_construction, M=M)
    index.add_items(embs_f32, np.arange(n))
    index.set_ef(ef_search)

    logger.info(f"  Index built: {index.get_current_count():,} vectors, dim={dim}")

    # ------------------------------------------------------------------ #
    # Step 3: Save artifacts
    # ------------------------------------------------------------------ #
    embs_path = out_path / "protein_embeddings.npy"
    np.save(embs_path, embs_f32)
    logger.info(f"  Saved embeddings: {embs_path}")

    ids_path = out_path / "protein_ids.pkl"
    with open(ids_path, 'wb') as f:
        pickle.dump(protein_ids, f)
    logger.info(f"  Saved protein IDs: {ids_path}")

    index_path = out_path / "protein_index.bin"
    index.save_index(str(index_path))
    logger.info(f"  Saved hnswlib index: {index_path}")

    summary = {
        "status": "success",
        "output_dir": str(out_path),
        "n_proteins": len(protein_ids),
        "embedding_shape": embs_f32.shape,
        "files": {
            "embeddings": str(embs_path),
            "protein_ids": str(ids_path),
            "index": str(index_path),
        },
    }

    logger.info(f"\nProtein index ready:")
    logger.info(f"  Proteins indexed : {len(protein_ids):,}")
    logger.info(f"  Embedding dim    : {dim}")
    logger.info(f"  embeddings       : {embs_path}")
    logger.info(f"  protein IDs      : {ids_path}")
    logger.info(f"  hnswlib index    : {index_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Build an hnswlib index from protein sequence embeddings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--pkl', required=True,
                        help='Protein embeddings pkl (sequence_embeddings key).')
    parser.add_argument('--out', default='protein_index',
                        help='Output directory (default: protein_index).')
    parser.add_argument('--ef-construction', type=int, default=200,
                        help='hnswlib ef_construction (default: 200).')
    parser.add_argument('--M', type=int, default=16,
                        help='hnswlib M connections per node (default: 16).')
    parser.add_argument('--ef-search', type=int, default=200,
                        help='hnswlib ef for query time (default: 200).')
    args = parser.parse_args()

    build_protein_index(
        pkl_path=args.pkl,
        output_dir=args.out,
        ef_construction=args.ef_construction,
        M=args.M,
        ef_search=args.ef_search,
    )


if __name__ == '__main__':
    main()
