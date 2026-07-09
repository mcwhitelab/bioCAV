#!/usr/bin/env python3
"""
Build a universal reference population for CAV training.

Randomly samples N cells from the CellxGene Census across all tissues and
donors, embeds them with Geneformer, and saves the result as a reference pkl.

This fixed random background is used as the negative set for every CAV in
run_cav_pipeline.py --reference-pkl mode, so all CAV directions are defined
relative to the same reference population and are directly comparable via
cosine similarity.

Usage
-----
python specific_scripts/build_reference_population.py \\
    --n-cells 10000 \\
    --out     reference_population/ \\
    --model   /path/to/geneformer_model \\
    --token-dict /path/to/token_dictionary.pkl

# Pass --delete-h5ad to remove reference.h5ad after embedding.

# Produces:
#   reference_population/embeddings/cells.pkl  — reference embeddings
#   reference_population/global_scaler_v1.pkl  — scaler fit on those embeddings

# Then use with run_cav_pipeline.py:
python specific_scripts/run_cav_pipeline.py \\
    --library       cav_library/<dataset_id>/ \\
    --model         /path/to/geneformer_model \\
    --dataset-id    <dataset_id> \\
    --reference-pkl reference_population/embeddings/cells.pkl \\
    --scaler-pkl    reference_population/global_scaler_v1.pkl
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
from src.utils.data_loader import load_sequence_embeddings
from src.utils.preprocessing import create_scaler


# ===========================================================================
# Step 1: sample soma_joinids
# ===========================================================================

def sample_random_joinids(n_cells, seed=42, census_version="stable"):
    """
    Randomly sample soma_joinids from the full human CellxGene Census.
    Reads all soma_joinids into memory (just integers), then random-samples.
    """
    import cellxgene_census

    logger.info(f"Opening CellxGene Census (version={census_version})...")
    census = cellxgene_census.open_soma(census_version=census_version)

    logger.info("Reading all human soma_joinids (one-time, may take a moment)...")
    all_ids = (
        census["census_data"]["homo_sapiens"]
        .obs.read(column_names=["soma_joinid"])
        .concat()
        .to_pandas()
        ["soma_joinid"]
        .values
    )
    census.close()

    logger.info(f"  Total cells in census: {len(all_ids):,}")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(all_ids, size=min(n_cells, len(all_ids)), replace=False)
    logger.info(f"  Sampled {len(chosen):,} cells (seed={seed})")
    return chosen.tolist()


# ===========================================================================
# Step 2: download h5ad
# ===========================================================================

def download_h5ad(soma_joinids, out_path, census_version="stable"):
    """Download h5ad for the sampled joinids and fix obs/var names for Geneformer."""
    import cellxgene_census

    out_path = Path(out_path)
    logger.info(f"Downloading h5ad for {len(soma_joinids):,} reference cells...")

    census = cellxgene_census.open_soma(census_version=census_version)
    adata = cellxgene_census.get_anndata(
        census=census,
        organism="Homo sapiens",
        obs_coords=soma_joinids,
    )
    census.close()

    # obs_names → soma_joinid strings (matches spans file format)
    adata.obs_names = adata.obs["soma_joinid"].astype(str)
    adata.obs_names.name = None
    if "soma_joinid" in adata.obs.columns:
        adata.obs = adata.obs.drop(columns=["soma_joinid"])

    # var_names → Ensembl IDs (Geneformer token dictionary uses Ensembl IDs)
    logger.info(f"  var.columns = {adata.var.columns.tolist()}")
    if "feature_id" in adata.var.columns:
        adata.var_names = adata.var["feature_id"].astype(str)
        adata.var_names.name = None
        adata.var = adata.var.drop(columns=["feature_id"])
        logger.info(f"  var_names set to Ensembl IDs, sample: {adata.var_names[:3].tolist()}")
    else:
        logger.warning("  'feature_id' not in var.columns — var_names may not match token dict")

    logger.info(f"  {adata.n_obs} cells × {adata.n_vars} genes")
    adata.write_h5ad(out_path)
    logger.info(f"  Saved: {out_path}")


# ===========================================================================
# Step 3: embed with Geneformer
# ===========================================================================


def embed(h5ad_path, out_dir, model_path, token_dict_path=None):
    """Run prepare_scrnaseq_embeddings.py to get a cells.pkl."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = out_dir / "cells.pkl"

    embed_script = Path(__file__).parent / "prepare_scrnaseq_embeddings.py"
    cmd = [
        sys.executable, str(embed_script),
        "--input", str(h5ad_path),
        "--model", str(model_path),
        "--out",   str(pkl_path),
    ]
    if token_dict_path:
        cmd += ["--token-dict", str(token_dict_path)]

    logger.info(f"Running Geneformer embedding: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    logger.info(f"  Saved: {pkl_path}")
    return pkl_path


# ===========================================================================
# Step 4: fit global scaler on reference embeddings
# ===========================================================================

def fit_global_scaler(embs, out_path):
    """Fit a StandardScaler on reference embeddings and save to out_path."""
    import joblib

    out_path = Path(out_path)
    logger.info(f"Fitting global scaler on {len(embs):,} cells x {embs.shape[1]} dims...")

    scaler, _ = create_scaler(embs)

    joblib.dump({"scaler": scaler}, out_path)
    logger.info(f"  Saved: {out_path}")
    return scaler


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build a universal reference population for CAV training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--n-cells", type=int, default=10000,
                        help="Cells to randomly sample from the Census (default: 10000)")
    parser.add_argument("--out", default="reference_population",
                        help="Output directory (default: reference_population)")
    parser.add_argument("--model", required=True,
                        help="Path to Geneformer model directory")
    parser.add_argument("--token-dict", default=None,
                        help="Path to Geneformer token dictionary pkl")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for cell sampling (default: 42)")
    parser.add_argument("--census-version", default="stable",
                        help="CellxGene Census version (default: stable)")
    parser.add_argument("--delete-h5ad", action="store_true",
                        help="Delete reference.h5ad after embedding to save disk space "
                             "(kept by default)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Sample soma_joinids (idempotent: skip if already saved)
    # ------------------------------------------------------------------ #
    ids_path = out_dir / "reference_joinids.txt"
    if ids_path.exists():
        logger.info(f"Loading existing joinids: {ids_path}")
        soma_joinids = [int(l.strip()) for l in ids_path.read_text().splitlines() if l.strip()]
        logger.info(f"  {len(soma_joinids):,} cells")
    else:
        soma_joinids = sample_random_joinids(
            args.n_cells, seed=args.seed, census_version=args.census_version
        )
        ids_path.write_text("\n".join(map(str, soma_joinids)) + "\n")
        logger.info(f"Saved joinids: {ids_path}")

    # ------------------------------------------------------------------ #
    # 2. Download h5ad (idempotent)
    # ------------------------------------------------------------------ #
    h5ad_path = out_dir / "reference.h5ad"
    if h5ad_path.exists():
        logger.info(f"h5ad already exists, skipping download: {h5ad_path}")
    else:
        download_h5ad(soma_joinids, h5ad_path, census_version=args.census_version)

    # ------------------------------------------------------------------ #
    # 3. Embed (idempotent)
    # ------------------------------------------------------------------ #
    pkl_path = out_dir / "embeddings" / "cells.pkl"
    if pkl_path.exists():
        logger.info(f"Embeddings already exist, skipping: {pkl_path}")
    else:
        embed(h5ad_path, out_dir / "embeddings", args.model, args.token_dict)

    # ------------------------------------------------------------------ #
    # 4. Fit global scaler (idempotent)
    # ------------------------------------------------------------------ #
    scaler_path = out_dir / "global_scaler_v1.pkl"
    embs, cell_ids = load_sequence_embeddings(str(pkl_path))
    logger.info(f"\nReference embeddings: {len(cell_ids):,} cells, dim={embs.shape[1]}")

    if scaler_path.exists():
        logger.info(f"Global scaler already exists, skipping: {scaler_path}")
    else:
        fit_global_scaler(embs, scaler_path)

    # ------------------------------------------------------------------ #
    # 5. Verify
    # ------------------------------------------------------------------ #
    logger.info("\nReference population ready:")
    logger.info(f"  Cells embedded : {len(cell_ids):,}")
    logger.info(f"  Embedding dim  : {embs.shape[1]}")
    logger.info(f"  embeddings pkl : {pkl_path}")
    logger.info(f"  global scaler pkl : {scaler_path}")

    if args.delete_h5ad and h5ad_path.exists():
        h5ad_path.unlink()
        logger.info(f"  Deleted h5ad")

    print(f"\nDone. Pass to run_cav_pipeline.py with:")
    print(f"  --reference-pkl {pkl_path}")
    print(f"  --scaler-pkl    {scaler_path}")


if __name__ == "__main__":
    main()
