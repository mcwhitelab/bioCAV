#!/usr/bin/env python3
"""
Orchestrate the full CAV training pipeline for a CellxGene Census dataset.

Runs five steps in sequence, skipping any that are already complete:
  1. collect        — gather all unique soma_joinids from spans files
  2. download       — fetch expression data from Census → data/cells.h5ad
  3. embed          — run Geneformer → embeddings/cells.pkl
  4. global_scaler  — fit a shared scaler (on reference population if --reference-pkl given)
  5. train          — train one CAV per concept using spans + embeddings
  6. evaluate        — score train + val sets, write results/evaluation.tsv

Usage
-----
python specific_scripts/run_cav_pipeline.py \
    --library  cav_library/944dedde-... \
    --model    /path/to/geneformer_model \
    --dataset-id 944dedde-...

# Universal reference negatives (recommended for comparable CAV directions):
python specific_scripts/run_cav_pipeline.py \
    --library       cav_library/944dedde-... \
    --model         /path/to/geneformer_model \
    --dataset-id    944dedde-... \
    --reference-pkl reference_population/embeddings/cells.pkl \
    --scaler-pkl    reference_population/global_scaler_v1.pkl

# Keep only CAVs + results (delete h5ad and pkl when done):
python specific_scripts/run_cav_pipeline.py \
    --library  cav_library/944dedde-... \
    --model    /path/to/geneformer_model \
    --dataset-id 944dedde-... \
    --slim

# Re-run from a specific step (e.g. after fixing a bug in training):
python specific_scripts/run_cav_pipeline.py \
    --library  cav_library/944dedde-... \
    --model    /path/to/geneformer_model \
    --dataset-id 944dedde-... \
    --from-step train
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
from src.train_cav import train_cav
from src.evaluate import load_cav_artifacts, compute_projections
from src.utils.data_loader import load_sequence_embeddings

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

STEPS = ["collect", "download", "embed", "global_scaler", "train", "evaluate"]

CAV_CONFIG = {
    "regularization_C": 1.0,
    "cv_folds": 5,
    "random_seed": 42,
    "class_weight": "balanced",   # up-weights positives to match neg count
}


# ===========================================================================
# Helpers
# ===========================================================================

def read_ids(path):
    """Read soma_joinid strings from a spans file, skip blank lines."""
    p = Path(path)
    if not p.exists():
        return []
    return [line.strip() for line in p.read_text().splitlines() if line.strip()]


def concept_dirs(spans_dir):
    """Yield (concept_name, dir_path) for each concept in spans/."""
    return sorted(
        (d.name, d) for d in spans_dir.iterdir() if d.is_dir()
    )


# ===========================================================================
# Step 1: collect
# ===========================================================================

def step_collect(lib_dir, pos_only=False, pos_n=None, seed=42):
    """
    Scan spans files and return a sorted list of unique soma_joinids.

    pos_only=True: only collect positive IDs (pos.txt + val_pos.txt).
    Use this with --reference-pkl so the download/embed steps skip neg cells
    that are never needed — negatives come from the reference population instead.

    pos_n: if set, subsample each pos/val_pos file to at most this many IDs.
    Useful when val_pos.txt is very large and you only need a fixed evaluation set.
    """
    import random as _random
    rng = _random.Random(seed)

    spans_dir = lib_dir / "spans"
    all_ids = set()
    n_concepts = 0
    files = (["pos.txt", "val_pos.txt"] if pos_only
             else ["pos.txt", "neg.txt", "val_pos.txt", "val_neg.txt"])
    pos_files = {"pos.txt", "val_pos.txt"}
    for concept_name, cdir in concept_dirs(spans_dir):
        for fname in files:
            ids = read_ids(cdir / fname)
            if pos_n is not None and fname in pos_files and len(ids) > pos_n:
                ids = rng.sample(ids, pos_n)
            all_ids.update(ids)
        n_concepts += 1
    ids = sorted(all_ids)
    mode = "pos-only (reference negatives)" if pos_only else "all"
    cap = f", capped at {pos_n} per pos file" if pos_n else ""
    logger.info(f"collect: {len(ids)} unique soma_joinids across {n_concepts} concepts ({mode}{cap})")
    return ids


# ===========================================================================
# Step 2: download
# ===========================================================================

def step_download(lib_dir, soma_joinids, dataset_id, census_version="stable"):
    """Download expression data from Census and save as h5ad."""
    import cellxgene_census

    h5ad_path = lib_dir / "data" / "cells.h5ad"
    h5ad_path.parent.mkdir(parents=True, exist_ok=True)

    # Census filter: integer soma_joinids
    id_ints = [int(i) for i in soma_joinids]
    value_filter = f"soma_joinid in {id_ints}"
    logger.info(f"download: querying census — {len(id_ints)} cells")
    logger.info(f"  Census query: {value_filter[:120]}{'...' if len(value_filter) > 120 else ''}")

    census = cellxgene_census.open_soma(census_version=census_version)
    adata = cellxgene_census.get_anndata(
        census,
        organism="Homo sapiens",
        obs_value_filter=value_filter,
    )
    census.close()

    # Set obs_names to soma_joinid strings so they match the spans files,
    # then drop the now-redundant soma_joinid column to avoid the anndata
    # "index.name also used by a column with different values" error.
    adata.obs_names = adata.obs["soma_joinid"].astype(str)
    adata.obs_names.name = None
    adata.obs = adata.obs.drop(columns=["soma_joinid"])

    # Census returns sequential integer var_names; Geneformer needs Ensembl IDs.
    # feature_id holds the Ensembl ID — set it as var_names and drop the column
    # to avoid anndata's "index.name also used by a column" write error.
    logger.info(f"download: var.columns = {adata.var.columns.tolist()}")
    logger.info(f"download: var_names sample = {adata.var_names[:3].tolist()}")
    if "feature_id" in adata.var.columns:
        adata.var_names = adata.var["feature_id"].astype(str)
        adata.var_names.name = None
        adata.var = adata.var.drop(columns=["feature_id"])
        logger.info(f"download: var_names set to Ensembl IDs, sample = {adata.var_names[:3].tolist()}")
    else:
        logger.warning(f"download: 'feature_id' not in var columns — "
                       f"var_names may not match token dictionary. "
                       f"var.columns = {adata.var.columns.tolist()}")

    logger.info(f"download: {adata.n_obs} cells x {adata.n_vars} genes")

    adata.write_h5ad(h5ad_path)
    logger.info(f"download: saved {h5ad_path}")
    return h5ad_path


# ===========================================================================
# Step 3: embed
# ===========================================================================

def step_embed(lib_dir, h5ad_path, model_path, token_dict_path=None):
    """Run Geneformer embedding script and return path to pkl."""
    emb_dir = lib_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = emb_dir / "cells.pkl"

    embed_script = Path(__file__).parent / "prepare_scrnaseq_embeddings.py"
    cmd = [
        sys.executable, str(embed_script),
        "--input",  str(h5ad_path),
        "--model",  str(model_path),
        "--out",    str(pkl_path),
    ]
    if token_dict_path:
        cmd += ["--token-dict", str(token_dict_path)]

    logger.info(f"embed: running {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    logger.info(f"embed: saved {pkl_path}")
    return pkl_path


# ===========================================================================
# Step 4: global_scaler  (optional — fit one shared scaler across all concepts)
# ===========================================================================

def step_global_scaler(lib_dir, pkl_path=None, scaler_pkl=None, reference_pkl=None):
    """
    Load or fit a single StandardScaler and save to lib_dir/global_scaler_v1.pkl.

    scaler_pkl (recommended): path to a pre-built scaler from
        build_reference_population.py.  Loaded directly — no fitting required.
        This is the standard mode when using a reference population, since the
        scaler is already fit on the same reference cells used as negatives.

    reference_pkl (legacy): fit on the fixed random reference population on the fly.
        Use only if you have a reference embeddings pkl but no pre-built scaler.

    pkl_path only (legacy): fit on all training embeddings (pos + neg across every
        concept), deduplicated by cell ID.
    """
    import joblib
    from src.utils.preprocessing import create_scaler

    out_path = lib_dir / "global_scaler_v1.pkl"

    if scaler_pkl is not None:
        logger.info(f"global_scaler: loading pre-built scaler from {scaler_pkl}")
        artifacts = joblib.load(scaler_pkl)
        scaler = artifacts["scaler"]
        joblib.dump({"scaler": scaler}, out_path)
        logger.info(f"global_scaler: cached to {out_path}")
        return scaler

    if reference_pkl is not None:
        logger.info(f"global_scaler: fitting on REFERENCE population: {reference_pkl}")
        X, ref_ids = load_sequence_embeddings(str(reference_pkl))
        logger.info(f"global_scaler: {len(ref_ids):,} reference cells, dim={X.shape[1]}")
    else:
        spans_dir = lib_dir / "spans"
        logger.info("global_scaler: collecting all training cell IDs...")
        all_ids = set()
        for _, cdir in concept_dirs(spans_dir):
            all_ids.update(read_ids(cdir / "pos.txt"))
            all_ids.update(read_ids(cdir / "neg.txt"))
        logger.info(f"global_scaler: {len(all_ids)} unique cells across all concepts")

        logger.info(f"global_scaler: loading embeddings from {pkl_path}")
        seq_embs, cell_ids = load_sequence_embeddings(str(pkl_path))
        id_to_idx = {cid: i for i, cid in enumerate(cell_ids)}

        valid_ids = [i for i in all_ids if i in id_to_idx]
        if len(valid_ids) < len(all_ids):
            logger.warning(f"global_scaler: {len(all_ids) - len(valid_ids)} IDs not found in pkl")
        X = seq_embs[[id_to_idx[i] for i in valid_ids]]
        logger.info(f"global_scaler: fitting on {len(X)} cells x {X.shape[1]} dims")

    scaler, _ = create_scaler(X)

    joblib.dump({"scaler": scaler}, out_path)
    logger.info(f"global_scaler: saved {out_path}")
    return scaler


def load_global_scaler(lib_dir):
    import joblib
    path = lib_dir / "global_scaler_v1.pkl"
    if not path.exists():
        return None
    artifacts = joblib.load(path)
    logger.info(f"global_scaler: loaded from {path}")
    return artifacts["scaler"]


# ===========================================================================
# Step 5: train
# ===========================================================================

def step_train(lib_dir, pkl_path, global_scaler=None,
               reference_embs=None, reference_ids=None, reference_n=1000, seed=42):
    """
    For each concept, extract embeddings from pkl and train a CAV.

    reference_embs: if provided, N cells are randomly sampled from this array
        as negatives instead of reading neg.txt.  Enables a universal negative
        set so all CAV directions are comparable.
    reference_ids: parallel list of cell ID strings for reference_embs.
        When provided, the sampled IDs are written to ref_neg_ids.txt alongside
        the CAV artifacts for full reproducibility.
    reference_n: how many reference cells to sample as negatives per concept.
    """
    spans_dir = lib_dir / "spans"
    cavs_dir  = lib_dir / "cavs"
    cavs_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"train: loading embeddings from {pkl_path}")
    seq_embs, cell_ids = load_sequence_embeddings(str(pkl_path))
    id_to_idx = {cid: i for i, cid in enumerate(cell_ids)}
    logger.info(f"train: {len(cell_ids)} cells, hidden_dim={seq_embs.shape[1]}")

    rng = np.random.default_rng(seed)

    for concept_name, cdir in concept_dirs(spans_dir):
        cav_out = cavs_dir / concept_name
        if (cav_out / "concept_v1.npy").exists():
            logger.info(f"  skip (already trained): {concept_name}")
            continue

        # Positive cells always come from the dataset pkl
        pos_ids = [i for i in read_ids(cdir / "pos.txt") if i in id_to_idx]
        if len(pos_ids) == 0:
            logger.warning(f"  skip (no positive embeddings): {concept_name}")
            continue
        pos_embs = seq_embs[[id_to_idx[i] for i in pos_ids]]

        # Negative cells: reference population OR spans neg.txt
        if reference_embs is not None:
            n_neg = min(reference_n, len(reference_embs))
            neg_idx = rng.choice(len(reference_embs), size=n_neg, replace=False)
            neg_embs = reference_embs[neg_idx]
            neg_label = f"{n_neg} reference"
            # Record exactly which reference cells were used as negatives
            cav_out.mkdir(parents=True, exist_ok=True)
            if reference_ids is not None:
                sampled_ids = [reference_ids[i] for i in neg_idx]
                (cav_out / "ref_neg_ids.txt").write_text(
                    "\n".join(map(str, sampled_ids)) + "\n"
                )
        else:
            neg_ids = [i for i in read_ids(cdir / "neg.txt") if i in id_to_idx]
            if len(neg_ids) == 0:
                logger.warning(f"  skip (no negative embeddings): {concept_name}")
                continue
            neg_embs = seq_embs[[id_to_idx[i] for i in neg_ids]]
            neg_label = f"{len(neg_ids)} dataset"

        # Save as .npy for train_cav
        cav_out.mkdir(parents=True, exist_ok=True)
        np.save(cav_out / "pos.npy", pos_embs)
        np.save(cav_out / "neg.npy", neg_embs)

        scaler_label = " [global scaler]" if global_scaler is not None else ""
        ref_label = " [ref neg]" if reference_embs is not None else ""
        logger.info(f"  training: {concept_name} "
                    f"({len(pos_ids)} pos, {neg_label} neg){scaler_label}{ref_label}")
        try:
            train_cav(
                embed_dir=str(cav_out),
                output_dir=str(cav_out),
                config=CAV_CONFIG,
                artifact_version="v1",
                scaler=global_scaler,
            )
        except Exception as e:
            logger.error(f"  FAILED: {concept_name}: {e}")
            continue

    logger.info("train: done")
    return id_to_idx, seq_embs, cell_ids


# ===========================================================================
# Step 5: evaluate
# ===========================================================================

def step_evaluate(lib_dir, id_to_idx, seq_embs,
                  reference_embs=None, reference_n=1000, seed=42):
    """
    Score train + val sets for every concept, write evaluation.tsv.

    reference_embs: if provided, reference cells are used as negatives for
        both train and val scoring (mirrors how the CAVs were trained).
        val_neg.txt cells are ignored when reference mode is active.
    """
    spans_dir   = lib_dir / "spans"
    cavs_dir    = lib_dir / "cavs"
    results_dir = lib_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    rows = []
    for concept_name, cdir in concept_dirs(spans_dir):
        cav_dir = cavs_dir / concept_name
        if not (cav_dir / "concept_v1.npy").exists():
            logger.warning(f"  skip (no CAV): {concept_name}")
            continue

        artifacts = load_cav_artifacts(str(cav_dir), version="v1")

        def score_split(pos_file, neg_file):
            pos_ids = [i for i in read_ids(pos_file) if i in id_to_idx]

            if reference_embs is not None:
                # Reference mode: sample from reference population as negatives
                n_neg = min(reference_n, len(reference_embs))
                neg_idx = rng.choice(len(reference_embs), size=n_neg, replace=False)
                neg_arr = reference_embs[neg_idx]
                if len(pos_ids) == 0:
                    return None, None, 0, n_neg
                pos_arr = seq_embs[[id_to_idx[i] for i in pos_ids]]
                embs = np.vstack([pos_arr, neg_arr])
                labels = np.array([1] * len(pos_ids) + [0] * n_neg)
                n_neg_out = n_neg
            else:
                neg_ids = [i for i in read_ids(neg_file) if i in id_to_idx]
                if len(pos_ids) == 0 or len(neg_ids) == 0:
                    return None, None, len(pos_ids), len(neg_ids)
                embs = seq_embs[[id_to_idx[i] for i in pos_ids + neg_ids]]
                labels = np.array([1] * len(pos_ids) + [0] * len(neg_ids))
                n_neg_out = len(neg_ids)

            scores = compute_projections(
                embs,
                artifacts["concept_cav"],
                artifacts["scaler"],
            )
            auroc = roc_auc_score(labels, scores)
            return auroc, scores[:len(pos_ids)].mean(), len(pos_ids), n_neg_out

        train_auroc, train_mean, n_pos_tr, n_neg_tr = score_split(
            cdir / "pos.txt", cdir / "neg.txt"
        )
        val_auroc, val_mean, n_pos_val, n_neg_val = score_split(
            cdir / "val_pos.txt", cdir / "val_neg.txt"
        )

        logger.info(
            f"  {concept_name:50s}  "
            f"train AUROC={train_auroc:.3f}  val AUROC={val_auroc:.3f}"
        )
        rows.append({
            "concept_name":  concept_name,
            "n_pos_train":   n_pos_tr,
            "n_neg_train":   n_neg_tr,
            "n_pos_val":     n_pos_val,
            "n_neg_val":     n_neg_val,
            "train_auroc":   round(train_auroc, 4) if train_auroc else None,
            "val_auroc":     round(val_auroc,   4) if val_auroc   else None,
        })

    df = pd.DataFrame(rows).sort_values("val_auroc", ascending=False)
    out_path = results_dir / "evaluation.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    logger.info(f"evaluate: saved {out_path}")
    print(f"\n{'='*70}")
    print(df.to_string(index=False))
    return df


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate full CAV pipeline for a Census dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--library", required=True,
        help="Path to cav_library/<dataset_id>/ (output of build_cav_library_agent.py)."
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to local Geneformer model directory."
    )
    parser.add_argument(
        "--token-dict", default=None,
        help="Path to Geneformer token dictionary pkl (e.g. token_dictionary_gc104M.pkl)."
    )
    parser.add_argument(
        "--dataset-id", required=True,
        help="CellxGene Census dataset_id (for the download step)."
    )
    parser.add_argument(
        "--census-version", default="stable",
        help="CellxGene Census version (default: stable)."
    )
    parser.add_argument(
        "--from-step", choices=STEPS, default="collect",
        help="Resume from this step, skipping earlier ones (default: collect)."
    )
    parser.add_argument(
        "--scaler-pkl", default=None,
        help="Path to a pre-built scaler pkl (from build_reference_population.py). "
             "When provided, the global_scaler step loads this directly instead of fitting. "
             "Recommended when using --reference-pkl."
    )
    parser.add_argument(
        "--reference-pkl", default=None,
        help="Path to a reference population pkl (from build_reference_population.py). "
             "If given, negatives for training and evaluation are sampled from this "
             "fixed random background instead of the per-concept neg.txt files. "
             "All CAV directions then live in the same coordinate system and are "
             "directly comparable via cosine similarity."
    )
    parser.add_argument(
        "--reference-n", type=int, default=1000,
        help="Cells to sample from the reference pkl as negatives per concept "
             "(default: 1000).  Only used when --reference-pkl is set."
    )
    parser.add_argument(
        "--pos-n", type=int, default=None,
        help="Subsample each pos.txt and val_pos.txt to at most this many cells "
             "before downloading/embedding.  Useful when val_pos files are very "
             "large and you only need a fixed training/evaluation set."
    )
    parser.add_argument(
        "--slim", action="store_true",
        help="Delete data/cells.h5ad and embeddings/cells.pkl after training "
             "to save disk space. CAVs and results are always kept."
    )
    args = parser.parse_args()

    lib_dir = Path(args.library)
    if not lib_dir.exists():
        raise FileNotFoundError(f"Library directory not found: {lib_dir}")

    start_idx = STEPS.index(args.from_step)

    # Load reference population embeddings once (if provided)
    reference_embs = None
    if args.reference_pkl:
        ref_pkl = Path(args.reference_pkl)
        if not ref_pkl.exists():
            raise FileNotFoundError(f"--reference-pkl not found: {ref_pkl}")
        logger.info(f"Reference mode: loading {ref_pkl}")
        reference_embs, ref_ids = load_sequence_embeddings(str(ref_pkl))
        logger.info(f"  {len(ref_ids):,} reference cells, dim={reference_embs.shape[1]}")

    # ------------------------------------------------------------------ #
    # Step 1: collect
    # ------------------------------------------------------------------ #
    soma_joinids = None
    if start_idx <= STEPS.index("collect"):
        # In reference mode only download positive cells — negatives come from reference
        soma_joinids = step_collect(lib_dir, pos_only=(reference_embs is not None), pos_n=args.pos_n)

    # ------------------------------------------------------------------ #
    # Step 2: download
    # ------------------------------------------------------------------ #
    h5ad_path = lib_dir / "data" / "cells.h5ad"
    if start_idx <= STEPS.index("download"):
        if h5ad_path.exists():
            logger.info(f"download: skipping (already exists): {h5ad_path}")
        else:
            if soma_joinids is None:
                soma_joinids = step_collect(lib_dir, pos_only=(reference_embs is not None), pos_n=args.pos_n)
            h5ad_path = step_download(
                lib_dir, soma_joinids, args.dataset_id,
                census_version=args.census_version
            )

    # ------------------------------------------------------------------ #
    # Step 3: embed
    # ------------------------------------------------------------------ #
    pkl_path = lib_dir / "embeddings" / "cells.pkl"
    if start_idx <= STEPS.index("embed"):
        if pkl_path.exists():
            logger.info(f"embed: skipping (already exists): {pkl_path}")
        else:
            pkl_path = step_embed(lib_dir, h5ad_path, args.model, args.token_dict)

    # ------------------------------------------------------------------ #
    # Step 4: global_scaler
    # ------------------------------------------------------------------ #
    global_scaler = None
    if start_idx <= STEPS.index("global_scaler"):
        if (lib_dir / "global_scaler_v1.pkl").exists():
            logger.info("global_scaler: skipping (already exists)")
            global_scaler = load_global_scaler(lib_dir)
        else:
            global_scaler = step_global_scaler(
                lib_dir,
                pkl_path=pkl_path,
                scaler_pkl=args.scaler_pkl,
                reference_pkl=args.reference_pkl,
            )
    elif (lib_dir / "global_scaler_v1.pkl").exists():
        global_scaler = load_global_scaler(lib_dir)

    # ------------------------------------------------------------------ #
    # Step 5: train
    # ------------------------------------------------------------------ #
    id_to_idx = seq_embs = None
    if start_idx <= STEPS.index("train"):
        id_to_idx, seq_embs, _ = step_train(
            lib_dir, pkl_path, global_scaler,
            reference_embs=reference_embs,
            reference_ids=ref_ids if args.reference_pkl else None,
            reference_n=args.reference_n,
        )

    # ------------------------------------------------------------------ #
    # Step 6: evaluate
    # ------------------------------------------------------------------ #
    if start_idx <= STEPS.index("evaluate"):
        if id_to_idx is None:
            logger.info("evaluate: loading embeddings...")
            seq_embs, cell_ids = load_sequence_embeddings(str(pkl_path))
            id_to_idx = {cid: i for i, cid in enumerate(cell_ids)}
        step_evaluate(
            lib_dir, id_to_idx, seq_embs,
            reference_embs=reference_embs,
            reference_n=args.reference_n,
        )

    # ------------------------------------------------------------------ #
    # Slim mode: remove large intermediates
    # ------------------------------------------------------------------ #
    if args.slim:
        for path in [h5ad_path, pkl_path]:
            if path.exists():
                path.unlink()
                logger.info(f"slim: deleted {path}")
        # Also remove .npy files saved alongside CAV artifacts
        for _, cdir in concept_dirs(lib_dir / "cavs"):
            for npy in ["pos.npy", "neg.npy"]:
                p = cdir / npy
                if p.exists():
                    p.unlink()
        logger.info("slim: intermediate files removed")

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
