#!/usr/bin/env python3
"""
cav_hierarchy.py — Build and use a three-level orthogonal CAV hierarchy.

CAV libraries built from multi-tissue / multi-disease datasets follow the naming
convention:

    {cell_type}__{tissue}__{disease}

This script decomposes the flat set of CAV directions into three orthogonal levels
using Gram-Schmidt projection:

    Level 0 (cell_type):  mean of all *__normal CAVs across tissues
                          → captures shared cell-identity signal
    Level 1 (tissue):     tissue-specific residual (after removing Level 0)
                          → captures what makes a tissue context unique
    Level 2 (disease):    disease residual (after removing Level 0 + Level 1)
                          → captures pure disease signal within a tissue + cell type

For matched normal/disease pairs (e.g. macrophage__lung__normal and
macrophage__lung__lung_cancer), a *disease-transition delta* is also computed:

    delta = disease_residual − normal_residual

Gene correlations computed at Level 2 use the delta direction (where available),
giving genes whose expression correlates with shifting from normal → disease state,
independent of cell-type and tissue context.

Prerequisites
-------------
All CAVs in the library must have been trained with the same global PCA
(--pca-pkl reference_population/global_pca_v1.pkl). The concept vectors are
then all in the same 128-D PCA space and can be compared directly.

Usage
-----
# Project cells and build hierarchy coordinates:
python specific_scripts/cav_hierarchy.py \\
    --lib-dir  cav_library/3f7c572c/ \\
    --pkl      embeddings/tumor_cells.pkl \\
    --out      results/hierarchy/

# Also compute gene correlations at each level:
python specific_scripts/cav_hierarchy.py \\
    --lib-dir  cav_library/3f7c572c/ \\
    --pkl      embeddings/tumor_cells.pkl \\
    --h5ad     data/tumor_cells.h5ad \\
    --out      results/hierarchy/

Outputs
-------
    cell_coordinates.tsv        — per-cell projection onto every hierarchy axis
    hierarchy_axes.json         — axis metadata (level, cell_type, tissue, disease)
    gene_correlations_L0.tsv    — genes correlated with cell-type axes (if --h5ad)
    gene_correlations_L1.tsv    — genes correlated with tissue axes (if --h5ad)
    gene_correlations_L2.tsv    — genes correlated with disease-transition axes (if --h5ad)
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
from src.utils.data_loader import load_sequence_embeddings
from src.utils.preprocessing import preprocess_embeddings

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ===========================================================================
# Parsing — uses library_structure.json when available
# ===========================================================================

def load_library_structure(lib_dir: Path) -> Optional[dict]:
    """Load library_structure.json produced by analyze_cav_library.py."""
    path = lib_dir / "library_structure.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def discover_cavs(lib_dir: Path, version: str = "v1") -> Dict[str, tuple]:
    """
    Scan lib_dir/cavs/ and return {cav_name: parts_tuple} for every trained CAV.

    Parts tuple meaning depends on library structure (from library_structure.json):
      2-part library:  (group, concept)
      3-part library:  (group, context, condition)

    Falls back to splitting on '__' if no library_structure.json exists.
    Skips CAVs whose name doesn't match the expected number of parts.
    """
    cavs_dir = lib_dir / "cavs"
    if not cavs_dir.exists():
        raise FileNotFoundError(f"No 'cavs/' directory found under {lib_dir}")

    structure = load_library_structure(lib_dir)
    if structure is not None:
        sep     = structure.get("separator", "__")
        n_parts = structure.get("n_levels")
        logger.info(f"discover_cavs: using library_structure.json "
                    f"({n_parts}-level, sep='{sep}')")
    else:
        sep     = "__"
        n_parts = None
        logger.info("discover_cavs: no library_structure.json — "
                    "accepting any name with 2+ parts")

    result = {}
    for d in sorted(cavs_dir.iterdir()):
        if not d.is_dir():
            continue
        npy = d / f"concept_{version}.npy"
        if not npy.exists():
            continue
        parts = d.name.split(sep)
        if n_parts is not None and len(parts) != n_parts:
            logger.debug(f"Skipping '{d.name}': expected {n_parts} parts, got {len(parts)}")
            continue
        if len(parts) < 2:
            logger.debug(f"Skipping '{d.name}': fewer than 2 parts")
            continue
        result[d.name] = tuple(parts)

    logger.info(f"Found {len(result)} CAVs in {cavs_dir}")
    return result


# ===========================================================================
# Loading CAV directions
# ===========================================================================

def load_cav_direction(cav_dir: Path, version: str = "v1") -> np.ndarray:
    """Load and unit-normalise the concept vector (in PCA space)."""
    v = np.load(cav_dir / f"concept_{version}.npy").astype(np.float64)
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        raise ValueError(f"Near-zero CAV vector in {cav_dir}")
    return v / norm


def load_all_directions(
    lib_dir: Path,
    cav_names: Dict[str, Tuple[str, str, str]],
    version: str = "v1",
) -> Dict[str, np.ndarray]:
    """
    Return {cav_name: unit_vector} for all named CAVs.
    Vectors are in the shared PCA space (128-D by default).
    """
    cavs_dir = lib_dir / "cavs"
    dirs = {}
    for name in cav_names:
        try:
            dirs[name] = load_cav_direction(cavs_dir / name, version)
        except Exception as e:
            logger.warning(f"Could not load {name}: {e}")
    return dirs


# ===========================================================================
# Gram-Schmidt orthogonalisation
# ===========================================================================

def orthogonalize(v: np.ndarray, basis: List[np.ndarray]) -> Optional[np.ndarray]:
    """
    Remove all components in the span of `basis` from `v`.
    Returns the unit-normalised residual, or None if v lies entirely in the span.
    """
    v = v.copy()
    for u in basis:
        v -= np.dot(v, u) * u
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-10 else None


# ===========================================================================
# Hierarchy construction
# ===========================================================================

def build_hierarchy(
    cav_names: Dict[str, tuple],
    cav_dirs: Dict[str, np.ndarray],
    lib_dir: Optional[Path] = None,
    level_order: str = "group-context-condition",
) -> Dict:
    """
    Construct the orthogonal basis from flat CAV directions.

    For 3-level libraries (group / context / condition):
        L0  — primary axis (mean of baseline CAVs across secondaries)
        L1  — secondary residual (within primary)
        L2  — condition residual (within primary + secondary)
        delta — condition_residual − baseline_residual for matched pairs

    level_order controls which field plays L0 vs L1:
        "group-context-condition"  (default) → L0=cell_type, L1=tissue
        "context-group-condition"            → L0=tissue,    L1=cell_type

    For 2-level libraries (group / condition):
        L0  — group/baseline direction
        L2  — condition residual (orthogonalized against L0)
        (L1 is empty)

    Returns a hierarchy dict with keys:
        'L0'      — {primary: unit_vector}
        'L1'      — {(primary, secondary): unit_vector}
        'L2'      — {(cell_type, tissue, disease): unit_vector}
        'delta'   — {(cell_type, tissue): unit_vector}
        'meta'    — structured metadata for reporting
    """
    # Determine level indices from library_structure.json if available
    structure      = load_library_structure(lib_dir) if lib_dir else None
    group_level    = structure["group_level"]    if structure else 0
    condition_level = structure["condition_level"] if structure else (len(next(iter(cav_names.values()))) - 1)
    baseline_vals  = set(structure["baseline_values"]) if structure else {"normal"}
    n_levels       = structure["n_levels"] if structure else len(next(iter(cav_names.values())))
    sep            = structure["separator"] if structure else "__"

    # Context level sits between group and condition (only meaningful for 3+ levels)
    has_context = n_levels >= 3
    context_level = [i for i in range(n_levels)
                     if i != group_level and i != condition_level][0] if has_context else None

    # Apply level_order: swap group and context roles if requested
    if has_context and level_order == "context-group-condition":
        logger.info("level_order=context-group-condition: swapping L0/L1 "
                    "(L0=tissue/context, L1=cell_type/group)")
        group_level, context_level = context_level, group_level
    else:
        logger.info(f"level_order=group-context-condition: L0=group, L1=context")

    def parts(name):
        return cav_names[name]

    def is_baseline(name):
        return parts(name)[condition_level] in baseline_vals

    # ------------------------------------------------------------------ #
    # Level 0: group directions (mean of baseline CAVs across all contexts)
    # ------------------------------------------------------------------ #
    baseline_by_group: Dict[str, List[np.ndarray]] = {}
    for name in cav_names:
        if is_baseline(name) and name in cav_dirs:
            group = parts(name)[group_level]
            baseline_by_group.setdefault(group, []).append(cav_dirs[name])

    L0: Dict[str, np.ndarray] = {}
    for group, vecs in baseline_by_group.items():
        mean_v = np.mean(vecs, axis=0)
        norm   = np.linalg.norm(mean_v)
        if norm > 1e-10:
            L0[group] = mean_v / norm

    logger.info(f"L0 (group axes): {len(L0)} — {sorted(L0)}")

    # ------------------------------------------------------------------ #
    # Level 1: context residuals (within each group) — 3-level libraries only
    # ------------------------------------------------------------------ #
    L1: Dict[tuple, np.ndarray] = {}
    if has_context:
        context_baselines: Dict[tuple, np.ndarray] = {}
        for name in cav_names:
            if is_baseline(name) and name in cav_dirs:
                key = (parts(name)[group_level], parts(name)[context_level])
                context_baselines[key] = cav_dirs[name]

        for (group, context), v in context_baselines.items():
            basis    = [L0[group]] if group in L0 else []
            residual = orthogonalize(v, basis)
            if residual is not None:
                L1[(group, context)] = residual
            else:
                logger.debug(f"L1: {group}__{context} residual is zero — skipping")

    logger.info(f"L1 (context axes): {len(L1)}")

    # ------------------------------------------------------------------ #
    # Level 2: condition residuals (within group + context)
    # ------------------------------------------------------------------ #
    L2: Dict[tuple, np.ndarray] = {}
    for name in cav_names:
        if name not in cav_dirs:
            continue
        p     = parts(name)
        group = p[group_level]
        basis = []
        if group in L0:
            basis.append(L0[group])
        if has_context:
            context = p[context_level]
            if (group, context) in L1:
                basis.append(L1[(group, context)])
            key = (group, context, p[condition_level])
        else:
            key = (group, p[condition_level])
        residual = orthogonalize(cav_dirs[name], basis)
        if residual is not None:
            L2[key] = residual
        else:
            logger.debug(f"L2: {name} residual is zero — skipping")

    logger.info(f"L2 (condition axes): {len(L2)}")

    # ------------------------------------------------------------------ #
    # Delta: condition_residual − baseline_residual for matched pairs
    # ------------------------------------------------------------------ #
    delta: Dict[tuple, np.ndarray] = {}
    for key, v_cond in L2.items():
        cond_val = key[-1]
        if cond_val in baseline_vals:
            continue
        # Build the baseline key by swapping the condition value
        baseline_key = key[:-1] + (next(iter(baseline_vals)),)
        if baseline_key in L2:
            d    = v_cond - L2[baseline_key]
            norm = np.linalg.norm(d)
            if norm > 1e-10:
                delta[key] = d / norm

    logger.info(f"Delta (condition-transition axes): {len(delta)}")

    # ------------------------------------------------------------------ #
    # Metadata
    # ------------------------------------------------------------------ #
    meta = {
        "cell_types": sorted(L0),
        "tissue_axes": [f"{ct}__{tissue}" for (ct, tissue) in sorted(L1)],
        "disease_axes": [f"{ct}__{tissue}__{disease}" for (ct, tissue, disease) in sorted(L2)],
        "delta_axes": [f"{ct}__{tissue}__{disease}" for (ct, tissue, disease) in sorted(delta)],
        "counts": {
            "L0": len(L0),
            "L1": len(L1),
            "L2": len(L2),
            "delta": len(delta),
        },
    }

    return {"L0": L0, "L1": L1, "L2": L2, "delta": delta, "meta": meta}


# ===========================================================================
# Cell projection
# ===========================================================================

def project_cells(
    embs_raw: np.ndarray,
    cell_ids: List[str],
    hierarchy: Dict,
    scaler,
    pca,
) -> pd.DataFrame:
    """
    Project each cell's embedding onto every axis in the hierarchy.

    Parameters
    ----------
    embs_raw : (n_cells, hidden_dim) raw Geneformer embeddings
    cell_ids : list of cell ID strings
    hierarchy: output of build_hierarchy()
    scaler, pca: fitted sklearn objects (global PCA)

    Returns
    -------
    DataFrame with columns:
        cell_id
        L0__{cell_type}                      for each L0 axis
        L1__{cell_type}__{tissue}            for each L1 axis
        L2__{cell_type}__{tissue}__{disease} for each L2 axis
        delta__{cell_type}__{tissue}__{disease} for each delta axis
    """
    # Transform raw embeddings into PCA space
    X = preprocess_embeddings(embs_raw, scaler, pca)   # (n, pca_dim)

    rows = {"cell_id": cell_ids}

    for ct, v in sorted(hierarchy["L0"].items()):
        rows[f"L0__{ct}"] = X @ v

    for (ct, tissue), v in sorted(hierarchy["L1"].items()):
        rows[f"L1__{ct}__{tissue}"] = X @ v

    for (ct, tissue, disease), v in sorted(hierarchy["L2"].items()):
        rows[f"L2__{ct}__{tissue}__{disease}"] = X @ v

    for (ct, tissue, disease), v in sorted(hierarchy["delta"].items()):
        rows[f"delta__{ct}__{tissue}__{disease}"] = X @ v

    return pd.DataFrame(rows)


# ===========================================================================
# Gene correlations
# ===========================================================================

def pearson_r_all_genes(
    X_expr: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    """
    Pearson r between each gene (column of X_expr) and a scalar score per cell.
    """
    X_c = X_expr - X_expr.mean(axis=0)
    y_c = scores - scores.mean()
    numerator  = X_c.T @ y_c
    norm_X     = np.linalg.norm(X_c, axis=0)
    norm_y     = float(np.linalg.norm(y_c))
    denom      = norm_X * norm_y
    return np.where(denom > 1e-12, numerator / denom, 0.0)


def compute_gene_correlations(
    hierarchy: Dict,
    embs_raw: np.ndarray,
    cell_ids: List[str],
    h5ad_path: str,
    scaler,
    pca,
    top_n: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """
    For each hierarchy axis, correlate the per-cell projection score with
    gene expression.

    Returns a dict {axis_label: gene_df} where gene_df has columns:
        ensembl_id, gene_symbol, pearson_r, rank, direction
    """
    import anndata as ad
    import scanpy as sc

    logger.info(f"Loading h5ad: {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    h5ad_set  = set(adata.obs_names)
    keep_mask = [c in h5ad_set for c in cell_ids]
    keep_ids  = [c for c, m in zip(cell_ids, keep_mask) if m]
    keep_idx  = [i for i, m in enumerate(keep_mask) if m]

    n_dropped = len(cell_ids) - len(keep_ids)
    if n_dropped:
        logger.warning(f"{n_dropped} cells not found in h5ad — using {len(keep_ids)}")

    X_emb_sub = embs_raw[keep_idx]
    X_pca     = preprocess_embeddings(X_emb_sub, scaler, pca)
    adata_sub = adata[keep_ids]
    X_expr    = (
        adata_sub.X.toarray()
        if hasattr(adata_sub.X, "toarray")
        else np.asarray(adata_sub.X)
    ).astype(np.float32)

    gene_ids   = list(adata.var_names)
    if "gene_symbol" in adata.var.columns:
        gene_symbols = list(adata.var["gene_symbol"])
    else:
        gene_symbols = gene_ids
        logger.warning("var['gene_symbol'] not found; using Ensembl IDs.")

    results: Dict[str, pd.DataFrame] = {}

    def _corr_df(scores, label):
        r = pearson_r_all_genes(X_expr, scores)
        df = pd.DataFrame({
            "ensembl_id":  gene_ids,
            "gene_symbol": gene_symbols,
            "pearson_r":   r,
            "axis":        label,
        })
        df["abs_r"] = df["pearson_r"].abs()
        df = df.sort_values("abs_r", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        df["direction"] = np.where(df["pearson_r"] > 0, "pos_concept", "neg_concept")
        df = df.drop(columns="abs_r")
        if top_n:
            df = df.head(top_n)
        return df

    # Level 0
    for ct, v in sorted(hierarchy["L0"].items()):
        label = f"L0__{ct}"
        results[label] = _corr_df(X_pca @ v, label)

    # Level 1
    for (ct, tissue), v in sorted(hierarchy["L1"].items()):
        label = f"L1__{ct}__{tissue}"
        results[label] = _corr_df(X_pca @ v, label)

    # Level 2 — use delta where available, otherwise the raw L2 residual
    for (ct, tissue, disease), v in sorted(hierarchy["L2"].items()):
        delta_v = hierarchy["delta"].get((ct, tissue, disease))
        use_v   = delta_v if delta_v is not None else v
        label   = f"L2__{ct}__{tissue}__{disease}"
        results[label] = _corr_df(X_pca @ use_v, label)

    return results


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build a hierarchical CAV decomposition and project cells.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--lib-dir", required=True,
                        help="CAV library directory (contains cavs/ and global_pca_v1.pkl).")
    parser.add_argument("--pkl", required=True,
                        help="Embedding pkl file for cells to project.")
    parser.add_argument("--out", required=True,
                        help="Output directory for results.")
    parser.add_argument("--h5ad", default=None,
                        help="AnnData h5ad file for gene correlation analysis (optional).")
    parser.add_argument("--version", default="v1",
                        help="CAV artifact version suffix (default: v1).")
    parser.add_argument("--top-n", type=int, default=None,
                        help="Keep only the top N genes by |pearson_r| in gene correlation "
                             "output (default: all genes).")
    parser.add_argument("--pca-pkl", default=None,
                        help="Path to global PCA pkl (scaler + PCA). Defaults to "
                             "lib_dir/global_pca_v1.pkl.")
    parser.add_argument("--level-order",
                        choices=["group-context-condition",
                                 "context-group-condition"],
                        default="group-context-condition",
                        help="Order of L0/L1 decomposition. "
                             "'group-context-condition' (default): L0=cell_type, L1=tissue. "
                             "'context-group-condition': L0=tissue, L1=cell_type. "
                             "L2 (disease) always comes last.")
    args = parser.parse_args()

    lib_dir = Path(args.lib_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load global PCA
    # ------------------------------------------------------------------ #
    import joblib

    pca_pkl = Path(args.pca_pkl) if args.pca_pkl else lib_dir / "global_pca_v1.pkl"
    if not pca_pkl.exists():
        raise FileNotFoundError(
            f"Global PCA not found: {pca_pkl}\n"
            f"Pass --pca-pkl or ensure lib_dir/global_pca_v1.pkl exists."
        )
    logger.info(f"Loading global PCA from {pca_pkl}")
    pca_artifacts = joblib.load(pca_pkl)
    scaler = pca_artifacts["scaler"]
    pca    = pca_artifacts["pca"]

    # ------------------------------------------------------------------ #
    # 2. Discover and load CAV directions
    # ------------------------------------------------------------------ #
    cav_names = discover_cavs(lib_dir, version=args.version)
    cav_dirs  = load_all_directions(lib_dir, cav_names, version=args.version)
    logger.info(f"Loaded {len(cav_dirs)} CAV directions")

    # ------------------------------------------------------------------ #
    # 3. Build hierarchy
    # ------------------------------------------------------------------ #
    hierarchy = build_hierarchy(cav_names, cav_dirs,
                               lib_dir=lib_dir,
                               level_order=args.level_order)

    meta_path = out_dir / "hierarchy_axes.json"
    with open(meta_path, "w") as f:
        json.dump(hierarchy["meta"], f, indent=2)
    logger.info(f"Saved axis metadata: {meta_path}")

    # ------------------------------------------------------------------ #
    # 4. Load embeddings and project cells
    # ------------------------------------------------------------------ #
    logger.info(f"Loading embeddings from {args.pkl}")
    embs_raw, cell_ids = load_sequence_embeddings(args.pkl)
    logger.info(f"  {len(cell_ids)} cells, hidden_dim={embs_raw.shape[1]}")

    coords_df = project_cells(embs_raw, cell_ids, hierarchy, scaler, pca)
    coords_path = out_dir / "cell_coordinates.tsv"
    coords_df.to_csv(coords_path, sep="\t", index=False, float_format="%.4f")
    logger.info(f"Saved cell coordinates: {coords_path}  ({coords_df.shape})")

    # ------------------------------------------------------------------ #
    # 5. Gene correlations (optional)
    # ------------------------------------------------------------------ #
    if args.h5ad:
        logger.info("Computing gene correlations at each level...")
        gene_results = compute_gene_correlations(
            hierarchy, embs_raw, cell_ids,
            args.h5ad, scaler, pca,
            top_n=args.top_n,
        )

        # Split by level and concatenate into per-level TSVs
        for level_prefix in ("L0", "L1", "L2"):
            subset = {k: v for k, v in gene_results.items()
                      if k.startswith(level_prefix + "__")}
            if not subset:
                continue
            combined = pd.concat(subset.values(), ignore_index=True)
            out_path = out_dir / f"gene_correlations_{level_prefix}.tsv"
            combined.to_csv(out_path, sep="\t", index=False)
            logger.info(f"Saved {level_prefix} gene correlations: {out_path} "
                        f"({len(subset)} axes, {len(combined)} rows)")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    m = hierarchy["meta"]["counts"]
    print(f"\nHierarchy summary:")
    print(f"  L0 cell-type axes    : {m['L0']}")
    print(f"  L1 tissue axes       : {m['L1']}")
    print(f"  L2 disease axes      : {m['L2']}")
    print(f"  Delta transition axes: {m['delta']}")
    print(f"\nCell coordinates : {coords_path}")
    if args.h5ad:
        print(f"Gene correlations: {out_dir}/gene_correlations_L{{0,1,2}}.tsv")


if __name__ == "__main__":
    main()
