#!/usr/bin/env python3
"""
cav_gene_correlation.py — Correlate per-cell L2 projection scores with gene expression.

For each CAV pair identified in library_structure.json, computes Pearson correlation
between each cell's L2 projection score and its gene expression across all genes.
Produces a ranked gene list per pair (analogous to differential expression), plus
an optional summary heatmap of the top N genes across all pairs.

Conceptual parallel:
  Standard DE:             log2FC  (cancer mean − normal mean)
  CAV gene correlation:    Pearson r  (L2 score vs. gene expression)

Because L2 scores have cell-type and tissue variation orthogonalized out, the
correlations reflect pure disease signal, not confounders.

Usage
-----
python specific_scripts/cav_gene_correlation.py \\
    --coords   cav_library/.../results/hierarchy/cell_coordinates.tsv \\
    --h5ad     cav_library/.../data/cells.h5ad \\
    --lib-dir  cav_library/.../ \\
    --group-col     cell_type \\
    --context-col   tissue \\
    --condition-col disease \\
    --level    L2 \\
    --out-dir  cav_library/.../results/gene_correlation/

Outputs
-------
  <out-dir>/
    <pair_name>.tsv          — ranked gene table: gene, r, pval, padj
    summary_heatmap.png      — top-N genes × all pairs heatmap of r values
    summary_top_genes.tsv    — wide table of r values used for heatmap
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict

import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers shared with cav_pair_spotlight.py
# ---------------------------------------------------------------------------

def load_library_structure(lib_dir: Path) -> Optional[dict]:
    path = lib_dir / "library_structure.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def norm(v: str) -> str:
    """Lowercase + unify underscore/space for fuzzy matching."""
    return v.lower().replace("_", " ").strip()


def safe_labels(series) -> np.ndarray:
    """Convert pandas series to string array, Categorical-safe."""
    return series.astype(str).replace("nan", "unknown").values


def parse_pair_parts(name: str, sep: str, group_level: int,
                     cond_level: int) -> tuple:
    """Return (group_val, context_val, cond_val) from a CAV name."""
    parts = name.split(sep)
    group_val   = parts[group_level]
    cond_val    = parts[cond_level]
    mid         = parts[group_level + 1 : cond_level]
    context_val = sep.join(mid) if mid else ""
    return group_val, context_val, cond_val


def cells_matching(obs: pd.DataFrame,
                   group_col: str, context_col: str, condition_col: str,
                   group_val: str, context_val: str, cond_val: str) -> np.ndarray:
    """Boolean mask for cells matching (group, context, condition)."""
    def match(col, val):
        if not col or col not in obs.columns:
            return np.ones(len(obs), bool)
        labels = safe_labels(obs[col])
        nval = norm(val)
        return np.array([norm(l) == nval for l in labels])
    return match(group_col, group_val) & \
           match(context_col, context_val) & \
           match(condition_col, cond_val)


# ---------------------------------------------------------------------------
# Core correlation
# ---------------------------------------------------------------------------

def load_expression(h5ad_path: str, gene_name_col: str = "",
                    normalize: bool = True, target_sum: float = 1e4) -> tuple:
    """
    Load gene expression matrix from h5ad.
    Returns (X_dense, gene_ids, cell_ids, gene_name_map).
    gene_name_map: dict {gene_id -> display_name} (empty if col not found).
    X is returned as float32 to reduce memory.

    normalize: if True (default), library-size normalize each cell to
    `target_sum` total counts, then log1p -- same convention used by
    de_pseudobulk.py's mixedlm/wilcoxon methods. Without this, cells with
    very different total RNA content (sequencing depth, doublets, cell
    size) can dominate the correlation for any gene, independent of that
    gene's own biology -- e.g. a gene can show a strong positive raw-count
    correlation with the CAV axis purely because higher-scoring cells
    happen to have far more total transcript captured overall, even while
    that gene's *proportional* share of the transcriptome goes down.
    """
    import anndata as ad
    import scipy.sparse as sp

    logger.info(f"Loading h5ad: {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)

    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)

    if normalize:
        lib_size = X.sum(axis=1, keepdims=True)
        lib_size[lib_size == 0] = 1.0
        X = X / lib_size * target_sum
        X = np.log1p(X)
        logger.info(f"  Library-size normalized (target_sum={target_sum:.0f}) + log1p")

    gene_ids = list(adata.var_names)
    cell_ids = list(adata.obs_names)
    logger.info(f"  Expression matrix: {X.shape[0]} cells × {X.shape[1]} genes")
    logger.info(f"  var columns: {list(adata.var.columns)}")

    # Try to find a human-readable gene name column
    gene_name_map = {}
    col = gene_name_col or ""
    if not col:
        for candidate in ["feature_name", "gene_name", "gene_symbol",
                          "symbol", "name", "Gene", "gene"]:
            if candidate in adata.var.columns:
                col = candidate
                break
    if col and col in adata.var.columns:
        gene_name_map = dict(zip(gene_ids, adata.var[col].astype(str).values))
        logger.info(f"  Using gene name column: '{col}'")
    else:
        logger.info("  No gene name column found — output will use var_names only")

    return X, gene_ids, cell_ids, gene_name_map


def load_obs_columns(h5ad_path: str, columns: List[str]) -> pd.DataFrame:
    """Load only obs metadata columns from h5ad."""
    import anndata as ad
    adata = ad.read_h5ad(h5ad_path, backed="r")
    obs = adata.obs[columns].copy()
    adata.file.close()
    return obs


def correlate_scores_with_genes(scores: np.ndarray,
                                 expr: np.ndarray,
                                 gene_names: List[str],
                                 min_cells: int = 10) -> pd.DataFrame:
    """
    Pearson r between a 1-D score vector and each column of expr.
    Returns DataFrame sorted by |r| descending, with r, pval, padj columns.
    """
    n = len(scores)
    if n < min_cells:
        logger.warning(f"  Only {n} cells — skipping (need ≥ {min_cells})")
        return pd.DataFrame()

    # Vectorised Pearson r: centre + normalise scores once, then matrix-multiply
    s = scores - scores.mean()
    s_std = s.std()
    if s_std == 0:
        logger.warning("  Score vector has zero variance — skipping")
        return pd.DataFrame()
    s_norm = s / s_std                                         # (n,)

    # Centre each gene across these cells
    e      = expr - expr.mean(axis=0, keepdims=True)          # (n, G)
    e_std  = e.std(axis=0)                                     # (G,)
    valid  = e_std > 0
    e[:, valid] /= e_std[valid]

    r_all  = (s_norm @ e) / n                                  # (G,)
    r_all[~valid] = np.nan

    # Two-tailed p-value from t distribution
    t_stat = r_all * np.sqrt((n - 2) / np.maximum(1 - r_all**2, 1e-10))
    pvals  = 2 * stats.t.sf(np.abs(t_stat), df=n - 2)

    # Benjamini-Hochberg FDR
    padj = _bh_correction(pvals)

    df = pd.DataFrame({
        "gene":  gene_names,
        "r":     r_all,
        "pval":  pvals,
        "padj":  padj,
    })
    df = df.dropna(subset=["r"])
    df = df.assign(abs_r=df["r"].abs()) \
           .sort_values("abs_r", ascending=False) \
           .drop(columns="abs_r") \
           .reset_index(drop=True)
    return df


def _bh_correction(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(pvals)
    order   = np.argsort(pvals)
    ranks   = np.empty_like(order)
    ranks[order] = np.arange(1, n + 1)
    padj = np.minimum(1.0, pvals * n / ranks)
    # Make monotone
    padj_sorted = padj[order]
    for i in range(n - 2, -1, -1):
        padj_sorted[i] = min(padj_sorted[i], padj_sorted[i + 1])
    padj[order] = padj_sorted
    return padj


# ---------------------------------------------------------------------------
# Summary heatmap
# ---------------------------------------------------------------------------

def make_summary_heatmap(r_matrix: pd.DataFrame, out_path: str,
                          top_n: int = 50,
                          figsize_w: float = 14, figsize_h: float = 10):
    """
    Heatmap of top_n genes (by max |r| across any pair) × all pairs.
    """
    if r_matrix.empty:
        logger.warning("No data for summary heatmap.")
        return

    # Select top_n genes by max absolute r across any pair
    max_abs_r = r_matrix.abs().max(axis=1)
    top_genes = max_abs_r.nlargest(top_n).index
    sub = r_matrix.loc[top_genes]

    # Shorten pair column names for readability
    sub.columns = [c.replace("__", " | ").replace("_", " ") for c in sub.columns]
    sub.index   = [g.replace("_", " ") for g in sub.index]

    fig, ax = plt.subplots(figsize=(figsize_w, figsize_h))
    sns.heatmap(
        sub,
        cmap="RdBu_r",
        center=0,
        vmin=-1, vmax=1,
        ax=ax,
        xticklabels=True,
        yticklabels=True,
        linewidths=0.2,
        linecolor="#dddddd",
    )
    ax.set_title(f"Top {top_n} genes by CAV L2 score correlation\n(red = higher in disease, blue = higher in normal)",
                 fontsize=11)
    ax.set_xlabel("CAV pair", fontsize=10)
    ax.set_ylabel("Gene", fontsize=10)
    ax.tick_params(axis="x", labelsize=7, rotation=45)
    ax.tick_params(axis="y", labelsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved summary heatmap: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Correlate CAV L2 scores with gene expression.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--coords", required=True,
                        help="cell_coordinates.tsv from cav_hierarchy.py")
    parser.add_argument("--h5ad", required=True,
                        help="h5ad file (must contain raw/normalised counts in .X)")
    parser.add_argument("--lib-dir", required=True,
                        help="CAV library dir containing library_structure.json")
    parser.add_argument("--group-col", default="cell_type",
                        help="obs column for group level (default: cell_type)")
    parser.add_argument("--context-col", default="tissue",
                        help="obs column for context level (default: tissue)")
    parser.add_argument("--condition-col", default="disease",
                        help="obs column for condition level (default: disease)")
    parser.add_argument("--level", default="L2",
                        choices=["L0", "L1", "L2", "delta"],
                        help="Which hierarchy level scores to use (default: L2)")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for per-pair TSVs and heatmap")
    parser.add_argument("--top-n", type=int, default=50,
                        help="Top N genes per pair to include in heatmap (default: 50)")
    parser.add_argument("--min-cells", type=int, default=10,
                        help="Skip pairs with fewer cells than this (default: 10)")
    parser.add_argument("--pairs", nargs="*", default=None,
                        help="Only run these baseline CAV names (default: all)")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Correlate against raw counts instead of library-size "
                             "normalized + log1p expression (default: normalize). "
                             "Only for reproducing old/pre-fix output -- raw counts "
                             "confound a gene's correlation with total per-cell "
                             "library size, which can differ hugely between "
                             "conditions for reasons unrelated to that gene.")
    parser.add_argument("--gene-name-col", default="",
                        help="adata.var column for human-readable gene names "
                             "(e.g. feature_name, gene_symbol). "
                             "Auto-detected if not specified.")
    parser.add_argument("--heatmap-width",  type=float, default=14.0)
    parser.add_argument("--heatmap-height", type=float, default=10.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Library structure
    # ------------------------------------------------------------------ #
    structure = load_library_structure(Path(args.lib_dir))
    if structure is None:
        logger.error("No library_structure.json — run analyze_cav_library.py first.")
        sys.exit(1)

    all_pairs = structure.get("pairs", [])
    sep       = structure.get("separator", "__")
    group_lvl = structure.get("group_level", 0)
    cond_lvl  = structure.get("condition_level", 2)

    if args.pairs:
        filter_set = set(args.pairs)
        all_pairs  = [p for p in all_pairs if p["baseline"] in filter_set]
    logger.info(f"Processing {len(all_pairs)} pairs")

    # ------------------------------------------------------------------ #
    # Load coordinates
    # ------------------------------------------------------------------ #
    logger.info(f"Loading coordinates: {args.coords}")
    coords = pd.read_csv(args.coords, sep="\t", index_col=0)

    feature_cols = [c for c in coords.columns if c.startswith(args.level + "__")]
    if not feature_cols:
        raise ValueError(f"No '{args.level}__' columns in {args.coords}")
    logger.info(f"Using {len(feature_cols)} '{args.level}' score columns")

    # ------------------------------------------------------------------ #
    # Load expression + obs
    # ------------------------------------------------------------------ #
    X_expr, gene_names, expr_cell_ids, gene_name_map = load_expression(
        args.h5ad, gene_name_col=args.gene_name_col, normalize=not args.no_normalize)
    expr_id_to_row = {cid: i for i, cid in enumerate(expr_cell_ids)}

    obs_cols = list({args.group_col, args.context_col, args.condition_col})
    obs = load_obs_columns(args.h5ad, obs_cols)

    # Normalise all indices to strings before aligning
    coords.index = coords.index.astype(str)
    obs.index    = obs.index.astype(str)
    expr_id_to_row = {str(k): v for k, v in expr_id_to_row.items()}

    shared_ids = coords.index.intersection(obs.index)
    logger.info(f"Shared cell IDs between coords and h5ad: {len(shared_ids)}")
    if len(shared_ids) == 0:
        logger.error("No shared cell IDs — check that cell_coordinates.tsv and h5ad use the same index.")
        sys.exit(1)

    coords_aligned = coords.reindex(shared_ids)
    obs_aligned    = obs.reindex(shared_ids)

    # Expression rows in the same order as shared_ids
    expr_rows = np.array([expr_id_to_row[cid] for cid in shared_ids
                          if cid in expr_id_to_row])
    shared_ids_expr = [cid for cid in shared_ids if cid in expr_id_to_row]
    if len(shared_ids_expr) < len(shared_ids):
        logger.warning(f"{len(shared_ids) - len(shared_ids_expr)} cells missing from expression matrix")

    coords_aligned = coords_aligned.reindex(shared_ids_expr)
    obs_aligned    = obs_aligned.reindex(shared_ids_expr)
    X_sub          = X_expr[expr_rows]

    logger.info(f"Final aligned: {len(shared_ids_expr)} cells")

    # ------------------------------------------------------------------ #
    # Per-pair correlation
    # ------------------------------------------------------------------ #
    r_collection: Dict[str, pd.Series] = {}   # pair_label -> r Series indexed by gene

    for pair in all_pairs:
        bname = pair["baseline"]
        cname = pair["condition"]

        g_val, ctx_val, b_val = parse_pair_parts(bname, sep, group_lvl, cond_lvl)
        _,     _,       c_val = parse_pair_parts(cname, sep, group_lvl, cond_lvl)

        # Find cells belonging to this pair (baseline OR condition)
        base_mask = cells_matching(obs_aligned,
                                   args.group_col, args.context_col, args.condition_col,
                                   g_val, ctx_val, b_val)
        cond_mask = cells_matching(obs_aligned,
                                   args.group_col, args.context_col, args.condition_col,
                                   g_val, ctx_val, c_val)
        pair_mask = base_mask | cond_mask

        n_base = base_mask.sum()
        n_cond = cond_mask.sum()
        n_pair = pair_mask.sum()

        if n_pair < args.min_cells:
            logger.warning(f"  {bname}: only {n_pair} cells total — skipping")
            continue

        pair_label = f"{g_val}__{ctx_val}__{b_val}_vs_{c_val}"
        logger.info(f"  {g_val} | {ctx_val} | {b_val} ({n_base}) vs {c_val} ({n_cond})")

        # L2 score column for this condition CAV
        score_col = f"{args.level}__{cname}"
        if score_col not in coords_aligned.columns:
            # Try baseline column
            score_col = f"{args.level}__{bname}"
        if score_col not in coords_aligned.columns:
            available = [c for c in coords_aligned.columns if args.level in c][:5]
            logger.warning(f"  Score column not found for {cname}. "
                           f"Tried: {args.level}__{cname} and {args.level}__{bname}. "
                           f"Sample available cols: {available}")
            continue
        logger.info(f"    using score column: {score_col}")

        scores = coords_aligned[score_col].values[pair_mask].astype(np.float32)
        expr   = X_sub[pair_mask]

        logger.info(f"    score range: [{scores.min():.3f}, {scores.max():.3f}]  "
                    f"expr shape: {expr.shape}")

        result_df = correlate_scores_with_genes(scores, expr, gene_names,
                                                min_cells=args.min_cells)
        if result_df.empty:
            logger.warning(f"  No gene correlations produced for {pair_label} — skipping")
            continue

        # Save per-pair TSV — insert gene_name column after gene if available
        if gene_name_map:
            result_df.insert(1, "gene_name",
                             result_df["gene"].map(gene_name_map).fillna(""))
        tsv_path = out_dir / f"{pair_label}.tsv"
        result_df.to_csv(tsv_path, sep="\t", index=False)
        logger.info(f"    → saved {tsv_path.name}  "
                    f"(top gene: {result_df.iloc[0]['gene']}, r={result_df.iloc[0]['r']:.3f})")

        # Store r series for heatmap
        r_series = result_df.set_index("gene")["r"]
        r_collection[pair_label] = r_series

    # ------------------------------------------------------------------ #
    # Summary heatmap
    # ------------------------------------------------------------------ #
    if r_collection:
        r_matrix = pd.DataFrame(r_collection).fillna(0)
        heatmap_path = out_dir / "summary_heatmap.png"
        make_summary_heatmap(
            r_matrix,
            str(heatmap_path),
            top_n=args.top_n,
            figsize_w=args.heatmap_width,
            figsize_h=args.heatmap_height,
        )

        # Also save the wide r-value table
        tsv_summary = out_dir / "summary_top_genes.tsv"
        max_abs_r = r_matrix.abs().max(axis=1)
        top_genes = max_abs_r.nlargest(args.top_n).index
        r_matrix.loc[top_genes].to_csv(tsv_summary, sep="\t")
        logger.info(f"Saved summary table: {tsv_summary}")
    else:
        logger.warning("No pairs produced results — no heatmap generated.")

    logger.info("Done.")


if __name__ == "__main__":
    main()
