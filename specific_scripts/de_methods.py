#!/usr/bin/env python3
"""
de_methods.py — Normal-vs-cancer differential expression for each
tissue × cell-type pair, using the same pair enumeration as
cav_gene_correlation.py (from library_structure.json).

(Renamed from de_pseudobulk.py 2026-09-13: the file offers three methods, and
the one the paper actually uses — --method mixedlm — is not pseudobulk. The old
name made the Methods section read as though a pseudobulk test had been run.)

Three methods available:
  --method deseq2    Pseudobulk DESeq2 via pydeseq2 (preferred).
                     Aggregates raw counts per donor, runs DESeq2.
                     Requires: pip install pydeseq2
  --method mixedlm   Per-gene linear mixed model, log2(CP10K + 1) ~ condition
                     with a random intercept per donor. This is the method the
                     paper's Figure 6 uses; see run_mixedlm_pair for why.
                     Requires: --donor-col
  --method wilcoxon  Per-cell Wilcoxon rank-sum via scanpy (fallback).
                     No donor column required.
                     log2FC = log2((mean_cancer + 1e-9) / (mean_normal + 1e-9))
                     on log1p-normalised counts.

Output mirrors cav_gene_correlation.py for direct comparison:
  <out-dir>/
    <pair_name>.tsv         — gene, gene_name, log2fc, pval, padj (sorted by |log2fc|)
    summary_heatmap.png     — top-N genes × pairs heatmap of log2FC
    summary_top_genes.tsv   — wide log2FC table used for heatmap

Usage
-----
python specific_scripts/de_methods.py \\
    --h5ad          cav_library/.../data/cells.h5ad \\
    --lib-dir       cav_library/.../ \\
    --group-col     cell_type \\
    --context-col   tissue \\
    --condition-col disease \\
    --donor-col     donor_id \\
    --out-dir       cav_library/.../results/de_vs_cav/

# Compare with CAV gene correlations:
python specific_scripts/de_methods.py \\
    ... \\
    --cav-dir  cav_library/.../results/gene_correlation/ \\
    --out-dir  cav_library/.../results/de_vs_cav/
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import scipy.stats as stats

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (shared with cav_gene_correlation.py)
# ---------------------------------------------------------------------------

def load_library_structure(lib_dir: Path) -> Optional[dict]:
    path = lib_dir / "library_structure.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def norm(v: str) -> str:
    return v.lower().replace("_", " ").strip()


def safe_labels(series) -> np.ndarray:
    return series.astype(str).replace("nan", "unknown").values


def parse_pair_parts(name: str, sep: str, group_level: int,
                     cond_level: int) -> tuple:
    parts = name.split(sep)
    group_val   = parts[group_level]
    cond_val    = parts[cond_level]
    mid         = parts[group_level + 1 : cond_level]
    context_val = sep.join(mid) if mid else ""
    return group_val, context_val, cond_val


def cells_matching(obs: pd.DataFrame,
                   group_col: str, context_col: str, condition_col: str,
                   group_val: str, context_val: str, cond_val: str) -> np.ndarray:
    def match(col, val):
        if not col or col not in obs.columns:
            return np.ones(len(obs), bool)
        labels = safe_labels(obs[col])
        nval = norm(val)
        return np.array([norm(l) == nval for l in labels])
    return (match(group_col, group_val) &
            match(context_col, context_val) &
            match(condition_col, cond_val))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_adata(h5ad_path: str, gene_name_col: str = ""):
    """Load AnnData, return adata + gene_name_map."""
    import anndata as ad
    logger.info(f"Loading h5ad: {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    logger.info(f"  {adata.n_obs} cells × {adata.n_vars} genes")
    logger.info(f"  obs columns: {list(adata.obs.columns)}")
    logger.info(f"  var columns: {list(adata.var.columns)}")

    gene_ids = list(adata.var_names)
    gene_name_map = {}
    col = gene_name_col
    if not col:
        for candidate in ["feature_name", "gene_name", "gene_symbol",
                          "symbol", "name", "Gene", "gene"]:
            if candidate in adata.var.columns:
                col = candidate
                break
    if col and col in adata.var.columns:
        gene_name_map = dict(zip(gene_ids, adata.var[col].astype(str).values))
        logger.info(f"  Gene name column: '{col}'")

    return adata, gene_name_map


# ---------------------------------------------------------------------------
# Pseudobulk DESeq2
# ---------------------------------------------------------------------------

def run_deseq2_pair(adata_pair, donor_col: str,
                    condition_col: str, baseline_value: str,
                    min_donors: int = 3) -> Optional[pd.DataFrame]:
    """
    Aggregate counts per donor, run DESeq2 (pydeseq2).
    Returns DataFrame with gene, log2fc, pval, padj or None if insufficient donors.
    """
    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
    except ImportError:
        raise ImportError("pydeseq2 not installed. Run: pip install pydeseq2")

    import scipy.sparse as sp

    obs = adata_pair.obs.copy()
    obs["_condition"] = obs[condition_col].astype(str)
    obs["_donor"]     = obs[donor_col].astype(str) if donor_col else "single_donor"

    # Aggregate raw counts per donor
    donors     = obs["_donor"].unique()
    conditions = obs["_condition"].unique()

    n_cond    = {c: obs[obs["_condition"] == c]["_donor"].nunique() for c in conditions}
    min_n     = min(n_cond.values()) if n_cond else 0
    if min_n < min_donors:
        logger.warning(f"  Insufficient donors per condition {dict(n_cond)} "
                       f"(need ≥ {min_donors}) — skipping DESeq2")
        return None

    rows, meta_rows = [], []
    for donor in sorted(donors):
        mask = (obs["_donor"] == donor).values
        X    = adata_pair.X[mask]
        if sp.issparse(X):
            X = X.toarray()
        rows.append(X.sum(axis=0))
        cond = obs.loc[obs["_donor"] == donor, "_condition"].iloc[0]
        meta_rows.append({"sample": donor, "condition": cond})

    counts_df = pd.DataFrame(
        np.vstack(rows).astype(int),
        index=[r["sample"] for r in meta_rows],
        columns=adata_pair.var_names,
    )
    meta_df = pd.DataFrame(meta_rows).set_index("sample")

    # Filter low-count genes
    keep = counts_df.sum(axis=0) >= len(donors)
    counts_df = counts_df.loc[:, keep]

    logger.info(f"  DESeq2: {len(donors)} donors, {counts_df.shape[1]} genes after filtering")

    dds = DeseqDataSet(
        counts=counts_df,
        metadata=meta_df,
        design="~condition",
        quiet=True,
    )
    dds.deseq2()

    # Determine the condition value that is NOT the baseline
    # Use normalised comparison to handle case/underscore/space mismatches
    unique_conds = meta_df["condition"].unique()
    logger.info(f"    condition values in pseudobulk: {list(unique_conds)}")
    cond_values = [v for v in unique_conds
                   if norm(str(v)) != norm(str(baseline_value))]
    # Also find the exact baseline string as it appears in the data
    baseline_actual = next(
        (v for v in unique_conds if norm(str(v)) == norm(str(baseline_value))),
        baseline_value,
    )
    if not cond_values:
        logger.warning(f"  Could not determine condition value for contrast. "
                       f"Unique values: {list(unique_conds)}, baseline: '{baseline_value}'")
        return None
    condition_value = cond_values[0]

    logger.info(f"    contrast: '{condition_value}' vs '{baseline_actual}'")
    ds = DeseqStats(
        dds,
        contrast=["condition", condition_value, baseline_actual],
        quiet=True,
    )
    ds.summary()

    res = ds.results_df.reset_index()
    res.columns = [c if c != "index" else "gene" for c in res.columns]
    if "gene" not in res.columns:
        res = res.rename(columns={res.columns[0]: "gene"})

    res = res.rename(columns={
        "log2FoldChange": "log2fc",
        "pvalue":         "pval",
        "padj":           "padj",
    })
    return res[["gene", "log2fc", "pval", "padj"]].dropna(subset=["pval"])


# ---------------------------------------------------------------------------
# Wilcoxon fallback
# ---------------------------------------------------------------------------

def run_wilcoxon_pair(adata_pair, condition_col: str,
                      baseline_value: str) -> pd.DataFrame:
    """
    Per-gene Wilcoxon rank-sum on log1p-normalised counts.
    Returns DataFrame with gene, log2fc, pval, padj.
    """
    import scipy.sparse as sp
    import scanpy as sc

    ad_tmp = adata_pair.copy()
    sc.pp.normalize_total(ad_tmp, target_sum=1e4)
    sc.pp.log1p(ad_tmp)

    obs = ad_tmp.obs[condition_col].astype(str).values
    is_baseline = np.array([norm(v) == norm(baseline_value) for v in obs])
    is_condition = ~is_baseline

    X = ad_tmp.X
    if sp.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)

    n_genes = X.shape[1]
    log2fc  = np.zeros(n_genes, dtype=np.float32)
    pvals   = np.ones(n_genes, dtype=np.float64)

    mean_b = X[is_baseline].mean(axis=0)
    mean_c = X[is_condition].mean(axis=0)
    log2fc  = np.log2((mean_c + 1e-9) / (mean_b + 1e-9))

    logger.info(f"  Wilcoxon: {is_baseline.sum()} baseline, "
                f"{is_condition.sum()} condition cells, {n_genes} genes")

    # Vectorize per-gene Wilcoxon is slow for 60k genes; use scipy
    for i in range(n_genes):
        b_vals = X[is_baseline, i]
        c_vals = X[is_condition, i]
        if b_vals.std() == 0 and c_vals.std() == 0:
            continue
        _, pvals[i] = stats.mannwhitneyu(c_vals, b_vals, alternative="two-sided")

    # BH correction
    from statsmodels.stats.multitest import multipletests
    _, padj, _, _ = multipletests(pvals, method="fdr_bh")

    return pd.DataFrame({
        "gene":   list(adata_pair.var_names),
        "log2fc": log2fc,
        "pval":   pvals,
        "padj":   padj,
    })


def run_mixedlm_pair(adata_pair, donor_col: str,
                     condition_col: str, baseline_value: str,
                     gene_ids: Optional[List[str]] = None,
                     min_donors: int = 3,
                     min_pct_cells: float = 0.01) -> Optional[pd.DataFrame]:
    """
    Per-gene linear mixed model: log2(CPM-normalised expr + 1) ~ condition,
    with a random intercept per donor.

    Rationale: naive per-cell tests (e.g. Wilcoxon on cells) treat every cell
    as an independent observation, which is pseudoreplication -- cells from
    the same donor share donor-level biology/technical variation, so p-values
    from a per-cell test are anti-conservative. Pseudobulk (DESeq2) is the
    conservative fix -- it collapses to one point per donor -- but that can
    make it badly underpowered when donor counts are modest (e.g. 9 vs. 13),
    even for a real, biologically large effect. A mixed model is the middle
    ground: it uses every cell (so it has much more power than pseudobulk)
    while still modelling donor as the correct unit of non-independence via
    the random intercept, so it doesn't overstate significance the way a
    naive per-cell test does.

    Returns DataFrame with gene, log2fc, pval, padj, converged, or None if
    insufficient donors.
    """
    import scipy.sparse as sp
    import statsmodels.formula.api as smf
    import warnings

    obs = adata_pair.obs.copy()
    obs["_condition"] = obs[condition_col].astype(str)
    obs["_donor"]     = obs[donor_col].astype(str)

    conditions = obs["_condition"].unique()
    n_cond = {c: obs[obs["_condition"] == c]["_donor"].nunique() for c in conditions}
    min_n  = min(n_cond.values()) if n_cond else 0
    if min_n < min_donors:
        logger.warning(f"  Insufficient donors per condition {dict(n_cond)} "
                       f"(need ≥ {min_donors}) — skipping mixed model")
        return None

    # Manual CPM-style total-count normalisation (avoids a scanpy import --
    # this environment has a scanpy/anndata version conflict unrelated to
    # this analysis). Library size is computed from ALL genes before any
    # gene subsetting below.
    X_full = adata_pair.X
    if sp.issparse(X_full):
        X_full = X_full.toarray()
    X_full = X_full.astype(np.float64)
    lib_size = X_full.sum(axis=1, keepdims=True)
    lib_size[lib_size == 0] = 1.0
    target_sum = 1e4
    X_norm = X_full / lib_size * target_sum

    gene_names_all = list(adata_pair.var_names)
    if gene_ids:
        keep_idx = [i for i, g in enumerate(gene_names_all) if g in set(gene_ids)]
        if not keep_idx:
            logger.warning("  None of the requested gene_ids found in var_names")
            return None
        X_full  = X_full[:, keep_idx]
        X_norm  = X_norm[:, keep_idx]
        gene_names_all = [gene_names_all[i] for i in keep_idx]

    # Detection-rate pre-filter: skip genes barely/never detected in this
    # pair -- a full mixed-model fit on an all-zero (or near-zero) gene is
    # wasted time and won't converge to anything meaningful anyway.
    n_cells   = X_full.shape[0]
    detected  = (X_full > 0).sum(axis=0)
    keep_expr = detected >= max(1, int(min_pct_cells * n_cells))
    if not keep_expr.all():
        logger.info(f"  Detection filter (≥{min_pct_cells:.0%} of {n_cells} cells): "
                    f"{keep_expr.sum()}/{len(keep_expr)} genes kept")
    X_norm = X_norm[:, keep_expr]
    gene_names_all = [g for g, k in zip(gene_names_all, keep_expr) if k]

    X = np.log2(X_norm + 1.0)

    is_baseline = np.array([norm(v) == norm(baseline_value) for v in obs["_condition"]])
    df_base = pd.DataFrame({
        "condition": (~is_baseline).astype(int),
        "donor":     obs["_donor"].values,
    })

    logger.info(f"  MixedLM: {(~is_baseline).sum()} condition, {is_baseline.sum()} "
                f"baseline cells across {obs['_donor'].nunique()} donors, "
                f"{len(gene_names_all)} genes")

    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, gene in enumerate(gene_names_all):
            df = df_base.copy()
            df["expr"] = X[:, i]
            if df["expr"].std() == 0:
                rows.append({"gene": gene, "log2fc": 0.0, "pval": 1.0,
                            "converged": False})
                continue
            try:
                model  = smf.mixedlm("expr ~ condition", df, groups=df["donor"])
                result = model.fit(reml=True, method="lbfgs")
                coef   = result.fe_params.get("condition", np.nan)
                pval   = result.pvalues.get("condition", np.nan)
                rows.append({"gene": gene, "log2fc": coef, "pval": pval,
                            "converged": result.converged})
            except Exception as e:
                rows.append({"gene": gene, "log2fc": np.nan, "pval": np.nan,
                            "converged": False})

    result_df = pd.DataFrame(rows).dropna(subset=["pval"])
    if result_df.empty:
        logger.warning("  No genes converged — skipping")
        return None

    from statsmodels.stats.multitest import multipletests
    _, padj, _, _ = multipletests(result_df["pval"], method="fdr_bh")
    result_df["padj"] = padj

    n_failed = (~result_df["converged"]).sum()
    if n_failed:
        logger.warning(f"  {n_failed}/{len(result_df)} genes did not converge")

    return result_df[["gene", "log2fc", "pval", "padj", "converged"]]


# ---------------------------------------------------------------------------
# Summary heatmap
# ---------------------------------------------------------------------------

def make_summary_heatmap(lfc_matrix: pd.DataFrame, out_path: str,
                          top_n: int = 50,
                          figsize_w: float = 14, figsize_h: float = 10):
    if lfc_matrix.empty:
        return
    max_abs = lfc_matrix.abs().max(axis=1)
    top_genes = max_abs.nlargest(top_n).index
    sub = lfc_matrix.loc[top_genes].copy()
    sub.columns = [c.replace("__", " | ").replace("_", " ") for c in sub.columns]
    sub.index   = [g.replace("_", " ") for g in sub.index]

    fig, ax = plt.subplots(figsize=(figsize_w, figsize_h))
    vmax = np.nanpercentile(lfc_matrix.abs().values, 95)
    sns.heatmap(sub, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                ax=ax, xticklabels=True, yticklabels=True,
                linewidths=0.2, linecolor="#dddddd")
    ax.set_title(f"Top {top_n} DE genes by |log2FC| (cancer vs. normal)\n"
                 f"red = higher in cancer, blue = higher in normal", fontsize=11)
    ax.set_xlabel("Pair", fontsize=10)
    ax.set_ylabel("Gene", fontsize=10)
    ax.tick_params(axis="x", labelsize=7, rotation=45)
    ax.tick_params(axis="y", labelsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved summary heatmap: {out_path}")


# ---------------------------------------------------------------------------
# CAV comparison
# ---------------------------------------------------------------------------

def continuum_genes_for_pair(cav: pd.DataFrame, top_n_each: int) -> pd.DataFrame:
    """Reproduce the exact gene selection used by cav_continuum_viz.py's
    load_gene_corr(top_n_each=...): top-N by positive r + top-N by negative r.
    Returns the selected rows (with a 'gene' column) from the CAV dataframe.
    """
    pos = cav[cav["r"] >= 0].nlargest(top_n_each, "r")
    neg = cav[cav["r"] <  0].nsmallest(top_n_each, "r")
    return pd.concat([pos, neg])


def compare_with_cav(de_dir: Path, cav_dir: Path, out_path: str,
                     top_n: int = 200,
                     lfc_threshold: float = 1.5,
                     label_top_n: int = 8,
                     highlight_genes: Optional[set] = None,
                     show_continuum_genes: bool = False,
                     continuum_top_n_each: int = 4,
                     continuum_top_n_each_overrides: Optional[Dict[str, int]] = None,
                     only_pairs: Optional[set] = None):
    """
    For each pair present in both de_dir and cav_dir, compute Spearman r
    between log2FC and CAV Pearson r across the top_n genes (by |log2FC|).
    Saves a scatter summary and a correlation table.

    Gene labelling:
      - Always labels the top label_top_n genes by distance from origin
        (high agreement in either direction).
      - Additionally highlights any genes in highlight_genes (by symbol).

    only_pairs, if given, restricts the comparison (and the figure grid) to
    that subset of pair labels -- useful for pulling out a handful of
    representative cases into their own figure without rerunning DE.
    """
    de_files  = {f.stem: f for f in de_dir.glob("*.tsv")}
    cav_files = {f.stem: f for f in cav_dir.glob("*.tsv")}
    shared    = sorted(set(de_files) & set(cav_files))
    if only_pairs:
        missing = only_pairs - set(shared)
        if missing:
            logger.warning(f"  --only-pairs not found in both dirs: {sorted(missing)}")
        shared = [p for p in shared if p in only_pairs]

    if not shared:
        logger.warning("No matching pair TSVs found between DE and CAV dirs.")
        return

    logger.info(f"Comparing {len(shared)} pairs (DE vs. CAV)")

    records  = []
    n_cols   = min(3, len(shared))
    n_rows   = max(1, (len(shared) + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 5 * n_rows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for idx, pair in enumerate(shared):
        try:
            de  = pd.read_csv(de_files[pair],  sep="\t")
            cav = pd.read_csv(cav_files[pair], sep="\t")
        except Exception as e:
            logger.warning(f"  {pair}: could not read file — {e}, skipping")
            continue
        if de.empty or cav.empty:
            logger.warning(f"  {pair}: empty file — skipping")
            continue
        if "log2fc" not in de.columns:
            logger.warning(f"  {pair}: DE file missing 'log2fc' column "
                           f"(has: {list(de.columns)}) — skipping")
            continue
        if "r" not in cav.columns:
            logger.warning(f"  {pair}: CAV file missing 'r' column "
                           f"(has: {list(cav.columns)}) — skipping")
            continue

        # Merge DE and CAV on gene ID; bring in gene_name from whichever has it
        de_cols  = ["gene", "log2fc"] + (["gene_name"] if "gene_name" in de.columns else [])
        cav_cols = ["gene", "r"]      + (["gene_name"] if "gene_name" in cav.columns else [])
        merged = (de[de_cols].rename(columns={"log2fc": "lfc"})
                  .merge(cav[cav_cols].rename(columns={"r": "cav_r",
                                                        "gene_name": "gene_name_cav"}),
                         on="gene", how="inner")
                  .dropna(subset=["lfc", "cav_r"]))

        # Resolve display name: prefer DE gene_name, fall back to CAV, then ID
        if "gene_name" in merged.columns:
            merged["display"] = merged["gene_name"].fillna(merged["gene"])
        elif "gene_name_cav" in merged.columns:
            merged["display"] = merged["gene_name_cav"].fillna(merged["gene"])
        else:
            merged["display"] = merged["gene"]

        if len(merged) < 10:
            logger.warning(f"  {pair}: only {len(merged)} shared genes — skipping")
            continue

        # Continuum genes: the exact top-N-each-by-r genes shown in the
        # cav_continuum_viz.py strip plots for this pair. Determined from
        # the CAV file directly (before the lfc threshold below) so they
        # appear on the scatter even if their DE swing is small.
        continuum_ids: set = set()
        if show_continuum_genes:
            n_each = continuum_top_n_each
            if continuum_top_n_each_overrides:
                for key, val in continuum_top_n_each_overrides.items():
                    if key in pair:
                        n_each = val
                        break
            continuum_df = continuum_genes_for_pair(cav, n_each)
            continuum_ids = set(continuum_df["gene"])

        # Restrict to genes passing |log2FC| threshold, but always keep
        # continuum genes and explicitly highlighted genes so they remain
        # visible on the plot regardless of their DE magnitude.
        keep_mask = merged["lfc"].abs() >= lfc_threshold
        if continuum_ids:
            keep_mask = keep_mask | merged["gene"].isin(continuum_ids)
        if highlight_genes:
            keep_mask = keep_mask | merged["display"].isin(highlight_genes)
        merged = merged.loc[keep_mask].copy()

        rho, pval = stats.spearmanr(merged["lfc"], merged["cav_r"])
        records.append({"pair": pair, "spearman_r": rho, "pval": pval,
                        "n_genes": len(merged)})
        logger.info(f"  {pair}: Spearman r={rho:.3f} (p={pval:.2e}, n={len(merged)})")

        ax = axes_flat[idx]

        is_highlight = (merged["display"].isin(highlight_genes)
                        if highlight_genes else pd.Series(False, index=merged.index))
        is_continuum = (merged["gene"].isin(continuum_ids) & ~is_highlight
                        if continuum_ids else pd.Series(False, index=merged.index))
        is_background = ~is_highlight & ~is_continuum

        # Colour points: highlighted (checkpoint) genes red stars,
        # continuum genes gold circles, everything else background blue
        ax.scatter(merged.loc[is_background, "lfc"],
                   merged.loc[is_background, "cav_r"],
                   s=6, alpha=0.4, color="#2277bb", zorder=2)
        if is_continuum.any():
            ax.scatter(merged.loc[is_continuum, "lfc"],
                       merged.loc[is_continuum, "cav_r"],
                       s=26, alpha=0.9, facecolor="#f0a500", edgecolor="#7a5300",
                       linewidths=0.6, zorder=3, marker="o")
        if is_highlight.any():
            ax.scatter(merged.loc[is_highlight, "lfc"],
                       merged.loc[is_highlight, "cav_r"],
                       s=30, alpha=0.9, color="#cc3333", zorder=4, marker="*")

        ax.axhline(0, color="#aaaaaa", lw=0.5)
        ax.axvline(0, color="#aaaaaa", lw=0.5)
        ax.set_xlabel("log2FC (DE)", fontsize=8)
        ax.set_ylabel("Pearson r (CAV)", fontsize=8)
        ax.set_title(f"{pair.replace('__', ' | ')}\nSpearman r={rho:.2f}",
                     fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)

        # Label genes: top label_top_n by Euclidean distance from origin
        # (captures high-agreement genes in any quadrant)
        # Plus any explicitly highlighted genes not already in the top set
        norm_lfc   = merged["lfc"]   / (merged["lfc"].abs().max()   + 1e-9)
        norm_r     = merged["cav_r"] / (merged["cav_r"].abs().max() + 1e-9)
        merged["dist"] = np.sqrt(norm_lfc**2 + norm_r**2)

        auto_label = set(merged.nlargest(label_top_n, "dist").index)
        hl_label   = set(merged[is_highlight].index) if is_highlight.any() else set()
        cont_label = set(merged[is_continuum].index) if is_continuum.any() else set()
        label_idx  = auto_label | hl_label | cont_label

        for i in label_idx:
            row = merged.loc[i]
            if highlight_genes and row["display"] in highlight_genes:
                color = "#cc3333"
            elif i in cont_label:
                color = "#7a5300"
            else:
                color = "#333333"
            ax.annotate(row["display"],
                        xy=(row["lfc"], row["cav_r"]),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=6, color=color,
                        arrowprops=dict(arrowstyle="-", color="#aaaaaa",
                                        lw=0.5))

    # Hide unused subplots
    for ax in axes_flat[len(shared):]:
        ax.set_visible(False)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2277bb",
               alpha=0.6, markersize=6, label="other genes (|log2FC| ≥ threshold)"),
    ]
    if show_continuum_genes:
        legend_handles.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#f0a500",
                   markeredgecolor="#7a5300", markersize=7,
                   label="top continuum gene (shown in strip plot)")
        )
    if highlight_genes:
        legend_handles.append(
            Line2D([0], [0], marker="*", color="w", markerfacecolor="#cc3333",
                   markersize=10, label="curated highlight gene")
        )

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    title = "DE log2FC vs. CAV gene correlation (top genes by |log2FC|)"
    fig.suptitle(title, fontsize=11, y=0.995)
    fig.legend(handles=legend_handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.975),
               ncol=len(legend_handles), fontsize=8, frameon=False)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved comparison scatter: {out_path}")

    summary = pd.DataFrame(records).sort_values("spearman_r", ascending=False)
    summary_path = Path(out_path).parent / "de_vs_cav_spearman.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    logger.info(f"Saved Spearman summary: {summary_path}")
    print("\nDE vs. CAV Spearman correlations:")
    print(summary.to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pseudobulk DE for each tissue×cell_type pair, "
                    "comparable to cav_gene_correlation.py output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--h5ad",     default=None,
                        help="AnnData h5ad. Not required when --compare-only is set.")
    parser.add_argument("--lib-dir",  required=True,
                        help="CAV library dir containing library_structure.json")
    parser.add_argument("--group-col",     default="cell_type")
    parser.add_argument("--context-col",   default="tissue")
    parser.add_argument("--condition-col", default="disease")
    parser.add_argument("--donor-col",     default="",
                        help="obs column for donor/sample ID (required for DESeq2).")
    parser.add_argument("--baseline-value", default="normal",
                        help="Value in condition-col that is baseline (default: normal)")
    parser.add_argument("--method",
                        choices=["deseq2", "wilcoxon", "mixedlm"], default="wilcoxon",
                        help="DE method: deseq2 (pseudobulk, needs --donor-col), "
                             "wilcoxon (per-cell, default), or mixedlm (per-cell "
                             "expression ~ condition with a donor random intercept -- "
                             "needs --donor-col, more powerful than pseudobulk with "
                             "few donors while still accounting for donor "
                             "non-independence, unlike wilcoxon).")
    parser.add_argument("--mixedlm-genes", nargs="*", default=None,
                        help="Restrict --method mixedlm to these gene symbols "
                             "(resolved via --gene-name-col). Omit to run "
                             "genome-wide (slower: ~0.1-0.2s/gene/pair).")
    parser.add_argument("--mixedlm-min-pct-cells", type=float, default=0.01,
                        help="Skip genes detected in fewer than this fraction "
                             "of cells in the pair before fitting (default: 0.01).")
    parser.add_argument("--out-dir",  required=True)
    parser.add_argument("--gene-name-col", default="",
                        help="adata.var column for human-readable gene names.")
    parser.add_argument("--min-cells",  type=int, default=10)
    parser.add_argument("--min-donors", type=int, default=3,
                        help="Minimum donors per condition for DESeq2 (default: 3).")
    parser.add_argument("--top-n",      type=int, default=50,
                        help="Top N genes for summary heatmap (default: 50).")
    parser.add_argument("--pairs", nargs="*", default=None,
                        help="Run only these baseline CAV names.")
    parser.add_argument("--cav-dir", default=None,
                        help="Directory of cav_gene_correlation.py output TSVs. "
                             "If provided, computes Spearman r between log2FC and CAV r.")
    parser.add_argument("--compare-only", action="store_true",
                        help="Skip DE computation entirely and jump straight to "
                             "the DE vs. CAV comparison plots. Requires --cav-dir "
                             "and --out-dir to already contain DE TSVs.")
    parser.add_argument("--lfc-threshold", type=float, default=1.5,
                        help="Minimum |log2FC| to include a gene in the comparison "
                             "scatter (default: 1.5).")
    parser.add_argument("--label-top-n", type=int, default=8,
                        help="Label the top N genes by agreement (distance from "
                             "origin in normalised space) on each scatter (default: 8).")
    parser.add_argument("--highlight-genes", nargs="*", default=None,
                        help="Gene symbols to highlight in red on the scatter "
                             "(e.g. PDCD1 LAG3 HAVCR2 TIGIT TOX CTLA4). "
                             "Can also pass a file path ending in .txt with one "
                             "gene per line.")
    parser.add_argument("--heatmap-width",  type=float, default=14.0)
    parser.add_argument("--heatmap-height", type=float, default=10.0)
    parser.add_argument("--show-continuum-genes", action="store_true",
                        help="Highlight (gold circles) the exact top-N-each-by-r "
                             "genes shown in the matching cav_continuum_viz.py "
                             "strip plot for each pair, so the two figures can "
                             "be cross-referenced gene-for-gene.")
    parser.add_argument("--continuum-top-n-each", type=int, default=4,
                        help="Default top-N-each (pos + neg by r) to mark as "
                             "continuum genes (default: 4, matching the lung/"
                             "breast/colorectal continuum figures).")
    parser.add_argument("--only-pairs", nargs="*", default=None,
                        help="Restrict the DE-vs-CAV comparison figure to these "
                             "pair labels (e.g. neutrophil__breast__normal_vs_breast_cancer), "
                             "pulling a subset out into its own figure without "
                             "rerunning DE. Applies in both normal and --compare-only mode.")
    parser.add_argument("--scatter-out", default=None,
                        help="Override the output path for the comparison scatter "
                             "PNG (and its sibling de_vs_cav_spearman.tsv, written "
                             "alongside it). Default: <out-dir>/de_vs_cav_scatter.png.")
    parser.add_argument("--continuum-top-n-each-override", nargs="*", default=None,
                        help="Per-pair-substring overrides as key=value pairs, "
                             "e.g. skin_epidermis=5 (melanoma continuum figure "
                             "used top-n-each 5, not the default 4).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve highlight genes (list of symbols or a .txt file)
    highlight_genes = None
    if args.highlight_genes:
        if len(args.highlight_genes) == 1 and args.highlight_genes[0].endswith(".txt"):
            highlight_genes = set(Path(args.highlight_genes[0]).read_text().splitlines())
            logger.info(f"Loaded {len(highlight_genes)} highlight genes from file")
        else:
            highlight_genes = set(args.highlight_genes)
            logger.info(f"Highlighting genes: {highlight_genes}")

    continuum_overrides = None
    if args.continuum_top_n_each_override:
        continuum_overrides = {}
        for kv in args.continuum_top_n_each_override:
            k, v = kv.split("=")
            continuum_overrides[k] = int(v)

    only_pairs = set(args.only_pairs) if args.only_pairs else None
    scatter_out_path = Path(args.scatter_out) if args.scatter_out else (out_dir / "de_vs_cav_scatter.png")
    if args.scatter_out:
        scatter_out_path.parent.mkdir(parents=True, exist_ok=True)

    # --compare-only: skip DE, go straight to comparison plots
    if args.compare_only:
        if not args.cav_dir:
            logger.error("--compare-only requires --cav-dir")
            sys.exit(1)
        compare_with_cav(
            de_dir          = out_dir,
            cav_dir         = Path(args.cav_dir),
            out_path        = str(scatter_out_path),
            lfc_threshold   = args.lfc_threshold,
            label_top_n     = args.label_top_n,
            highlight_genes = highlight_genes,
            show_continuum_genes = args.show_continuum_genes,
            continuum_top_n_each = args.continuum_top_n_each,
            continuum_top_n_each_overrides = continuum_overrides,
            only_pairs      = only_pairs,
        )
        logger.info("Done (compare-only).")
        return

    # Library structure
    structure = load_library_structure(Path(args.lib_dir))
    if structure is None:
        logger.error("No library_structure.json — run analyze_cav_library.py first.")
        sys.exit(1)

    all_pairs   = structure.get("pairs", [])
    sep         = structure.get("separator", "__")
    group_lvl   = structure.get("group_level", 0)
    cond_lvl    = structure.get("condition_level", 2)
    baseline_vals = set(structure.get("baseline_values", ["normal"]))

    if args.pairs:
        filter_set = set(args.pairs)
        all_pairs  = [p for p in all_pairs if p["baseline"] in filter_set]
    logger.info(f"Processing {len(all_pairs)} pairs via {args.method}")

    # Load h5ad
    adata, gene_name_map = load_adata(args.h5ad, args.gene_name_col)
    adata.obs.index = adata.obs.index.astype(str)

    mixedlm_gene_ids = None
    if args.method == "mixedlm" and args.mixedlm_genes:
        name_to_id = {v: k for k, v in gene_name_map.items()}
        mixedlm_gene_ids = [name_to_id[g] for g in args.mixedlm_genes if g in name_to_id]
        missing = [g for g in args.mixedlm_genes if g not in name_to_id]
        if missing:
            logger.warning(f"  --mixedlm-genes not found: {missing}")
        logger.info(f"  Restricting mixedlm to {len(mixedlm_gene_ids)} genes")

    # Normalise obs index for matching
    lfc_collection: Dict[str, pd.Series] = {}

    for pair in all_pairs:
        bname = pair["baseline"]
        cname = pair["condition"]

        g_val, ctx_val, b_val = parse_pair_parts(bname, sep, group_lvl, cond_lvl)
        _,     _,       c_val = parse_pair_parts(cname, sep, group_lvl, cond_lvl)

        pair_label = f"{g_val}__{ctx_val}__{b_val}_vs_{c_val}"

        base_mask = cells_matching(adata.obs, args.group_col, args.context_col,
                                   args.condition_col, g_val, ctx_val, b_val)
        cond_mask = cells_matching(adata.obs, args.group_col, args.context_col,
                                   args.condition_col, g_val, ctx_val, c_val)
        pair_mask = base_mask | cond_mask

        n_base = base_mask.sum()
        n_cond = cond_mask.sum()
        n_pair = pair_mask.sum()

        if n_pair < args.min_cells:
            logger.warning(f"  {pair_label}: only {n_pair} cells — skipping")
            continue

        logger.info(f"  {g_val} | {ctx_val} | {b_val} ({n_base}) vs {c_val} ({n_cond})")

        adata_pair = adata[pair_mask].copy()

        try:
            if args.method == "deseq2":
                if not args.donor_col:
                    logger.error("--donor-col required for DESeq2")
                    sys.exit(1)
                result_df = run_deseq2_pair(
                    adata_pair, args.donor_col, args.condition_col,
                    args.baseline_value, min_donors=args.min_donors,
                )
            elif args.method == "mixedlm":
                if not args.donor_col:
                    logger.error("--donor-col required for mixedlm")
                    sys.exit(1)
                result_df = run_mixedlm_pair(
                    adata_pair, args.donor_col, args.condition_col,
                    args.baseline_value, gene_ids=mixedlm_gene_ids,
                    min_donors=args.min_donors,
                    min_pct_cells=args.mixedlm_min_pct_cells,
                )
            else:
                result_df = run_wilcoxon_pair(
                    adata_pair, args.condition_col, args.baseline_value,
                )
        except Exception as e:
            logger.warning(f"  {pair_label}: DE failed — {e}")
            continue

        if result_df is None or result_df.empty:
            logger.warning(f"  {pair_label}: no results")
            continue

        # Sort by |log2fc| descending
        result_df = (result_df
                     .assign(abs_lfc=result_df["log2fc"].abs())
                     .sort_values("abs_lfc", ascending=False)
                     .drop(columns="abs_lfc")
                     .reset_index(drop=True))

        if gene_name_map:
            result_df.insert(1, "gene_name",
                             result_df["gene"].map(gene_name_map).fillna(""))

        tsv_path = out_dir / f"{pair_label}.tsv"
        result_df.to_csv(tsv_path, sep="\t", index=False)
        top = result_df.iloc[0]
        logger.info(f"    → {tsv_path.name}  "
                    f"top: {top.get('gene_name', top['gene'])}  "
                    f"log2fc={top['log2fc']:.3f}")

        lfc_series = result_df.set_index("gene")["log2fc"]
        lfc_collection[pair_label] = lfc_series

    # Summary heatmap
    if lfc_collection:
        lfc_matrix = pd.DataFrame(lfc_collection).fillna(0)
        make_summary_heatmap(
            lfc_matrix, str(out_dir / "summary_heatmap.png"),
            top_n=args.top_n,
            figsize_w=args.heatmap_width, figsize_h=args.heatmap_height,
        )
        top_n_genes = lfc_matrix.abs().max(axis=1).nlargest(args.top_n).index
        lfc_matrix.loc[top_n_genes].to_csv(
            out_dir / "summary_top_genes.tsv", sep="\t")
        logger.info("Saved summary files.")
    else:
        logger.warning("No pairs produced results.")

    # CAV comparison
    if args.cav_dir:
        compare_with_cav(
            de_dir          = out_dir,
            cav_dir         = Path(args.cav_dir),
            out_path        = str(scatter_out_path),
            lfc_threshold   = args.lfc_threshold,
            label_top_n     = args.label_top_n,
            highlight_genes = highlight_genes,
            show_continuum_genes = args.show_continuum_genes,
            continuum_top_n_each = args.continuum_top_n_each,
            continuum_top_n_each_overrides = continuum_overrides,
            only_pairs      = only_pairs,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
