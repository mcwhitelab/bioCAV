#!/usr/bin/env python3
"""
cav_pair_spotlight.py — Per-pair spotlight plots in global CAV-space embedding.

Reads library_structure.json to discover all matched baseline/condition pairs,
computes a single global 2-D embedding (t-SNE / PCA / UMAP) from the chosen
hierarchy level, then for each pair produces one plot:

    • All cells → light gray background
    • Baseline cells (e.g. normal) → blue foreground
    • Condition cells (e.g. lung_cancer) → red foreground

Cell matching uses normalized string comparison (lowercase, underscore ↔ space)
against obs metadata columns.

Usage
-----
python specific_scripts/cav_pair_spotlight.py \\
    --coords        cav_library/.../results/hierarchy/cell_coordinates.tsv \\
    --lib-dir       cav_library/.../ \\
    --obs           cav_library/.../data/cells.h5ad \\
    --group-col     cell_type \\
    --context-col   tissue \\
    --condition-col disease \\
    --level         L2 \\
    --reducer       tsne \\
    --out-dir       cav_library/.../results/pair_spotlights/
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
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
from src.utils.data_loader import load_sequence_embeddings

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_library_structure(lib_dir: Path) -> Optional[dict]:
    path = lib_dir / "library_structure.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_obs(h5ad_path: str, columns: List[str]) -> pd.DataFrame:
    import anndata as ad
    adata = ad.read_h5ad(h5ad_path, backed="r")
    obs = adata.obs[columns].copy()
    adata.file.close()
    return obs


def make_reducer(name: str):
    if name == "tsne":
        return TSNE(n_components=2, random_state=42,
                    perplexity=30, max_iter=1000, init="pca", learning_rate="auto")
    elif name == "pca":
        return PCA(n_components=2, random_state=42)
    elif name == "umap":
        try:
            from umap import UMAP
        except ImportError:
            raise ImportError("Install umap-learn or use --reducer tsne / pca")
        return UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
    else:
        raise ValueError(f"Unknown reducer: {name}")


def safe_labels(series) -> np.ndarray:
    """Convert pandas series to string array, Categorical-safe."""
    return series.astype(str).replace("nan", "unknown").values


def norm(v: str) -> str:
    """Lowercase + unify underscore/space for fuzzy matching."""
    return v.lower().replace("_", " ").strip()


def cells_matching(aligned: pd.DataFrame,
                   group_col: str, context_col: str, condition_col: str,
                   group_val: str, context_val: str, cond_val: str) -> np.ndarray:
    """Boolean mask for cells matching (group, context, condition) using normalized comparison."""
    def match(col, val):
        if not col or col not in aligned.columns:
            return np.ones(len(aligned), bool)
        labels = safe_labels(aligned[col])
        nval = norm(val)
        return np.array([norm(l) == nval for l in labels])

    return match(group_col, group_val) & \
           match(context_col, context_val) & \
           match(condition_col, cond_val)


def parse_pair_parts(name: str, sep: str, group_level: int,
                     cond_level: int) -> tuple:
    """Return (group_val, context_val, cond_val) from a CAV name."""
    parts = name.split(sep)
    group_val   = parts[group_level]
    cond_val    = parts[cond_level]
    mid         = parts[group_level + 1 : cond_level]
    context_val = sep.join(mid) if mid else ""
    return group_val, context_val, cond_val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Per-pair spotlight plots in global CAV-space embedding.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--coords", required=True,
                        help="cell_coordinates.tsv from cav_hierarchy.py")
    parser.add_argument("--lib-dir", required=True,
                        help="CAV library directory containing library_structure.json")
    parser.add_argument("--obs", required=True,
                        help="h5ad file for cell metadata (obs columns)")
    parser.add_argument("--group-col", default="cell_type",
                        help="obs column for CAV group level (default: cell_type)")
    parser.add_argument("--context-col", default="tissue",
                        help="obs column for CAV context level (default: tissue)")
    parser.add_argument("--condition-col", default="disease",
                        help="obs column for CAV condition level (default: disease)")
    parser.add_argument("--level", default="L2",
                        choices=["L0", "L1", "L2", "delta"],
                        help="Hierarchy level to embed (default: L2)")
    parser.add_argument("--reducer", default="tsne",
                        choices=["umap", "tsne", "pca"],
                        help="Dimensionality reduction method (default: tsne)")
    parser.add_argument("--max-cells", type=int, default=20000,
                        help="Subsample for embedding (default: 20000)")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for per-pair PNG files")
    parser.add_argument("--panel-size", type=float, default=5.0,
                        help="Figure width/height in inches (default: 5.0)")
    parser.add_argument("--pairs", nargs="*", default=None,
                        help="Optional: only plot these baseline CAV names "
                             "(e.g. B_cell__breast__normal). Default: all pairs.")
    args = parser.parse_args()

    lib_dir = Path(args.lib_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load library structure
    # ------------------------------------------------------------------ #
    structure = load_library_structure(lib_dir)
    if structure is None:
        logger.error("No library_structure.json found — run analyze_cav_library.py first.")
        sys.exit(1)

    all_pairs  = structure.get("pairs", [])
    sep        = structure.get("separator", "__")
    group_lvl  = structure.get("group_level", 0)
    cond_lvl   = structure.get("condition_level", 2)

    if args.pairs:
        filter_set = set(args.pairs)
        all_pairs  = [p for p in all_pairs if p["baseline"] in filter_set]
    logger.info(f"Processing {len(all_pairs)} pairs")

    # ------------------------------------------------------------------ #
    # Load coordinates + compute global embedding ONCE
    # ------------------------------------------------------------------ #
    logger.info(f"Loading coordinates: {args.coords}")
    coords = pd.read_csv(args.coords, sep="\t", index_col=0)

    feature_cols = [c for c in coords.columns if c.startswith(args.level + "__")]
    if not feature_cols:
        raise ValueError(f"No '{args.level}__' columns in {args.coords}")
    logger.info(f"Using {len(feature_cols)} '{args.level}' axes")

    rng = np.random.default_rng(42)
    n   = min(args.max_cells, len(coords))
    idx = rng.choice(len(coords), n, replace=False)
    X   = coords[feature_cols].iloc[idx].values

    logger.info(f"Running {args.reducer.upper()} on {n} × {len(feature_cols)}...")
    emb = make_reducer(args.reducer).fit_transform(X)
    logger.info("Embedding complete.")

    # ------------------------------------------------------------------ #
    # Load obs metadata
    # ------------------------------------------------------------------ #
    obs_cols = list({args.group_col, args.context_col, args.condition_col})
    try:
        obs     = load_obs(args.obs, obs_cols)
        aligned = obs.reindex(coords.index.astype(str)).iloc[idx]
    except Exception as e:
        logger.error(f"Could not load obs metadata: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Per-pair spotlight plots
    # ------------------------------------------------------------------ #
    BLUE   = "#1f77b4"
    RED    = "#d62728"
    GRAY   = "#dddddd"
    fs     = args.panel_size

    xl, yl = {
        "umap": ("UMAP 1", "UMAP 2"),
        "tsne": ("t-SNE 1", "t-SNE 2"),
        "pca":  ("PC 1",    "PC 2"),
    }.get(args.reducer, ("Dim 1", "Dim 2"))

    n_saved = 0
    for pair in all_pairs:
        bname = pair["baseline"]
        cname = pair["condition"]

        g_val, ctx_val, b_val = parse_pair_parts(bname, sep, group_lvl, cond_lvl)
        _,     _,       c_val = parse_pair_parts(cname, sep, group_lvl, cond_lvl)

        base_mask = cells_matching(aligned,
                                   args.group_col, args.context_col, args.condition_col,
                                   g_val, ctx_val, b_val)
        cond_mask = cells_matching(aligned,
                                   args.group_col, args.context_col, args.condition_col,
                                   g_val, ctx_val, c_val)

        n_base = base_mask.sum()
        n_cond = cond_mask.sum()

        if n_base == 0 and n_cond == 0:
            logger.warning(f"No cells matched for {bname} / {cname} — skipping")
            continue

        logger.info(f"{g_val} | {ctx_val} | {b_val} ({n_base})  vs  {c_val} ({n_cond})")

        neither = ~base_mask & ~cond_mask

        fig, ax = plt.subplots(figsize=(fs, fs))

        # Gray background (all other cells)
        ax.scatter(emb[neither, 0], emb[neither, 1],
                   c=GRAY, s=3, alpha=0.2, linewidths=0, rasterized=True)

        # Baseline (blue)
        if n_base > 0:
            ax.scatter(emb[base_mask, 0], emb[base_mask, 1],
                       c=BLUE, s=12, alpha=0.75, linewidths=0,
                       label=f"{b_val.replace('_',' ')}  (n={n_base})",
                       rasterized=True)

        # Condition (red)
        if n_cond > 0:
            ax.scatter(emb[cond_mask, 0], emb[cond_mask, 1],
                       c=RED, s=12, alpha=0.75, linewidths=0,
                       label=f"{c_val.replace('_',' ')}  (n={n_cond})",
                       rasterized=True)

        ax.set_xlabel(xl, fontsize=10)
        ax.set_ylabel(yl, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(
            f"{g_val.replace('_',' ')}  ·  {ctx_val.replace('_',' ')}\n"
            f"{b_val.replace('_',' ')}  vs  {c_val.replace('_',' ')}",
            fontsize=10
        )
        ax.legend(fontsize=8, frameon=False, loc="upper right",
                  markerscale=1.5)

        fname = f"{bname}_vs_{c_val}.png"
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close()
        n_saved += 1

    logger.info(f"Saved {n_saved} spotlight plots to {out_dir}")


if __name__ == "__main__":
    main()
