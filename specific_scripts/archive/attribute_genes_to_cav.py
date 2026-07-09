#!/usr/bin/env python3
"""
Gene attribution: rank genes by their alignment with a trained CAV direction.

For each gene, computes the Pearson correlation between its normalized
expression level across training cells and each cell's CAV projection score.
Genes whose expression most strongly tracks the CAV direction are attributed
as the concept's top contributors.

Positive pearson_r  -> gene is more expressed in concept-positive cells
                       (e.g. highly expressed in menstrual-phase cells)
Negative pearson_r  -> gene is more expressed in concept-negative cells
                       (e.g. downregulated during menstrual phase)

No per-gene token embeddings are required -- only the sequence_embeddings
pkl, the trained CAV artifacts, and the original h5ad for expression values.

Usage
-----
python specific_scripts/attribute_genes_to_cav.py \\
    --cav-dir  cavs/C1_menstrual/ \\
    --pkl      embeddings/scrnaseq/c1.pkl \\
    --h5ad     data/GSE111976_C1_geneformer_ready.h5ad \\
    --pos-spans spans/c1_menstrual_pos.txt \\
    --neg-spans spans/c1_menstrual_neg.txt \\
    --out      cavs/C1_menstrual/gene_attribution.tsv

Loop over all three CAVs:
    for cav in C1_menstrual C1_prolif C1_secretory; do
        python specific_scripts/attribute_genes_to_cav.py \\
            --cav-dir  cavs/${cav}/ \\
            --pkl      embeddings/scrnaseq/c1.pkl \\
            --h5ad     data/GSE111976_C1_geneformer_ready.h5ad \\
            --pos-spans spans/${cav}_pos.txt \\
            --neg-spans spans/${cav}_neg.txt \\
            --out      cavs/${cav}/gene_attribution.tsv \\
            --top-n 200
    done
"""

import sys
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Archived: not called by any live pipeline. Also predates the scaler-only
# simplification of compute_projections()/load_cav_artifacts() -- the
# artifacts['pca'] calls below are broken (that key no longer exists) and
# were not fixed, since this script isn't in use.
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "core"))

from src.evaluate import load_cav_artifacts, compute_projections
from src.utils.data_loader import load_sequence_embeddings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_span_ids(spans_path):
    """
    Read cell IDs from a spans file.
    Handles whole-sequence mode (one ID per line) and tab-separated formats
    (uses only the first column).  Skips blank lines and '#' comments.
    """
    ids = []
    with open(spans_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            ids.append(line.split('\t')[0])
    return ids


def pearson_r_all_genes(X_expr, cav_scores):
    """
    Compute Pearson r between each gene (column of X_expr) and cav_scores.

    Parameters
    ----------
    X_expr    : (n_cells, n_genes)  float32 normalized expression
    cav_scores: (n_cells,)          float32 CAV projection score per cell

    Returns
    -------
    r : (n_genes,)  Pearson correlation coefficient per gene
    """
    X_c = X_expr - X_expr.mean(axis=0)          # center genes
    y_c = cav_scores - cav_scores.mean()         # center scores

    numerator   = X_c.T @ y_c                   # (n_genes,)
    norm_X      = np.linalg.norm(X_c, axis=0)   # (n_genes,)
    norm_y      = float(np.linalg.norm(y_c))

    # Avoid divide-by-zero for genes with zero variance
    denom = norm_X * norm_y
    r = np.where(denom > 1e-12, numerator / denom, 0.0)
    return r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Rank genes by Pearson correlation between their expression "
            "and a cell's CAV projection score."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--cav-dir', required=True,
                        help='Directory with trained CAV artifacts '
                             '(concept_v1.npy, scaler_v1.pkl, pca_v1.pkl).')
    parser.add_argument('--pkl', required=True,
                        help='Embedding pkl with sequence_embeddings for all cells.')
    parser.add_argument('--h5ad', required=True,
                        help='AnnData .h5ad file (raw counts, var_names = Ensembl IDs). '
                             'var["gene_symbol"] used when present.')
    parser.add_argument('--pos-spans', required=True,
                        help='Spans file listing concept-positive cell IDs.')
    parser.add_argument('--neg-spans', required=True,
                        help='Spans file listing concept-negative cell IDs.')
    parser.add_argument('--out', required=True,
                        help='Output TSV file path.')
    parser.add_argument('--version', default='v1',
                        help='CAV artifact version suffix (default: v1).')
    parser.add_argument('--top-n', type=int, default=None,
                        help='Output only the top N genes by |pearson_r| '
                             '(default: all genes).')
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # 1. CAV artifacts
    # ------------------------------------------------------------------ #
    logger.info(f"Loading CAV artifacts from {args.cav_dir}")
    artifacts = load_cav_artifacts(args.cav_dir, version=args.version)

    # ------------------------------------------------------------------ #
    # 2. Sequence embeddings
    # ------------------------------------------------------------------ #
    logger.info(f"Loading embeddings from {args.pkl}")
    seq_embs, all_ids = load_sequence_embeddings(args.pkl)
    id_to_idx = {cid: i for i, cid in enumerate(all_ids)}
    logger.info(f"PKL: {len(all_ids)} cells, hidden_dim={seq_embs.shape[1]}")

    # ------------------------------------------------------------------ #
    # 3. Spans (positive / negative cell ID lists)
    # ------------------------------------------------------------------ #
    pos_ids = load_span_ids(args.pos_spans)
    neg_ids = load_span_ids(args.neg_spans)
    logger.info(f"Spans loaded: {len(pos_ids)} pos, {len(neg_ids)} neg")

    # Filter to cells present in the pkl
    pos_ids = [c for c in pos_ids if c in id_to_idx]
    neg_ids = [c for c in neg_ids if c in id_to_idx]
    logger.info(f"After PKL intersection: {len(pos_ids)} pos, {len(neg_ids)} neg")

    all_cell_ids = pos_ids + neg_ids
    if not all_cell_ids:
        raise ValueError("No cells found after intersecting spans with PKL. "
                         "Check that cell IDs match between spans files and pkl.info.")

    # ------------------------------------------------------------------ #
    # 4. CAV projection scores for all selected cells
    # ------------------------------------------------------------------ #
    indices_emb = [id_to_idx[c] for c in all_cell_ids]
    X_emb = seq_embs[indices_emb]                                   # (n, hidden_dim)
    cav_scores = compute_projections(
        X_emb,
        artifacts['concept_cav'],
        artifacts['scaler'],
        artifacts['pca'],
    ).astype(np.float32)                                             # (n,)

    mean_pos = cav_scores[:len(pos_ids)].mean()
    mean_neg = cav_scores[len(pos_ids):].mean()
    logger.info(
        f"CAV scores — pos mean: {mean_pos:.3f}, neg mean: {mean_neg:.3f} "
        f"(separation: {mean_pos - mean_neg:.3f})"
    )

    # ------------------------------------------------------------------ #
    # 5. h5ad: normalize and align cells
    # ------------------------------------------------------------------ #
    import anndata as ad
    import scanpy as sc

    logger.info(f"Loading h5ad: {args.h5ad}")
    adata = ad.read_h5ad(args.h5ad)
    logger.info(f"AnnData: {adata.n_obs} cells x {adata.n_vars} genes")

    # Normalize the same way as the embedding step
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    h5ad_cells = set(adata.obs_names)
    matched_ids = [c for c in all_cell_ids if c in h5ad_cells]

    n_missing = len(all_cell_ids) - len(matched_ids)
    if n_missing:
        logger.warning(
            f"{n_missing} cells from spans not found in h5ad obs_names — "
            f"proceeding with {len(matched_ids)} matched cells."
        )

    if not matched_ids:
        raise ValueError("No cells match between spans and h5ad. "
                         "Check that obs_names in h5ad match the cell IDs in spans files.")

    # Recompute CAV scores for matched cells only (some may have been dropped)
    matched_indices_emb = [id_to_idx[c] for c in matched_ids]
    X_emb_matched = seq_embs[matched_indices_emb]
    cav_scores_matched = compute_projections(
        X_emb_matched,
        artifacts['concept_cav'],
        artifacts['scaler'],
        artifacts['pca'],
    ).astype(np.float32)

    # Expression matrix for matched cells
    adata_matched = adata[matched_ids]
    X_expr = (
        adata_matched.X.toarray()
        if hasattr(adata_matched.X, 'toarray')
        else np.asarray(adata_matched.X)
    ).astype(np.float32)                                             # (n_matched, n_genes)

    logger.info(
        f"Computing Pearson r: {X_expr.shape[1]} genes x {X_expr.shape[0]} cells"
    )

    # ------------------------------------------------------------------ #
    # 6. Per-gene Pearson correlation
    # ------------------------------------------------------------------ #
    r_values = pearson_r_all_genes(X_expr, cav_scores_matched)

    # ------------------------------------------------------------------ #
    # 7. Build output dataframe
    # ------------------------------------------------------------------ #
    gene_ids = list(adata.var_names)

    if 'gene_symbol' in adata.var.columns:
        gene_symbols = list(adata.var['gene_symbol'])
    else:
        gene_symbols = gene_ids
        logger.warning("var['gene_symbol'] not found; using Ensembl IDs as gene symbols.")

    df = pd.DataFrame({
        'ensembl_id':  gene_ids,
        'gene_symbol': gene_symbols,
        'pearson_r':   r_values,
    })
    df['abs_r'] = df['pearson_r'].abs()
    df = df.sort_values('abs_r', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1
    df['direction'] = np.where(df['pearson_r'] > 0, 'pos_concept', 'neg_concept')
    df = df.drop(columns='abs_r')

    if args.top_n is not None:
        df = df.head(args.top_n)
        logger.info(f"Keeping top {args.top_n} genes by |pearson_r|")

    # ------------------------------------------------------------------ #
    # 8. Save
    # ------------------------------------------------------------------ #
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep='\t', index=False)
    logger.info(f"Saved gene attribution TSV: {out_path}")

    # Preview
    print("\n--- Top 20 genes ---")
    print(df.head(20).to_string(index=False))

    # Also show top genes in each direction separately
    top_pos = df[df['direction'] == 'pos_concept'].head(10)
    top_neg = df[df['direction'] == 'neg_concept'].head(10)
    print("\n--- Top 10 concept-positive genes (higher expression → higher CAV score) ---")
    print(top_pos[['rank', 'gene_symbol', 'ensembl_id', 'pearson_r']].to_string(index=False))
    print("\n--- Top 10 concept-negative genes (lower expression → higher CAV score) ---")
    print(top_neg[['rank', 'gene_symbol', 'ensembl_id', 'pearson_r']].to_string(index=False))


if __name__ == '__main__':
    main()
