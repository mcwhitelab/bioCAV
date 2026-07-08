#!/usr/bin/env python3
"""
compare_tool_temporal.py

Compare an external tool's predictions against CAV temporal holdout results.

The external tool provides a wide-format sparse CSV:
  - Header row: protein_id, GO:0000001, GO:0000016, ...
  - Data rows:  protein_id, [score or empty], ...
  - Empty cells = not predicted (below threshold).

The tool output is assumed to cover only the validation proteins; test negatives
are not in the file. Two complementary metrics are reported:

  1. Spearman correlation between CAV score and tool score across all
     (protein, GO term) validation pairs — requires no negatives.

  2. AUC (val positives vs test negatives), where test negatives receive
     tool score = 0.  This is reasonable because the tool did not flag those
     proteins; it becomes optimistic only if those proteins were never shown
     to the tool.  Treat accordingly.

  3. Recall at threshold: fraction of true val pairs where tool score > 0.

Usage
-----
python specific_scripts/compare_tool_temporal.py \\
    --tool-predictions  path/to/external_predictions.csv \\
    --results           results/temporal_eval_mf/eval_temporal_results.tsv \\
    --per-term-summary  results/temporal_eval_mf/eval_temporal_per_term_summary.tsv \\
    --out-dir           results/temporal_eval_mf/ \\
    --output            results/temporal_eval_mf/tool_comparison.tsv
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats as spstats
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_SKIP_PARTS = {"sp", "tr", "sw", "ref"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalise_pid(pid: str) -> str:
    """'sp|Q15185|TEBP_HUMAN' → 'Q15185'  |  'Q15185' → 'Q15185'"""
    parts = [p for p in str(pid).split("|") if p and p not in _SKIP_PARTS]
    return parts[0] if parts else str(pid)


def compute_auc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return np.nan
    y_true  = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_score = np.concatenate([pos_scores, neg_scores])
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return np.nan
    return float(roc_auc_score(y_true, y_score))


# ---------------------------------------------------------------------------
# Load external tool predictions → long-format DataFrame
# ---------------------------------------------------------------------------

def load_tool_predictions(csv_path: str, go_terms_needed: set) -> pd.DataFrame:
    """Read the wide CSV, keep only relevant GO columns, melt to long format.

    Returns a DataFrame with columns: protein_id (normalised), go_term, tool_score.
    Only rows where tool_score > 0 are returned (sparse representation).
    """
    logger.info(f"Reading external tool predictions: {csv_path}")
    wide = pd.read_csv(csv_path)

    pid_col   = wide.columns[0]
    go_cols   = [c for c in wide.columns[1:] if c in go_terms_needed]
    n_all_go  = wide.shape[1] - 1
    logger.info(f"  GO terms in file : {n_all_go}  |  overlapping with eval : {len(go_cols)}")
    logger.info(f"  Proteins in file : {len(wide)}")

    # Normalise protein IDs
    wide[pid_col] = wide[pid_col].astype(str).apply(normalise_pid)

    # Keep only relevant GO columns + protein ID
    wide = wide[[pid_col] + go_cols].copy()
    wide = wide.rename(columns={pid_col: "protein_id"})

    # Melt to long format; drop NaN / zero entries
    long = wide.melt(id_vars="protein_id", var_name="go_term", value_name="tool_score")
    long = long.dropna(subset=["tool_score"])
    long = long[long["tool_score"] > 0].reset_index(drop=True)
    long["tool_score"] = long["tool_score"].astype(float)

    logger.info(f"  Non-zero (protein, GO) predictions: {len(long)}")
    return long


def load_tool_predictions_long(tsv_path: str, go_terms_needed: set) -> pd.DataFrame:
    """Read a headerless 3-column TSV (full_fasta_header, go_term, score).

    Returns a DataFrame with columns: protein_id (normalised), go_term, tool_score.
    Only rows where tool_score > 0 are returned. Duplicate (protein, GO) pairs
    are resolved by keeping the maximum score.
    """
    logger.info(f"Reading long-format tool predictions: {tsv_path}")
    df = pd.read_csv(
        tsv_path, sep="\t", header=None,
        names=["protein_id", "go_term", "tool_score"],
    )
    logger.info(f"  Total rows: {len(df)}")

    df["protein_id"] = df["protein_id"].astype(str).apply(normalise_pid)
    df["tool_score"] = pd.to_numeric(df["tool_score"], errors="coerce")

    df = df[df["go_term"].isin(go_terms_needed)].copy()
    logger.info(f"  GO terms matching eval: {df['go_term'].nunique()}")

    df = df.dropna(subset=["tool_score"])
    df = df[df["tool_score"] > 0].reset_index(drop=True)

    df = df.groupby(["protein_id", "go_term"], as_index=False)["tool_score"].max()

    logger.info(f"  Non-zero (protein, GO) predictions: {len(df)}")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tool-predictions", required=True,
                        help="Predictions file from the external tool.")
    parser.add_argument("--tool-format", choices=["wide_csv", "long_tsv"], default="wide_csv",
                        help="File format: wide_csv (DeepGOWeb default) or long_tsv "
                             "(headerless 3-column TSV: full_header, go_term, score).")
    parser.add_argument("--results", required=True,
                        help="Path to eval_temporal_results.tsv (our method).")
    parser.add_argument("--per-term-summary", required=True,
                        help="Path to eval_temporal_per_term_summary.tsv (has CAV AUC).")
    parser.add_argument("--out-dir", required=True,
                        help="Directory containing {go_id}_test_neg_scores.tsv files.")
    parser.add_argument("--output", required=True,
                        help="Output TSV path for per-term comparison table.")
    # Optional — needed only for the per-protein rank scatter figure
    parser.add_argument("--val-pkl", default=None,
                        help="PKL of val protein embeddings (enables rank scatter).")
    parser.add_argument("--go-base-dirs", nargs="+", default=None,
                        help="Base dirs containing GO_XXXXXXX subdirs (glob patterns ok).")
    parser.add_argument("--scaler-pkl", default=None,
                        help="Shared reference population scaler pkl.")
    parser.add_argument("--version", default="v1",
                        help="CAV artifact version suffix (default: v1).")
    parser.add_argument("--figure-data-dir", default=None,
                        help="If provided, write figure-ready CSVs to this directory.")
    parser.add_argument("--label", default=None,
                        help="Short label for this run (e.g. 'mf', 'bp', 'cc') appended "
                             "to figure-data CSV filenames so multiple ontology runs "
                             "coexist in the same directory.")
    parser.add_argument("--llr-threshold", type=float, default=1.0,
                        help="LLR threshold for 'above threshold' in rank histogram (default: 1).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    # ------------------------------------------------------------------
    # Load our results
    # ------------------------------------------------------------------
    results = pd.read_csv(args.results, sep="\t")
    go_terms = set(results["go_term"].unique())
    logger.info(f"Loaded {len(results)} val pairs across {len(go_terms)} GO terms")

    summary = pd.read_csv(args.per_term_summary, sep="\t")
    summary = summary[summary["go_term"].isin(go_terms)].reset_index(drop=True)
    logger.info(f"Per-term summary restricted to {len(summary)} CAV-covered GO terms")

    results["protein_id_norm"] = results["protein_id"].apply(normalise_pid)

    # ------------------------------------------------------------------
    # Load tool predictions (long format, only nonzero scores)
    # ------------------------------------------------------------------
    if args.tool_format == "long_tsv":
        tool_long = load_tool_predictions_long(args.tool_predictions, go_terms_needed=go_terms)
    else:
        tool_long = load_tool_predictions(args.tool_predictions, go_terms_needed=go_terms)

    # ------------------------------------------------------------------
    # Join tool scores onto our validation pairs
    # ------------------------------------------------------------------
    # results has one row per (protein, go_term) where protein truly has that term
    merged = results.merge(
        tool_long,
        left_on=["protein_id_norm", "go_term"],
        right_on=["protein_id",     "go_term"],
        how="left",
    )
    merged["tool_score"] = merged["tool_score"].fillna(0.0)

    # ------------------------------------------------------------------
    # Global Spearman: CAV score vs tool score across all val pairs
    # ------------------------------------------------------------------
    r_global, p_global = spstats.spearmanr(
        merged["val_cav_score"], merged["tool_score"]
    )
    overall_recall = float((merged["tool_score"] > 0).mean())

    # ------------------------------------------------------------------
    # Per-GO-term stats
    # ------------------------------------------------------------------
    rows = []
    pr_data_cav  = []   # list of (recall, precision) per GO term, for macro-avg PR
    pr_data_tool = []
    pool_pos_cav, pool_neg_cav = [], []   # pooled CAV scores across all GO terms
    pool_pos_llr, pool_neg_llr = [], []   # pooled LLR scores across all GO terms
    pool_neg_tool              = []        # tool scores for negatives (parallel to pool_neg_llr)

    for go_id, grp in merged.groupby("go_term"):
        val_cav  = grp["val_cav_score"].values.astype(float)
        val_tool = grp["tool_score"].values.astype(float)

        n_nonzero = int((val_tool > 0).sum())
        recall    = float(n_nonzero / len(grp)) if len(grp) else np.nan

        # Spearman per term (only meaningful when ≥3 pairs and tool has variance)
        if len(grp) >= 3 and val_tool.std() > 0:
            r_term, p_term = spstats.spearmanr(val_cav, val_tool)
        else:
            r_term, p_term = np.nan, np.nan

        # AUC + PR: val positives vs test negatives.
        # Use real tool scores for negatives when available (e.g. DeepGOSE scores
        # both splits); fall back to 0 for proteins absent from tool output.
        neg_file = out_dir / f"{go_id}_test_neg_scores.tsv"
        if not neg_file.exists():
            neg_tool = np.array([])
            neg_cav  = np.array([])
        else:
            neg_df   = pd.read_csv(neg_file, sep="\t")
            neg_pids = neg_df["protein_id"].apply(normalise_pid).tolist()
            neg_cav  = neg_df["cav_score"].values.astype(float)
            pid_to_score = (
                tool_long[tool_long["go_term"] == go_id]
                .set_index("protein_id")["tool_score"]
                .to_dict()
            )
            neg_tool = np.array([pid_to_score.get(p, 0.0) for p in neg_pids])

        tool_auc = compute_auc(val_tool, neg_tool)

        # Collect per-term PR curve data for macro-averaging; also compute AUPR
        tool_aupr = np.nan
        if len(neg_cav) > 0:
            y_true = np.concatenate([np.ones(len(val_cav)), np.zeros(len(neg_cav))])
            if 0 < y_true.sum() < len(y_true):
                tool_aupr = float(average_precision_score(
                    y_true, np.concatenate([val_tool, neg_tool])
                ))
                for scores, store in [
                    (np.concatenate([val_cav,  neg_cav]),  pr_data_cav),
                    (np.concatenate([val_tool, neg_tool]), pr_data_tool),
                ]:
                    prec, rec, _ = precision_recall_curve(y_true, scores)
                    # sort by ascending recall for interpolation
                    idx = np.argsort(rec)
                    store.append((rec[idx], prec[idx]))

        # Collect pooled scores for density plots
        pool_pos_cav.extend(val_cav.tolist())
        pool_pos_llr.extend(grp["llr"].values.astype(float).tolist())
        if len(neg_cav) > 0:
            pool_neg_cav.extend(neg_cav.tolist())
            row0 = grp.iloc[0]
            pos_mu    = float(row0.get("pos_mean", np.nan))
            pos_sigma = float(row0.get("pos_std",  np.nan))
            neg_mu    = float(row0.get("neg_mean", np.nan))
            neg_sigma = float(row0.get("neg_std",  np.nan))
            if all(np.isfinite([pos_mu, pos_sigma, neg_mu, neg_sigma])) \
                    and pos_sigma > 0 and neg_sigma > 0:
                llr_neg = (spstats.norm.logpdf(neg_cav, pos_mu, pos_sigma) -
                           spstats.norm.logpdf(neg_cav, neg_mu, neg_sigma))
                pool_neg_llr.extend(llr_neg.tolist())
                pool_neg_tool.extend(neg_tool.tolist())

        rows.append({
            "go_term":                      go_id,
            "n_val_proteins":               len(grp),
            "tool_auc":                     tool_auc,
            "tool_aupr":                    tool_aupr,
            "tool_recall_at_threshold":     recall,
            "tool_n_val_nonzero":           n_nonzero,
            "spearman_r_cav_vs_tool":       r_term,
            "spearman_p_cav_vs_tool":       p_term,
        })

    term_df = pd.DataFrame(rows)

    # Merge with per-term summary (adds CAV AUC, go_term_name, depth, etc.)
    # Drop n_val_proteins from term_df — summary already has it
    compare = summary.merge(
        term_df.drop(columns=["n_val_proteins", "n_test_neg"], errors="ignore"),
        on="go_term", how="left",
    )
    compare["auc_diff_cav_minus_tool"] = (
        compare["auc_val_vs_test_neg"] - compare["tool_auc"]
    )

    compare.sort_values("auc_val_vs_test_neg", ascending=False, inplace=True)
    compare.to_csv(args.output, sep="\t", index=False, float_format="%.4f")
    logger.info(f"Comparison saved to {args.output}")

    # ------------------------------------------------------------------
    # Print global summary
    # ------------------------------------------------------------------
    valid = compare.dropna(subset=["auc_val_vs_test_neg", "tool_auc"])
    n = len(valid)

    cav_macro  = float(valid["auc_val_vs_test_neg"].mean())
    tool_macro = float(valid["tool_auc"].mean())
    mean_diff  = float(valid["auc_diff_cav_minus_tool"].mean())
    n_cav_wins  = int((valid["auc_diff_cav_minus_tool"] > 0).sum())
    n_tool_wins = int((valid["auc_diff_cav_minus_tool"] < 0).sum())

    print(f"\n{'='*60}")
    print(f"External tool vs CAV — temporal holdout comparison")
    print(f"{'='*60}")
    print(f"  GO terms compared                 : {n}")
    print(f"  Val (protein, GO) pairs           : {len(merged)}")

    print(f"\n--- Score agreement (no negatives needed) ---")
    print(f"  Spearman r  CAV vs tool score     : {r_global:+.3f}  (p={p_global:.2e})")
    print(f"  Tool recall at threshold          : {overall_recall:.1%}  "
          f"(val pairs with tool score > 0)")

    print(f"\n--- AUC comparison ---")
    print(f"  Macro-AUC  CAV                    : {cav_macro:.3f}")
    print(f"  Macro-AUC  tool (0-fill negs)     : {tool_macro:.3f}")
    print(f"  Mean AUC diff (CAV − tool)        : {mean_diff:+.3f}")
    print(f"  GO terms where CAV wins           : {n_cav_wins} / {n}")
    print(f"  GO terms where tool wins          : {n_tool_wins} / {n}")

    print(f"\n--- CAV AUC distribution ---")
    print(valid["auc_val_vs_test_neg"].describe().to_string())
    print(f"\n--- Tool AUC distribution ---")
    print(valid["tool_auc"].describe().to_string())

    # Display columns for tables — only keep those present in the merged DataFrame
    wanted = (
        ["go_term", "go_term_name", "n_val_proteins",
         "auc_val_vs_test_neg", "tool_auc",
         "auc_diff_cav_minus_tool", "tool_recall_at_threshold",
         "spearman_r_cav_vs_tool", "depth", "n_ancestors"]
    )
    display = [c for c in wanted if c in compare.columns]

    print(f"\n--- Top 10 GO terms by CAV AUC ---")
    print(compare.head(10)[display].to_string(index=False))

    print(f"\n--- Bottom 10 GO terms by CAV AUC ---")
    print(compare.tail(10)[display].to_string(index=False))

    print(f"\n--- GO terms where tool most outperforms CAV ---")
    print(compare.nsmallest(10, "auc_diff_cav_minus_tool")[display].to_string(index=False))

    make_figures(
        compare, merged, out_dir, pr_data_cav, pr_data_tool,
        pool_pos_cav=np.array(pool_pos_cav),
        pool_neg_cav=np.array(pool_neg_cav),
        pool_pos_llr=np.array(pool_pos_llr),
        pool_neg_llr=np.array(pool_neg_llr),
        pool_neg_tool=np.array(pool_neg_tool),
        figure_data_dir=args.figure_data_dir,
        label=args.label,
    )

    if args.figure_data_dir:
        fig_dir = Path(args.figure_data_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        label_suffix = f"_{args.label}" if args.label else ""

        compare.to_csv(
            fig_dir / f"temporal_tool_comparison{label_suffix}.csv",
            index=False, float_format="%.4f"
        )

        pair_cols = ["protein_id", "go_term", "val_cav_score", "llr",
                     "tool_score", "test_pos_percentile", "test_neg_percentile"]
        merged[[c for c in pair_cols if c in merged.columns]].to_csv(
            fig_dir / f"temporal_protein_pairs{label_suffix}.csv",
            index=False, float_format="%.4f"
        )

        logger.info(f"Figure data written to {fig_dir}")

    # ------------------------------------------------------------------
    # Per-protein rank scatter (optional — requires val embeddings + GO dirs)
    # ------------------------------------------------------------------
    if args.val_pkl and args.go_base_dirs and args.scaler_pkl:
        import glob as _glob
        go_base_dirs = [
            Path(d)
            for pattern in args.go_base_dirs
            for d in sorted(_glob.glob(pattern))
        ]
        make_rank_scatter(
            results=results,
            tool_predictions=args.tool_predictions,
            tool_format=args.tool_format,
            go_terms=go_terms,
            val_pkl=args.val_pkl,
            go_base_dirs=go_base_dirs,
            scaler_pkl=args.scaler_pkl,
            version=args.version,
            out_dir=out_dir,
            figure_data_dir=args.figure_data_dir,
            label=args.label,
            llr_threshold=args.llr_threshold,
        )
    else:
        logger.info("Skipping rank scatter (pass --val-pkl, --go-base-dirs, --scaler-pkl to enable)")


# ---------------------------------------------------------------------------
# Figures  (one PDF per plot)
# ---------------------------------------------------------------------------

RCPARAMS = {
    "font.size":        10,
    "axes.labelsize":   11,
    "axes.titlesize":   11,
    "legend.fontsize":   9,
    "figure.dpi":       150,
}

def _scatter_base(ax, tool_auc, cav_auc):
    """Draw diagonal reference line and set square equal-aspect limits."""
    lo = min(tool_auc.min(), cav_auc.min()) - 0.04
    hi = 1.03
    diag = np.linspace(lo, hi, 200)
    ax.plot(diag, diag, "--", color="0.6", lw=1, zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Tool AUC")
    ax.set_ylabel("CAV AUC")


def make_rank_scatter(
    results: pd.DataFrame,
    tool_predictions: str,
    tool_format: str,
    go_terms: set,
    val_pkl: str,
    go_base_dirs: list,
    scaler_pkl: str,
    version: str,
    out_dir: Path,
    figure_data_dir: str | None = None,
    label: str | None = None,
    llr_threshold: float = 1.0,
) -> None:
    """
    For each (val protein, true GO term) pair, compute:
      - CAV rank: rank of the true GO term's CAV score among all GO term CAVs
      - Tool rank: rank of the true GO term's tool score among all GO term tool scores
    Then plot a scatter of tool rank vs CAV rank.
    Both rankings are restricted to the GO terms with trained CAVs.
    """
    import glob as _glob
    import joblib
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
    from src.utils.data_loader import load_sequence_embeddings

    CAV_SUBDIR = "random_positive_train_max1000_cav"

    # ------------------------------------------------------------------
    # Load all GO term CAV vectors (restricted to go_terms)
    # ------------------------------------------------------------------
    cav_vectors = []
    cav_go_ids  = []
    for base in go_base_dirs:
        for go_dir in sorted(base.glob("GO_*")):
            go_id = go_dir.name.replace("GO_", "GO:")
            if go_id not in go_terms:
                continue
            cav_file = go_dir / CAV_SUBDIR / f"concept_{version}.npy"
            if cav_file.exists():
                cav_vectors.append(np.load(cav_file))
                cav_go_ids.append(go_id)

    if not cav_vectors:
        logger.warning("No CAV vectors found — skipping rank scatter")
        return

    n_go = len(cav_vectors)
    logger.info(f"Loaded {n_go} GO term CAV vectors for rank scoring")
    cav_matrix   = np.stack(cav_vectors, axis=0)   # (n_go, dim)
    cav_id_to_col = {gid: i for i, gid in enumerate(cav_go_ids)}

    # ------------------------------------------------------------------
    # Load val embeddings + scaler, compute score matrix
    # ------------------------------------------------------------------
    logger.info(f"Loading val embeddings: {val_pkl}")
    val_embs, val_ids = load_sequence_embeddings(val_pkl)

    _SKIP = {"sp", "tr", "sw", "ref"}
    id_to_idx = {}
    for i, sid in enumerate(val_ids):
        id_to_idx[sid] = i
        for part in sid.split("|"):
            if part and part not in _SKIP:
                id_to_idx.setdefault(part, i)

    scaler = joblib.load(scaler_pkl)
    val_preprocessed = scaler.transform(val_embs)          # (n_val, dim)
    score_matrix     = val_preprocessed @ cav_matrix.T     # (n_val, n_go)
    logger.info(f"Score matrix: {score_matrix.shape}")

    # ------------------------------------------------------------------
    # Re-load tool predictions restricted to ALL CAV GO terms (not just
    # val annotation GO terms) so rank scoring is fair across the full
    # set of terms the model knows about.
    # ------------------------------------------------------------------
    cav_go_set = set(cav_go_ids)
    logger.info(f"Re-loading tool predictions for {len(cav_go_set)} CAV GO terms")
    if tool_format == "long_tsv":
        tool_long = load_tool_predictions_long(tool_predictions, go_terms_needed=cav_go_set)
    else:
        tool_long = load_tool_predictions(tool_predictions, go_terms_needed=cav_go_set)
    logger.info(f"Tool predictions loaded: {len(tool_long)} rows")

    # ------------------------------------------------------------------
    # Tool score lookup: protein_id → {go_term: score}
    # ------------------------------------------------------------------
    tool_by_protein: dict[str, dict[str, float]] = {}
    for _, row in tool_long.iterrows():
        tool_by_protein.setdefault(row["protein_id"], {})[row["go_term"]] = float(row["tool_score"])

    # Pre-build array of tool scores for all go_ids in cav order (zeros where tool is silent)
    # We'll slice per-protein from this
    go_col_order = cav_go_ids   # same ordering as cav_matrix columns

    # ------------------------------------------------------------------
    # Compute per-(protein, GO term) ranks
    # ------------------------------------------------------------------
    # Restrict to val proteins only (those whose embeddings were loaded)
    val_pid_set = set(id_to_idx.keys())

    results_norm = results.copy()
    results_norm["_pid_norm"] = results_norm["protein_id"].apply(normalise_pid)
    results_norm = results_norm[results_norm["_pid_norm"].isin(val_pid_set)].reset_index(drop=True)
    logger.info(f"Val proteins in results: {results_norm['_pid_norm'].nunique()} proteins, "
                f"{len(results_norm)} rows")

    rows = []
    for _, pair in results_norm.iterrows():
        pid   = pair["_pid_norm"]
        go_id = pair["go_term"]

        idx = id_to_idx.get(pid)
        col = cav_id_to_col.get(go_id)
        if idx is None or col is None:
            continue

        # CAV rank
        cav_scores      = score_matrix[idx]              # (n_go,)
        true_cav_score  = float(cav_scores[col])
        cav_rank        = int((cav_scores > true_cav_score).sum()) + 1

        # Tool rank (fill missing with 0)
        prot_tool = tool_by_protein.get(pid, {})
        tool_scores_all = np.array([prot_tool.get(g, 0.0) for g in go_col_order])
        true_tool_score = float(prot_tool.get(go_id, 0.0))
        tool_rank       = int((tool_scores_all > true_tool_score).sum()) + 1

        rows.append({
            "protein_id":     pid,
            "go_term":        go_id,
            "cav_rank":       cav_rank,
            "tool_rank":      tool_rank,
            "tool_predicted": true_tool_score > 0,
            "llr":            float(pair.get("llr", np.nan)),
            "n_go_terms":     n_go,
        })

    if not rows:
        logger.warning("No rank data computed — skipping scatter")
        return

    rank_df = pd.DataFrame(rows)
    rank_file = out_dir / "go_specificity_ranks.tsv"
    rank_df.to_csv(rank_file, sep="\t", index=False)
    logger.info(f"Saved rank table: {rank_file}")

    if figure_data_dir:
        fig_dir = Path(figure_data_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        lsuffix = f"_{label}" if label else ""
        rank_df.to_csv(fig_dir / f"go_specificity_ranks{lsuffix}.csv", index=False)
        logger.info(f"Figure data: go_specificity_ranks{lsuffix}.csv → {fig_dir}")

    # Sanity check: for "not predicted" pairs, do those proteins have ANY tool predictions?
    not_pred = rank_df[~rank_df["tool_predicted"]]
    if len(not_pred) > 0:
        tool_proteins = set(tool_long["protein_id"].unique())
        n_with_any    = not_pred["protein_id"].isin(tool_proteins).sum()
        n_total_np    = len(not_pred)
        n_proteins_np = not_pred["protein_id"].nunique()
        print(f"\n{'='*55}")
        print(f"Sanity check — 'not predicted' pairs: {n_total_np} pairs, {n_proteins_np} unique proteins")
        print(f"  Of those proteins, {n_with_any}/{n_total_np} pairs belong to proteins")
        print(f"  that DO have at least one tool prediction (for any GO term).")
        print(f"  => tool is silent on specific GO term but active on protein: "
              f"{n_with_any} ({100*n_with_any/n_total_np:.1f}%)")
        print(f"  => tool makes no predictions for protein at all:             "
              f"{n_total_np - n_with_any} ({100*(n_total_np-n_with_any)/n_total_np:.1f}%)")
        silent_proteins = not_pred[~not_pred["protein_id"].isin(tool_proteins)]["protein_id"].unique()
        if len(silent_proteins) > 0:
            preview = silent_proteins[:10]
            print(f"\n  Preview of proteins with zero tool predictions for any CAV GO term ({len(silent_proteins)} total):")
            for pid in preview:
                print(f"    {pid}")
            if len(silent_proteins) > 10:
                print(f"    ... ({len(silent_proteins) - 10} more)")
        print(f"{'='*55}")

    # Summary
    n_pairs = len(rank_df)
    pct_cav1  = (rank_df["cav_rank"]  == 1).mean() * 100
    pct_tool1 = (rank_df["tool_rank"] == 1).mean() * 100
    med_cav   = rank_df["cav_rank"].median()
    med_tool  = rank_df["tool_rank"].median()
    print(f"\n{'='*55}")
    print(f"GO term specificity ranks  ({n_pairs} pairs, {n_go} GO terms)")
    print(f"  CAV  — median rank: {med_cav:.0f}  |  rank=1: {pct_cav1:.1f}%")
    print(f"  Tool — median rank: {med_tool:.0f}  |  rank=1: {pct_tool1:.1f}%")
    print(f"{'='*55}")

    # ------------------------------------------------------------------
    # Rank histograms: CAV (LLR > 1 only) and tool (predicted only),
    # with a "below threshold" / "not predicted" bar for the rest.
    # All proportions use n_total as denominator.
    # ------------------------------------------------------------------
    plt.rcParams.update(RCPARAMS)
    n_total   = len(rank_df)
    llr_vals  = rank_df["llr"].values
    predicted = rank_df["tool_predicted"].values

    above_thresh    = llr_vals > llr_threshold
    cav_ranks_above = rank_df.loc[above_thresh, "cav_rank"].values
    n_cav_below     = (~above_thresh).sum()

    tool_ranks_pred = rank_df.loc[predicted, "tool_rank"].values
    n_not_predicted = (~predicted).sum()

    # top-3 rates out of ALL pairs (including below-threshold / not-predicted)
    pct_cav_top3  = (above_thresh & (rank_df["cav_rank"].values  <= 3)).sum() / n_total * 100
    pct_tool_top3 = (predicted     & (rank_df["tool_rank"].values <= 3)).sum() / n_total * 100

    # Layout: rank bins 1..n_go, then gap, then extra bar
    not_pred_x = n_go + 3
    gap_x      = n_go + 2

    fig, (ax_cav, ax_tool) = plt.subplots(1, 2, figsize=(9, 4), sharey=True)

    # --- left: CAV ---
    cav_counts = np.bincount(cav_ranks_above, minlength=n_go + 1)[1:n_go + 1] / n_total
    ax_cav.bar(np.arange(1, n_go + 1), cav_counts,
               width=1.0, color="#1f77b4", alpha=0.75, edgecolor="none")
    ax_cav.axvline(gap_x - 0.5, color="0.6", lw=0.8, ls=":")
    ax_cav.bar(not_pred_x, n_cav_below / n_total,
               width=1.0, color="#1f77b4", alpha=0.35, edgecolor="none")
    ax_cav.set_ylabel("Proportion")
    ax_cav.set_xlabel("Rank of true GO term  (rank 1 = top score)")
    ax_cav.set_title(f"CAV  (LLR > {llr_threshold})", fontsize=10)
    ax_cav.set_xlim(0, not_pred_x + 1.5)

    # --- right: external tool ---
    tool_counts = np.bincount(tool_ranks_pred, minlength=n_go + 1)[1:n_go + 1] / n_total
    ax_tool.bar(np.arange(1, n_go + 1), tool_counts,
                width=1.0, color="#d62728", alpha=0.75, edgecolor="none")
    ax_tool.axvline(gap_x - 0.5, color="0.6", lw=0.8, ls=":")
    ax_tool.bar(not_pred_x, n_not_predicted / n_total,
                width=1.0, color="#d62728", alpha=0.35, edgecolor="none")
    ax_tool.set_xlabel("Rank of true GO term  (rank 1 = top score)")
    ax_tool.set_title("External tool", fontsize=10)
    ax_tool.set_xlim(0, not_pred_x + 1.5)

    fig.suptitle(
        f"GO term specificity rank distribution\n"
        f"({n_total} val protein–GO pairs, {n_go} GO terms)",
        fontsize=10,
    )

    # Shared x-ticks, different last label per panel
    base_ticks  = list(np.linspace(1, n_go, 6).astype(int))
    base_labels = [str(t) for t in base_ticks]
    for ax, last_lbl in [(ax_cav, "Below\nthresh."), (ax_tool, "Not\npred.")]:
        ax.set_xticks(base_ticks + [not_pred_x])
        ax.set_xticklabels(base_labels + [last_lbl], fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.9])
    p = out_dir / "fig_specificity_rank_density.pdf"
    fig.savefig(p)
    plt.close(fig)
    logger.info(f"Saved {p}")


def _macro_pr(pr_curves, recall_grid):
    """Interpolate per-term PR curves onto a common recall grid, return (mean, std)."""
    interped = []
    for rec, prec in pr_curves:
        interped.append(np.interp(recall_grid, rec, prec))
    arr = np.array(interped)
    return arr.mean(axis=0), arr.std(axis=0)


def make_figures(
    compare: pd.DataFrame,
    merged: pd.DataFrame,
    out_dir: Path,
    pr_data_cav: list | None = None,
    pr_data_tool: list | None = None,
    pool_pos_cav: np.ndarray | None = None,
    pool_neg_cav: np.ndarray | None = None,
    pool_pos_llr: np.ndarray | None = None,
    pool_neg_llr: np.ndarray | None = None,
    pool_neg_tool: np.ndarray | None = None,
    figure_data_dir: str | None = None,
    label: str | None = None,
) -> None:
    plt.rcParams.update(RCPARAMS)

    valid = compare.dropna(subset=["auc_val_vs_test_neg", "tool_auc"]).copy()
    cav_auc  = valid["auc_val_vs_test_neg"].values
    tool_auc = valid["tool_auc"].values
    n        = len(valid)

    # ------------------------------------------------------------------
    # 1. Scatter — coloured by GO depth  (skipped if depth not present)
    # ------------------------------------------------------------------
    if "depth" in valid.columns and valid["depth"].notna().any():
        dv = valid.dropna(subset=["depth"])
        fig, ax = plt.subplots(figsize=(5, 5))
        sc = ax.scatter(dv["tool_auc"], dv["auc_val_vs_test_neg"],
                        c=dv["depth"], cmap="RdYlBu_r",
                        s=35, alpha=0.85, linewidths=0.4, edgecolors="k", zorder=2)
        _scatter_base(ax, dv["tool_auc"].values, dv["auc_val_vs_test_neg"].values)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("GO term depth")
        ax.set_title(f"CAV vs tool AUC per GO term  (n={len(dv)})")
        fig.tight_layout()
        p = out_dir / "fig_scatter_auc_depth.pdf"
        fig.savefig(p)
        plt.close(fig)
        logger.info(f"Saved {p}")

    # ------------------------------------------------------------------
    # 3. Scatter — dot size proportional to number of validation proteins
    # ------------------------------------------------------------------
    nval = valid["n_val_proteins"].values
    size_min, size_max = 20, 200
    nval_lo, nval_hi   = nval.min(), nval.max()
    if nval_hi > nval_lo:
        sizes = size_min + (size_max - size_min) * (nval - nval_lo) / (nval_hi - nval_lo)
    else:
        sizes = np.full(len(nval), (size_min + size_max) / 2)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(tool_auc, cav_auc, s=sizes, alpha=0.65,
               color="steelblue", linewidths=0.4, edgecolors="k", zorder=2)
    _scatter_base(ax, tool_auc, cav_auc)

    # Size legend: pick ~3 representative values
    legend_ns = sorted({nval_lo, int(np.median(nval)), nval_hi})
    legend_handles = []
    for nv in legend_ns:
        s = size_min + (size_max - size_min) * (nv - nval_lo) / max(nval_hi - nval_lo, 1)
        legend_handles.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue",
                   markeredgecolor="k", markeredgewidth=0.4,
                   markersize=(s / np.pi) ** 0.5 * 2, label=f"n = {int(nv)}")
        )
    ax.legend(handles=legend_handles, title="Val proteins", frameon=False,
              loc="upper left", fontsize=8)
    ax.set_title(f"CAV vs tool AUC per GO term  (n={n})")
    fig.tight_layout()
    p = out_dir / "fig_scatter_auc_nval.pdf"
    fig.savefig(p)
    plt.close(fig)
    logger.info(f"Saved {p}")

    # ------------------------------------------------------------------
    # 4. Paired KDE — AUC distribution for CAV vs tool
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    for values, label, color in [
        (cav_auc,  "CAV",           "#2166ac"),
        (tool_auc, "External tool", "#d6604d"),
    ]:
        kde   = spstats.gaussian_kde(values, bw_method="scott")
        x_grid = np.linspace(0.2, 1.05, 300)
        y_kde  = kde(x_grid)
        ax.plot(x_grid, y_kde, lw=2, color=color, label=label)
        ax.fill_between(x_grid, y_kde, alpha=0.15, color=color)
        ax.axvline(float(np.mean(values)), lw=1.2, ls="--", color=color)

    ax.set_xlabel("AUC")
    ax.set_ylabel("Density")
    ax.set_title(f"AUC distribution across {n} GO terms")
    ax.legend(frameon=False)
    ax.set_xlim(0.2, 1.05)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    p = out_dir / "fig_auc_distributions.pdf"
    fig.savefig(p)
    plt.close(fig)
    logger.info(f"Saved {p}")

    # ------------------------------------------------------------------
    # 4b. Paired KDE — AUPR distribution for CAV vs tool
    # ------------------------------------------------------------------
    if pr_data_cav and pr_data_tool:
        cav_aupr  = np.array([np.trapezoid(prec, rec) for rec, prec in pr_data_cav])
        tool_aupr = np.array([np.trapezoid(prec, rec) for rec, prec in pr_data_tool])
        n_pr = len(cav_aupr)
        fig, ax = plt.subplots(figsize=(6, 4))
        for values, label, color in [
            (cav_aupr,  "CAV",           "#2166ac"),
            (tool_aupr, "External tool", "#d6604d"),
        ]:
            kde    = spstats.gaussian_kde(values, bw_method="scott")
            x_grid = np.linspace(0, 1.05, 300)
            y_kde  = kde(x_grid)
            ax.plot(x_grid, y_kde, lw=2, color=color, label=label)
            ax.fill_between(x_grid, y_kde, alpha=0.15, color=color)
            ax.axvline(float(np.mean(values)), lw=1.2, ls="--", color=color)
        ax.set_xlabel("AUPR")
        ax.set_ylabel("Density")
        ax.set_title(f"AUPR distribution across {n_pr} GO terms")
        ax.legend(frameon=False)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        p = out_dir / "fig_aupr_distributions.pdf"
        fig.savefig(p)
        plt.close(fig)
        logger.info(f"Saved {p}")

    # ------------------------------------------------------------------
    # 5. Violin — side-by-side AUC distributions
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(4, 5))
    parts = ax.violinplot([cav_auc, tool_auc], positions=[1, 2],
                          showmedians=True, showextrema=False)
    colors = ["#2166ac", "#d6604d"]
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("k")
    parts["cmedians"].set_linewidth(2)

    # Overlay individual points as a strip
    rng = np.random.default_rng(42)
    for vals, pos, color in [(cav_auc, 1, "#2166ac"), (tool_auc, 2, "#d6604d")]:
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(pos + jitter, vals, s=12, color=color, alpha=0.5,
                   linewidths=0.3, edgecolors="k", zorder=3)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["CAV", "External tool"])
    ax.set_ylabel("AUC")
    ax.set_ylim(0.2, 1.05)
    ax.set_title(f"AUC distributions\n({n} GO terms)")
    fig.tight_layout()
    p = out_dir / "fig_auc_violin.pdf"
    fig.savefig(p)
    plt.close(fig)
    logger.info(f"Saved {p}")

    # ------------------------------------------------------------------
    # 6 & 7. Protein-level scatters: CAV score vs tool score,
    #         LLR vs tool score  (one point per validation protein-GO pair)
    # ------------------------------------------------------------------
    tool_score  = merged["tool_score"].values
    predicted   = tool_score > 0          # tool fired on this pair
    colors_pt   = np.where(predicted, "#2166ac", "#bbbbbb")
    alpha_pt    = np.where(predicted, 0.7, 0.35)

    # legend proxies
    legend_handles_pt = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2166ac",
               markeredgecolor="k", markeredgewidth=0.3, markersize=6,
               label="Tool predicted"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#bbbbbb",
               markeredgecolor="k", markeredgewidth=0.3, markersize=6,
               label="Tool silent (score = 0)"),
    ]

    for y_col, y_label, fname in [
        ("val_cav_score", "CAV score",         "fig_protein_cav_vs_tool.pdf"),
        ("llr",           "Log-likelihood ratio (LLR)", "fig_protein_llr_vs_tool.pdf"),
    ]:
        y_vals = merged[y_col].values.astype(float)

        fig, ax = plt.subplots(figsize=(5, 5))

        # Plot silent (grey) first so predicted (blue) sits on top
        for mask, c, a in [
            (~predicted, "#bbbbbb", 0.35),
            ( predicted, "#2166ac", 0.7),
        ]:
            ax.scatter(tool_score[mask], y_vals[mask],
                       s=18, color=c, alpha=a,
                       linewidths=0.3, edgecolors="k", zorder=2)

        ax.axvline(0, color="0.7", lw=0.8, ls="--", zorder=0)
        ax.axhline(0, color="0.7", lw=0.8, ls="--", zorder=0)
        ax.set_xlabel("Tool score")
        ax.set_ylabel(y_label)
        ax.set_title(f"Per-protein: {y_label} vs tool score\n"
                     f"(n={len(y_vals)} validation pairs)")
        if y_col == "llr":
            linthresh = max(1.0, float(np.percentile(np.abs(y_vals[y_vals != 0]), 10)))
            ax.set_yscale("symlog", linthresh=linthresh)
            ax.set_ylabel(f"{y_label}  (symlog scale)")
        ax.legend(handles=legend_handles_pt, frameon=False, fontsize=8)

        fig.tight_layout()
        p = out_dir / fname
        fig.savefig(p)
        plt.close(fig)
        logger.info(f"Saved {p}")

    # ------------------------------------------------------------------
    # 8 & 9.  2-D hexbin density versions of the same protein-level plots
    # ------------------------------------------------------------------
    def symlog_transform(vals: np.ndarray, linthresh: float) -> np.ndarray:
        """Map values to symlog space so hexbin bins are uniform."""
        return np.sign(vals) * np.log1p(np.abs(vals) / linthresh)

    def symlog_ticks(linthresh: float, raw_vals: np.ndarray):
        """Return (transformed_positions, labels) using clean powers of 10."""
        mag = max(np.abs(raw_vals).max(), linthresh)
        pos_ticks = [0]
        v = 1.0
        while v <= mag * 1.05:
            pos_ticks.append(v)
            v *= 10
        raw_ticks = sorted({-t for t in pos_ticks if t != 0} | set(pos_ticks))
        transformed = [symlog_transform(np.array([t]), linthresh)[0] for t in raw_ticks]
        labels = ["0" if t == 0 else str(int(t)) for t in raw_ticks]
        return transformed, labels

    for y_col, y_label, fname in [
        ("val_cav_score", "CAV score",             "fig_protein_cav_vs_tool_hexbin.pdf"),
        ("llr",           "Log-likelihood ratio (LLR)", "fig_protein_llr_vs_tool_hexbin.pdf"),
    ]:
        y_vals = merged[y_col].values.astype(float)

        if y_col == "llr":
            linthresh = max(1.0, float(np.percentile(np.abs(y_vals[y_vals != 0]), 10)))
            y_plot    = symlog_transform(y_vals, linthresh)
            tick_pos, tick_labels = symlog_ticks(linthresh, y_vals)
            y_axis_label = f"{y_label}  (symlog scale)"
            hline_y = symlog_transform(np.array([0.0]), linthresh)[0]
        else:
            y_plot       = y_vals
            tick_pos     = tick_labels = None
            y_axis_label = y_label
            hline_y      = 0.0

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_facecolor("#fef9c3")
        hb = ax.hexbin(tool_score, y_plot, gridsize=30, cmap="Blues",
                       mincnt=1, linewidths=0.2)
        fig.colorbar(hb, ax=ax, label="Count")

        # 2-D KDE contour overlay
        try:
            kde2d  = spstats.gaussian_kde(
                np.vstack([tool_score, y_plot]),
                bw_method="scott",
            )
            xg = np.linspace(tool_score.min(), tool_score.max(), 80)
            yg = np.linspace(y_plot.min(),    y_plot.max(),    80)
            Xg, Yg = np.meshgrid(xg, yg)
            Zg = kde2d(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
            ax.contour(Xg, Yg, Zg, levels=5,
                       colors="navy", alpha=0.55, linewidths=0.9, zorder=3)
        except Exception as e:
            logger.warning(f"KDE contour failed: {e}")

        ax.axvline(0,       color="0.5", lw=0.8, ls="--", zorder=0)
        ax.axhline(hline_y, color="0.5", lw=0.8, ls="--", zorder=0)
        if tick_pos is not None:
            ax.set_yticks(tick_pos)
            ax.set_yticklabels(tick_labels)
        ax.set_xlabel("Tool score")
        ax.set_ylabel(y_axis_label)
        ax.set_title(f"Per-protein: {y_label} vs tool score\n"
                     f"(n={len(y_vals)} validation pairs, 2-D density)")
        fig.tight_layout()
        p = out_dir / fname
        fig.savefig(p)
        plt.close(fig)
        logger.info(f"Saved {p}")

    # ------------------------------------------------------------------
    # 10.  Filled 2-D KDE: LLR vs tool score only
    # ------------------------------------------------------------------
    from matplotlib.colors import LinearSegmentedColormap

    y_vals_llr = merged["llr"].values.astype(float)
    linthresh   = max(1.0, float(np.percentile(np.abs(y_vals_llr[y_vals_llr != 0]), 10)))
    y_plot_llr  = symlog_transform(y_vals_llr, linthresh)
    tick_pos_llr, tick_labels_llr = symlog_ticks(linthresh, y_vals_llr)
    hline_llr   = symlog_transform(np.array([0.0]), linthresh)[0]

    # Colormap: background yellow → dark blue
    yl_to_blue = LinearSegmentedColormap.from_list(
        "yl_blue", ["#fef9c3", "#d0e8f5", "#2166ac", "#0a1f44"], N=256
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_facecolor("#fef9c3")

    try:
        kde2d = spstats.gaussian_kde(
            np.vstack([tool_score, y_plot_llr]), bw_method="scott"
        )
        xg = np.linspace(tool_score.min(), tool_score.max(), 120)
        yg = np.linspace(y_plot_llr.min(), y_plot_llr.max(), 120)
        Xg, Yg = np.meshgrid(xg, yg)
        Zg = kde2d(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
        ax.contourf(Xg, Yg, Zg, levels=12, cmap=yl_to_blue, zorder=1)
        ax.contour( Xg, Yg, Zg, levels=12, colors="white",
                    alpha=0.25, linewidths=0.5, zorder=2)
    except Exception as e:
        logger.warning(f"Filled KDE failed: {e}")

    ax.axvline(0,        color="0.4", lw=0.8, ls="--", zorder=3)
    ax.axhline(hline_llr, color="0.4", lw=0.8, ls="--", zorder=3)
    ax.set_yticks(tick_pos_llr)
    ax.set_yticklabels(tick_labels_llr)
    ax.set_xlabel("Tool score")
    ax.set_ylabel("Log-likelihood ratio (LLR)  (symlog scale)")
    ax.set_title(f"Per-protein: LLR vs tool score\n"
                 f"(n={len(y_vals_llr)} validation pairs, filled KDE)")
    fig.tight_layout()
    p = out_dir / "fig_protein_llr_vs_tool_filled_kde.pdf"
    fig.savefig(p)
    plt.close(fig)
    logger.info(f"Saved {p}")

    # ------------------------------------------------------------------
    # 10b. Same filled KDE but for negative test proteins
    # ------------------------------------------------------------------
    if pool_neg_tool is not None and len(pool_neg_llr) > 0 and len(pool_neg_tool) == len(pool_neg_llr):
        neg_tool_arr = pool_neg_tool.astype(float)
        neg_llr_arr  = pool_neg_llr.astype(float)

        linthresh_neg = max(1.0, float(np.percentile(np.abs(neg_llr_arr[neg_llr_arr != 0]), 10)))
        y_plot_neg    = symlog_transform(neg_llr_arr, linthresh_neg)
        tick_pos_neg, tick_labels_neg = symlog_ticks(linthresh_neg, neg_llr_arr)
        hline_neg     = symlog_transform(np.array([0.0]), linthresh_neg)[0]

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_facecolor("#fef9c3")
        try:
            kde2d_neg = spstats.gaussian_kde(
                np.vstack([neg_tool_arr, y_plot_neg]), bw_method="scott"
            )
            xg = np.linspace(neg_tool_arr.min(), neg_tool_arr.max(), 120)
            yg = np.linspace(y_plot_neg.min(),   y_plot_neg.max(),   120)
            Xg, Yg = np.meshgrid(xg, yg)
            Zg = kde2d_neg(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
            ax.contourf(Xg, Yg, Zg, levels=12, cmap=yl_to_blue, zorder=1)
            ax.contour( Xg, Yg, Zg, levels=12, colors="white",
                        alpha=0.25, linewidths=0.5, zorder=2)
        except Exception as e:
            logger.warning(f"Filled KDE (negatives) failed: {e}")
        ax.axvline(0,        color="0.4", lw=0.8, ls="--", zorder=3)
        ax.axhline(hline_neg, color="0.4", lw=0.8, ls="--", zorder=3)
        ax.set_yticks(tick_pos_neg)
        ax.set_yticklabels(tick_labels_neg)
        ax.set_xlabel("Tool score")
        ax.set_ylabel("Log-likelihood ratio (LLR)  (symlog scale)")
        ax.set_title(f"Negatives: LLR vs tool score\n"
                     f"(n={len(neg_llr_arr):,} negative test pairs, filled KDE)")
        fig.tight_layout()
        p = out_dir / "fig_neg_llr_vs_tool_filled_kde.pdf"
        fig.savefig(p)
        plt.close(fig)
        logger.info(f"Saved {p}")

    # ------------------------------------------------------------------
    # 11 & 12.  1-D KDE split by tool-predicted vs tool-silent
    #           Shows distribution of y-variable for each group separately
    # ------------------------------------------------------------------
    for y_col, y_label, fname in [
        ("val_cav_score", "CAV score",             "fig_protein_cav_split_kde.pdf"),
        ("llr",           "Log-likelihood ratio (LLR)", "fig_protein_llr_split_kde.pdf"),
    ]:
        y_vals = merged[y_col].values.astype(float)

        fig, ax = plt.subplots(figsize=(6, 4))
        for mask, label, color in [
            (predicted,  f"Tool predicted  (n={predicted.sum()})",  "#2166ac"),
            (~predicted, f"Tool silent  (n={(~predicted).sum()})",  "#bbbbbb"),
        ]:
            vals = y_vals[mask]
            if len(vals) < 3:
                continue
            if y_col == "llr":
                # Clip extreme outliers for KDE stability (plot range only)
                lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
                x_grid = np.linspace(lo, hi, 400)
            else:
                x_grid = np.linspace(vals.min() - 0.05, vals.max() + 0.05, 400)
            kde   = spstats.gaussian_kde(vals, bw_method="scott")
            y_kde = kde(x_grid)
            ax.plot(x_grid, y_kde, lw=2, color=color, label=label)
            ax.fill_between(x_grid, y_kde, alpha=0.2, color=color)
            ax.axvline(float(np.median(vals)), lw=1.2, ls="--", color=color)

        ax.axvline(0, color="0.5", lw=0.8, ls=":", zorder=0)
        ax.set_xlabel(y_label)
        ax.set_ylabel("Density")
        ax.set_title(f"Distribution of {y_label}\nby tool prediction status")
        ax.legend(frameon=False)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        p = out_dir / fname
        fig.savefig(p)
        plt.close(fig)
        logger.info(f"Saved {p}")

    # ------------------------------------------------------------------
    # 13.  CAV score vs LLR per validation pair, coloured by tool score
    # ------------------------------------------------------------------
    cav_scores = merged["val_cav_score"].values.astype(float)
    llr_vals   = merged["llr"].values.astype(float)
    tool_vals  = merged["tool_score"].values.astype(float)

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(
        cav_scores, llr_vals,
        c=tool_vals, cmap="plasma",
        s=16, alpha=0.7, linewidths=0, vmin=0, vmax=max(tool_vals.max(), 1e-6),
    )
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Tool score")
    ax.axvline(0, color="0.7", lw=0.8, ls="--", zorder=0)
    ax.axhline(0, color="0.7", lw=0.8, ls="--", zorder=0)
    ax.set_xlabel("CAV score")
    ax.set_ylabel("Log-likelihood ratio (LLR)")
    ax.set_title(
        f"CAV score vs LLR  (n={len(cav_scores)} val pairs)\n"
        f"coloured by tool score"
    )
    linthresh = max(1.0, float(np.percentile(np.abs(llr_vals[llr_vals != 0]), 10)))
    ax.set_yscale("symlog", linthresh=linthresh)
    fig.tight_layout()
    p = out_dir / "fig_cav_score_vs_llr_tool_color.pdf"
    fig.savefig(p)
    plt.close(fig)
    logger.info(f"Saved {p}")

    # ------------------------------------------------------------------
    # 14.  Density plots: CAV score and LLR for positives vs negatives
    # ------------------------------------------------------------------
    if pool_pos_cav is not None and len(pool_pos_cav) > 0 and len(pool_neg_cav) > 0:
        for pos_vals, neg_vals, xlabel, fname, lo, hi in [
            (pool_pos_cav, pool_neg_cav,
             "CAV score", "fig_density_cav_score_pos_neg.pdf",
             float(np.percentile(np.concatenate([pool_pos_cav, pool_neg_cav]), 0.5)),
             float(np.percentile(np.concatenate([pool_pos_cav, pool_neg_cav]), 99.5))),
            (pool_pos_llr, pool_neg_llr,
             "Log-likelihood ratio (LLR)", "fig_density_llr_pos_neg.pdf", -50, 50),
        ]:
            if len(pos_vals) == 0 or len(neg_vals) == 0:
                continue
            fig, ax = plt.subplots(figsize=(6, 4))
            for vals, label, color in [
                (pos_vals, f"Positives  (n={len(pos_vals):,})", "#2166ac"),
                (neg_vals, f"Negatives  (n={len(neg_vals):,})", "#d6604d"),
            ]:
                x_grid = np.linspace(lo, hi, 400)
                v_clip = np.clip(vals, lo, hi)
                kde    = spstats.gaussian_kde(v_clip, bw_method="scott")
                y_kde  = kde(x_grid)
                ax.plot(x_grid, y_kde, lw=2, color=color, label=label)
                ax.fill_between(x_grid, y_kde, alpha=0.2, color=color)
                ax.axvline(float(np.median(vals)), lw=1, ls="--", color=color)
            ax.axvline(0, color="0.5", lw=0.8, ls=":", zorder=0)
            ax.set_xlim(lo, hi)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Density")
            ax.set_title(f"Pooled {xlabel} distribution\n(across all GO terms)")
            ax.legend(frameon=False)
            ax.set_ylim(bottom=0)
            fig.tight_layout()
            p = out_dir / fname
            fig.savefig(p)
            plt.close(fig)
            logger.info(f"Saved {p}")

    # ------------------------------------------------------------------
    # 15.  Macro-averaged precision-recall curve (CAV vs external tool)
    #      Only produced when pr_data_cav / pr_data_tool were collected,
    #      i.e. when the neg-score TSVs exist and have a cav_score column.
    # ------------------------------------------------------------------
    if pr_data_cav and pr_data_tool:
        recall_grid = np.linspace(0, 1, 101)
        mean_cav,  std_cav  = _macro_pr(pr_data_cav,  recall_grid)
        mean_tool, std_tool = _macro_pr(pr_data_tool, recall_grid)
        n_terms = len(pr_data_cav)

        # mAP = area under the macro-averaged PR curve
        map_cav  = float(np.trapezoid(mean_cav,  recall_grid))
        map_tool = float(np.trapezoid(mean_tool, recall_grid))

        fig, ax = plt.subplots(figsize=(6, 5))
        for mean, std, color, label in [
            (mean_cav,  std_cav,  "#2166ac", f"CAV  (mAP={map_cav:.3f})"),
            (mean_tool, std_tool, "#d6604d", f"External tool  (mAP={map_tool:.3f})"),
        ]:
            ax.plot(recall_grid, mean, lw=2, color=color, label=label)
            ax.fill_between(recall_grid,
                            np.clip(mean - std, 0, 1),
                            np.clip(mean + std, 0, 1),
                            alpha=0.18, color=color)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Macro-averaged PR curve  ({n_terms} GO terms)")
        ax.legend(frameon=False)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        p = out_dir / "fig_pr_curve_macro_avg.pdf"
        fig.savefig(p)
        plt.close(fig)
        logger.info(f"Saved {p}")

        if figure_data_dir:
            fig_dir = Path(figure_data_dir)
            fig_dir.mkdir(parents=True, exist_ok=True)
            lsuffix = f"_{label}" if label else ""
            pd.DataFrame({
                "recall":            recall_grid,
                "cav_precision":     mean_cav,
                "cav_precision_sd":  std_cav,
                "tool_precision":    mean_tool,
                "tool_precision_sd": std_tool,
            }).to_csv(fig_dir / f"temporal_pr_curves{lsuffix}.csv", index=False, float_format="%.4f")
            logger.info(f"Figure data: temporal_pr_curves{lsuffix}.csv → {fig_dir}")

    if figure_data_dir and pool_pos_cav is not None and len(pool_pos_cav) > 0:
        fig_dir = Path(figure_data_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        lsuffix = f"_{label}" if label else ""
        density_dfs = []
        if len(pool_pos_cav):
            density_dfs.append(pd.DataFrame({
                "label": "positive",
                "cav_score": pool_pos_cav,
                "llr": pool_pos_llr if pool_pos_llr is not None and len(pool_pos_llr) == len(pool_pos_cav) else np.nan,
            }))
        if len(pool_neg_cav):
            density_dfs.append(pd.DataFrame({
                "label": "negative",
                "cav_score": pool_neg_cav,
                "llr": pool_neg_llr if pool_neg_llr is not None and len(pool_neg_llr) == len(pool_neg_cav) else np.nan,
            }))
        if density_dfs:
            pd.concat(density_dfs, ignore_index=True).to_csv(
                fig_dir / f"temporal_pos_neg_density{lsuffix}.csv", index=False, float_format="%.4f"
            )
            logger.info(f"Figure data: temporal_pos_neg_density{lsuffix}.csv → {fig_dir}")


if __name__ == "__main__":
    main()
