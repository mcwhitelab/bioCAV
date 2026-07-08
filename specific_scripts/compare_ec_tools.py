#!/usr/bin/env python3
"""
compare_ec_tools.py

Compare multiple EC-number prediction tools against each other and against
CAV eval results (from eval_ec.py).

The tool prediction file has:
  - id         : protein identifier
  - ec_number  : true EC number(s), comma-separated (gold standard)
  - one column per tool: predicted EC number(s), comma-separated; "-" = no prediction

Matching uses the truth's own specificity level:
  - If the truth is '4.2.3' (3-level), a prediction of '4.2.3.158' gets credit.
  - If the truth is '4.2.3.158' (4-level), the prediction must match all 4 levels.

Wildcard true labels (4.2.3.-) are normalised by stripping trailing '-' levels.

If --cav-results is supplied, CAV recall is added at LLR > 0 (configurable).
All comparisons are restricted to (protein, EC) pairs covered by CAV.

Usage
-----
python specific_scripts/compare_ec_tools.py \\
    --tool-predictions  data/price149_predictions.csv \\
    --out-dir           results/ec_eval/ \\
    --cav-results       results/ec_eval/eval_ec_results.tsv
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Columns in the tool CSV that are NOT tool predictions
NON_TOOL_COLS = {"id", "ec_number"}


# ---------------------------------------------------------------------------
# EC matching helpers
# ---------------------------------------------------------------------------

def strip_wildcards(ec: str) -> str:
    """'4.2.3.-' → '4.2.3'   |   '4.2.3.158' → '4.2.3.158'"""
    parts = ec.strip().split(".")
    while parts and parts[-1].strip() in ("-", ""):
        parts.pop()
    return ".".join(parts)


def ec_to_level(ec: str, level: int) -> str | None:
    """Return EC truncated to `level` significant levels, or None if too shallow.

    '4.2.3.158' at level 3 → '4.2.3'
    '4.2.3'     at level 4 → None  (not specific enough)
    """
    norm = strip_wildcards(ec)
    parts = norm.split(".")
    if len(parts) >= level:
        return ".".join(parts[:level])
    return None


def parse_ec_field(field: str) -> list[str]:
    """Split a comma-separated EC field into individual EC strings, drop '-'."""
    if not field or str(field).strip() in ("-", "nan", ""):
        return []
    return [s.strip() for s in str(field).split(",") if s.strip() not in ("-", "")]


def matches_exact(predicted_field: str, true_field: str) -> bool:
    """Match at the truth's own specificity level.

    If the truth is '4.2.3' (3-level), a prediction of '4.2.3.158' gets
    credit because its first 3 levels match — the tool is being rewarded
    for predicting a valid specific sub-entry of an underspecified truth.
    If the truth is '4.2.3.158' (4-level), the prediction must match all 4.
    """
    preds = parse_ec_field(predicted_field)
    trues = parse_ec_field(true_field)
    if not preds or not trues:
        return False
    for true_ec in trues:
        true_norm  = strip_wildcards(true_ec)
        true_level = len(true_norm.split("."))
        true_at    = true_norm                           # already at its own level
        for pred in preds:
            pred_at = ec_to_level(pred, true_level)      # truncate prediction to same level
            if pred_at == true_at:
                return True
    return False


def predicted_anything(field: str) -> bool:
    return len(parse_ec_field(str(field))) > 0


# ---------------------------------------------------------------------------
# Load and explode tool predictions
# ---------------------------------------------------------------------------

def load_and_explode(csv_path: str) -> tuple[pd.DataFrame, list[str]]:
    """Read tool CSV, explode multi-EC true labels to one row per (protein, EC).

    Returns (long_df, tool_col_names).
    long_df has columns: protein_id, ec_number, true_ec_norm, <tool cols...>
    """
    df = pd.read_csv(csv_path, dtype=str).fillna("-")
    df = df.rename(columns={"id": "protein_id"})

    tool_cols = [c for c in df.columns if c not in NON_TOOL_COLS and c != "protein_id"]

    # Explode multi-EC true labels
    df["ec_number"] = df["ec_number"].astype(str).str.strip().str.strip('"')
    df["ec_number"] = df["ec_number"].str.split(",")
    df = df.explode("ec_number")
    df["ec_number"] = df["ec_number"].str.strip()
    df = df[df["ec_number"].apply(lambda x: bool(parse_ec_field(x)))].reset_index(drop=True)

    # Normalised true EC (wildcards stripped)
    df["true_ec_norm"] = df["ec_number"].apply(strip_wildcards)

    logger.info(
        f"Loaded {df['protein_id'].nunique()} proteins, "
        f"{df['true_ec_norm'].nunique()} unique EC terms, "
        f"{len(df)} (protein, EC) pairs"
    )
    logger.info(f"Tools found: {tool_cols}")
    return df, tool_cols


# ---------------------------------------------------------------------------
# Per-pair match computation
# ---------------------------------------------------------------------------

def compute_matches(df: pd.DataFrame, tool_cols: list[str]) -> pd.DataFrame:
    """Add boolean match columns for each tool."""
    for col in tool_cols:
        safe = col.replace("-", "_").replace(" ", "_")
        df[f"{safe}__exact"]     = df.apply(
            lambda r, c=col: matches_exact(r[c], r["ec_number"]), axis=1
        )
        df[f"{safe}__predicted"] = df[col].apply(predicted_anything)
    return df


# ---------------------------------------------------------------------------
# Per-tool summary
# ---------------------------------------------------------------------------

def tool_summary(df: pd.DataFrame, tool_cols: list[str]) -> pd.DataFrame:
    rows = []
    n_total = len(df)
    for col in tool_cols:
        safe   = col.replace("-", "_").replace(" ", "_")
        n_pred  = df[f"{safe}__predicted"].sum()
        n_exact = df[f"{safe}__exact"].sum()
        rows.append({
            "tool":            col,
            "n_pairs":         n_total,
            "n_predicted":     int(n_pred),
            "coverage":        round(n_pred  / n_total, 4),
            "n_exact_match":   int(n_exact),
            "recall_exact":    round(n_exact / n_total, 4),
            "precision_exact": round(n_exact / n_pred,  4) if n_pred else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("recall_exact", ascending=False)


# ---------------------------------------------------------------------------
# CAV integration
# ---------------------------------------------------------------------------

def add_cav_metrics(
    df: pd.DataFrame,
    cav_results_path: str,
    llr_threshold: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    """Join CAV eval results onto the per-pair DataFrame.

    Returns (merged_df, cav_summary_dict).
    """
    cav = pd.read_csv(cav_results_path, sep="\t")
    logger.info(f"Loaded {len(cav)} CAV result rows from {cav_results_path}")

    # Normalise CAV ec_number for join
    cav["true_ec_norm"] = cav["ec_number"].apply(strip_wildcards)

    merged = df.merge(
        cav[["protein_id", "true_ec_norm", "llr",
             "val_cav_score", "test_pos_percentile", "test_neg_percentile"]],
        on=["protein_id", "true_ec_norm"],
        how="left",
    )
    n_matched = merged["llr"].notna().sum()
    logger.info(f"CAV scores joined for {n_matched} / {len(merged)} pairs")

    merged["cav__llr_positive"] = merged["llr"] > llr_threshold

    n_total   = len(merged)
    n_scored  = int(merged["llr"].notna().sum())
    n_exact   = int(merged["cav__llr_positive"].sum())

    cav_summary = {
        "tool":            "CAV",
        "n_pairs":         n_total,
        "n_predicted":     n_scored,
        "coverage":        round(n_scored / n_total, 4),
        "n_exact_match":   n_exact,
        "recall_exact":    round(n_exact / n_total, 4),
        "precision_exact": round(n_exact / n_scored, 4) if n_scored else float("nan"),
        "note":            f"LLR>{llr_threshold}",
    }

    return merged, cav_summary


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

RCPARAMS = {"font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10}


def make_figures(summary: pd.DataFrame, detail_df: pd.DataFrame, out_dir: Path,
                 figure_data_dir: str | None = None,
                 rank_df: pd.DataFrame | None = None,
                 tool_cols: list | None = None,
                 llr_threshold: float = 0.0) -> None:
    plt.rcParams.update(RCPARAMS)
    from matplotlib.patches import Patch

    # ------------------------------------------------------------------
    # 1. Horizontal bar: recall (exact) — full-coverage tools only
    # ------------------------------------------------------------------
    cav_row = summary[summary["tool"] == "CAV"]
    n_cav   = int(cav_row["n_predicted"].iloc[0]) if len(cav_row) else None

    plot_summary = summary.copy()
    if n_cav is not None:
        plot_summary = plot_summary[plot_summary["n_predicted"] == n_cav]

    plot_summary = plot_summary.sort_values("recall_exact", ascending=True)
    tools  = plot_summary["tool"].tolist()
    y      = np.arange(len(tools))
    colors = ["#d6604d" if t == "CAV" else "#2166ac" for t in tools]

    fig, ax = plt.subplots(figsize=(7, max(4, len(tools) * 0.4)))
    ax.barh(y, plot_summary["recall_exact"], color=colors, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(tools, fontsize=8)
    ax.set_xlabel("Recall  (fraction of gold-standard pairs)")
    ax.set_xlim(0, 1.05)
    ax.axvline(0.5, color="0.6", lw=0.8, ls="--")
    legend_elements = [
        Patch(facecolor="#2166ac", label="Other tools"),
        Patch(facecolor="#d6604d", label="CAV (LLR > 0)"),
    ]
    ax.legend(handles=legend_elements, frameon=False)
    ax.set_title("EC prediction recall across tools (full-coverage only)")
    fig.tight_layout()
    p = out_dir / "fig_ec_tool_recall.pdf"
    fig.savefig(p)
    plt.close(fig)
    logger.info(f"Saved {p}")

    # ------------------------------------------------------------------
    # 2. Scatter: coverage vs recall_exact
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 5))
    is_cav = summary["tool"] == "CAV"
    ax.scatter(
        summary.loc[~is_cav, "coverage"],
        summary.loc[~is_cav, "recall_exact"],
        s=40, color="#2166ac", alpha=0.8, linewidths=0.4, edgecolors="k",
        label="Other tools", zorder=2,
    )
    if is_cav.any():
        ax.scatter(
            summary.loc[is_cav, "coverage"],
            summary.loc[is_cav, "recall_exact"],
            s=80, color="#d6604d", marker="*", linewidths=0.5, edgecolors="k",
            label="CAV (LLR > 0)", zorder=3,
        )
    for _, row in summary.iterrows():
        ax.annotate(row["tool"], (row["coverage"], row["recall_exact"]),
                    fontsize=6, xytext=(3, 2), textcoords="offset points")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, zorder=0)
    ax.set_xlabel("Coverage (fraction of pairs with any prediction)")
    ax.set_ylabel("Recall (exact match)")
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(frameon=False)
    ax.set_title("EC prediction tools: coverage vs recall")
    fig.tight_layout()
    p = out_dir / "fig_ec_coverage_vs_recall.pdf"
    fig.savefig(p)
    plt.close(fig)
    logger.info(f"Saved {p}")

    # ------------------------------------------------------------------
    # 3. Recall vs LLR threshold curve
    # ------------------------------------------------------------------
    llr_vals = detail_df["llr"].dropna().values
    if len(llr_vals) > 0:
        n_total    = len(llr_vals)
        thresholds = np.linspace(llr_vals.min() - 0.5, llr_vals.max() + 0.5, 300)
        recalls    = [(llr_vals > t).sum() / n_total for t in thresholds]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(thresholds, recalls, color="#2166ac", lw=2)
        ax.set_xlabel("LLR threshold")
        ax.set_ylabel("Recall  (fraction of val pairs above threshold)")
        ax.set_ylim(-0.02, 1.05)
        fig.tight_layout()
        p = out_dir / "fig_ec_recall_vs_llr.pdf"
        fig.savefig(p)
        plt.close(fig)
        logger.info(f"Saved {p}")
    else:
        logger.warning("No LLR values available — skipping recall-vs-threshold figure")

    # ------------------------------------------------------------------
    # Export figure-ready CSVs
    # ------------------------------------------------------------------
    if figure_data_dir:
        fig_dir = Path(figure_data_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)

        summary.to_csv(fig_dir / "ec_tool_comparison_summary.csv", index=False, float_format="%.4f")
        logger.info(f"Figure data: ec_tool_comparison_summary.csv → {fig_dir}")

        if len(llr_vals) > 0:
            pd.DataFrame({"llr_threshold": thresholds, "recall": recalls}).to_csv(
                fig_dir / "ec_recall_vs_llr.csv", index=False, float_format="%.4f"
            )
            logger.info(f"Figure data: ec_recall_vs_llr.csv → {fig_dir}")

        # Per-EC-term recall for full-coverage tools + CAV
        # "Full coverage" = same n_predicted as CAV (matches the existing bar-chart filter)
        safe_to_tool = {t.replace("-", "_").replace(" ", "_"): t
                        for t in summary["tool"].tolist() if t != "CAV"}
        full_cov_safe = {t.replace("-", "_").replace(" ", "_")
                         for t in plot_summary["tool"].tolist() if t != "CAV"}
        exact_cols = [c for c in detail_df.columns if c.endswith("__exact")]
        has_cav = "cav__llr_positive" in detail_df.columns

        rows = []
        for ec, grp in detail_df.groupby("true_ec_norm"):
            n = len(grp)
            if has_cav:
                rows.append({"ec_number": ec, "n_proteins": n, "tool": "CAV",
                              "recall": float(grp["cav__llr_positive"].mean())})
            for col in exact_cols:
                tool_safe = col[: -len("__exact")]
                if tool_safe not in full_cov_safe:
                    continue
                rows.append({"ec_number": ec, "n_proteins": n,
                              "tool": safe_to_tool.get(tool_safe, tool_safe),
                              "recall": float(grp[col].mean())})

        if rows:
            pd.DataFrame(rows).to_csv(
                fig_dir / "ec_per_term_recall.csv", index=False, float_format="%.4f"
            )
            logger.info(f"Figure data: ec_per_term_recall.csv → {fig_dir}")

    if rank_df is None or tool_cols is None:
        return

    # ------------------------------------------------------------------
    # Join rank data onto detail_df
    # ------------------------------------------------------------------
    ranked = detail_df.merge(
        rank_df[["protein_id", "true_ec_norm", "rank", "n_cavs"]],
        on=["protein_id", "true_ec_norm"],
        how="inner",
    )
    if len(ranked) == 0:
        logger.warning("No rows after joining rank data — skipping rank figures")
        return

    n_cavs = int(ranked["n_cavs"].iloc[0])
    has_llr = "llr" in ranked.columns

    for tool in tool_cols:
        safe    = tool.replace("-", "_").replace(" ", "_")
        correct_col = f"{safe}__exact"
        if correct_col not in ranked.columns:
            continue

        correct   = ranked[correct_col].astype(bool)
        ranks_all = ranked["rank"].values

        # ------------------------------------------------------------------
        # Figure 1: rank histogram colored by tool correctness
        # ------------------------------------------------------------------
        n_total    = len(ranked)
        ranks_right = ranked.loc[correct,  "rank"].values
        ranks_wrong = ranked.loc[~correct, "rank"].values

        max_show   = min(50, n_cavs)
        bin_edges  = np.arange(1, max_show + 2)
        not_shown_x = max_show + 3

        counts_right = np.bincount(ranks_right[ranks_right <= max_show],
                                   minlength=max_show + 1)[1:max_show + 1] / n_total
        counts_wrong = np.bincount(ranks_wrong[ranks_wrong <= max_show],
                                   minlength=max_show + 1)[1:max_show + 1] / n_total
        n_beyond = (ranks_all > max_show).sum()

        fig, ax = plt.subplots(figsize=(7, 3.5))
        x_bins = np.arange(1, max_show + 1)
        ax.bar(x_bins, counts_wrong, width=1.0, color="#d62728", alpha=0.75,
               edgecolor="none", label=f"Tool incorrect")
        ax.bar(x_bins, counts_right, width=1.0, color="#2ca02c", alpha=0.75,
               edgecolor="none", bottom=counts_wrong, label=f"Tool correct")
        ax.axvline(max_show + 1.5, color="0.6", lw=0.8, ls=":")
        ax.bar(not_shown_x, n_beyond / n_total, width=1.0,
               color="0.6", alpha=0.5, edgecolor="none", label=f"Rank > {max_show}")
        ax.set_xlabel(f"CAV rank of true EC  (rank 1 = top score, out of {n_cavs})")
        ax.set_ylabel("Proportion")
        ax.set_title(f"CAV specificity rank, colored by {tool} correctness")
        ax.set_xlim(0, not_shown_x + 1.5)
        ticks = list(np.linspace(1, max_show, 6).astype(int)) + [not_shown_x]
        labels = [str(t) for t in np.linspace(1, max_show, 6).astype(int)] + [f">{max_show}"]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=8)
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = out_dir / f"fig_ec_rank_hist_{safe}.pdf"
        fig.savefig(p); plt.close(fig)
        logger.info(f"Saved {p}")

        # ------------------------------------------------------------------
        # Figure 2: LLR distribution — tool-correct vs tool-incorrect
        # ------------------------------------------------------------------
        if has_llr:
            llr_right = ranked.loc[correct  & ranked["llr"].notna(), "llr"].values
            llr_wrong = ranked.loc[~correct & ranked["llr"].notna(), "llr"].values

            fig, ax = plt.subplots(figsize=(5, 3.5))
            import scipy.stats as spstats
            for vals, label, color in [
                (llr_wrong, f"Tool incorrect (n={len(llr_wrong)})", "#d62728"),
                (llr_right, f"Tool correct (n={len(llr_right)})",   "#2ca02c"),
            ]:
                if len(vals) < 3:
                    continue
                kde = spstats.gaussian_kde(vals, bw_method="scott")
                xg  = np.linspace(vals.min() - 1, vals.max() + 1, 400)
                ax.plot(xg, kde(xg), lw=2, color=color, label=label)
                ax.fill_between(xg, kde(xg), alpha=0.18, color=color)
            ax.axvline(llr_threshold, color="0.4", lw=0.8, ls="--",
                       label=f"LLR threshold ({llr_threshold})")
            ax.set_xlabel("CAV log-likelihood ratio (LLR)")
            ax.set_ylabel("Density")
            ax.set_title(f"CAV LLR by {tool} prediction correctness")
            ax.legend(fontsize=8)
            fig.tight_layout()
            p = out_dir / f"fig_ec_llr_by_tool_{safe}.pdf"
            fig.savefig(p); plt.close(fig)
            logger.info(f"Saved {p}")

        # ------------------------------------------------------------------
        # Figure 3: 2×2 complementarity (stacked bar per tool)
        # ------------------------------------------------------------------
        if has_llr:
            cav_pos  = ranked["llr"] > llr_threshold
            tool_pos = correct

            n_both      = (cav_pos  &  tool_pos).sum()
            n_cav_only  = (cav_pos  & ~tool_pos).sum()
            n_tool_only = (~cav_pos &  tool_pos).sum()
            n_neither   = (~cav_pos & ~tool_pos).sum()

            categories = ["Both", "CAV only", f"{tool} only", "Neither"]
            counts     = [n_both, n_cav_only, n_tool_only, n_neither]
            colors_bar = ["#4dac26", "#1f77b4", "#d62728", "#bdbdbd"]

            fig, ax = plt.subplots(figsize=(4, 4))
            bottom = 0
            for cat, cnt, col in zip(categories, counts, colors_bar):
                pct = cnt / n_total * 100
                ax.bar(0, pct, bottom=bottom, color=col, label=f"{cat} ({pct:.1f}%)")
                if pct > 3:
                    ax.text(0, bottom + pct / 2, f"{cat}\n{pct:.1f}%",
                            ha="center", va="center", fontsize=8, color="white",
                            fontweight="bold")
                bottom += pct
            ax.set_xlim(-0.6, 0.6)
            ax.set_ylim(0, 100)
            ax.set_ylabel("% of (protein, EC) pairs")
            ax.set_xticks([])
            ax.set_title(f"CAV vs {tool}\n(LLR > {llr_threshold})")
            ax.legend(fontsize=7, loc="lower right")
            fig.tight_layout()
            p = out_dir / f"fig_ec_complementarity_{safe}.pdf"
            fig.savefig(p); plt.close(fig)
            logger.info(f"Saved {p}")


# ---------------------------------------------------------------------------
# EC rank computation (mirrors specificity_ec_rank.py logic)
# ---------------------------------------------------------------------------

def compute_ec_ranks(
    detail_df: pd.DataFrame,
    val_pkl: str,
    ec_base_dirs: list,
    scaler_pkl: str,
    version: str,
) -> pd.DataFrame | None:
    """Score each val protein against all EC CAVs and return per-pair rank.

    Returns a DataFrame with columns: protein_id, true_ec_norm, rank, n_cavs.
    Returns None if prerequisites are missing.
    """
    import glob as _glob
    import sys
    import joblib
    sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
    from src.utils.data_loader import load_sequence_embeddings

    CAV_SUBDIR = "random_positive_train_max1000_cav"

    # Load val embeddings
    logger.info(f"Loading val embeddings: {val_pkl}")
    val_embs, val_ids = load_sequence_embeddings(val_pkl)

    _SKIP = {"sp", "tr", "sw", "ref"}
    id_to_idx: dict[str, int] = {}
    for i, sid in enumerate(val_ids):
        id_to_idx[sid] = i
        for part in sid.split("|"):
            if part and part not in _SKIP:
                id_to_idx.setdefault(part, i)

    # Load all EC CAVs
    base_dirs = [Path(d) for pattern in ec_base_dirs for d in sorted(_glob.glob(pattern))]
    cav_vectors, cav_ids = [], []
    for base in base_dirs:
        for ec_dir in sorted(base.glob("ecNo_*")):
            cav_file = ec_dir / CAV_SUBDIR / f"concept_{version}.npy"
            if cav_file.exists():
                cav_vectors.append(np.load(cav_file))
                cav_ids.append(ec_dir.name)   # e.g. "ecNo_1-1-1-1"

    if not cav_vectors:
        logger.warning("No EC CAVs found — skipping rank computation")
        return None

    n_cavs = len(cav_vectors)
    logger.info(f"Loaded {n_cavs} EC CAVs")
    cav_matrix    = np.stack(cav_vectors, axis=0)          # (n_cavs, dim)
    cav_id_to_col = {cid: i for i, cid in enumerate(cav_ids)}

    scaler        = joblib.load(scaler_pkl)
    val_preprocessed = scaler.transform(val_embs)          # (n_val, dim)
    score_matrix  = val_preprocessed @ cav_matrix.T        # (n_val, n_cavs)

    # Derive ec_cav_id from true_ec_norm ("1.1.1.1" → "ecNo_1-1-1-1")
    rows = []
    for _, pair in detail_df[["protein_id", "true_ec_norm"]].drop_duplicates().iterrows():
        pid        = pair["protein_id"]
        ec_norm    = pair["true_ec_norm"]
        ec_cav_id  = "ecNo_" + ec_norm.replace(".", "-")

        idx = id_to_idx.get(pid)
        col = cav_id_to_col.get(ec_cav_id)
        if idx is None or col is None:
            continue

        scores     = score_matrix[idx]
        true_score = float(scores[col])
        rank       = int((scores > true_score).sum()) + 1
        rows.append({"protein_id": pid, "true_ec_norm": ec_norm,
                     "rank": rank, "n_cavs": n_cavs})

    if not rows:
        logger.warning("No rank data computed")
        return None

    rank_df = pd.DataFrame(rows)
    logger.info(f"Computed ranks for {len(rank_df)} (protein, EC) pairs")
    return rank_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tool-predictions", required=True,
                        help="Wide-format CSV with tool predictions and ec_number column.")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for results and figures.")
    parser.add_argument("--cav-results", default=None,
                        help="Path to eval_ec_results.tsv (from eval_ec.py). Optional.")
    parser.add_argument("--exclude", default=None,
                        help="Optional TSV with ec_number and protein_id columns to exclude "
                             "(e.g. train/val overlap pairs).")
    parser.add_argument("--llr-threshold", type=float, default=0.0,
                        help="LLR threshold for CAV binary prediction (default: 0).")
    parser.add_argument("--figure-data-dir", default=None,
                        help="If provided, write figure-ready CSVs to this directory.")
    # Optional — needed only for rank-based comparison figures
    parser.add_argument("--val-pkl", default=None,
                        help="PKL of val protein embeddings (enables rank figures).")
    parser.add_argument("--ec-base-dirs", nargs="+", default=None,
                        help="Base dirs containing ecNo_* subdirs (glob patterns ok).")
    parser.add_argument("--scaler-pkl", default=None,
                        help="Shared reference population scaler pkl.")
    parser.add_argument("--version", default="v1",
                        help="CAV artifact version suffix (default: v1).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load predictions
    # ------------------------------------------------------------------
    df, tool_cols = load_and_explode(args.tool_predictions)

    # ------------------------------------------------------------------
    # Restrict to CAV-covered pairs (must happen before computing matches)
    # ------------------------------------------------------------------
    if args.cav_results:
        cav_pre = pd.read_csv(args.cav_results, sep="\t")
        cav_pre["true_ec_norm"] = cav_pre["ec_number"].apply(strip_wildcards)
        cav_pairs = set(zip(cav_pre["protein_id"], cav_pre["true_ec_norm"]))
        n_before = len(df)
        df = df[df.apply(
            lambda r: (r["protein_id"], r["true_ec_norm"]) in cav_pairs, axis=1
        )].reset_index(drop=True)
        logger.info(
            f"Restricted from {n_before} to {len(df)} pairs "
            f"with CAV coverage ({cav_pre['ec_number'].nunique()} EC terms)"
        )

    if args.exclude:
        excl = pd.read_csv(args.exclude, sep="\t")[["ec_number", "protein_id"]]
        excl["_drop"] = True
        n_before = len(df)
        df = df.merge(excl, on=["ec_number", "protein_id"], how="left")
        df = df[df["_drop"].isna()].drop(columns="_drop").reset_index(drop=True)
        logger.info(f"Excluded {n_before - len(df)} pairs via --exclude; "
                    f"{len(df)} pairs remain")

    if len(df) == 0:
        logger.error(
            "No (protein, EC) pairs remain after filtering. "
            "This usually means protein IDs in the tool predictions CSV do not match "
            "those in the CAV results TSV. "
            "Check ID formats — e.g. 'sp|Q15185|GENE_HUMAN' vs 'Q15185'."
        )
        raise SystemExit(1)

    df = compute_matches(df, tool_cols)

    # ------------------------------------------------------------------
    # Per-tool summary
    # ------------------------------------------------------------------
    summary = tool_summary(df, tool_cols)

    # ------------------------------------------------------------------
    # CAV integration (optional)
    # ------------------------------------------------------------------
    if args.cav_results:
        df, cav_row = add_cav_metrics(
            df, args.cav_results,
            llr_threshold=args.llr_threshold,
        )
        summary = pd.concat([summary, pd.DataFrame([cav_row])], ignore_index=True)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    summary_file = out_dir / "ec_tool_comparison_summary.tsv"
    summary.to_csv(summary_file, sep="\t", index=False, float_format="%.4f")
    logger.info(f"Summary saved to {summary_file}")

    detail_cols = ["protein_id", "ec_number", "true_ec_norm"]
    for col in tool_cols:
        safe = col.replace("-", "_").replace(" ", "_")
        detail_cols += [f"{safe}__exact"]
    if args.cav_results:
        detail_cols += ["llr", "val_cav_score", "test_pos_percentile",
                        "test_neg_percentile", "cav__llr_positive"]

    detail_file = out_dir / "ec_tool_comparison_detail.tsv"
    df[[c for c in detail_cols if c in df.columns]].to_csv(
        detail_file, sep="\t", index=False, float_format="%.4f"
    )
    logger.info(f"Per-pair detail saved to {detail_file}")

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"EC tool comparison — {len(df)} (protein, EC) pairs")
    print(f"{'='*70}")
    display_cols = ["tool", "n_predicted", "coverage",
                    "recall_exact", "precision_exact"]
    if "note" in summary.columns:
        display_cols.append("note")
    print(summary[[c for c in display_cols if c in summary.columns]].to_string(index=False))

    rank_df = None
    if args.val_pkl and args.ec_base_dirs and args.scaler_pkl:
        rank_df = compute_ec_ranks(
            detail_df=df,
            val_pkl=args.val_pkl,
            ec_base_dirs=args.ec_base_dirs,
            scaler_pkl=args.scaler_pkl,
            version=args.version,
        )
    else:
        logger.info("Skipping rank figures (pass --val-pkl, --ec-base-dirs, --scaler-pkl to enable)")

    make_figures(summary, df, out_dir, figure_data_dir=args.figure_data_dir,
                 rank_df=rank_df, tool_cols=tool_cols,
                 llr_threshold=args.llr_threshold)


if __name__ == "__main__":
    main()
