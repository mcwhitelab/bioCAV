#!/usr/bin/env python3
"""
eval_ec.py

Evaluate an EC-number CAV library against a held-out gold standard.

For each (protein, EC number) pair in the reformatted gold standard TSV
(produced by reformat_ec_goldstandard.py):
  1. Embed validation proteins (cached as a single pkl under --eval-embed-dir).
  2. Retrieve + embed per-term test positives and test negatives if their
     pkls are not already on disk.
  3. Score all three sets against the EC term's CAV (pos/neg distributions
     cached as TSVs under --out-dir).
  4. Fit a Gaussian to each background distribution and compute the
     log-likelihood ratio (LLR) for each validation protein's CAV score.

Expected directory layout per EC term:
  {ec_base_dir}/ecNo_4-2-3-158/
      random_positive_train_max1000_cav/   CAV artifacts
      random_positive_test_max1000.span    test positive accession list
      random_negative_test_max1000.span    test negative accession list

Usage
-----
python specific_scripts/eval_ec.py \\
    --gold-standard  data/ec_gold_standard_long.tsv \\
    --ec-base-dirs   /xdisk/.../ec_dataset_part* \\
    --out-dir        results/ec_eval/ \\
    --min-n          3
"""

import argparse
import glob
import logging
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats

sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
from src.evaluate import load_cav_artifacts, compute_projections
from src.utils.data_loader import load_sequence_embeddings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

CAV_SUBDIR = "random_positive_train_max1000_cav"
POS_SPAN   = "random_positive_test_max1000.span"
NEG_SPAN   = "random_negative_test_max1000.span"

_SKIP_ID_PARTS = {"sp", "tr", "sw", "ref"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_ec_dir(ec_cav_id: str, ec_base_dirs: list) -> Path | None:
    """Return the first base dir that contains a subdirectory for ec_cav_id."""
    for base in ec_base_dirs:
        d = base / ec_cav_id
        if d.is_dir():
            return d
    return None


def ensure_fasta_and_pkl(
    span_file: Path,
    retrieve_script: str,
    embed_script: str,
    model: str,
) -> Path:
    """Run retrieve_fasta.py and hf_embed_new.py if outputs not already on disk."""
    fasta_file = Path(str(span_file) + ".fasta")
    pkl_file   = Path(str(fasta_file) + ".pkl")

    if not fasta_file.exists():
        logger.info(f"  Retrieving FASTA: {fasta_file}")
        subprocess.run(
            [sys.executable, retrieve_script, str(span_file), str(fasta_file)],
            check=True,
        )
    else:
        logger.info(f"  FASTA exists, skipping: {fasta_file.name}")

    if not pkl_file.exists():
        logger.info(f"  Embedding: {pkl_file.name}")
        subprocess.run(
            [
                sys.executable, embed_script,
                "-f", str(fasta_file),
                "-o", str(pkl_file),
                "--get_sequence_embedding",
                "--strat", "mean",
                "-l", "-11",
                "-m", model,
                "-b", "1",
                "--max_length", "2048",
            ],
            check=True,
        )
    else:
        logger.info(f"  PKL exists, skipping: {pkl_file.name}")

    return pkl_file


def score_pkl(pkl_path: Path, cav_artifacts: dict, scaler) -> tuple:
    """Load pkl and return (scores array, protein_id list)."""
    embs, ids = load_sequence_embeddings(str(pkl_path))
    scores = compute_projections(embs, cav_artifacts["concept_cav"], scaler)
    return scores.astype(float), ids


def log_prob_normal(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return -math.inf
    return float(stats.norm.logpdf(x, loc=mu, scale=sigma))


# ---------------------------------------------------------------------------
# Gold standard loading
# ---------------------------------------------------------------------------

def load_gold_standard(path: str, min_n: int) -> pd.DataFrame:
    """Read the long-format TSV from reformat_ec_goldstandard.py.

    Expected columns: protein_id, ec_number_raw, ec_number, ec_cav_id
    Returns deduplicated (protein_id, ec_number, ec_cav_id) rows for EC
    terms with >= min_n unique proteins.
    """
    df = pd.read_csv(path, sep="\t", dtype=str)
    logger.info(f"Loaded {len(df)} rows from {path}")

    required = {"protein_id", "ec_number", "ec_cav_id"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Gold standard is missing columns: {missing}. "
            f"Run reformat_ec_goldstandard.py first."
        )

    df = df[["protein_id", "ec_number", "ec_cav_id"]].drop_duplicates()

    logger.info(
        f"  {df['ec_number'].nunique()} EC terms | "
        f"{df['protein_id'].nunique()} unique proteins | "
        f"{len(df)} pairs"
    )

    # Keep only EC terms with >= min_n unique proteins
    counts = df.groupby("ec_number")["protein_id"].nunique()
    keep   = counts[counts >= min_n].index
    dropped = df["ec_number"].nunique() - len(keep)
    df = df[df["ec_number"].isin(keep)].reset_index(drop=True)

    logger.info(
        f"After min_n={min_n} filter: {df['ec_number'].nunique()} EC terms "
        f"({dropped} dropped) | {df['protein_id'].nunique()} proteins | "
        f"{len(df)} pairs"
    )
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gold-standard", required=True,
                        help="Long-format TSV from reformat_ec_goldstandard.py.")
    parser.add_argument("--ec-base-dirs", nargs="+", required=True,
                        help="Base directories containing ecNo_* subdirs. "
                             "Glob patterns (e.g. /path/ec_dataset_part*) are expanded.")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for results and cached score files.")
    parser.add_argument("--eval-embed-dir", default=None,
                        help="Directory for validation protein FASTA and PKL. "
                             "Defaults to --out-dir/eval_embeddings.")
    parser.add_argument("--val-fasta", default=None,
                        help="Pre-built FASTA for validation proteins. When provided, "
                             "the retrieve step is skipped (use for non-UniProt IDs "
                             "e.g. RefSeq). The FASTA from reformat_ec_goldstandard.py "
                             "can be used directly here.")
    parser.add_argument("--min-n", type=int, default=3,
                        help="Min validation proteins per EC term (default: 3).")
    parser.add_argument("--retrieve-script",
                        default=str(Path(__file__).parent / "retrieve_fasta.py"),
                        help="Path to retrieve_fasta.py.")
    parser.add_argument("--embed-script",
                        default="/groups/clairemcwhite/claire_workspace/github/mcwlab_utils/hf_embed_new.py",
                        help="Path to hf_embed_new.py.")
    parser.add_argument("--model",
                        default="/groups/clairemcwhite/models/ESMplusplus_large",
                        help="Model directory for embedding.")
    parser.add_argument("--version", default="v1",
                        help="CAV artifact version suffix (default: v1).")
    parser.add_argument("--scaler-pkl",
                        default="reference_population/scaler_v1.pkl",
                        help="Shared reference population scaler pkl "
                             "(default: reference_population/scaler_v1.pkl).")
    args = parser.parse_args()

    import joblib
    scaler = joblib.load(args.scaler_pkl)
    logger.info(f"Loaded shared scaler from {args.scaler_pkl}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_embed_dir = Path(args.eval_embed_dir) if args.eval_embed_dir \
                     else out_dir / "eval_embeddings"
    eval_embed_dir.mkdir(parents=True, exist_ok=True)

    # Expand glob patterns in --ec-base-dirs
    ec_base_dirs = []
    for pattern in args.ec_base_dirs:
        expanded = sorted(glob.glob(pattern))
        if not expanded:
            parser.error(f"--ec-base-dirs pattern matched nothing: {pattern}")
        ec_base_dirs.extend(Path(d) for d in expanded)
    logger.info(f"EC base dirs ({len(ec_base_dirs)}): {[str(d) for d in ec_base_dirs]}")

    # ------------------------------------------------------------------
    # Load gold standard
    # ------------------------------------------------------------------
    gold     = load_gold_standard(args.gold_standard, args.min_n)
    ec_terms = sorted(gold["ec_number"].unique())

    # ------------------------------------------------------------------
    # Pre-scan: keep only EC terms that have a CAV on disk
    # ------------------------------------------------------------------
    logger.info("Pre-scanning for available CAVs...")
    valid_ec_terms  = []
    skipped_no_dir  = []
    skipped_no_cav  = []

    for ec_num in ec_terms:
        ec_cav_id = gold[gold["ec_number"] == ec_num]["ec_cav_id"].iloc[0]
        ec_dir    = find_ec_dir(ec_cav_id, ec_base_dirs)
        if ec_dir is None:
            skipped_no_dir.append((ec_num, ec_cav_id))
            continue
        if not (ec_dir / CAV_SUBDIR).is_dir():
            skipped_no_cav.append((ec_num, ec_cav_id))
            continue
        valid_ec_terms.append(ec_num)

    logger.info(
        f"  {len(valid_ec_terms)} EC terms have CAVs | "
        f"{len(skipped_no_dir)} no directory | "
        f"{len(skipped_no_cav)} no CAV subdir"
    )

    gold     = gold[gold["ec_number"].isin(valid_ec_terms)].copy()
    ec_terms = valid_ec_terms
    logger.info(f"  Proteins to retrieve after CAV filter: {gold['protein_id'].nunique()}")

    # ------------------------------------------------------------------
    # Build / load validation protein embeddings
    # ------------------------------------------------------------------
    val_proteins = sorted(gold["protein_id"].unique())

    logger.info("Ensuring validation protein embeddings exist...")
    if args.val_fasta:
        # Non-UniProt IDs (e.g. RefSeq): FASTA already provided, skip retrieve step
        val_fasta = Path(args.val_fasta)
        if not val_fasta.exists():
            raise FileNotFoundError(f"--val-fasta not found: {val_fasta}")
        val_pkl = Path(str(val_fasta) + ".pkl")
        if not val_pkl.exists():
            logger.info(f"  Embedding provided FASTA: {val_fasta.name}")
            import subprocess, sys
            subprocess.run(
                [
                    sys.executable, args.embed_script,
                    "-f", str(val_fasta),
                    "-o", str(val_pkl),
                    "--get_sequence_embedding",
                    "--strat", "mean",
                    "-l", "-11",
                    "-m", args.model,
                    "-b", "1",
                    "--max_length", "2048",
                ],
                check=True,
            )
        else:
            logger.info(f"  PKL exists, skipping: {val_pkl.name}")
    else:
        # UniProt IDs: fetch FASTA then embed
        val_span = eval_embed_dir / "val_proteins.span"
        if not val_span.exists():
            val_span.write_text("\n".join(val_proteins) + "\n")
            logger.info(f"Wrote validation span: {val_span} ({len(val_proteins)} proteins)")
        val_pkl = ensure_fasta_and_pkl(
            val_span, args.retrieve_script, args.embed_script, args.model
        )
    val_embs, val_ids = load_sequence_embeddings(str(val_pkl))

    # Build lookup handling both bare accessions and full FASTA-header IDs
    val_id_to_idx = {}
    for i, sid in enumerate(val_ids):
        val_id_to_idx[sid] = i
        for part in sid.split("|"):
            if part and part not in _SKIP_ID_PARTS:
                val_id_to_idx.setdefault(part, i)
    logger.info(f"Validation embeddings loaded: {len(val_ids)} proteins")

    missing_val = set(val_proteins) - set(val_id_to_idx)
    if missing_val:
        logger.warning(
            f"{len(missing_val)} validation proteins not in pkl — likely failed retrieval "
            f"(e.g. {sorted(missing_val)[:3]})"
        )

    # ------------------------------------------------------------------
    # Evaluate each EC term — results written progressively
    # ------------------------------------------------------------------
    out_file = out_dir / "eval_ec_results.tsv"

    # Resume: find (ec_number, protein_id) pairs already written
    completed_pairs = set()
    if out_file.exists():
        try:
            existing = pd.read_csv(out_file, sep="\t", usecols=["ec_number", "protein_id"])
            completed_pairs = set(zip(existing["ec_number"], existing["protein_id"]))
            logger.info(f"Resuming: {len(completed_pairs)} pairs already in {out_file}")
        except Exception as e:
            logger.warning(f"Could not read existing output file: {e}")

    header_written = out_file.exists()
    n_written = 0

    for ec_num in ec_terms:
        ec_cav_id = gold[gold["ec_number"] == ec_num]["ec_cav_id"].iloc[0]
        ec_dir    = find_ec_dir(ec_cav_id, ec_base_dirs)
        cav_dir   = ec_dir / CAV_SUBDIR

        val_proteins_for_term = gold[gold["ec_number"] == ec_num]["protein_id"].tolist()
        remaining = [p for p in val_proteins_for_term
                     if (ec_num, p) not in completed_pairs]
        if not remaining:
            logger.info(f"{ec_num}: all {len(val_proteins_for_term)} proteins done — skipping")
            continue
        if len(remaining) < len(val_proteins_for_term):
            logger.info(f"{ec_num}: {len(val_proteins_for_term) - len(remaining)} done, "
                        f"{len(remaining)} remaining")

        logger.info(f"Processing {ec_num} ({ec_cav_id}, {len(remaining)} validation proteins)")

        # -------------------------------------------------------
        # Test pos / neg score distributions (cached per EC term)
        # -------------------------------------------------------
        pos_scores_file = out_dir / f"{ec_cav_id}_test_pos_scores.tsv"
        neg_scores_file = out_dir / f"{ec_cav_id}_test_neg_scores.tsv"

        cav_artifacts = load_cav_artifacts(str(cav_dir), version=args.version)

        if pos_scores_file.exists() and neg_scores_file.exists():
            logger.info(f"  Loading cached pos/neg scores")
            pos_scores = pd.read_csv(pos_scores_file, sep="\t")["cav_score"].values.astype(float)
            neg_scores = pd.read_csv(neg_scores_file, sep="\t")["cav_score"].values.astype(float)
        else:
            pos_pkl = ensure_fasta_and_pkl(
                ec_dir / POS_SPAN, args.retrieve_script, args.embed_script, args.model
            )
            neg_pkl = ensure_fasta_and_pkl(
                ec_dir / NEG_SPAN, args.retrieve_script, args.embed_script, args.model
            )

            pos_scores, pos_ids = score_pkl(pos_pkl, cav_artifacts, scaler)
            neg_scores, neg_ids = score_pkl(neg_pkl, cav_artifacts, scaler)

            pd.DataFrame({"protein_id": pos_ids, "cav_score": pos_scores}).to_csv(
                pos_scores_file, sep="\t", index=False, float_format="%.6f"
            )
            pd.DataFrame({"protein_id": neg_ids, "cav_score": neg_scores}).to_csv(
                neg_scores_file, sep="\t", index=False, float_format="%.6f"
            )
            logger.info(f"  Scored {len(pos_ids)} positives, {len(neg_ids)} negatives")

        pos_mu, pos_sigma = float(pos_scores.mean()), float(pos_scores.std())
        neg_mu, neg_sigma = float(neg_scores.mean()), float(neg_scores.std())

        # -------------------------------------------------------
        # Score each validation protein against this EC CAV
        # -------------------------------------------------------
        ec_term_rows = []

        for protein_id in remaining:
            idx = val_id_to_idx.get(protein_id)
            if idx is None:
                logger.warning(f"  {protein_id} not in validation pkl — skipping")
                continue

            val_score = float(
                compute_projections(
                    val_embs[idx : idx + 1],
                    cav_artifacts["concept_cav"],
                    scaler,
                )[0]
            )

            lp_pos = log_prob_normal(val_score, pos_mu, pos_sigma)
            lp_neg = log_prob_normal(val_score, neg_mu, neg_sigma)
            llr    = lp_pos - lp_neg

            pos_zscore = (val_score - pos_mu) / pos_sigma if pos_sigma > 0 else math.nan
            neg_zscore = (val_score - neg_mu) / neg_sigma if neg_sigma > 0 else math.nan

            ec_term_rows.append({
                "ec_number":            ec_num,
                "ec_cav_id":            ec_cav_id,
                "protein_id":           protein_id,
                "val_cav_score":        val_score,
                "pos_n":                len(pos_scores),
                "pos_mean":             pos_mu,
                "pos_std":              pos_sigma,
                "pos_median":           float(np.median(pos_scores)),
                "pos_min":              float(pos_scores.min()),
                "pos_max":              float(pos_scores.max()),
                "neg_n":                len(neg_scores),
                "neg_mean":             neg_mu,
                "neg_std":              neg_sigma,
                "neg_median":           float(np.median(neg_scores)),
                "neg_min":              float(neg_scores.min()),
                "neg_max":              float(neg_scores.max()),
                "log_prob_test_pos":    lp_pos,
                "log_prob_test_neg":    lp_neg,
                "llr":                  llr,
                "test_pos_zscore":      pos_zscore,
                "test_neg_zscore":      neg_zscore,
                "test_pos_percentile":  float(stats.percentileofscore(pos_scores, val_score)),
                "test_neg_percentile":  float(stats.percentileofscore(neg_scores, val_score)),
            })

        # Append this EC term's rows immediately
        if ec_term_rows:
            ec_df = pd.DataFrame(ec_term_rows)
            ec_df.to_csv(
                out_file, sep="\t", index=False,
                mode="a", header=not header_written,
                float_format="%.6f",
            )
            header_written = True
            n_written += len(ec_term_rows)
            logger.info(f"  Wrote {len(ec_term_rows)} rows ({n_written} total)")

    # ------------------------------------------------------------------
    # Write no-CAV report
    # ------------------------------------------------------------------
    no_cav_file = out_dir / "no_cav_available.tsv"
    if skipped_no_dir or skipped_no_cav:
        no_cav_df = pd.DataFrame(
            [(ec, cid, "no_directory") for ec, cid in skipped_no_dir] +
            [(ec, cid, "no_cav_subdir") for ec, cid in skipped_no_cav],
            columns=["ec_number", "ec_cav_id", "reason"],
        )
        no_cav_df.to_csv(no_cav_file, sep="\t", index=False)
        logger.warning(
            f"No CAV for {len(skipped_no_dir) + len(skipped_no_cav)} terms — "
            f"written to {no_cav_file}"
        )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Results: {out_file}")
    if out_file.exists():
        results = pd.read_csv(out_file, sep="\t")
        print(f"  protein-term pairs  : {len(results)}")
        print(f"  EC terms evaluated  : {results['ec_number'].nunique()}")
        print(f"  unique proteins     : {results['protein_id'].nunique()}")
        print(f"\nLLR summary:")
        print(results["llr"].describe().to_string())
        print(f"\ntest_pos_percentile summary:")
        print(results["test_pos_percentile"].describe().to_string())


if __name__ == "__main__":
    main()
