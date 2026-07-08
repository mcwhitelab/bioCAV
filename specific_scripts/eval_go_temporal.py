#!/usr/bin/env python3
"""
eval_go_temporal.py

Evaluate a GO-term CAV library against experimental annotations added after
the training cutoff (temporal holdout evaluation).

For each protein-GO term pair in the gold standard:
  1. Embed test proteins (cached as a single pkl under --out-dir).
  2. Retrieve + embed the per-term training positives and test negatives
     if their pkls are not already on disk.
  3. Score positives and negatives against the GO term's CAV (cached as
     TSVs under --out-dir).
  4. Fit a Gaussian to each background distribution and compute the
     log-likelihood ratio (LLR) for the test protein's CAV score.

Expected directory layout per GO term:
  {go_base_dir}/GO_XXXXXXX/
      random_positive_train_max1000_cav/       CAV artifacts
      random_positive_train_max1000.span       positive accession list
      random_negative_test_max1000.span        negative accession list

Usage
-----
python specific_scripts/eval_go_temporal.py \\
    --gold-standard  annotations_post2021.tsv \\
    --go-base-dirs   /xdisk/.../go_dataset_part1 /xdisk/.../go_dataset_part2 \\
    --out-dir        results/temporal_eval/ \\
    --min-n          5 \\
    --date-cutoff    20211231
"""

import argparse
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

# ECO IDs for direct / experimental assay evidence (as stored in GPAD Evidence_Code column)
DIRECT_ASSAY_CODES = {
    "ECO:0000314",  # IDA  - Inferred from Direct Assay
    "ECO:0000315",  # IMP  - Inferred from Mutant Phenotype
    "ECO:0000316",  # IGI  - Inferred from Genetic Interaction
    "ECO:0000353",  # IPI  - Inferred from Physical Interaction
    "ECO:0000270",  # IEP  - Inferred from Expression Pattern
    "ECO:0000269",  # EXP  - Inferred from Experiment
    "ECO:0007005",  # HDA  - High-throughput Direct Assay
    "ECO:0007001",  # HMP  - High-throughput Mutant Phenotype
    "ECO:0007003",  # HGI  - High-throughput Genetic Interaction
    "ECO:0007007",  # HEP  - High-throughput Expression Pattern
}

CAV_SUBDIR = "random_positive_train_max1000_cav"
POS_SPAN   = "random_positive_test_max1000.span"
NEG_SPAN   = "random_negative_test_max1000.span"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def go_id_to_dirname(go_id: str) -> str:
    """GO:0000036 → GO_0000036"""
    return go_id.replace(":", "_")


def find_go_dir(go_id: str, go_base_dirs: list) -> Path | None:
    """Return the first base dir that contains a subdirectory for go_id."""
    dirname = go_id_to_dirname(go_id)
    for base in go_base_dirs:
        d = base / dirname
        if d.is_dir():
            return d
    return None


def ensure_fasta_and_pkl(
    span_file: Path,
    retrieve_script: str,
    embed_script: str,
    model: str,
) -> Path:
    """
    Run retrieve_fasta.py and hf_embed_new.py if outputs are not already
    on disk.  Returns the path to the .fasta.pkl file.
    """
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
    """Log probability density of x under N(mu, sigma). Returns -inf if sigma=0."""
    if sigma <= 0:
        return -math.inf
    return float(stats.norm.logpdf(x, loc=mu, scale=sigma))


# ---------------------------------------------------------------------------
# Gold standard loading
# ---------------------------------------------------------------------------

COL_NAMES = [
    "DB", "DB_Object_ID", "Qualifier", "GO_ID", "DB_Reference",
    "Evidence_Code", "With_From", "Interacting_taxon_ID", "Date",
    "Assigned_by", "Annotation_Extension", "Annotation_Properties",
]


def load_gold_standard(path: str, date_cutoff: int, min_n: int,
                       filter_evidence: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, header=None, names=COL_NAMES,
                     comment="!")

    # Keep only UniProtKB entries — other DBs (e.g. IntAct CPX-XXXX) can't be fetched via UniProt
    before = len(df)
    df = df[df["DB"] == "UniProtKB"].copy()
    logger.info(f"UniProtKB filter: {before} → {len(df)} rows "
                f"(dropped {before - len(df)} non-UniProtKB entries)")

    if filter_evidence:
        before = len(df)
        df = df[df["Evidence_Code"].isin(DIRECT_ASSAY_CODES)].copy()
        logger.info(f"Evidence filter: {before} → {len(df)} rows")

    # Post-cutoff dates only
    df["Date"] = pd.to_numeric(df["Date"], errors="coerce")
    logger.info(f"Date range before filter: {int(df['Date'].min())} – {int(df['Date'].max())} "
                f"({len(df)} rows, {df['GO_ID'].nunique()} GO terms, {df['DB_Object_ID'].nunique()} proteins)")
    df = df[df["Date"] > date_cutoff]
    logger.info(f"Date range after  filter (> {date_cutoff}): {int(df['Date'].min())} – {int(df['Date'].max())} "
                f"({len(df)} rows, {df['GO_ID'].nunique()} GO terms, {df['DB_Object_ID'].nunique()} proteins)")

    df = df.rename(columns={"DB_Object_ID": "protein_id", "GO_ID": "go_term"})
    df = df[["protein_id", "go_term"]].drop_duplicates()

    # Normalise accessions to bare UniProt format:
    #   "UniProtKB:P0C6Y4-PRO_0000037376" → "P0C6Y4"
    # Strip optional "DB:" prefix, then strip isoform/chain suffix after "-"
    # (standard UniProt accessions never contain "-").
    df["protein_id"] = (
        df["protein_id"]
        .str.split(":").str[-1]   # drop "UniProtKB:" prefix if present
        .str.split("-").str[0]    # drop "-PRO_XXXXX" / "-2" isoform suffixes
    )

    # Keep only GO terms with >= min_n unique proteins
    counts = df.groupby("go_term")["protein_id"].nunique()
    keep   = counts[counts >= min_n].index
    df     = df[df["go_term"].isin(keep)]

    logger.info(
        f"Gold standard after filters: {len(df)} protein-term pairs | "
        f"{df['go_term'].nunique()} GO terms | "
        f"{df['protein_id'].nunique()} unique proteins"
    )
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gold-standard", required=True,
                        help="GO annotation TSV with post-cutoff experimental annotations.")
    parser.add_argument("--go-base-dirs", nargs="+", required=True,
                        help="Base directories containing GO_XXXXXXX subdirs.")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for results and cached score files.")
    parser.add_argument("--eval-embed-dir", default=None,
                        help="Directory to store FASTA and PKL for gold-standard test proteins. "
                             "Defaults to --out-dir/eval_embeddings.")
    parser.add_argument("--min-n", type=int, default=5,
                        help="Min test proteins per GO term (default: 5).")
    parser.add_argument("--date-cutoff", type=int, default=20211231,
                        help="Keep annotations with DATE > this value YYYYMMDD (default: 20211231).")
    parser.add_argument("--filter-evidence", action="store_true",
                        help="Restrict to direct/experimental ECO evidence codes. "
                             "Off by default (use when your file is not already pre-filtered).")
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
                        help="Shared reference population scaler pkl used for all CAV projections "
                             "(default: reference_population/scaler_v1.pkl).")
    args = parser.parse_args()

    import joblib
    scaler = joblib.load(args.scaler_pkl)
    logger.info(f"Loaded shared scaler from {args.scaler_pkl}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_embed_dir = Path(args.eval_embed_dir) if args.eval_embed_dir else out_dir / "eval_embeddings"
    eval_embed_dir.mkdir(parents=True, exist_ok=True)
    import glob
    go_base_dirs = []
    for pattern in args.go_base_dirs:
        expanded = sorted(glob.glob(pattern))
        if not expanded:
            parser.error(f"--go-base-dirs pattern matched nothing: {pattern}")
        go_base_dirs.extend(Path(d) for d in expanded)
    logger.info(f"GO base dirs ({len(go_base_dirs)}): {[str(d) for d in go_base_dirs]}")

    # ------------------------------------------------------------------
    # Load gold standard (date filter applied here)
    # ------------------------------------------------------------------
    gold     = load_gold_standard(args.gold_standard, args.date_cutoff, args.min_n,
                                  filter_evidence=args.filter_evidence)
    go_terms = sorted(gold["go_term"].unique())
    logger.info(f"GO terms after date/min-n filter: {len(go_terms)}")

    # ------------------------------------------------------------------
    # Pre-scan: keep only GO terms that have a CAV on disk
    # ------------------------------------------------------------------
    logger.info("Pre-scanning for available CAVs...")
    valid_go_terms  = []
    skipped_no_dir  = []
    skipped_no_cav  = []
    for go_id in go_terms:
        go_dir = find_go_dir(go_id, go_base_dirs)
        if go_dir is None:
            skipped_no_dir.append(go_id)
            continue
        if not (go_dir / CAV_SUBDIR).is_dir():
            skipped_no_cav.append(go_id)
            continue
        valid_go_terms.append(go_id)

    logger.info(f"  {len(valid_go_terms)} GO terms have CAVs | "
                f"{len(skipped_no_dir)} no directory | "
                f"{len(skipped_no_cav)} no CAV dir")

    gold     = gold[gold["go_term"].isin(valid_go_terms)].copy()
    go_terms = valid_go_terms
    logger.info(f"  Proteins to retrieve after CAV filter: {gold['protein_id'].nunique()}")

    # ------------------------------------------------------------------
    # Build / load validation protein embeddings
    # (proteins from the post-cutoff annotation file being evaluated)
    # ------------------------------------------------------------------
    val_proteins = sorted(gold["protein_id"].unique())
    val_span     = eval_embed_dir / "val_proteins.span"
    if not val_span.exists():
        val_span.write_text("\n".join(val_proteins) + "\n")
        logger.info(f"Wrote validation span: {val_span} ({len(val_proteins)} proteins)")

    logger.info(f"Ensuring validation protein embeddings exist (eval_embed_dir={eval_embed_dir})...")
    val_pkl = ensure_fasta_and_pkl(
        val_span, args.retrieve_script, args.embed_script, args.model
    )
    val_embs, val_ids = load_sequence_embeddings(str(val_pkl))
    # Build lookup that handles both bare accessions ("Q15185") and full
    # FASTA-header IDs ("sp|Q15185|TEBP_HUMAN") by indexing every |-part.
    _SKIP = {"sp", "tr", "sw", "ref"}
    val_id_to_idx = {}
    for i, sid in enumerate(val_ids):
        val_id_to_idx[sid] = i
        for part in sid.split("|"):
            if part and part not in _SKIP:
                val_id_to_idx.setdefault(part, i)
    logger.info(f"Validation embeddings loaded: {len(val_ids)} proteins")

    missing_val = set(val_proteins) - set(val_ids)
    if missing_val:
        logger.warning(
            f"{len(missing_val)} validation proteins not found in pkl — likely failed retrieval "
            f"(e.g. {sorted(missing_val)[:3]})"
        )

    # ------------------------------------------------------------------
    # Evaluate each GO term — results written progressively
    # ------------------------------------------------------------------
    out_file = out_dir / "eval_temporal_results.tsv"

    # Resume: find protein-term pairs already written to the output file
    completed_pairs = set()
    if out_file.exists():
        try:
            existing = pd.read_csv(out_file, sep="\t", usecols=["go_term", "protein_id"])
            completed_pairs = set(zip(existing["go_term"], existing["protein_id"]))
            logger.info(
                f"Resuming: {len(completed_pairs)} protein-term pairs already in {out_file}"
            )
        except Exception as e:
            logger.warning(f"Could not read existing output file: {e}")

    header_written = out_file.exists()
    n_written = 0

    for go_id in go_terms:
        go_dir  = find_go_dir(go_id, go_base_dirs)
        cav_dir = go_dir / CAV_SUBDIR

        val_proteins_for_term = gold[gold["go_term"] == go_id]["protein_id"].tolist()
        remaining = [p for p in val_proteins_for_term if (go_id, p) not in completed_pairs]
        if not remaining:
            logger.info(f"{go_id}: all {len(val_proteins_for_term)} validation proteins already in output — skipping")
            continue
        if len(remaining) < len(val_proteins_for_term):
            logger.info(f"{go_id}: {len(val_proteins_for_term) - len(remaining)} validation proteins already done, "
                        f"{len(remaining)} remaining")

        logger.info(f"Processing {go_id} ({len(remaining)} validation proteins)")

        # -------------------------------------------------------
        # Test pos / neg CAV score distributions (cached per GO term)
        # -------------------------------------------------------
        pos_scores_file = out_dir / f"{go_id}_test_pos_scores.tsv"
        neg_scores_file = out_dir / f"{go_id}_test_neg_scores.tsv"

        cav_artifacts = load_cav_artifacts(str(cav_dir), version=args.version)

        if pos_scores_file.exists() and neg_scores_file.exists():
            logger.info(f"  Loading cached pos/neg scores")
            pos_scores = pd.read_csv(pos_scores_file, sep="\t")["cav_score"].values.astype(float)
            neg_scores = pd.read_csv(neg_scores_file, sep="\t")["cav_score"].values.astype(float)
        else:
            pos_pkl = ensure_fasta_and_pkl(
                go_dir / POS_SPAN, args.retrieve_script, args.embed_script, args.model
            )
            neg_pkl = ensure_fasta_and_pkl(
                go_dir / NEG_SPAN, args.retrieve_script, args.embed_script, args.model
            )

            pos_scores, pos_ids = score_pkl(pos_pkl, cav_artifacts, scaler)
            neg_scores, neg_ids = score_pkl(neg_pkl, cav_artifacts, scaler)

            pd.DataFrame({"protein_id": pos_ids, "cav_score": pos_scores}).to_csv(
                pos_scores_file, sep="\t", index=False, float_format="%.6f"
            )
            pd.DataFrame({"protein_id": neg_ids, "cav_score": neg_scores}).to_csv(
                neg_scores_file, sep="\t", index=False, float_format="%.6f"
            )
            logger.info(
                f"  Scored {len(pos_ids)} positives, {len(neg_ids)} negatives"
            )

        # Gaussian parameters for background distributions
        pos_mu, pos_sigma = float(pos_scores.mean()), float(pos_scores.std())
        neg_mu, neg_sigma = float(neg_scores.mean()), float(neg_scores.std())

        # -------------------------------------------------------
        # Score each validation protein against this GO term's CAV
        # and compare to test pos/neg distributions
        # -------------------------------------------------------
        go_term_rows = []

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

            go_term_rows.append({
                "go_term":          go_id,
                "protein_id":       protein_id,
                "val_cav_score":    val_score,
                # Positive background
                "pos_n":            len(pos_scores),
                "pos_mean":         pos_mu,
                "pos_std":          pos_sigma,
                "pos_median":       float(np.median(pos_scores)),
                "pos_min":          float(pos_scores.min()),
                "pos_max":          float(pos_scores.max()),
                # Negative background
                "neg_n":            len(neg_scores),
                "neg_mean":         neg_mu,
                "neg_std":          neg_sigma,
                "neg_median":       float(np.median(neg_scores)),
                "neg_min":          float(neg_scores.min()),
                "neg_max":          float(neg_scores.max()),
                # Comparison to test pos/neg distributions
                "log_prob_test_pos":    lp_pos,
                "log_prob_test_neg":    lp_neg,
                "llr":                  llr,
                "test_pos_zscore":      pos_zscore,
                "test_neg_zscore":      neg_zscore,
                "test_pos_percentile":  float(stats.percentileofscore(pos_scores, val_score)),
                "test_neg_percentile":  float(stats.percentileofscore(neg_scores, val_score)),
            })

        # Append this GO term's rows immediately
        if go_term_rows:
            go_df = pd.DataFrame(go_term_rows)
            go_df.to_csv(
                out_file, sep="\t", index=False,
                mode="a", header=not header_written,
                float_format="%.6f",
            )
            header_written = True
            n_written += len(go_term_rows)
            logger.info(f"  Wrote {len(go_term_rows)} rows to {out_file} ({n_written} total)")

    # Write no-CAV report
    no_cav_terms = skipped_no_dir + skipped_no_cav
    no_cav_file  = out_dir / "no_cav_available.tsv"
    if no_cav_terms:
        no_cav_df = pd.DataFrame({
            "go_term": no_cav_terms,
            "reason":  (["no_directory"] * len(skipped_no_dir) +
                        ["no_cav_dir"]  * len(skipped_no_cav)),
        })
        no_cav_df.to_csv(no_cav_file, sep="\t", index=False)
        logger.warning(f"No CAV for {len(no_cav_terms)} terms — written to {no_cav_file}")

    print(f"\n{'='*60}")
    print(f"Results: {out_file}")
    if out_file.exists():
        results = pd.read_csv(out_file, sep="\t")
        print(f"  protein-term pairs total     : {len(results)}")
        print(f"  GO terms total               : {results['go_term'].nunique()}")
        print(f"  unique proteins total        : {results['protein_id'].nunique()}")
        print(f"\nLLR summary:")
        print(results["llr"].describe().to_string())
        print(f"\ntest_pos_percentile summary:")
        print(results["test_pos_percentile"].describe().to_string())


if __name__ == "__main__":
    main()
