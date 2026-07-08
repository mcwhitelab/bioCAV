#!/usr/bin/env python3
"""
Query a protein embedding pkl for proteins most aligned with one or more CAVs.

Scores every protein by its projection (dot product) onto each CAV direction
and returns a ranked TSV with one score column per CAV.

--cav accepts one or more paths (directories or bare .npy files).

Usage
-----
# Pre-embedded pkl, single CAV:
python specific_scripts/query_proteins_by_cav.py \\
    --cav  cavs/my_concept/ \\
    --pkl  embeddings/proteins.pkl \\
    --out  results/top_proteins.tsv

# On-the-fly embedding, multiple CAVs:
python specific_scripts/query_proteins_by_cav.py \\
    --cav  cavs/concept_a/ cavs/concept_b/ \\
    --embed-fasta  proteome.fasta \\
    --embed-script /path/to/hf_embed_new.py \\
    --model        /path/to/ESMplusplus_large \\
    --out  results/top_proteins.tsv
"""

import os
import sys
import subprocess
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from src.utils.data_loader import load_sequence_embeddings, load_embeddings_pkl, load_spans
from src.detect import sliding_window_scan

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_cav(cav_path: str, version: str = 'v1'):
    """
    Load CAV vector and optional scaler.

    Returns (cav_vec, scaler). scaler is None when given a bare .npy.
    """
    p = Path(cav_path)

    if p.is_dir():
        from src.evaluate import load_cav_artifacts
        artifacts = load_cav_artifacts(str(p), version=version)
        return artifacts['concept_cav'], artifacts['scaler']

    cav_vec = np.load(p)
    logger.warning(
        "Loading CAV from a bare .npy — no scaler will be applied. "
        "Make sure the embeddings and CAV are already in the same space."
    )
    return cav_vec, None


def cav_name(cav_path: str, depth: int = 1) -> str:
    """Return a short label for a CAV path.

    depth=1 (default): immediate directory/file name
    depth=2: parent_dirname  e.g. GO_0000124_random_positive_train_max1000_cav
    depth=3: grandparent_parent_dirname, etc.
    """
    p = Path(cav_path).resolve()
    parts = list(p.parts)
    stem = p.stem if p.is_file() else p.name
    if depth <= 1:
        return stem
    ancestors = parts[-(depth):-1]  # depth-1 components above the leaf
    return "_".join(ancestors + [stem])


def score_proteins(embs: np.ndarray, cav_vec: np.ndarray, scaler) -> np.ndarray:
    """Apply scaler and return per-protein projection scores."""
    X = embs.astype('float32')
    if scaler is not None:
        X = scaler.transform(X)
    return X @ cav_vec


def parse_fasta_lengths(fasta_path: str) -> dict:
    """Return {accession: length} parsed from a FASTA file."""
    lengths = {}
    current_id = None
    current_len = 0
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None:
                    lengths[current_id] = current_len
                current_id = line[1:].split()[0]
                current_len = 0
            else:
                current_len += len(line)
    if current_id is not None:
        lengths[current_id] = current_len
    return lengths


def span_window_size(spans_path: str) -> int:
    """Return the max residue width across all entries in a spans file."""
    spans = load_spans(spans_path)
    widths = []
    for _, span in spans:
        if span is None:
            continue
        if len(span) == 1:
            widths.append(1)
        elif len(span) == 2:
            start, end = span
            widths.append(end - start + 1)
        else:
            widths.append(len(span))
    if not widths:
        raise ValueError(f"No positional spans found in {spans_path} — cannot infer window size.")
    return max(widths)


def sliding_window_scores(
    aa_embs: np.ndarray,
    protein_ids: list,
    protein_lengths: dict,
    window_size: int,
    cav_artifacts: dict,
) -> np.ndarray:
    """Score each protein by the max window projection onto the CAV direction."""
    scores = np.empty(len(protein_ids), dtype='float32')
    missing = []
    for i, pid in enumerate(protein_ids):
        seq_len = protein_lengths.get(pid)
        if seq_len is None:
            missing.append(pid)
            seq_len = None
        proj, _ = sliding_window_scan(aa_embs[i], cav_artifacts, window_size, seq_len=seq_len)
        scores[i] = proj.max() if len(proj) > 0 else 0.0
    if missing:
        logger.warning(
            f"{len(missing)} proteins not found in FASTA; padded length used for those entries."
        )
    return scores


def length_correct_scores(
    raw_scores: np.ndarray,
    protein_ids: list,
    protein_lengths: dict,
    K: int,
    ref_mean: np.ndarray,
    cav_vec: np.ndarray,
    scaler,
) -> np.ndarray:
    """
    Correct CAV scores for motif dilution in whole-sequence embeddings.

    Models full_seq ≈ (K/L)*span + ((L-K)/L)*μ, rearranges to estimate the
    span embedding, and scores that instead.
    """
    mu = ref_mean.reshape(1, -1).astype('float32')
    if scaler is not None:
        mu = scaler.transform(mu)
    mu_proj = (mu @ cav_vec).item()

    corrected = np.empty_like(raw_scores)
    missing = []
    for i, pid in enumerate(protein_ids):
        L = protein_lengths.get(pid)
        if L is None:
            missing.append(pid)
            corrected[i] = raw_scores[i]
            continue
        corrected[i] = (L / K) * raw_scores[i] - ((L - K) / K) * mu_proj

    if missing:
        logger.warning(
            f"{len(missing)} proteins not found in FASTA (lengths unknown); "
            "raw scores used for those entries."
        )
    return corrected


def embed_fasta(fasta_path: str, embed_script: str, model: str,
                layer: int, batch: int, max_length: int) -> str:
    """Run hf_embed_new.py on fasta_path; return path to the output pkl."""
    pkl_out = fasta_path + ".pkl"
    if os.path.exists(pkl_out):
        logger.info(f"Skipping embedding: {pkl_out} already exists")
        return pkl_out

    logger.info(f"Embedding {fasta_path} → {pkl_out}")
    cmd = [
        sys.executable, embed_script,
        "-f", fasta_path,
        "-o", pkl_out,
        "--get_aa_embeddings",
        "--get_sequence_embedding",
        "--strat", "mean",
        "-l", str(layer),
        "-m", model,
        "-b", str(batch),
        "--max_length", str(max_length),
    ]
    subprocess.run(cmd, check=True)
    return pkl_out


def main():
    parser = argparse.ArgumentParser(
        description="Rank proteins by projection onto one or more CAV directions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # CAV inputs
    parser.add_argument('--cav', required=True, nargs='+',
                        help='One or more CAV artifact directories or bare .npy files.')
    parser.add_argument('--version', default='v1',
                        help='CAV artifact version suffix (default: v1).')
    parser.add_argument('--cav-name-depth', type=int, default=1,
                        help='Number of path components (from the right) to use as the '
                             'CAV column name, joined with "_". '
                             '1=dir name only (default), 2=parent_dirname, etc. '
                             'Useful when many CAVs share the same directory name '
                             '(e.g. random_positive_train_max1000_cav) and the GO/EC '
                             'term ID lives one level up.')
    parser.add_argument('--scaler-pkl', default=None,
                        help='Path to a shared reference scaler (joblib pkl). '
                             'If provided, all CAVs are scored with a single '
                             'matrix multiply instead of a per-CAV loop.')
    parser.add_argument('--cav-name-strip', default=None,
                        help='Strip this suffix from each CAV name after depth assembly. '
                             'E.g. "_positive_train_max1000_cav" to turn '
                             '"GO_0000124_random_positive_train_max1000_cav" → "GO_0000124_random".')

    # Embedding source — exactly one of --pkl or --embed-fasta required
    emb_group = parser.add_mutually_exclusive_group(required=True)
    emb_group.add_argument('--pkl',
                           help='Pre-computed protein embeddings pkl.')
    emb_group.add_argument('--embed-fasta',
                           help='FASTA to embed on the fly (produces <fasta>.pkl).')

    # On-the-fly embedding parameters
    parser.add_argument('--embed-script', default=None,
                        help='Path to hf_embed_new.py (required with --embed-fasta).')
    parser.add_argument('--model', default=None,
                        help='Model directory for embedding (required with --embed-fasta).')
    parser.add_argument('--embed-layer', type=int, default=-11,
                        help='Layer index for embedding (default: -11).')
    parser.add_argument('--embed-batch', type=int, default=1,
                        help='Batch size for embedding (default: 1).')
    parser.add_argument('--embed-max-length', type=int, default=2048,
                        help='Max sequence length for embedding (default: 2048).')

    # Output / ranking
    parser.add_argument('--out', required=True, help='Output TSV path.')
    parser.add_argument('--k', type=int, default=100,
                        help='Number of top proteins to return (default: 100; -1 for all).')
    parser.add_argument('--sort-cav', type=int, default=0,
                        help='Index of the CAV to sort results by (default: 0, the first).')

    # Sliding-window mode
    parser.add_argument('--sliding-window', action='store_true',
                        help='Score each protein by its max sliding-window projection. '
                             'Requires aa_embeddings in --pkl/--embed-fasta, --spans, and --fasta.')
    parser.add_argument('--spans', default=None,
                        help='Spans file used for CAV training; sets the window size.')
    parser.add_argument('--pre-projected', action='store_true',
                        help='aa_embeddings are already in scaler space; skip scaler.')

    # Length correction
    parser.add_argument('--length-correct', action='store_true',
                        help='Correct scores for motif dilution. '
                             'Requires --span-length, --fasta, and --ref-pkl.')
    parser.add_argument('--span-length', type=int, default=None,
                        help='Length K of the motif span (number of residues).')
    parser.add_argument('--fasta', default=None,
                        help='FASTA for the search proteome (protein length lookup).')
    parser.add_argument('--ref-pkl', default=None,
                        help='Reference negative pkl for background mean embedding.')
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Resolve embedding pkl
    # ------------------------------------------------------------------ #
    if args.embed_fasta:
        if not args.embed_script or not args.model:
            parser.error("--embed-fasta requires --embed-script and --model.")
        pkl_path = embed_fasta(
            args.embed_fasta, args.embed_script, args.model,
            args.embed_layer, args.embed_batch, args.embed_max_length,
        )
        # Also use the embed-fasta as the fasta for length lookup if not set
        if args.fasta is None:
            args.fasta = args.embed_fasta
    else:
        pkl_path = args.pkl

    # ------------------------------------------------------------------ #
    # Load CAVs
    # ------------------------------------------------------------------ #
    cavs = []
    for cav_path in args.cav:
        logger.info(f"Loading CAV from {cav_path}")
        cav_vec, scaler = load_cav(cav_path, version=args.version)
        name = cav_name(cav_path, depth=args.cav_name_depth)
        if args.cav_name_strip and name.endswith(args.cav_name_strip):
            name = name[: -len(args.cav_name_strip)]
        logger.info(f"  {name}: dim={cav_vec.shape[0]}")
        cavs.append((name, cav_vec, scaler))

    if args.sort_cav >= len(cavs):
        parser.error(f"--sort-cav {args.sort_cav} is out of range ({len(cavs)} CAVs).")

    # ------------------------------------------------------------------ #
    # Load embeddings and score all CAVs
    # ------------------------------------------------------------------ #
    if args.sliding_window:
        if args.spans is None:
            raise ValueError("--sliding-window requires --spans to infer window size.")
        window_size = span_window_size(args.spans)
        logger.info(f"Sliding-window mode: window_size={window_size} (from {args.spans})")

        emb_dict, protein_ids = load_embeddings_pkl(pkl_path)
        if 'aa_embeddings' not in emb_dict:
            raise ValueError(f"{pkl_path} does not contain aa_embeddings. "
                             "Re-embed with --get_aa_embeddings.")
        aa_embs = emb_dict['aa_embeddings']
        pre_projected = emb_dict.get('aa_pca_projected', False) or args.pre_projected
        logger.info(f"  {len(protein_ids):,} proteins, shape={aa_embs.shape}"
                    + (" (pre-projected, skipping scaler)" if pre_projected else ""))

        protein_lengths = parse_fasta_lengths(args.fasta) if args.fasta else {}

        all_scores = []
        for name, cav_vec, scaler in cavs:
            cav_artifacts = {
                'concept_cav': cav_vec,
                'scaler': None if pre_projected else scaler,
            }
            scores = sliding_window_scores(
                aa_embs, protein_ids, protein_lengths, window_size, cav_artifacts
            )
            logger.info(f"  [{name}] min={scores.min():.4f}, mean={scores.mean():.4f}, max={scores.max():.4f}")
            all_scores.append((name, scores))
    else:
        logger.info(f"Loading embeddings from {pkl_path}")
        embs, protein_ids = load_sequence_embeddings(pkl_path)
        logger.info(f"  {len(protein_ids):,} proteins, hidden_dim={embs.shape[1]}")

        all_scores = []
        if args.scaler_pkl:
            import joblib
            shared_scaler = joblib.load(args.scaler_pkl)
            logger.info(f"Using shared scaler from {args.scaler_pkl} — batching all CAV dot products")
            X = shared_scaler.transform(embs.astype('float32'))
            cav_matrix = np.stack([cav_vec for _, cav_vec, _ in cavs], axis=1)  # (dim, n_cavs)
            score_matrix = X @ cav_matrix                                         # (n_proteins, n_cavs)
            for i, (name, _, _) in enumerate(cavs):
                scores = score_matrix[:, i]
                logger.info(f"  [{name}] min={scores.min():.4f}, mean={scores.mean():.4f}, max={scores.max():.4f}")
                all_scores.append((name, scores))
        else:
            for name, cav_vec, scaler in cavs:
                scores = score_proteins(embs, cav_vec, scaler)
                logger.info(f"  [{name}] min={scores.min():.4f}, mean={scores.mean():.4f}, max={scores.max():.4f}")
                all_scores.append((name, scores))

    # ------------------------------------------------------------------ #
    # Length correction (optional, applied to each CAV)
    # ------------------------------------------------------------------ #
    if args.length_correct:
        missing = [a for a in ('span_length', 'fasta', 'ref_pkl')
                   if getattr(args, a) is None]
        if missing:
            raise ValueError(
                f"--length-correct requires: {', '.join('--' + a.replace('_', '-') for a in missing)}"
            )
        protein_lengths = parse_fasta_lengths(args.fasta)
        logger.info(f"Loaded lengths for {len(protein_lengths):,} proteins from {args.fasta}")
        ref_embs, _ = load_sequence_embeddings(args.ref_pkl)
        ref_mean = ref_embs.mean(axis=0)
        logger.info(f"Reference mean from {len(ref_embs):,} sequences ({args.ref_pkl})")

        corrected = []
        for name, scores, (_, cav_vec, scaler) in zip(
            [n for n, _ in all_scores],
            [s for _, s in all_scores],
            cavs,
        ):
            scores = length_correct_scores(
                scores, protein_ids, protein_lengths,
                K=args.span_length,
                ref_mean=ref_mean,
                cav_vec=cav_vec,
                scaler=scaler,
            )
            corrected.append((name, scores))
        all_scores = corrected

    # ------------------------------------------------------------------ #
    # Rank and select top-k (sort by the chosen CAV)
    # ------------------------------------------------------------------ #
    sort_scores = all_scores[args.sort_cav][1]
    order = np.argsort(sort_scores)[::-1]
    if args.k > 0:
        order = order[:args.k]

    df = pd.DataFrame({'protein_id': [protein_ids[i] for i in order]})
    for name, scores in all_scores:
        df[name] = scores[order].astype(float)

    # ------------------------------------------------------------------ #
    # Save wide + long
    # ------------------------------------------------------------------ #
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep='\t', index=False, float_format='%.4f')
    logger.info(f"Saved {len(df):,} results (wide) to {out_path}")

    cav_cols = [name for name, _ in all_scores]
    long_df = df.melt(id_vars='protein_id', value_vars=cav_cols,
                      var_name='cav', value_name='score')
    long_path = out_path.with_suffix('.long.tsv')
    long_df.to_csv(long_path, sep='\t', index=False, float_format='%.4f')
    logger.info(f"Saved {len(long_df):,} results (long) to {long_path}")

    print(f"\n--- Top {min(10, len(df))} proteins ---")
    print(df.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
