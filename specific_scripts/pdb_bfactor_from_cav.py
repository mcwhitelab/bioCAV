#!/usr/bin/env python3
"""
Paint a PDB structure's B-factor column with per-residue CAV alignment scores.

Extracts the sequence directly from the PDB chain, embeds it fresh (so the
embedding always matches exactly the structure being painted, not a
separately-maintained FASTA/pkl for a possibly different sequence), computes
a CAV projection score at every residue — optionally smoothed with the same
sliding-window scan query_proteins_by_cav.py uses for --sliding-window — and
writes a new PDB file with those scores in the B-factor column, ready to
color-by-attribute in ChimeraX/PyMOL.

The extracted sequence and its embedding pkl are cached next to the PDB
(<pdb>.fasta / <pdb>.fasta.pkl) and reused on subsequent runs.

Usage
-----
# Windowed scan, inferring window size from the CAV's training spans file
# (matches query_proteins_by_cav.py --sliding-window):
python specific_scripts/pdb_bfactor_from_cav.py \\
    --pdb          RLK5_ARATH_AF.pdb \\
    --cav-dir      fastas/pairs/lrr/RLK5_ARATH.fasta_concept \\
    --spans        fastas/pairs/lrr/RLK5_ARATH.fasta.span \\
    --embed-script /path/to/hf_embed_new.py \\
    --model        /path/to/ESMplusplus_large \\
    --out          RLK5_ARATH_cav_bfactor.pdb

# Per-residue, no windowing (--window-size 1):
python specific_scripts/pdb_bfactor_from_cav.py \\
    --pdb RLK5_ARATH_AF.pdb --cav-dir ... \\
    --embed-script ... --model ... \\
    --window-size 1 --out RLK5_ARATH_cav_bfactor.pdb

Then in ChimeraX:
    open RLK5_ARATH_cav_bfactor.pdb
    color byattribute bfactor
"""

import sys
import subprocess
import argparse
import logging
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from src.evaluate import load_cav_artifacts
from src.detect import sliding_window_scan, per_position_scan
from src.utils.data_loader import load_embeddings_pkl

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def span_window_size(spans_path: str) -> int:
    """Return the max residue width across all entries in a spans file."""
    from src.utils.data_loader import load_spans
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


def window_scores_to_per_residue(scores, spans, seq_len):
    """
    Average overlapping window scores onto each residue position.
    Residues not covered by any window (edge effect when stride/window_size
    don't evenly divide seq_len) are filled with the nearest covered score.
    """
    sums = np.zeros(seq_len, dtype='float64')
    counts = np.zeros(seq_len, dtype='int32')
    for (start, end), score in zip(spans, scores):
        sums[start:end] += score
        counts[start:end] += 1

    covered = counts > 0
    per_residue = np.full(seq_len, np.nan, dtype='float64')
    per_residue[covered] = sums[covered] / counts[covered]

    if not covered.all():
        idx = np.arange(seq_len)
        covered_idx = idx[covered]
        for i in idx[~covered]:
            nearest = covered_idx[np.argmin(np.abs(covered_idx - i))]
            per_residue[i] = per_residue[nearest]

    return per_residue


def load_pdb_chain(pdb_path: str, chain_id: str = None):
    """Parse a PDB file and return (structure, chain, residues)."""
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import is_aa

    pdb_parser = PDBParser(QUIET=True)
    structure = pdb_parser.get_structure('structure', pdb_path)
    model = next(structure.get_models())

    if chain_id:
        if chain_id not in model:
            raise ValueError(f"Chain '{chain_id}' not found in {pdb_path}. "
                              f"Available: {[c.id for c in model]}")
        chain = model[chain_id]
    else:
        chain = next(model.get_chains())
        logger.info(f"No --chain given, using first chain: '{chain.id}'")

    residues = [r for r in chain if is_aa(r, standard=False)]
    logger.info(f"PDB chain '{chain.id}': {len(residues)} residues")

    return structure, chain, residues


def pdb_chain_sequence(residues) -> str:
    """One-letter sequence for a list of Bio.PDB residues, in chain order."""
    from Bio.PDB.Polypeptide import protein_letters_3to1_extended
    seq = []
    for r in residues:
        seq.append(protein_letters_3to1_extended.get(r.get_resname(), 'X'))
    return ''.join(seq)


def embed_pdb_sequence(pdb_path: str, sequence: str, seq_id: str,
                       embed_script: str, model: str,
                       layer: int, batch: int, max_length: int) -> str:
    """
    Write the PDB's sequence to a FASTA next to it, embed it with
    hf_embed_new.py, and return the resulting pkl path. Both the FASTA and
    the pkl are cached and reused on subsequent runs.
    """
    fasta_path = Path(pdb_path).with_suffix('.fasta')
    pkl_path = fasta_path.with_suffix('.fasta.pkl')

    if not fasta_path.exists():
        with open(fasta_path, 'w') as f:
            f.write(f">{seq_id}\n{sequence}\n")
        logger.info(f"Wrote PDB-derived sequence: {fasta_path}")
    else:
        logger.info(f"Reusing existing sequence FASTA: {fasta_path}")

    if pkl_path.exists():
        logger.info(f"Skipping embedding: {pkl_path} already exists")
        return str(pkl_path)

    logger.info(f"Embedding {fasta_path} -> {pkl_path}")
    cmd = [
        sys.executable, embed_script,
        "-f", str(fasta_path),
        "-o", str(pkl_path),
        "--get_aa_embeddings",
        "--get_sequence_embedding",
        "--strat", "mean",
        "-l", str(layer),
        "-m", model,
        "-b", str(batch),
        "--max_length", str(max_length),
    ]
    subprocess.run(cmd, check=True)
    return str(pkl_path)


def main():
    parser = argparse.ArgumentParser(
        description="Paint a PDB's B-factor column with per-residue CAV scores.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--pdb', required=True, help='Input PDB file (e.g. AlphaFold model).')
    parser.add_argument('--cav-dir', required=True, help='Trained CAV artifact directory.')
    parser.add_argument('--version', default='v1', help='CAV artifact version (default: v1).')
    parser.add_argument('--window-size', type=int, default=None,
                        help='Sliding-window size in residues. Mutually exclusive with --spans. '
                             'Use 1 for unsmoothed per-residue scoring.')
    parser.add_argument('--spans', default=None,
                        help='Spans file used for CAV training; infers window size from it '
                             '(same convention as query_proteins_by_cav.py --sliding-window). '
                             'Mutually exclusive with --window-size.')
    parser.add_argument('--stride', type=int, default=1, help='Window stride (default: 1).')
    parser.add_argument('--chain', default=None,
                        help='PDB chain ID to paint (default: first chain in the structure).')

    # On-the-fly embedding of the PDB's own sequence
    parser.add_argument('--embed-script', required=True,
                        help='Path to hf_embed_new.py.')
    parser.add_argument('--model', required=True,
                        help='Model directory for embedding.')
    parser.add_argument('--embed-layer', type=int, default=-11,
                        help='Layer index for embedding (default: -11).')
    parser.add_argument('--embed-batch', type=int, default=1,
                        help='Batch size for embedding (default: 1).')
    parser.add_argument('--embed-max-length', type=int, default=2048,
                        help='Max sequence length for embedding (default: 2048).')

    parser.add_argument('--out', required=True, help='Output PDB path.')
    args = parser.parse_args()

    if args.window_size is not None and args.spans is not None:
        parser.error("Pass only one of --window-size / --spans.")

    # ------------------------------------------------------------------ #
    # 1. Parse the PDB and pull its sequence directly (source of truth)
    # ------------------------------------------------------------------ #
    structure, chain, residues = load_pdb_chain(args.pdb, args.chain)
    seq_len = len(residues)
    sequence = pdb_chain_sequence(residues)
    seq_id = f"{Path(args.pdb).stem}_{chain.id}"

    # ------------------------------------------------------------------ #
    # 2. Embed that sequence fresh
    # ------------------------------------------------------------------ #
    pkl_path = embed_pdb_sequence(
        args.pdb, sequence, seq_id,
        args.embed_script, args.model,
        args.embed_layer, args.embed_batch, args.embed_max_length,
    )

    embeddings, sample_ids = load_embeddings_pkl(pkl_path)
    if 'aa_embeddings' not in embeddings:
        raise ValueError(f"{pkl_path} does not contain aa_embeddings.")
    idx = sample_ids.index(seq_id)
    aa_emb = embeddings['aa_embeddings'][idx]

    # ------------------------------------------------------------------ #
    # 3. Score every residue against the CAV
    # ------------------------------------------------------------------ #
    artifacts = load_cav_artifacts(args.cav_dir, version=args.version)

    window_size = args.window_size
    if args.spans:
        window_size = span_window_size(args.spans)
        logger.info(f"Inferred window size {window_size} from {args.spans}")

    if window_size and window_size > 1:
        scores, spans = sliding_window_scan(
            aa_emb, artifacts, window_size=window_size, stride=args.stride, seq_len=seq_len
        )
        per_residue = window_scores_to_per_residue(scores, spans, seq_len)
        logger.info(f"Sliding-window scan: {len(scores)} windows (size={window_size}, stride={args.stride})")
    else:
        per_residue = per_position_scan(aa_emb, artifacts, seq_len=seq_len)
        logger.info("Per-residue scan (no windowing)")

    logger.info(f"Score range: min={per_residue.min():.4f}, mean={per_residue.mean():.4f}, "
                f"max={per_residue.max():.4f}")

    # ------------------------------------------------------------------ #
    # 4. Paint the PDB B-factor column
    # ------------------------------------------------------------------ #
    for residue, score in zip(residues, per_residue):
        for atom in residue:
            atom.set_bfactor(float(score))

    from Bio.PDB import PDBIO
    io = PDBIO()
    io.set_structure(structure)
    io.save(args.out)
    logger.info(f"Saved: {args.out}")

    print(f"\nIn ChimeraX:")
    print(f"  open {args.out}")
    print(f"  color byattribute bfactor")


if __name__ == '__main__':
    main()
