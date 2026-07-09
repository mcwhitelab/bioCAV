#!/usr/bin/env python3
"""
Paint a PDB structure's B-factor column with per-residue CAV alignment scores.

Computes a CAV projection score at every residue of a protein — optionally
smoothed with the same sliding-window scan query_proteins_by_cav.py uses for
--sliding-window — and writes a new PDB file with those scores in the
B-factor column, ready to color-by-attribute in ChimeraX/PyMOL.

The residue count and order come directly from the PDB chain itself (no
FASTA needed): residue i of the chain is scored against position i of the
--pkl's aa_embeddings, so this assumes the PDB is a single-chain structure
of the same protein the embeddings were computed from (true for an
AlphaFold model), with no missing internal residues.

Usage
-----
# Windowed scan, inferring window size from the CAV's training spans file
# (matches query_proteins_by_cav.py --sliding-window):
python specific_scripts/pdb_bfactor_from_cav.py \\
    --pdb       RLK5_ARATH_AF.pdb \\
    --pkl       fastas/pairs/lrr/RLK5_ARATH.fasta.pkl \\
    --cav-dir   fastas/pairs/lrr/RLK5_ARATH.fasta_concept \\
    --spans     fastas/pairs/lrr/RLK5_ARATH.fasta.span \\
    --out       RLK5_ARATH_cav_bfactor.pdb

# Per-residue, no windowing (--window-size 1):
python specific_scripts/pdb_bfactor_from_cav.py \\
    --pdb RLK5_ARATH_AF.pdb --pkl ... --cav-dir ... \\
    --window-size 1 --out RLK5_ARATH_cav_bfactor.pdb

Then in ChimeraX:
    open RLK5_ARATH_cav_bfactor.pdb
    color byattribute bfactor
"""

import sys
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


def main():
    parser = argparse.ArgumentParser(
        description="Paint a PDB's B-factor column with per-residue CAV scores.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--pdb', required=True, help='Input PDB file (e.g. AlphaFold model).')
    parser.add_argument('--pkl', required=True, help='Pkl with aa_embeddings for the protein.')
    parser.add_argument('--accession', default=None,
                        help='Accession/ID to select from --pkl. Auto-detected if the pkl '
                             'has exactly one entry.')
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
    parser.add_argument('--out', required=True, help='Output PDB path.')
    args = parser.parse_args()

    if args.window_size is not None and args.spans is not None:
        parser.error("Pass only one of --window-size / --spans.")

    # ------------------------------------------------------------------ #
    # 1. Parse the PDB — its residue count/order is the source of truth
    # ------------------------------------------------------------------ #
    structure, chain, residues = load_pdb_chain(args.pdb, args.chain)
    seq_len = len(residues)

    # ------------------------------------------------------------------ #
    # 2. Resolve accession and load embeddings for that protein
    # ------------------------------------------------------------------ #
    embeddings, sample_ids = load_embeddings_pkl(args.pkl)
    if 'aa_embeddings' not in embeddings:
        raise ValueError(f"{args.pkl} does not contain aa_embeddings. "
                          "Re-embed with --get_aa_embeddings.")

    if args.accession:
        accession = args.accession
        if accession not in sample_ids:
            raise ValueError(f"{accession} not found in {args.pkl}. "
                              f"Available: {sample_ids[:5]}{'...' if len(sample_ids) > 5 else ''}")
    else:
        if len(sample_ids) != 1:
            raise ValueError(f"{args.pkl} has {len(sample_ids)} entries — pass --accession "
                              f"to pick one. Available: {sample_ids[:10]}")
        accession = sample_ids[0]
        logger.info(f"Using the only entry in {args.pkl}: {accession}")

    idx = sample_ids.index(accession)
    aa_emb = embeddings['aa_embeddings'][idx]

    if seq_len > aa_emb.shape[0]:
        raise ValueError(
            f"PDB chain '{chain.id}' has {seq_len} residues but {accession}'s aa_embeddings "
            f"only cover {aa_emb.shape[0]} positions — wrong PDB/pkl pair, or the embedding "
            f"was truncated (check --max_length on the original embed step)."
        )

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
