#!/usr/bin/env python3
"""
Scan sequences with a sliding window against a trained CAV.

For each sequence in the pkl, slides a window of fixed size across the
aa_embeddings, mean-pools each window, projects onto the CAV direction,
and outputs a score per window position.

Use this for regional features (TM helices, motifs, binding sites) where
a smoothed window score is more meaningful than per-residue scoring.
The window size should roughly match the expected feature size
(e.g. ~20 for TM helices, ~30-80 for protein domains).

Output TSV columns:
    accession   — sample identifier
    start       — window start position (0-based)
    end         — window end position (half-open)
    score       — CAV projection score for this window

Examples
--------
# TM helix localization (window ~20 residues):
python scripts/scan_sequence.py \\
    --pkl temporal_positive_validation.fasta.pkl \\
    --cav-dir ./cavs/GO_0005887/ \\
    --window-size 20 \\
    --out ./cavs/GO_0005887/validation_scan.tsv

# Larger domain (e.g. 40 residues), stride of 5:
python scripts/scan_sequence.py \\
    --pkl temporal_positive_validation.fasta.pkl \\
    --cav-dir ./cavs/GO_0005887/ \\
    --window-size 40 \\
    --stride 5 \\
    --out ./cavs/GO_0005887/validation_scan.tsv
"""

import sys
import csv
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from src.evaluate import load_cav_artifacts
from src.detect import sliding_window_scan
from src.utils.data_loader import load_embeddings_pkl

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Sliding window CAV scan over aa_embeddings"
    )
    parser.add_argument('--pkl', required=True,
                        help='Path to .pkl file containing aa_embeddings')
    parser.add_argument('--info', default=None,
                        help='Path to ID file (auto-discovered if omitted)')
    parser.add_argument('--cav-dir', required=True,
                        help='Directory containing trained CAV artifacts')
    parser.add_argument('--window-size', type=int, required=True,
                        help='Window size in residues (e.g. 20 for TM helices)')
    parser.add_argument('--stride', type=int, default=1,
                        help='Stride between windows (default: 1)')
    parser.add_argument('--out', required=True,
                        help='Output TSV path')
    parser.add_argument('--version', default='v1',
                        help='CAV artifact version (default: v1)')
    parser.add_argument('--subset',
                        help='Optional file with one accession per line to restrict output')
    args = parser.parse_args()

    embeddings, sample_ids = load_embeddings_pkl(args.pkl, args.info)

    if 'aa_embeddings' not in embeddings:
        raise ValueError(
            f"pkl does not contain 'aa_embeddings'. "
            f"Available keys: {list(embeddings.keys())}. "
            f"Re-run your embedding script requesting per-residue token embeddings."
        )

    aa_emb = embeddings['aa_embeddings']

    subset = None
    if args.subset:
        with open(args.subset) as f:
            subset = {line.strip() for line in f if line.strip()}
        logger.info(f"Restricting to {len(subset)} samples from {args.subset}")

    if subset is not None:
        indices = [i for i, sid in enumerate(sample_ids) if sid in subset]
        filtered_ids = [sample_ids[i] for i in indices]
        aa_emb = aa_emb[indices]
    else:
        filtered_ids = sample_ids

    artifacts = load_cav_artifacts(args.cav_dir, version=args.version)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_windows = 0
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['accession', 'start', 'end', 'score'])

        for sid, row in zip(filtered_ids, aa_emb):
            scores, spans = sliding_window_scan(
                row,
                artifacts,
                window_size=args.window_size,
                stride=args.stride
            )
            for (start, end), score in zip(spans, scores):
                writer.writerow([sid, start, end, f'{score:.6f}'])
            total_windows += len(scores)

    logger.info(
        f"Wrote {total_windows} windows across {len(filtered_ids)} sequences to {out_path}"
    )


if __name__ == '__main__':
    main()
