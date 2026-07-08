#!/usr/bin/env python3
"""
Score each token position (residue / gene) in a set of samples against a trained CAV.

Loads aa_embeddings from a .pkl file, applies a trained CAV's scaler/PCA, and
projects each token onto the CAV direction. Outputs a TSV with one row per
(sample, position) pair.

Output columns:
    accession   — sample identifier
    position    — 0-based token index
    name        — feature name (only present if --names is provided)
    score       — CAV projection score (higher = more concept-like)

Examples
--------
# Score all residues in positional order:
python scripts/run_attribution.py \\
    --pkl query_embeddings.pkl \\
    --cav-dir ./cavs/my_concept/ \\
    --out attribution_scores.tsv

# Single-cell: rank genes per cell by CAV alignment (check against marker genes):
python scripts/run_attribution.py \\
    --pkl cells.pkl \\
    --cav-dir ./cavs/Tcell/ \\
    --names gene_vocab.txt \\
    --sort-descending \\
    --out ranked_genes.tsv

# Protein: rank residues per protein (e.g. drug binding site discovery):
python scripts/run_attribution.py \\
    --pkl proteins.pkl \\
    --cav-dir ./cavs/binding_site/ \\
    --sort-descending \\
    --out ranked_residues.tsv

# Restrict to a subset of samples:
python scripts/run_attribution.py \\
    --pkl query_embeddings.pkl \\
    --cav-dir ./cavs/my_concept/ \\
    --subset my_accessions.txt \\
    --out attribution_scores.tsv
"""

import sys
import csv
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from src.utils.data_loader import load_embeddings_pkl
from src.evaluate import load_cav_artifacts
from src.attribution import compute_batch_attributions

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Per-position CAV attribution scoring"
    )
    parser.add_argument('--pkl', required=True,
                        help='Path to .pkl embedding file')
    parser.add_argument('--info', default=None,
                        help='Path to ID file (auto-discovered if omitted)')
    parser.add_argument('--cav-dir', required=True,
                        help='Directory containing trained CAV artifacts')
    parser.add_argument('--out', required=True,
                        help='Output TSV path')
    parser.add_argument('--version', default='v1',
                        help='CAV artifact version (default: v1)')
    parser.add_argument('--subset',
                        help='Optional file with one accession per line to restrict output')
    parser.add_argument('--sort-descending', action='store_true',
                        help='Sort output by score descending within each sample '
                             '(default: positional order). Use for gene ranking '
                             'in single-cell or residue ranking in proteins.')
    parser.add_argument('--names',
                        help='Optional file with one feature name per line '
                             '(position index = line index). Adds a name column '
                             'to output. E.g. gene vocabulary file for single-cell.')
    args = parser.parse_args()

    embeddings, sample_ids = load_embeddings_pkl(args.pkl, args.info)

    if 'aa_embeddings' not in embeddings:
        raise ValueError(
            "pkl file does not contain 'aa_embeddings'. "
            "Attribution requires token-level embeddings."
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

    names = None
    if args.names:
        with open(args.names) as f:
            names = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(names)} feature names from {args.names}")

    cav_artifacts = load_cav_artifacts(args.cav_dir, version=args.version)

    results = compute_batch_attributions(aa_emb, filtered_ids, cav_artifacts)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')

        header = ['accession', 'position']
        if names is not None:
            header.append('name')
        header.append('score')
        writer.writerow(header)

        for sid, scores in results:
            rows = list(enumerate(scores))

            if args.sort_descending:
                rows = sorted(rows, key=lambda x: x[1], reverse=True)

            for pos, score in rows:
                row = [sid, pos]
                if names is not None:
                    row.append(names[pos] if pos < len(names) else '')
                row.append(f'{score:.6f}')
                writer.writerow(row)

    logger.info(
        f"Wrote {sum(len(s) for _, s in results)} rows "
        f"({len(results)} samples) to {out_path}"
    )


if __name__ == '__main__':
    main()
