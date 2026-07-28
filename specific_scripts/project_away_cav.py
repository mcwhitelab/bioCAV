#!/usr/bin/env python3
"""
Remove one or more CAV directions from a target CAV (Gram-Schmidt orthogonalization).

Answers questions like "What is MYC-like after removing generic bHLH-ness?"
by subtracting the HLH component from the MYC CAV direction and renormalizing.
Because Gram-Schmidt only removes the *correlated* component of the target
with each away direction (target - (target . away) * away), this doubles as
a "remove the common signal between two CAVs" tool: e.g. two interaction-
interface CAVs each trained from a single example are often not specific
enough on their own (they pick up shared/generic interface signal along with
whatever is unique to that pair). Passing one as --target and the other as
--away leaves only the target-specific residual.

The output is a new CAV artifact directory that can be passed directly to
query_proteins_by_cav.py. The scaler and PCA are copied from the target CAV,
so this only makes sense when all CAVs share the same preprocessing (e.g.
trained with a global scaler/PCA via run_cav_pipeline.py --pca-pkl, or with
the same --scaler-pkl via train_cav_from_span.sh). The scaler is checked for
a match between target and away CAVs before projecting.

Usage
-----
# Project HLH out of MYC:
python specific_scripts/project_away_cav.py \\
    --target cavs/MYC/ \\
    --away   cavs/HLH/ \\
    --out    cavs/MYC_minus_HLH/

# Project away multiple directions at once:
python specific_scripts/project_away_cav.py \\
    --target cavs/MYC/ \\
    --away   cavs/HLH/ cavs/MAX/ \\
    --out    cavs/MYC_minus_HLH_MAX/

# Then query proteins with the residual CAV:
python specific_scripts/query_proteins_by_cav.py \\
    --cav  cavs/MYC_minus_HLH/ \\
    --pkl  embeddings/proteins.pkl \\
    --out  results/MYC_specific_hits.tsv
"""

import sys
import json
import shutil
import argparse
import logging
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
from src.evaluate import load_cav_artifacts

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def scalers_match(scaler_a, scaler_b) -> bool:
    """
    True if two fitted StandardScalers describe the same preprocessing space.

    CAV vectors are only comparable (and Gram-Schmidt only meaningful) if
    target and away were embedded/scaled identically — e.g. both trained
    with the same --scaler-pkl.
    """
    return (
        np.allclose(scaler_a.mean_, scaler_b.mean_)
        and np.allclose(scaler_a.scale_, scaler_b.scale_)
    )


def orthogonalize(target: np.ndarray, directions: list[np.ndarray]) -> np.ndarray:
    """
    Remove each direction from target via Gram-Schmidt and renormalize.

    Each direction is assumed to be unit-norm (CAV vectors are normalized at
    training time). The directions are applied sequentially, so the order
    matters only if they are not mutually orthogonal — in practice the
    difference is negligible.
    """
    v = target.copy().astype(float)
    for d in directions:
        d = d.astype(float)
        d = d / (np.linalg.norm(d) + 1e-10)
        v = v - np.dot(v, d) * d

    norm = np.linalg.norm(v)
    if norm < 1e-8:
        raise ValueError(
            "Residual CAV vector is near-zero after projection — the target "
            "direction lies almost entirely within the projected-away subspace."
        )
    return v / norm


def main():
    parser = argparse.ArgumentParser(
        description="Project one or more CAV directions out of a target CAV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--target', required=True,
                        help='CAV artifact directory to project away from (e.g. MYC).')
    parser.add_argument('--away', nargs='+', required=True,
                        help='CAV artifact directory(ies) whose directions to remove (e.g. HLH).')
    parser.add_argument('--out', required=True,
                        help='Output directory for the residual CAV artifacts.')
    parser.add_argument('--version', default='v1',
                        help='Artifact version suffix to read and write (default: v1).')
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Load target
    # ------------------------------------------------------------------ #
    logger.info(f"Loading target CAV: {args.target}")
    target_artifacts = load_cav_artifacts(args.target, version=args.version)
    target_vec = target_artifacts['concept_cav']
    logger.info(f"  Target dim: {target_vec.shape[0]}")

    # ------------------------------------------------------------------ #
    # Load directions to project away
    # ------------------------------------------------------------------ #
    away_vecs = []
    for away_dir in args.away:
        logger.info(f"Loading away CAV: {away_dir}")
        away_artifacts = load_cav_artifacts(away_dir, version=args.version)
        away_vec = away_artifacts['concept_cav']

        if not scalers_match(target_artifacts['scaler'], away_artifacts['scaler']):
            raise ValueError(
                f"Scaler mismatch between target ({args.target}) and away "
                f"({away_dir}) — they were not trained with the same "
                f"preprocessing, so their CAV directions don't live in the "
                f"same space and can't be validly subtracted. Retrain both "
                f"with the same --scaler-pkl."
            )

        cos_sim = float(np.dot(target_vec, away_vec))
        logger.info(f"  Cosine similarity to target: {cos_sim:.4f}  "
                    f"(the correlated/common component along this axis will be removed)")
        away_vecs.append(away_vec)

    # ------------------------------------------------------------------ #
    # Orthogonalize
    # ------------------------------------------------------------------ #
    residual_vec = orthogonalize(target_vec, away_vecs)
    removed = 1.0 - float(np.dot(residual_vec, target_vec / (np.linalg.norm(target_vec) + 1e-10)))
    logger.info(f"Residual CAV computed (fraction of direction removed: {removed:.4f})")

    # ------------------------------------------------------------------ #
    # Save artifacts — copy scaler/PCA from target, write new concept .npy
    # ------------------------------------------------------------------ #
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    concept_file = out_path / f"concept_{args.version}.npy"
    np.save(concept_file, residual_vec.astype(np.float32))
    logger.info(f"  Saved residual CAV: {concept_file}")

    for suffix in [f"scaler_{args.version}.pkl", f"pca_{args.version}.pkl"]:
        src = Path(args.target) / suffix
        if src.exists():
            shutil.copy2(src, out_path / suffix)
            logger.info(f"  Copied {suffix} from target")

    report = {
        'artifact_version': args.version,
        'timestamp': datetime.now().isoformat(),
        'operation': 'gram_schmidt_orthogonalization',
        'target_cav': str(args.target),
        'projected_away': args.away,
        'cosine_sim_target_residual': float(np.dot(
            target_vec / (np.linalg.norm(target_vec) + 1e-10), residual_vec
        )),
    }
    report_file = out_path / f"report_{args.version}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"  Saved report: {report_file}")

    logger.info(f"\nResidual CAV ready in {out_path}")
    logger.info(f"  Pass to query_proteins_by_cav.py with --cav {out_path}")


if __name__ == '__main__':
    main()
