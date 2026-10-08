"""
CAV training module with versioned artifacts.
"""

import numpy as np
import json
import joblib
import logging
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import hashlib

from .utils.preprocessing import create_scaler

logger = logging.getLogger(__name__)

# lbfgs at sklearn's default tol=1e-4 stops after ~15-50 iterations on these nearly separable
# problems, and the CAV direction then depends on the stopping rule. Fit to convergence.
DEFAULT_TOL = 1e-10
DEFAULT_MAX_ITER = 100000


def default_C(n_train: int) -> float:
    """C = 1 / n_train.

    sklearn's objective is 0.5*||w||^2 + C * sum_i loss_i (a SUMMED loss), so the effective
    regularisation is ~1/(C*n). C = 1/n keeps C*n = 1 whatever the training-set size. With the
    shared 10k background this is ~1e-4, the value used for the GO/EC/Pfam libraries; a
    fixed C=1e-4 on a small custom set (e.g. n=32) shrinks the CAV to the class-mean difference.
    """
    return 1.0 / n_train


def load_embeddings(
    embed_dir: str,
    pos_file: str = "pos.npy",
    neg_file: str = "neg.npy"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load positive and negative embeddings.

    Args:
        embed_dir: Directory containing embeddings
        pos_file: Positive embeddings filename
        neg_file: Negative embeddings filename

    Returns:
        Tuple of (X, y)
    """
    embed_path = Path(embed_dir)

    pos_emb = np.load(embed_path / pos_file)
    neg_emb = np.load(embed_path / neg_file)

    logger.info(f"Loaded {len(pos_emb)} positive, {len(neg_emb)} negative embeddings")

    X = np.vstack([pos_emb, neg_emb])
    y = np.hstack([np.ones(len(pos_emb)), np.zeros(len(neg_emb))])

    return X, y


def train_concept_cav(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: Optional[float] = None,
    cv_folds: int = 5,
    random_seed: int = 42,
    class_weight: str = "balanced",
    tol: float = DEFAULT_TOL,
    max_iter: int = DEFAULT_MAX_ITER,
) -> Tuple[LogisticRegression, Dict]:
    """
    Train concept CAV using logistic regression.

    Args:
        X_train: Training embeddings (preprocessed)
        y_train: Labels
        C: Inverse regularization strength. None (default) = 1 / n_train
            (see default_C); the same C is used for the CV folds and the final fit.
        cv_folds: Number of CV folds
        random_seed: Random seed
        class_weight: Passed to LogisticRegression.  "balanced" (default)
            re-weights each sample by n_total / (2 * n_class) so that
            gradient contributions are equal from positives and negatives
            regardless of how many reference cells were used.  This is
            especially important with a large fixed reference population
            (e.g. 10k cells) vs. a small positive set (e.g. 200 cells).
        tol, max_iter: lbfgs stopping rule; defaults fit to convergence.

    Returns:
        Tuple of (fitted_model, metrics_dict)
    """
    C_rule = "1/n_train" if C is None else "user"
    if C is None:
        C = default_C(len(X_train))
    logger.info(f"LogisticRegression C = {C:.3g} ({C_rule}, n_train = {len(X_train)})")
    clf = LogisticRegression(
        C=C,
        tol=tol,
        max_iter=max_iter,
        random_state=random_seed,
        solver='lbfgs',
        class_weight=class_weight,
    )

    if cv_folds >= 2:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
        cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='roc_auc')
        cv_auroc_mean = float(cv_scores.mean())
        cv_auroc_std = float(cv_scores.std())
        cv_auroc_scores = cv_scores.tolist()
    else:
        # cv_folds=0: skip cross-validation (e.g. only 1 positive example)
        cv_auroc_mean = None
        cv_auroc_std = None
        cv_auroc_scores = []

    clf.fit(X_train, y_train)
    n_iter = int(clf.n_iter_[0])
    if n_iter >= max_iter:
        logger.warning(f"lbfgs hit max_iter={max_iter} without converging (tol={tol})")

    y_pred_proba = clf.predict_proba(X_train)[:, 1]
    train_auroc = roc_auc_score(y_train, y_pred_proba)
    train_auprc = average_precision_score(y_train, y_pred_proba)

    metrics = {
        'cv_auroc_mean': cv_auroc_mean,
        'cv_auroc_std': cv_auroc_std,
        'cv_auroc_scores': cv_auroc_scores,
        'train_auroc': float(train_auroc),
        'train_auprc': float(train_auprc),
        'n_train': len(X_train),
        'n_positive': int(y_train.sum()),
        'n_negative': int((1 - y_train).sum()),
        'C': float(C),
        'C_rule': C_rule,
        'tol': tol,
        'max_iter': max_iter,
        'n_iter': n_iter,
        'converged': n_iter < max_iter,
    }

    if cv_auroc_mean is not None:
        logger.info(
            f"CAV trained: CV AUROC = {cv_auroc_mean:.3f} ± {cv_auroc_std:.3f}, "
            f"Train AUROC = {train_auroc:.3f}"
        )
    else:
        logger.info(f"CAV trained (no CV): Train AUROC = {train_auroc:.3f}")

    return clf, metrics


def extract_cav_vector(model: LogisticRegression) -> np.ndarray:
    """
    Extract and normalize CAV vector from trained model.

    Args:
        model: Fitted logistic regression model

    Returns:
        Normalized CAV vector
    """
    cav = model.coef_[0]
    cav_norm = cav / (np.linalg.norm(cav) + 1e-10)
    return cav_norm


def compute_artifact_hash(data: np.ndarray) -> str:
    hasher = hashlib.sha256()
    hasher.update(data.tobytes())
    return hasher.hexdigest()[:12]


def fold_pca_cav(pca_cav: np.ndarray, pca: Any) -> np.ndarray:
    """Map a CAV trained in PCA space back to the full (scaled) embedding space.

    A score in PC space is w . (P (z - mu_pca)) = (P^T w) . z - const, so P^T w ranks
    identically to the PC-space CAV and can be scored with the ordinary scaler-only path.
    """
    folded = pca.components_.T @ pca_cav
    return folded / (np.linalg.norm(folded) + 1e-10)


def save_cav_artifacts(
    output_dir: str,
    concept_cav: np.ndarray,
    scaler: Any,
    concept_metrics: Dict,
    config: Dict,
    artifact_version: str = "v1",
    pca_cav: Optional[np.ndarray] = None,
) -> None:
    """Save CAV artifacts with versioning and manifest."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(output_path / f"concept_{artifact_version}.npy", concept_cav)
    if pca_cav is not None:
        np.save(output_path / f"concept_pca_{artifact_version}.npy", pca_cav)
    joblib.dump(scaler, output_path / f"scaler_{artifact_version}.pkl")

    report = {
        'artifact_version': artifact_version,
        'timestamp': datetime.now().isoformat(),
        'concept_metrics': concept_metrics,
        'config': config,
        'artifact_hashes': {
            'concept_cav': compute_artifact_hash(concept_cav),
            'scaler': compute_artifact_hash(scaler.mean_),
        }
    }
    if pca_cav is not None:
        report['artifact_hashes']['concept_pca'] = compute_artifact_hash(pca_cav)

    with open(output_path / f"report_{artifact_version}.json", 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"Saved CAV artifacts to {output_path}")
    create_manifest(output_path, artifact_version)


def create_manifest(output_dir: Path, version: str) -> None:
    artifact_files = {
        'concept_cav': f"concept_{version}.npy",
        'concept_pca': f"concept_pca_{version}.npy",
        'scaler': f"scaler_{version}.pkl",
        'report': f"report_{version}.json",
    }

    manifest = {
        'version': version,
        'created': datetime.now().isoformat(),
        'files': {},
    }

    for key, filename in artifact_files.items():
        filepath = output_dir / filename
        if filepath.exists():
            manifest['files'][key] = {
                'filename': filename,
                'size_bytes': filepath.stat().st_size,
                'exists': True,
            }

    with open(output_dir / f"manifest_{version}.json", 'w') as f:
        json.dump(manifest, f, indent=2)


def train_cav(
    embed_dir: str,
    output_dir: str,
    config: Dict,
    artifact_version: str = "v1",
    holdout_fraction: float = 0.0,
    scaler=None,
    pca=None,
) -> None:
    """
    Complete CAV training pipeline.

    Args:
        embed_dir: Directory containing pos.npy and neg.npy
        output_dir: Output directory for artifacts
        config: Configuration dictionary
        artifact_version: Version string
        holdout_fraction: Fraction of data to hold out for evaluation (0 = no holdout)
        scaler: Optional pre-fit StandardScaler. If provided, skips fitting a new one.
        pca: Optional sklearn PCA fit on reference embeddings already standardized by
            `scaler`. If provided, the CAV is trained on the top-k PC scores and then
            folded back to the full space (fold_pca_cav), so the saved concept vector
            is scored exactly like a no-PCA CAV.
    """
    X, y = load_embeddings(embed_dir)

    if holdout_fraction > 0.0:
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X, y,
            test_size=holdout_fraction,
            stratify=y,
            random_state=config.get('random_seed', 42)
        )
        logger.info(f"Holdout split: {len(X_train)} train, {len(X_holdout)} holdout")
    else:
        X_train, y_train = X, y
        X_holdout, y_holdout = None, None

    if scaler is None:
        scaler, preprocess_meta = create_scaler(X_train)
    else:
        preprocess_meta = {'global_scaler': True}
        logger.info("Using pre-fit global scaler")

    X_scaled = scaler.transform(X_train)
    if pca is not None:
        X_fit = pca.transform(X_scaled)
        logger.info(f"Training in PCA space: {pca.n_components_} components, "
                    f"{pca.explained_variance_ratio_.sum():.1%} of reference variance")
    else:
        X_fit = X_scaled

    concept_model, concept_metrics = train_concept_cav(
        X_fit,
        y_train,
        C=config.get('regularization_C'),
        cv_folds=config.get('cv_folds', 5),
        random_seed=config.get('random_seed', 42),
        class_weight=config.get('class_weight', 'balanced'),
    )

    pca_cav = None
    if pca is not None:
        pca_cav = extract_cav_vector(concept_model)
        concept_cav = fold_pca_cav(pca_cav, pca)
        concept_metrics.update({
            'pca_dim': int(pca.n_components_),
            'pca_var_explained': float(pca.explained_variance_ratio_.sum()),
        })
    else:
        concept_cav = extract_cav_vector(concept_model)

    if X_holdout is not None:
        X_holdout_scaled = scaler.transform(X_holdout)
        cav_vec = concept_cav
        holdout_scores = X_holdout_scaled @ cav_vec
        holdout_auroc = roc_auc_score(y_holdout, holdout_scores)
        holdout_auprc = average_precision_score(y_holdout, holdout_scores)
        concept_metrics.update({
            'holdout_auroc': float(holdout_auroc),
            'holdout_auprc': float(holdout_auprc),
            'n_holdout': len(X_holdout),
            'holdout_fraction': holdout_fraction,
        })
        logger.info(f"Holdout AUROC = {holdout_auroc:.3f}, AUPRC = {holdout_auprc:.3f}")

    concept_metrics.update(preprocess_meta)

    save_cav_artifacts(
        output_dir=output_dir,
        concept_cav=concept_cav,
        scaler=scaler,
        concept_metrics=concept_metrics,
        config=config,
        artifact_version=artifact_version,
        pca_cav=pca_cav,
    )
