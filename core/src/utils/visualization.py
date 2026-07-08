"""Visualization utilities for TCAV analysis"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from typing import Dict, List, Optional, Tuple
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def plot_roc_pr_curves(
    metrics: Dict,
    output_path: str,
    title: str = "ROC and PR Curves"
) -> None:
    """
    Plot ROC and Precision-Recall curves.

    Args:
        metrics: Metrics dict from evaluate_projection_performance
        output_path: Path to save figure
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    if 'fpr' in metrics and 'tpr' in metrics:
        ax1.plot(
            metrics['fpr'],
            metrics['tpr'],
            label=f"AUC={metrics['auroc']:.3f}",
            linewidth=2
        )

    if 'precision' in metrics and 'recall' in metrics:
        ax2.plot(
            metrics['recall'],
            metrics['precision'],
            label=f"AUC={metrics['auprc']:.3f}",
            linewidth=2
        )

    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_random_cav_comparison(
    concept_auroc: float,
    random_aurocs: List[float],
    output_path: str,
    label: str = "Concept CAV",
    alpha: float = 0.05
) -> None:
    """
    Plot concept CAV AUROC vs random CAV distribution.

    Args:
        concept_auroc: AUROC of concept CAV
        random_aurocs: List of AUROCs from random CAVs
        output_path: Path to save figure
        label: Label for the concept CAV
        alpha: Significance level
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        random_aurocs,
        bins=15,
        alpha=0.6,
        color='gray',
        edgecolor='black',
        label=f'Random CAVs (n={len(random_aurocs)})'
    )

    ax.axvline(
        concept_auroc,
        color='red',
        linewidth=3,
        linestyle='--',
        label=f'{label} (AUROC={concept_auroc:.3f})'
    )

    random_mean = np.mean(random_aurocs)
    random_std = np.std(random_aurocs)
    z_score = (concept_auroc - random_mean) / (random_std + 1e-10)
    threshold_95 = np.percentile(random_aurocs, 95)

    ax.axvline(
        threshold_95,
        color='orange',
        linewidth=2,
        linestyle=':',
        label=f'95th percentile ({threshold_95:.3f})'
    )

    stats_text = (
        f"Random CAVs: μ={random_mean:.3f}, σ={random_std:.3f}\n"
        f"Z-score: {z_score:.2f}\n"
        f"Significant: {'YES' if concept_auroc > threshold_95 else 'NO'}"
    )
    ax.text(
        0.05, 0.95, stats_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    ax.set_xlabel('AUROC', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Concept CAV vs Random CAVs', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_localization_heatmap(
    projection_scores: np.ndarray,
    annotated_span: Optional[Tuple[int, int]],
    accession: str,
    output_path: str,
    sequence: Optional[str] = None
) -> None:
    """
    Plot per-position projection scores for a single sample.

    Args:
        projection_scores: Projection score per position
        annotated_span: (start, end) of annotated region (0-based, half-open), or None
        accession: Sample identifier
        output_path: Path to save figure
        sequence: Optional sequence string for residue labels
    """
    fig, ax = plt.subplots(figsize=(16, 4))

    positions = np.arange(len(projection_scores))
    ax.plot(positions, projection_scores, linewidth=2, color='steelblue')
    ax.fill_between(positions, projection_scores, alpha=0.3, color='steelblue')

    if annotated_span is not None:
        start, end = annotated_span
        ax.axvspan(start, end, alpha=0.2, color='red', label='Annotated region')

    top_idx = np.argmax(projection_scores)
    ax.scatter(
        [top_idx], [projection_scores[top_idx]],
        color='red', s=100, zorder=5, marker='*',
        label=f'Top prediction (pos {top_idx})'
    )

    ax.set_ylabel('CAV Projection', fontsize=11)
    ax.set_xlabel('Position', fontsize=11)
    ax.set_xlim(0, len(projection_scores))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{accession} — CAV Localization', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_multi_cav_heatmap(
    cav_scores: Dict[str, np.ndarray],
    accession: str,
    output_path: str,
    top_n: int = 10,
    sequence: Optional[str] = None,
) -> None:
    """
    Plot per-position projection scores for the top N CAVs as a heatmap.

    Rows are CAVs ranked by their peak score across the sequence (highest first).
    Useful for visualizing which domains, motifs, or features a sequence activates.

    Args:
        cav_scores: Dict mapping CAV name -> 1D array of per-position projection scores
        accession: Sequence identifier (used in title)
        output_path: Path to save figure
        top_n: Number of top CAVs to show
        sequence: Optional sequence string for x-axis residue labels
    """
    # Rank CAVs by their max score across positions
    ranked = sorted(cav_scores.items(), key=lambda kv: float(np.max(kv[1])), reverse=True)
    top = ranked[:top_n]

    cav_names = [name for name, _ in top]
    matrix = np.stack([scores for _, scores in top], axis=0)  # (top_n, seq_len)

    seq_len = matrix.shape[1]
    fig_width = max(14, seq_len // 20)
    fig_height = max(4, top_n * 0.5 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sns.heatmap(
        matrix,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        yticklabels=cav_names,
        xticklabels=False,
        cbar_kws={"label": "CAV Projection", "shrink": 0.6},
        linewidths=0,
    )

    if sequence is not None and seq_len <= 500:
        ax.set_xticks(np.arange(seq_len) + 0.5)
        ax.set_xticklabels(list(sequence), fontsize=6, rotation=0)

    ax.set_xlabel("Position", fontsize=11)
    ax.set_ylabel("CAV (ranked by peak score)", fontsize=11)
    ax.set_title(f"{accession} — Top {top_n} CAV Activations", fontsize=13, fontweight="bold")
    ax.tick_params(axis="y", labelsize=9)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_threshold_selection(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    selected_threshold: float,
    output_path: str,
    method: str = "f1_max",
    title: str = "Threshold Selection"
) -> None:
    """
    Plot threshold selection visualization.

    Args:
        y_true: True labels
        y_scores: Prediction scores
        selected_threshold: Chosen threshold
        output_path: Path to save figure
        method: Threshold selection method
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    ax1.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC={roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1)

    idx = np.argmin(np.abs(thresholds_roc - selected_threshold))
    ax1.scatter(
        [fpr[idx]], [tpr[idx]],
        color='red', s=150, zorder=5, marker='*',
        label=f'Selected (t={selected_threshold:.3f})'
    )

    ax1.set_xlabel('False Positive Rate', fontsize=11)
    ax1.set_ylabel('True Positive Rate', fontsize=11)
    ax1.set_title('ROC Curve with Selected Threshold', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)

    ax2.plot(thresholds_pr, precision[:-1], label='Precision', linewidth=2)
    ax2.plot(thresholds_pr, recall[:-1], label='Recall', linewidth=2)
    ax2.plot(thresholds_pr, f1_scores, label='F1 Score', linewidth=2, linestyle='--')
    ax2.axvline(
        selected_threshold,
        color='red', linewidth=2, linestyle=':',
        label='Selected threshold'
    )

    ax2.set_xlabel('Threshold', fontsize=11)
    ax2.set_ylabel('Score', fontsize=11)
    ax2.set_title(f'Threshold Selection ({method})', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
