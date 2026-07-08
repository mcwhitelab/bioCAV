"""
Data loading utilities for TCAV.

Embedding files
---------------
.pkl  — dict with keys:
    "sequence_embeddings": np.ndarray, shape (n_samples, hidden_dim)
    "aa_embeddings":       np.ndarray, shape (n_samples, max_seq_len, hidden_dim)
                           (padded; use seq_len from spans or the array's dim 1)

.pkl.info — plain text, one sample ID per line (no header).
    Row index in the arrays corresponds to line index in this file.

Spans files
-----------
Tab-separated, one entry per line, no header.
Each line is one of:
    accession                       → whole-sequence mode (no span)
    accession  TAB  pos             → single-position mode (PTM etc.)
    accession  TAB  start  TAB  end → window mode (1-indexed, inclusive: [start, end])
    accession  TAB  p1,p2,p3,...    → position-list mode (mean of listed positions)

All positions are 1-indexed (matching UniProt annotation conventions).

The same accession may appear on multiple lines — each line produces a
separate embedding vector. This allows multiple negative windows from one
protein, or multiple PTM sites from one sequence.

Lines beginning with '#' are treated as comments and ignored.
"""

import pickle
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding pkl loading
# ---------------------------------------------------------------------------

def _find_info_file(pkl_path: Path) -> Path:
    """
    Auto-discover the companion ID file for a pkl.
    Tries extensions in order: .seqnames, .info, .pkl.info
    """
    candidates = [
        pkl_path.with_suffix('.seqnames'),          # foo.pkl.seqnames
        Path(str(pkl_path) + '.seqnames'),           # foo.pkl → foo.pkl.seqnames
        pkl_path.with_suffix('.info'),               # foo.pkl.info (if .pkl is last suffix)
        Path(str(pkl_path) + '.info'),               # foo.pkl → foo.pkl.info
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find a companion ID file for {pkl_path}. "
        f"Tried: {[str(c) for c in candidates]}"
    )


def _read_ids(info_path: Path) -> List[str]:
    ids = []
    with open(info_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(line)
    return ids


def load_embeddings_pkl(
    pkl_path: str,
    info_path: str = None
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Load embedding pkl and companion ID file.

    Args:
        pkl_path:  Path to .pkl file containing embedding arrays.
        info_path: Path to companion ID file (one sample ID per line).
                   If omitted, auto-discovered by trying .seqnames then .info
                   extensions next to the pkl.

    Returns:
        Tuple of (embeddings_dict, sample_ids)
        embeddings_dict has keys "sequence_embeddings" and/or "aa_embeddings".
    """
    pkl_path = Path(pkl_path)

    if info_path is None:
        info_path = _find_info_file(pkl_path)
    else:
        info_path = Path(info_path)

    with open(pkl_path, 'rb') as f:
        embeddings = pickle.load(f)

    sample_ids = _read_ids(info_path)

    # Validate row counts
    for key, arr in embeddings.items():
        if len(arr) != len(sample_ids):
            raise ValueError(
                f"{key}: array has {len(arr)} rows but ID file has "
                f"{len(sample_ids)} IDs ({info_path})"
            )

    logger.info(
        f"Loaded {len(sample_ids)} samples from {pkl_path} "
        f"(keys: {list(embeddings.keys())})"
    )
    return embeddings, sample_ids


def load_sequence_embeddings(
    pkl_path: str,
    info_path: str = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Convenience loader for pkls that contain only sequence_embeddings.
    Returns the embedding matrix and sample IDs directly.

    Args:
        pkl_path:  Path to .pkl file.
        info_path: Path to companion ID file (auto-discovered if omitted).

    Returns:
        Tuple of (embeddings array of shape (n, hidden_dim), sample_ids list)
    """
    embeddings, sample_ids = load_embeddings_pkl(pkl_path, info_path)

    if 'sequence_embeddings' not in embeddings:
        raise ValueError(
            f"{pkl_path} does not contain 'sequence_embeddings'. "
            f"Available keys: {list(embeddings.keys())}"
        )

    return embeddings['sequence_embeddings'], sample_ids


# ---------------------------------------------------------------------------
# Spans file loading
# ---------------------------------------------------------------------------

SpanEntry = Optional[Tuple[int, ...]]  # None | (pos,) | (start, end) | (p1, p2, p3, ...)

# A spans list is a sequence of (accession, span) pairs.
# Multiple entries with the same accession are allowed and each produces
# a separate embedding vector (e.g. multiple negative windows per protein).
SpansList = List[Tuple[str, SpanEntry]]


def load_spans(spans_path: str) -> SpansList:
    """
    Parse a spans file into an ordered list of (accession, span) pairs.

    Multiple lines with the same accession are each kept as separate entries,
    allowing multiple windows or positions per protein.

    Each span is one of:
        None              — whole-sequence mode (no span columns)
        (pos,)            — single-position mode (one span column, 1-indexed)
        (start, end)      — window mode, inclusive (two span columns, 1-indexed)
        (p1, p2, p3, ...) — position-list mode (second column is comma-separated, 1-indexed)
    """
    spans: SpansList = []

    with open(spans_path, 'r') as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split('\t')
            accession = parts[0]

            if len(parts) == 1:
                spans.append((accession, None))
            elif len(parts) == 2:
                col = parts[1]
                if ',' in col:
                    positions = tuple(int(p) for p in col.split(','))
                    spans.append((accession, positions))
                else:
                    spans.append((accession, (int(col),)))
            elif len(parts) == 3:
                spans.append((accession, (int(parts[1]), int(parts[2]))))
            else:
                raise ValueError(
                    f"{spans_path}:{lineno}: expected 1–3 tab-separated columns, "
                    f"got {len(parts)}"
                )

    logger.info(f"Loaded {len(spans)} span entries from {spans_path}")
    return spans


# ---------------------------------------------------------------------------
# Span pooling
# ---------------------------------------------------------------------------

def pool_span(
    aa_embeddings_row: np.ndarray,
    span: SpanEntry,
    sequence_embedding_row: Optional[np.ndarray] = None,
    seq_len: Optional[int] = None
) -> np.ndarray:
    """
    Extract a fixed-size embedding vector from a single sample's token embeddings.

    Args:
        aa_embeddings_row:      Shape (max_seq_len, hidden_dim). May be padded.
        span:                   None | (pos,) | (start, end) | (p1, p2, ...) from load_spans().
                                All positions are 1-indexed (matching UniProt conventions).
        sequence_embedding_row: Shape (hidden_dim,). Used for whole-sequence mode
                                if present; falls back to mean-pooling aa_embeddings.
        seq_len:                Actual (unpadded) sequence length. If None, uses
                                aa_embeddings_row.shape[0].

    Returns:
        1-D embedding vector of shape (hidden_dim,).
    """
    if seq_len is None:
        seq_len = aa_embeddings_row.shape[0]

    if span is None:
        # Whole-sequence mode
        if sequence_embedding_row is not None:
            return sequence_embedding_row
        return aa_embeddings_row[:seq_len].mean(axis=0)

    if len(span) == 1:
        # Single-position mode (1-indexed)
        pos = span[0]
        if pos < 1 or pos > seq_len:
            raise ValueError(
                f"Position {pos} out of range for sequence of length {seq_len} (1-indexed)"
            )
        return aa_embeddings_row[pos - 1]

    if len(span) == 2:
        # Window mode (1-indexed, inclusive [start, end])
        start, end = span
        if start < 1 or end > seq_len or start > end:
            raise ValueError(
                f"Span [{start}, {end}] invalid for sequence of length {seq_len} (1-indexed, inclusive)"
            )
        return aa_embeddings_row[start - 1:end].mean(axis=0)

    # Position-list mode: mean of specified positions (1-indexed)
    for pos in span:
        if pos < 1 or pos > seq_len:
            raise ValueError(
                f"Position {pos} out of range for sequence of length {seq_len} (1-indexed)"
            )
    return aa_embeddings_row[[p - 1 for p in span]].mean(axis=0)


# ---------------------------------------------------------------------------
# Build pooled embedding matrix for CAV training
# ---------------------------------------------------------------------------

def build_embedding_matrix(
    embeddings: Dict[str, np.ndarray],
    sample_ids: List[str],
    spans: Optional[SpansList] = None
) -> np.ndarray:
    """
    Build a (n_entries, hidden_dim) matrix by pooling each span entry.

    Each entry in spans produces one row in the output matrix. The same
    accession may appear multiple times (e.g. multiple negative windows from
    one protein), yielding one row per entry.

    If spans is None, each sample in sample_ids is used once in
    whole-sequence mode.

    Args:
        embeddings:  Dict from load_embeddings_pkl().
        sample_ids:  Ordered list of IDs (row index = position in arrays).
        spans:       List of (accession, span) pairs from load_spans(),
                     or None for whole-sequence mode on all samples.

    Returns:
        2-D array of shape (n_entries, hidden_dim).
    """
    aa_emb = embeddings.get('aa_embeddings')
    seq_emb = embeddings.get('sequence_embeddings')

    # Exact index
    id_to_idx = {sid: i for i, sid in enumerate(sample_ids)}
    # Secondary index: each '|'-delimited part → row index (skips db prefixes like sp/tr)
    _SKIP = {'sp', 'tr', 'sw', 'ref'}
    part_to_idx = {}
    for i, sid in enumerate(sample_ids):
        for part in sid.split('|'):
            if part and part not in _SKIP:
                part_to_idx.setdefault(part, i)

    def _lookup(acc):
        if acc in id_to_idx:
            return id_to_idx[acc]
        # Try parts of acc itself (e.g. "sp|A2Y9C2|LAC20" → try "A2Y9C2")
        for part in acc.split('|'):
            if part in id_to_idx:
                return id_to_idx[part]
            if part in part_to_idx:
                return part_to_idx[part]
        # Try acc as a part token of any sample_id
        return part_to_idx.get(acc)

    vectors = []
    skipped = 0

    entries = spans if spans is not None else [(sid, None) for sid in sample_ids]

    for accession, span in entries:
        idx = _lookup(accession)
        if idx is None:
            logger.warning(f"'{accession}' in spans not found in pkl — skipping")
            skipped += 1
            continue
        row_aa = aa_emb[idx] if aa_emb is not None else None
        row_seq = seq_emb[idx] if seq_emb is not None else None

        if row_aa is None and span is not None:
            raise ValueError(
                f"Span specified for '{accession}' but pkl has no aa_embeddings"
            )
        if row_aa is None:
            if row_seq is None:
                raise ValueError(
                    "pkl has neither aa_embeddings nor sequence_embeddings"
                )
            vectors.append(row_seq)
        else:
            vectors.append(pool_span(row_aa, span, sequence_embedding_row=row_seq))

    if skipped:
        logger.warning(f"Skipped {skipped} accessions not found in pkl")

    if not vectors:
        raise ValueError("No valid samples found after matching spans to pkl")

    result = np.vstack(vectors)
    logger.info(f"Built embedding matrix: {result.shape}")
    return result
