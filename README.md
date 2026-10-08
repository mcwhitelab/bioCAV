# Protein Motif Detection using Concept Activation Vectors (CAV)

## Overview

We present an automated approach for identifying and annotating motifs and domains in protein sequences, using pretrained Protein Language Models (PLMs) and Concept Activation Vectors (CAVs), adapted from interpretability research in computer vision. We treat motifs as conceptual entities and represent them through learned CAVs in PLM embedding space by training simple linear classifiers to distinguish motif-containing from non-motif sequences.

## Key Features

- **Interpretable motif detection** using CAVs in protein language model embedding space
- **Multiple motif detection** in single sequences with precise localization
- **Layer-specific analysis** to identify optimal representation layers
- **Flexible PLM support** (ESM++, ESM2, ESM-C)

## Repository Structure

```
├── core/                    # Core CAV implementation
│   └── src/                 # Source modules (attribution, detection, training, evaluation)
├── scripts/                 # Core pipeline scripts
│   ├── fit_reference_scaler.py      # Fit shared scaler on reference population (setup)
│   ├── fit_global_pca.py            # Optional: fit scaler+PCA on reference population (--pca-pkl)
│   ├── prepare_embeddings.py        # Pool per-residue embeddings by span
│   ├── train_cav_from_embeddings.py # Train a CAV from positive/negative matrices
│   ├── evaluate_cav.py              # Evaluate CAV performance
│   ├── scan_sequence.py             # Sliding-window scan of a single sequence
│   └── run_attribution.py           # Attribution analysis
├── specific_scripts/        # Specialized analysis and visualization scripts
│   ├── query_proteins_by_cav.py     # Rank/scan a FASTA against trained CAV(s)
│   ├── build_reference_population.py# Build the reference negative set
│   └── pdb_bfactor_from_cav.py      # Write CAV scores into PDB B-factors
├── docs/                    # Additional documentation
├── examples/                # Example input files (see Examples section)
├── test_data/               # Full benchmark dataset (60 Pfam motifs)
└── test_data_small/         # Small subset for quick validation
```

## Installation

```bash
git clone <repository-url>
cd bioCAV
pip install -r core/requirements.txt
```

## Prerequisites

Two resources live outside this repository and must be provided:

| Resource | What it is |
|----------|------------|
| **PLM weights** | A local HuggingFace model directory, e.g. `ESMplusplus_large` |
| **Embedding script** | `hf_embed_new.py` from `mcwlab_utils`, used for all embedding steps |

Set these once; every command below refers to them:

```bash
export BIOCAV_REPO=/path/to/bioCAV
export ESM_MODEL_DIR=/path/to/ESMplusplus_large
export HF_EMBED_SCRIPT=/path/to/mcwlab_utils/hf_embed_new.py
export REF_DIR=/path/to/reference_population     # created in Step 00
```

> **Lab members:** the concrete cluster paths, shared conda environment, and
> the location of the prebuilt reference population are in
> [`docs/internal_cluster_paths.md`](docs/internal_cluster_paths.md) — you do
> not need to rebuild Step 00.

## Pipeline

A one-time setup (Step 00) followed by five per-concept steps (01–05).

### Step 00 — Build the reference population (one-time setup)

**00a. Embed a large background set** of random proteins to serve as negatives for all CAV training.

```bash
mkdir -p $REF_DIR

python $HF_EMBED_SCRIPT \
    -f neg_10000.fasta \
    -o $REF_DIR/neg_10000.pkl \
    --get_sequence_embeddings \
    --strategy mean \
    -l -11 \
    -m $ESM_MODEL_DIR \
    -b 1 \
    --max_length 2048
```

**00b. Fit the shared scaler.** Every CAV is trained in this one coordinate system, so that scores from different CAVs are directly comparable.

```bash
python $BIOCAV_REPO/scripts/fit_reference_scaler.py \
    --neg $REF_DIR/neg_10000.pkl \
    --out $REF_DIR/scaler_v1.pkl
```

> Step 00b is **required** — `train_cav_from_embeddings.py --scaler-pkl` in
> Step 03 consumes `scaler_v1.pkl`. Skipping it means every CAV fits its own
> scaler and scores are no longer comparable between concepts.

**00c (optional). Fit a reference PCA** if you plan to train CAVs from very few
positives (see [Regularization and PCA](#regularization-and-pca)). The bundle
holds a scaler fit on the same reference set, so it matches `scaler_v1.pkl`.

```bash
python $BIOCAV_REPO/scripts/fit_global_pca.py \
    --pkl     $REF_DIR/neg_10000.pkl \
    --out     $REF_DIR/global_pca_128.pkl \
    --pca-dim 128
```

### Step 01 — Embed input FASTA

Embed the protein(s) that define your concept. Per-residue (`--get_aa_embeddings`) output is required for span pooling in Step 02.

```bash
python $HF_EMBED_SCRIPT \
    -f $INPUT_FASTA \
    -o ${INPUT_FASTA}.pkl \
    --get_aa_embeddings \
    --get_sequence_embeddings \
    --strategy mean \
    -l -11 \
    -m $ESM_MODEL_DIR \
    -b 1 \
    --max_length 2048
```

### Step 02 — Pool embeddings by span → positive matrix

Extract the region(s) defined in your span file and pool them into a positive embedding matrix.

```bash
python $BIOCAV_REPO/scripts/prepare_embeddings.py \
    --pkl   ${INPUT_FASTA}.pkl \
    --info  ${INPUT_FASTA}.pkl.seqnames \
    --spans $SPAN_FILE \
    --out   ${SPAN_FILE%.span}_pos.npy
```

### Step 03 — Train CAV

Train a linear classifier separating the positive span embeddings from the reference negatives.

```bash
python $BIOCAV_REPO/scripts/train_cav_from_embeddings.py \
    --pos        ${SPAN_FILE%.span}_pos.npy \
    --neg        $REF_DIR/neg_10000.pkl \
    --out        $concept_dir \
    --cv-folds   0 \
    --scaler-pkl $REF_DIR/scaler_v1.pkl
```

**Choosing `--cv-folds`:** `0` skips cross-validation entirely — use it when you have one or very few positive examples (CV is meaningless there). With a substantial positive set (tens or more), use `5` to get a CV AUROC alongside the train AUROC.

#### Regularization and PCA

**`--C` (default: 1/n_train).** The CAV is an L2-regularized logistic regression.
sklearn sums the loss over training samples, so the effective regularization is
roughly 1/(C·n): a fixed `C` regularizes a 50-sample problem far more heavily
than a 10,000-sample one. The default `C = 1/n_train` holds C·n = 1 constant.
With the 10k reference negatives that is ≈1e-4, the value used for the GO, EC
and Pfam CAV libraries in the paper. Override it only deliberately. If you train
against a small custom negative set, keep the default: a fixed `C=1e-4` with
n≈30 regularizes so hard that the CAV collapses to the difference of class
means. `report_v1.json` records `C` and whether it came from the default
(`C_rule`).

**Convergence.** The solver runs to convergence (lbfgs, `tol=1e-10`,
`max_iter=1e5`). These problems are close to linearly separable, and at
sklearn's default `tol=1e-4` lbfgs stops after a few dozen iterations, at a
point where the CAV *direction* depends on the stopping rule rather than on
the data. `report_v1.json` records `n_iter` and `converged`. A warning is
logged if `max_iter` is hit.

**`--pca-pkl` (optional).** With very few positives (a single protein, or a
handful of spans), the full-dimensional direction can lock onto features unique to
those few sequences rather than the shared concept. `--pca-pkl` trains the CAV
in the top-k principal components of the reference population (from Step 00c),
which restricts it to axes of variation the background actually spans:

```bash
python $BIOCAV_REPO/scripts/train_cav_from_embeddings.py \
    --pos     ${SPAN_FILE%.span}_pos.npy \
    --neg     $REF_DIR/neg_10000.pkl \
    --out     ${concept_dir}_pca128 \
    --cv-folds 0 \
    --scaler-pkl $REF_DIR/scaler_v1.pkl \
    --pca-pkl    $REF_DIR/global_pca_128.pkl
```

The k-d direction is folded back into the full space (`Pᵀw`, unit-normalized)
and saved as `concept_v1.npy`, which ranks identically to the PC-space CAV. A
PCA CAV is therefore scored exactly like any other (Step 05, `--scaler-pkl`
batch mode included). The k-d vector is kept as `concept_pca_v1.npy`, and
`pca_dim` / `pca_var_explained` are recorded in the report. If `--scaler-pkl`
is also given, it must match the bundle's scaler.

Choosing k. For the single-positive MYC bHLH concept in the paper, k ≤ 16
was too coarse to separate the concept and results were stable from about
k = 96 to 160. k = 128 is a reasonable starting point. With a new
few-positive concept, try a small sweep (e.g. 32, 64, 128, 256, no PCA)
against a labelled control set before trusting one setting. Concepts with
many positives (the GO/EC/Pfam libraries) are trained without PCA.

### Step 04 — Embed search FASTA

Embed the sequences you want to rank. Skipped automatically if the `.pkl` exists.

```bash
if [ -f "$search_pkl" ]; then
    echo "Skipping embedding: $search_pkl already exists"
else
    python $HF_EMBED_SCRIPT \
        -f $SEARCH_FASTA \
        -o $search_pkl \
        --get_aa_embeddings \
        --get_sequence_embeddings \
        --strategy mean \
        -l -11 \
        -m $ESM_MODEL_DIR \
        -b 1 \
        --max_length 2048
fi
```

> **`--get_aa_embeddings` is required here** if you intend to use
> `--sliding-window` in Step 05. Without per-residue embeddings the query
> script exits with *"does not contain aa_embeddings"*. Sequence-level
> embeddings alone suffice only for whole-sequence scoring.

> **On-the-fly embedding:** for small search sets you can skip Step 04 and pass
> `--embed-fasta $SEARCH_FASTA --embed-script $HF_EMBED_SCRIPT --model $ESM_MODEL_DIR`
> to the query script instead of `--pkl`. It writes `<fasta>.pkl` as a side effect.

### Step 05 — Query search FASTA against CAV

Score and rank every sequence — or every sliding window — against the trained CAV.

```bash
python $BIOCAV_REPO/specific_scripts/query_proteins_by_cav.py \
    --cav   $concept_dir \
    --pkl   $search_pkl \
    --out   $out_tsv \
    --k     -1 \
    --sliding-window \
    --spans $SPAN_FILE \
    --fasta $SEARCH_FASTA
```

| Flag | Effect |
|------|--------|
| `--k -1` | Return all results (default `100`) |
| `--sliding-window` | Score each protein by its **best** window instead of its whole-sequence embedding. Requires `aa_embeddings` in the pkl |
| `--spans` | **Sets the sliding-window size**, taken as the widest span in the file — not merely cosmetic |
| `--fasta` | Supplies protein lengths, so windows stop at the true sequence end rather than running into padding |

> **Window size is inherited from your span file.** A 39-residue concept scans
> with a 39-residue window. If you want a different window, supply a spans file
> of that width — this silently determines what the scan can detect.

### Full example wrapper

```bash
# -------------
# Inputs
# -------------
SPAN_FILE=examples/RLK5_ARATH.fasta.span
INPUT_FASTA=examples/RLK5_ARATH.fasta
SEARCH_FASTA=/path/to/search_proteome.fasta

# -------------
# Paths (see Prerequisites)
# -------------
export BIOCAV_REPO=/path/to/bioCAV
export ESM_MODEL_DIR=/path/to/ESMplusplus_large
export HF_EMBED_SCRIPT=/path/to/mcwlab_utils/hf_embed_new.py
export REF_DIR=/path/to/reference_population

concept_dir=${SPAN_FILE%.span}_concept
search_pkl=${SEARCH_FASTA}.pkl
out_tsv=${SPAN_FILE%.span}_query_results.tsv

# Step 01
if [ ! -f "${INPUT_FASTA}.pkl" ]; then
    python $HF_EMBED_SCRIPT \
        -f $INPUT_FASTA -o ${INPUT_FASTA}.pkl \
        --get_aa_embeddings --get_sequence_embeddings \
        --strategy mean -l -11 \
        -m $ESM_MODEL_DIR -b 1 --max_length 2048
fi

# Step 02
python $BIOCAV_REPO/scripts/prepare_embeddings.py \
    --pkl ${INPUT_FASTA}.pkl --info ${INPUT_FASTA}.pkl.seqnames \
    --spans $SPAN_FILE --out ${SPAN_FILE%.span}_pos.npy

# Step 03
python $BIOCAV_REPO/scripts/train_cav_from_embeddings.py \
    --pos ${SPAN_FILE%.span}_pos.npy --neg $REF_DIR/neg_10000.pkl \
    --out $concept_dir --cv-folds 0 --scaler-pkl $REF_DIR/scaler_v1.pkl

# Step 04
if [ ! -f "$search_pkl" ]; then
    python $HF_EMBED_SCRIPT \
        -f $SEARCH_FASTA -o $search_pkl \
        --get_aa_embeddings --get_sequence_embeddings \
        --strategy mean -l -11 \
        -m $ESM_MODEL_DIR -b 1 --max_length 2048
fi

# Step 05
python $BIOCAV_REPO/specific_scripts/query_proteins_by_cav.py \
    --cav $concept_dir --pkl $search_pkl --out $out_tsv \
    --k -1 --sliding-window --spans $SPAN_FILE --fasta $SEARCH_FASTA
```

## Advanced options

### Querying many CAVs at once

`--cav` accepts multiple artifact directories. Passing a shared `--scaler-pkl`
replaces the per-CAV loop with a single matrix multiply, which matters for
library-scale runs (thousands of CAVs).

```bash
python $BIOCAV_REPO/specific_scripts/query_proteins_by_cav.py \
    --cav cavs/GO_0000124/ cavs/GO_0003677/ cavs/GO_0005524/ \
    --pkl $search_pkl \
    --scaler-pkl $REF_DIR/scaler_v1.pkl \
    --cav-name-depth 2 \
    --cav-name-strip _positive_train_max1000_cav \
    --out $out_tsv
```

`--cav-name-depth N` builds each output column name from the last *N* path
components (useful when many CAVs share a directory name and the term ID lives
one level up); `--cav-name-strip` removes a shared suffix from those names.

### Length correction (motif dilution)

Whole-sequence embeddings dilute a short motif's signal in proportion to protein
length, so long proteins score lower for the same motif. `--length-correct`
estimates the span embedding from the whole-sequence embedding and scores that
instead.

```bash
python $BIOCAV_REPO/specific_scripts/query_proteins_by_cav.py \
    --cav $concept_dir --pkl $search_pkl --out $out_tsv \
    --length-correct --span-length 39 \
    --fasta $SEARCH_FASTA --ref-pkl $REF_DIR/neg_10000.pkl
```

Note the converse bias in `--sliding-window` mode: a *max* over windows grows
with the number of windows, so long proteins gain an advantage from having more
chances. When ranking across a proteome, check whether score correlates with
length before trusting the ordering.

### Other scripts

| Script | Purpose |
|--------|---------|
| `scripts/evaluate_cav.py` | Evaluate a trained CAV against labelled positives/negatives |
| `scripts/scan_sequence.py` | Sliding-window scan of one sequence (per-residue profile) |
| `scripts/run_attribution.py` | Attribution analysis for a CAV |
| `specific_scripts/pdb_bfactor_from_cav.py` | Write CAV scores into PDB B-factors for structural visualisation |

## Examples

The `examples/` directory contains input files for a receptor-like kinase domain concept:

| File | Description |
|------|-------------|
| `examples/RLK5_ARATH.fasta` | Input protein sequence (RLK5, *A. thaliana*) |
| `examples/RLK5_ARATH.fasta.span` | Span defining the kinase domain (residues 404–591) |

```bash
SPAN_FILE=examples/RLK5_ARATH.fasta.span
INPUT_FASTA=examples/RLK5_ARATH.fasta
```

## Input Formats

### FASTA

Standard FASTA. The accession in the header must match the accession used in the span file.

```
>sp|P47735|RLK5_ARATH Receptor-like protein kinase 5 ...
MLYCLILLLC...
```

### Span file

Tab-separated, one entry per line, no header. All positions are **1-indexed**;
window ranges are **inclusive**. Lines beginning with `#` are comments. Four
modes are supported:

```
accession                          # whole-sequence (no span)
accession	pos                       # single position (e.g. a PTM site)
accession	start	end                 # window [start, end], inclusive
accession	p1,p2,p3,...              # position list — mean of exactly these residues
```

```
sp|P47735|RLK5_ARATH	404	591
```

**The same accession may appear on multiple lines** — each line yields a
separate embedding vector. This is how you supply several negative windows from
one protein, or multiple PTM sites from one sequence.

**Position-list mode is the right choice for alignment-derived spans.** When a
region is mapped onto a homolog through an alignment, `start`–`end` also sweeps
in any inserted residues that fall between the aligned columns; a position list
pools only the residues that genuinely correspond.

## Models Supported

- **ESM++** (`ESMplusplus_large`) — recommended for best performance
- **ESM2** (`ESM2_t33_650M_UR50D`, `ESM2_t36_3B_UR50D`)
- **ESM-C** (`ESMC_300M`, `ESMC_600M`)

Layer `-11` (second-to-last transformer layer) is used throughout, following the arxiv paper.

## Citation

If you use this code, please cite our paper:
```
[Citation information to be added]
```
