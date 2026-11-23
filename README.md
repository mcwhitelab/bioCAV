# bioCAV: biological concepts as directions in foundation model embeddings

Code for training, scoring and localizing concept activation vectors (CAVs) in
pretrained biological foundation models. Accompanies:

> **Representing biological concepts as directions in foundation model embeddings.**
> Ahmad Shamail and Claire D. McWhite, The University of Arizona.
> bioRxiv 2026. [doi:10.64898/2026.10.10.758204](https://www.biorxiv.org/content/10.64898/2026.10.10.758204v1.full)

## Overview

A CAV is the unit weight vector of a linear classifier that separates examples
of a concept from a common background population in a model's embedding space.
A concept can be anything you can supply examples of: a Pfam domain, a GO term,
an EC number, a binding interface defined by a handful of residues in one
protein, or a cell state. Scoring a new protein (or a window of it, or a cell)
is a dot product with that vector.

All CAVs here are trained against one shared background and standardized with
one shared scaler, so scores are comparable across concepts and a whole library
can be applied with a single matrix multiply.

- **This repository** is the method: embedding pooling, CAV training, scoring,
  sliding-window localization and per-residue attribution.
- **Pretrained libraries** (GO, EC, Pfam) are on Zenodo, see
  [Pretrained CAV libraries](#pretrained-cav-libraries).
- **Paper analyses and figures** are in
  [mcwhitelab/bioCAV-paper](https://github.com/mcwhitelab/bioCAV-paper).

## What you can do with it

| Task | Where |
|------|-------|
| Score proteins against the released GO / EC / Pfam CAVs | [Pretrained CAV libraries](#pretrained-cav-libraries) |
| Train a CAV for your own concept from a FASTA and a span file | [Pipeline](#pipeline), Steps 01–03 |
| Rank a proteome by a CAV, whole-sequence or best window | Steps 04–05 |
| Train from one or a few positives (PCA-restricted CAVs) | [Regularization and PCA](#regularization-and-pca) |
| Localize a concept along a sequence or onto a structure | [Other scripts](#other-scripts) |

## Repository structure

```
├── core/src/                # Library code: training, detection, evaluation, attribution, data loading
├── scripts/                 # The core pipeline (documented below)
│   ├── fit_reference_scaler.py      # Fit the shared scaler on the reference population
│   ├── fit_global_pca.py            # Optional: scaler + PCA bundle for --pca-pkl
│   ├── prepare_embeddings.py        # Pool per-residue embeddings by span
│   ├── train_cav_from_embeddings.py # Train a CAV from positive/negative matrices
│   ├── evaluate_cav.py              # AUROC/AUPRC on held-out positives and negatives
│   ├── scan_sequence.py             # Sliding-window scan of sequences
│   └── run_attribution.py           # Per-residue (or per-gene) projection onto a CAV
├── specific_scripts/        # Query tools and paper-specific utilities (see "Other scripts")
├── docs/run_single_cav.md   # Walkthrough for the one-command wrapper script
├── examples/                # Example input files (see Examples)
├── test_data/               # Pfam test sets: 69 families, 50 sequences each
└── test_data_small/         # The same layout with 10 sequences per family
```

## Installation

```bash
git clone https://github.com/mcwhitelab/bioCAV.git
cd bioCAV
pip install -r core/requirements.txt
```

The CAV code itself needs only `numpy`, `scikit-learn`, `scipy`, `pandas` and
`joblib`. `torch` and `transformers` are needed for embedding. The single-cell
utilities in `specific_scripts/` additionally use `scanpy`, `anndata` and
`cellxgene_census`, which are not in `requirements.txt`.

## Pretrained CAV libraries

The libraries used in the paper are released on Zenodo
(DOI [10.5281/zenodo.23284980](https://doi.org/10.5281/zenodo.23284980), CC BY 4.0):

| Library | CAVs | Positives |
|---------|------|-----------|
| Gene Ontology | 6,678 | proFAB GO term sets |
| Enzyme Commission | 953 | proFAB EC sets |
| Pfam | 27,472 | Pfam domain windows |
| Background | n/a | 10,000 UniRef50 proteins, their embeddings and the shared scaler |

Each library is one `.npz` holding a stacked matrix of unit-norm CAVs and the
shared scaler. No bioCAV code is needed to use them:

```python
import numpy as np
z = np.load("go_cavs.npz", allow_pickle=False)        # or ec_cavs.npz / pfam_cavs.npz
X = (emb - z["scaler_mean"]) / z["scaler_scale"]      # emb: (n_proteins, 1152), embedded as in Step 04
scores = X @ z["W"].T                                 # column j is CAV z["ids"][j]
```

A score is a projection, not a probability. Compare it across proteins for one
CAV, or across CAVs for one protein.

If you only want to train new CAVs that are comparable with these, download the
background files and skip Step 00: unzip `background_compat.zip`, which holds
the `neg_10000.pkl` and `scaler_v1.pkl` the pipeline reads, into `$REF_DIR`.

## Prerequisites

Two resources live outside this repository and must be provided:

| Resource | What it is |
|----------|------------|
| **PLM weights** | A local HuggingFace model directory. The paper uses [`Synthyra/ESMplusplus_large`](https://huggingface.co/Synthyra/ESMplusplus_large) (ESM-C 600M, 36 blocks, d = 1152) |
| **Embedding script** | `hf_embed_new.py` from [`mcwhitelab/mcwlab_utils`](https://github.com/mcwhitelab/mcwlab_utils), tag `cav`, used for all embedding steps |

Set these once; every command below refers to them:

```bash
export BIOCAV_REPO=/path/to/bioCAV
export ESM_MODEL_DIR=/path/to/ESMplusplus_large
export HF_EMBED_SCRIPT=/path/to/mcwlab_utils/hf_embed_new.py
export REF_DIR=/path/to/reference_population     # created in Step 00
```

> **Lab members:** the concrete cluster paths, shared conda environment and the
> prebuilt reference population are listed in `docs/internal_cluster_paths.md`
> in the lab checkout (not in the public repository). You do not need to
> rebuild Step 00.

## Pipeline

A one-time setup (Step 00) followed by five per-concept steps (01–05).

### Step 00 — Build the reference population (one-time setup)

Skip this step if you downloaded the background from Zenodo. Rebuilding it
gives a different coordinate system, and CAVs trained in it are not comparable
with the released libraries.

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

> **Which layer is `-l -11`?** `hf_embed_new.py` resolves negative layer
> indices against the number of transformer blocks (36 for ESM++ large), so
> `-l -11` selects `hidden_states[25]`: the output of block 25 of 36, where
> index 0 is the embedding layer. If you embed with your own code, take
> `hidden_states[25]`, strip BOS/EOS and mean-pool over residues. Indexing the
> 37-entry `hidden_states` tuple with `[-11]` gives layer 26, which does not
> match the released CAVs.

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

General-purpose:

| Script | Purpose |
|--------|---------|
| `scripts/evaluate_cav.py` | Evaluate a trained CAV against held-out positives/negatives (AUROC, AUPRC, threshold) |
| `scripts/scan_sequence.py` | Sliding-window scan, one score per window |
| `scripts/run_attribution.py` | Project each residue (or gene token) onto a CAV |
| `specific_scripts/pdb_bfactor_from_cav.py` | Write per-residue CAV scores into PDB B-factors for structural visualisation |
| `specific_scripts/combine_cav_scores.py` | Filter or re-rank by a second CAV (e.g. "kinase-like, then rank by concept X") |
| `specific_scripts/project_away_cav.py` | Remove one or more CAV directions from another (Gram-Schmidt) |
| `specific_scripts/build_protein_index.py` | hnswlib nearest-neighbour index over sequence embeddings, for embedding-similarity baselines |
| `specific_scripts/train_cav_from_span.sh` | One-command wrapper for Steps 01–03, see [`docs/run_single_cav.md`](docs/run_single_cav.md). A template with lab paths: copy and edit before use |

Used for the paper's evaluations (drivers are in
[bioCAV-paper](https://github.com/mcwhitelab/bioCAV-paper)):

| Scripts | Purpose |
|---------|---------|
| `format_proFAB.py`, `retrieve_fasta.py`, `retrieve_annotations.py` | Build GO/EC positive sets from proFAB and UniProt |
| `eval_go_temporal.py`, `summarize_temporal_eval.py`, `compare_tool_temporal.py`, `inspect_go_predictions.py` | GO temporal-holdout evaluation and comparison with external predictors |
| `eval_ec.py`, `summarize_ec_eval.py`, `compare_ec_tools.py`, `reformat_ec_goldstandard.py` | EC evaluation and tool comparison |
| `prepare_scrnaseq_embeddings.py`, `make_scrnaseq_spans.py`, `build_reference_population.py`, `run_cav_pipeline.py` | Single-cell CAVs: Geneformer embeddings, cell-level span files, CellxGene Census background |
| `cav_hierarchy.py`, `cav_gene_correlation.py`, `de_methods.py`, `cav_purity.py`, `cav_sample_size.py`, `cav_pair_spotlight.py` | Single-cell analyses: cell type / tissue / disease axes, gene correlation, differential expression |

The remaining files in `specific_scripts/` are exploratory or one-off utilities.

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

## Models

The CAV code operates on embedding matrices and is model-agnostic. Everything
in the paper, and every released protein CAV, uses:

- **Proteins:** ESM++ large (`Synthyra/ESMplusplus_large`, a HuggingFace port
  of ESM-C 600M), `hidden_states[25]` (`-l -11`, see the note in Step 00),
  mean-pooled over residues.
- **Single cells:** Geneformer V2, via `specific_scripts/prepare_scrnaseq_embeddings.py`.

`hf_embed_new.py` also supports other HuggingFace protein language models
(ESM2, BERT- and T5-style models). CAVs are specific to the model, layer and
scaler they were trained with, so the released libraries cannot be applied to
embeddings from a different model or layer.

## Citation

If you use this code or the CAV libraries, please cite:

```
Shamail A, McWhite CD. Representing biological concepts as directions in
foundation model embeddings. bioRxiv 2026.10.10.758204 (2026).
https://www.biorxiv.org/content/10.64898/2026.10.10.758204v1.full

Data: https://doi.org/10.5281/zenodo.23284980
```

## License

MIT, see [LICENSE](LICENSE). The released CAV libraries are CC BY 4.0.
