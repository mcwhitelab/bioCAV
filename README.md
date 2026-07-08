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
│   ├── prepare_embeddings.py        # Pool per-residue embeddings by span
│   ├── train_cav_from_embeddings.py # Train a CAV from positive/negative matrices
│   ├── fit_global_pca.py            # Fit PCA on reference population
│   ├── evaluate_cav.py              # Evaluate CAV performance
│   ├── scan_sequence.py             # Sliding-window scan of a sequence
│   └── run_attribution.py           # Attribution analysis
├── specific_scripts/        # Specialized analysis and visualization scripts
│   └── query_proteins_by_cav.py    # Rank/scan a FASTA against a trained CAV
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

## Pipeline

The pipeline has a one-time setup step (Step 00) followed by five per-concept steps (Steps 01–05).

### Step 00 — Build reference negative population (one-time setup)

Embed a large set of random/background proteins to use as negatives for all CAV training. Output lives in `reference_population/`.

```bash
conda activate /groups/clairemcwhite/envs/core_pkgs4

f=/groups/clairemcwhite/ahmad_workspace/esm_c/neg_data/neg_10000.fasta

# Select layer -11 (second-to-last layer), as in the paper
python /groups/clairemcwhite/claire_workspace/github/mcwlab_utils/hf_embed_new.py \
    -f $f \
    -o reference_population/neg_10000.pkl \
    -ss mean \
    -s \
    -l -11 \
    -m /groups/clairemcwhite/models/ESMplusplus_large \
    -b 1 \
    --max_length 2048
```

### Step 01 — Embed input FASTA

Embed the protein(s) that define your concept. Both per-residue (AA) and sequence-level embeddings are needed for span pooling in Step 02.

```bash
python /groups/clairemcwhite/claire_workspace/github/mcwlab_utils/hf_embed_new.py \
    -f $INPUT_FASTA \
    -o ${INPUT_FASTA}.pkl \
    --get_aa_embeddings \
    --get_sequence_embedding \
    --strat mean \
    -l -11 \
    -m /groups/clairemcwhite/models/ESMplusplus_large \
    -b 1 \
    --max_length 2048
```

### Step 02 — Pool embeddings by span → positive matrix

Extract the region(s) defined in your span file and pool them into a positive embedding matrix.

```bash
python $biocav_dir/scripts/prepare_embeddings.py \
    --pkl   ${INPUT_FASTA}.pkl \
    --info  ${INPUT_FASTA}.pkl.seqnames \
    --spans $SPAN_FILE \
    --out   ${SPAN_FILE%.spans}_pos.npy
```

### Step 03 — Train CAV

Train a linear classifier (CAV) separating the positive span embeddings from the reference negative population.

```bash
python $biocav_dir/scripts/train_cav_from_embeddings.py \
    --pos     ${SPAN_FILE%.spans}_pos.npy \
    --neg     $ref_neg \
    --out     $concept_dir \
    --cv-folds 0 \
    --pca-pkl  $ref_pca
```

### Step 04 — Embed search FASTA

Embed the sequences you want to rank. Only sequence-level embeddings are needed here. This step is skipped automatically if the `.pkl` already exists.

```bash
if [ -f "$search_pkl" ]; then
    echo "Skipping embedding: $search_pkl already exists"
else
    python /groups/clairemcwhite/claire_workspace/github/mcwlab_utils/hf_embed_new.py \
        -f $SEARCH_FASTA \
        -o $search_pkl \
        --get_sequence_embedding \
        --strat mean \
        -l -11 \
        -m /groups/clairemcwhite/models/ESMplusplus_large \
        -b 1 \
        --max_length 2048
fi
```

> **On-the-fly embedding:** For small search FASTAs you can skip pre-computing the pkl and pass `--fasta` directly to the query script (Step 05); it will embed on the fly.

### Step 05 — Query search FASTA against CAV

Score and rank every sequence (or every sliding window) in the search set against the trained CAV.

```bash
python $biocav_dir/specific_scripts/query_proteins_by_cav.py \
    --cav   $concept_dir \
    --pkl   $search_pkl \
    --out   $out_tsv \
    --k     -1 \
    --sliding-window \
    --spans $SPAN_FILE \
    --fasta $SEARCH_FASTA
```

`--k -1` returns all results; `--sliding-window` enables per-window scoring; `--spans` and `--fasta` are used to annotate known positive regions in the output.

### Full example wrapper

A complete wrapper that sets variables and runs Steps 01–05 end-to-end:

```bash
# -------------
# Inputs — replace these placeholders before running
# -------------
SPAN_FILE=fastas/pairs/myc_mad/MYC_HUMAN.fasta.spans
INPUT_FASTA=fastas/pairs/myc_mad/MYC_HUMAN.fasta
SEARCH_FASTA=/xdisk/clairemcwhite/clairemcwhite/uniprot_human_all.fasta

# -------------
# Paths
# -------------
source ~/.bashrc
conda activate /groups/clairemcwhite/envs/core_pkgs4
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

biocav_dir=/groups/clairemcwhite/cav_workspace/bioCAV
ref_neg=reference_population/neg_10000.pkl
ref_pca=reference_population/global_pca_v1.pkl

concept_dir=${SPAN_FILE%.spans}_concept
search_pkl=${SEARCH_FASTA}.pkl
out_tsv=${SPAN_FILE%.spans}_query_results.tsv

# Step 01
if [ -f "${INPUT_FASTA}.pkl" ]; then
    echo "Skipping embedding: ${INPUT_FASTA}.pkl already exists"
else
    python /groups/clairemcwhite/claire_workspace/github/mcwlab_utils/hf_embed_new.py \
        -f $INPUT_FASTA -o ${INPUT_FASTA}.pkl \
        --get_aa_embeddings --get_sequence_embedding \
        --strat mean -l -11 \
        -m /groups/clairemcwhite/models/ESMplusplus_large \
        -b 1 --max_length 2048
fi

# Step 02
python $biocav_dir/scripts/prepare_embeddings.py \
    --pkl ${INPUT_FASTA}.pkl --info ${INPUT_FASTA}.pkl.seqnames \
    --spans $SPAN_FILE --out ${SPAN_FILE%.spans}_pos.npy

# Step 03
python $biocav_dir/scripts/train_cav_from_embeddings.py \
    --pos ${SPAN_FILE%.spans}_pos.npy --neg $ref_neg \
    --out $concept_dir --cv-folds 0 --pca-pkl $ref_pca

# Step 04
if [ -f "$search_pkl" ]; then
    echo "Skipping embedding: $search_pkl already exists"
else
    python /groups/clairemcwhite/claire_workspace/github/mcwlab_utils/hf_embed_new.py \
        -f $SEARCH_FASTA -o $search_pkl \
        --get_sequence_embedding --strat mean -l -11 \
        -m /groups/clairemcwhite/models/ESMplusplus_large \
        -b 1 --max_length 2048
fi

# Step 05
python $biocav_dir/specific_scripts/query_proteins_by_cav.py \
    --cav $concept_dir --pkl $search_pkl --out $out_tsv \
    --k -1 --sliding-window --spans $SPAN_FILE --fasta $SEARCH_FASTA
```

## Examples

The `examples/` directory contains input files for a receptor-like kinase domain concept:

| File | Description |
|------|-------------|
| `examples/RLK5_ARATH.fasta` | Input protein sequence (RLK5, *A. thaliana*) |
| `examples/RLK5_ARATH.fasta.span` | Span defining the kinase domain (residues 404–591) |

Run the pipeline with these files by setting:
```bash
SPAN_FILE=examples/RLK5_ARATH.fasta.span
INPUT_FASTA=examples/RLK5_ARATH.fasta
```

## Input Formats

### FASTA
Standard FASTA format. The accession in the header must match the accession used in the span file.

```
>sp|P47735|RLK5_ARATH Receptor-like protein kinase 5 ...
MLYCLILLLC...
```

### Span file (`.span`)
Tab-separated, one span per line:

```
<accession>	<start>	<end>
```

The accession must match the FASTA header (full `sp|...|...` form or just the bare ID, depending on your embed script). Start and end are 1-based, inclusive residue positions.

```
sp|P47735|RLK5_ARATH	404	591
```

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
