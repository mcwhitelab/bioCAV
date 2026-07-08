#!/usr/bin/env bash
# Train a CAV from a span file of positive protein IDs.
# Usage: bash train_cav_from_span.sh <SPAN_FILE>
# Example: bash train_cav_from_span.sh go_dataset_part6/GO_0035639/random_positive_train_max1000.span

SPAN_FILE=$1
FEW_EXEMPLARS=0
ONLY_DOWNLOAD=0
SCALER_PKL=""
args=("$@")
for i in "${!args[@]}"; do
    [ "${args[$i]}" = "--few-exemplars" ] && FEW_EXEMPLARS=1
    [ "${args[$i]}" = "--only-download" ] && ONLY_DOWNLOAD=1
    if [ "${args[$i]}" = "--scaler-pkl" ]; then
        SCALER_PKL="${args[$((i+1))]}"
    fi
done
CV_FOLDS=$([ "$FEW_EXEMPLARS" -eq 1 ] && echo 0 || echo 5)

# -------------
### Paths
# -------------
source ~/.bashrc
set -euo pipefail
conda activate /groups/clairemcwhite/envs/core_pkgs4
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# This script is copied out of the repo before use (see docs/run_single_cav.md),
# so biocav_dir can't be self-located from this file's own path — it's a real
# hardcoded path to the current bioCAV checkout.
biocav_dir=/groups/clairemcwhite/cav_workspace/bioCAV
ref_neg=reference_population/neg_10000.pkl
embed_script=/groups/clairemcwhite/claire_workspace/github/mcwlab_utils/hf_embed_new.py
model=/groups/clairemcwhite/models/ESMplusplus_large

# Derived paths
FASTA_FILE=${SPAN_FILE}.fasta
PKL_FILE=${FASTA_FILE}.pkl
POS_NPY=${SPAN_FILE}.pos.npy
CAV_DIR=${SPAN_FILE%.span}_cav

# -------------
### Step 1: Retrieve FASTA for positive training proteins
# -------------
if [ -f "$FASTA_FILE" ]; then
    echo "Skipping FASTA retrieval: $FASTA_FILE already exists"
else
    python $biocav_dir/specific_scripts/retrieve_fasta.py \
        $SPAN_FILE $FASTA_FILE
fi

[ "$ONLY_DOWNLOAD" -eq 1 ] && { echo "Done (--only-download)."; exit 0; }

# -------------
### Step 2: Embed FASTA
# -------------
if [ -f "$PKL_FILE" ]; then
    echo "Skipping embedding: $PKL_FILE already exists"
else
    python $embed_script \
        -f $FASTA_FILE \
        -o $PKL_FILE \
        --get_aa_embeddings \
        --get_sequence_embedding \
        --strat mean \
        -l -11 \
        -m $model \
        -b 1 \
        --max_length 2048
fi

# -------------
### Step 3: Pool embeddings by span → positive embedding matrix
# -------------
python $biocav_dir/scripts/prepare_embeddings.py \
    --pkl  $PKL_FILE \
    --info ${PKL_FILE}.seqnames \
    --spans $SPAN_FILE \
    --out  $POS_NPY

# -------------
### Step 4: Train CAV
# -------------
python $biocav_dir/scripts/train_cav_from_embeddings.py \
    --pos  $POS_NPY \
    --neg  $ref_neg \
    --out  $CAV_DIR \
    --cv-folds $CV_FOLDS \
    ${SCALER_PKL:+--scaler-pkl "$SCALER_PKL"}

# -------------
### Step 5: Clean up intermediate embedding files
# -------------
rm -f $PKL_FILE ${PKL_FILE}.seqnames $POS_NPY
rm -f ${CAV_DIR}/pos.npy ${CAV_DIR}/neg.npy
