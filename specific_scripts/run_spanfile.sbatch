#! /bin/bash

# -------------
### Directives
# -------------
#SBATCH --job-name=span_cav
#SBATCH --account=mcwhite
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL

# -------------
### Inputs — replace these placeholders before running
# -------------
# SPAN_FILE   : tab-separated spans file defining positive examples
#               (accession [TAB pos] [TAB end] or accession TAB p1,p2,...)
# INPUT_FASTA : fasta containing the sequences referenced in SPAN_FILE
# SEARCH_FASTA: fasta of sequences to rank against the trained CAV

SPAN_FILE=SPAN_FILE
INPUT_FASTA=INPUT_FASTA
SEARCH_FASTA=SEARCH_FASTA

# -------------
### Paths
# -------------
source ~/.bashrc
conda activate /groups/clairemcwhite/envs/core_pkgs4
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# This script is copied out of the repo before use (see docs/run_single_cav.md),
# so biocav_dir can't be self-located from this file's own path — it's a real
# hardcoded path to the current bioCAV checkout.
biocav_dir=/groups/clairemcwhite/cav_workspace/bioCAV
ref_neg=reference_population/neg_10000.pkl
ref_pca=reference_population/global_pca_v1.pkl

# Derive output names from inputs
concept_dir=${SPAN_FILE%.spans}_concept
search_pkl=${SEARCH_FASTA}.pkl
out_tsv=${SPAN_FILE%.spans}_query_results.tsv

# -------------
### Step 1: Embed input fasta
# -------------
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

# -------------
### Step 2: Pool embeddings by span → positive embedding matrix
# -------------
python $biocav_dir/scripts/prepare_embeddings.py \
    --pkl  ${INPUT_FASTA}.pkl \
    --info ${INPUT_FASTA}.pkl.seqnames \
    --spans $SPAN_FILE \
    --out  ${SPAN_FILE%.spans}_pos.npy

# -------------
### Step 3: Train CAV
# -------------
python $biocav_dir/scripts/train_cav_from_embeddings.py \
    --pos    ${SPAN_FILE%.spans}_pos.npy \
    --neg    $ref_neg \
    --out    $concept_dir \
    --cv-folds 0 \
    --pca-pkl  $ref_pca

# -------------
### Step 4: Embed search fasta (skipped if pkl already exists)
# -------------
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

# -------------
### Step 5: Query search fasta against CAV
# -------------
python $biocav_dir/specific_scripts/query_proteins_by_cav.py \
    --cav  $concept_dir \
    --pkl  $search_pkl \
    --out  $out_tsv \
    --k -1
