# Running a single CAV from a span file

This guide walks through training a CAV (Concept Activation Vector) for a
custom set of positive proteins and querying a FASTA of interest against it.

> **Note on shell scripts:** `train_cav_from_span.sh` and `run_spanfile.sbatch`
> in `specific_scripts/` are templates — do not edit or run them in place.
> Copy them to your working directory first:
> ```bash
> cp $biocav_dir/specific_scripts/train_cav_from_span.sh .
> cp $biocav_dir/specific_scripts/run_spanfile.sbatch .
> ```
> This keeps the originals intact and lets you customise paths for your project.

---

## Inputs

### Span file

A span file defines the positive training examples. Each line is one protein
(or one protein region), tab-separated:

```
accession [TAB start TAB end]
```

- **Whole-sequence CAV** (most common): provide only the accession — the full
  sequence is used.
- **Domain/region CAV**: add residue positions (1-based, inclusive) to focus
  the CAV on a specific region.

**Example** (`my_concept.span`):

```
Q9UKV3
P47735  404  591
O15054
```

Accessions must be UniProt accessions in `sp|ACC|GENE` or bare `ACC` format.

---

## Steps

### 1. Run `train_cav_from_span.sh`

This script handles FASTA retrieval, embedding, and CAV training in one call.
It requires two files from the shared reference population (run from the
directory that contains `reference_population/`):

```
reference_population/neg_10000.pkl   ← negative training set (hardcoded in script)
reference_population/scaler_v1.pkl   ← shared scaler (passed via --scaler-pkl)
```

```bash
bash $biocav_dir/specific_scripts/train_cav_from_span.sh \
    my_concept.span \
    --scaler-pkl reference_population/scaler_v1.pkl
```

**What it does internally:**

| Step | Script | Output |
|------|--------|--------|
| 1 | `retrieve_fasta.py` | `my_concept.span.fasta` |
| 2 | `hf_embed_new.py` | `my_concept.span.fasta.pkl` |
| 3 | `prepare_embeddings.py` | `my_concept.span.pos.npy` |
| 4 | `train_cav_from_embeddings.py` | `my_concept_cav/` directory |
| 5 | cleanup | removes intermediate `.pkl` and `.npy` files |

The shared reference negative population is loaded from:
```
reference_population/neg_10000.pkl
```

The trained CAV is written to `my_concept_cav/` containing:
```
my_concept_cav/
    concept_v1.npy      ← CAV direction vector
    scaler_v1.pkl       ← fitted scaler (shares reference population)
    report_v1.json      ← training metrics (accuracy, AUC)
    random_*_v1.npy     ← random-direction null CAVs
```

---

### 2. Query proteins against the CAV

Score a FASTA of proteins against the trained CAV:

```bash
python $biocav_dir/specific_scripts/query_proteins_by_cav.py \
    --cav  my_concept_cav/ \
    --pkl  my_search_proteins.fasta.pkl \
    --out  my_concept_query_results.tsv \
    --k    -1
```

`--k -1` returns scores for all proteins (ranked). Set `--k 100` to return
only the top 100.

If the search FASTA has not been embedded yet, embed it first:

```bash
python $embed_script \
    -f  my_search_proteins.fasta \
    -o  my_search_proteins.fasta.pkl \
    --get_sequence_embedding \
    --strat mean \
    -l -11 \
    -m $model \
    -b 1 \
    --max_length 2048
```

where:
```bash
embed_script=/groups/clairemcwhite/claire_workspace/github/mcwlab_utils/hf_embed_new.py
model=/groups/clairemcwhite/models/ESMplusplus_large
```

---

## Running on the cluster (SLURM)

For large FASTAs or many proteins, use the SLURM template. Copy it to your
working directory before editing — do not run it directly from `specific_scripts/`:

```bash
cp $biocav_dir/specific_scripts/run_spanfile.sbatch my_job.sbatch
# Edit SPAN_FILE, INPUT_FASTA, SEARCH_FASTA at the top of my_job.sbatch
sbatch my_job.sbatch
```

---

## Reference population

All CAVs in this project share the same reference negative population, ensuring
scores are comparable across concepts (EC, GO, PFAM, PPI, motifs):

```
reference_population/neg_10000.pkl   ← 10,000 randomly sampled background proteins
                                        used as the negative training set (hardcoded)
reference_population/scaler_v1.pkl   ← StandardScaler fit on the reference population
                                        passed via --scaler-pkl
```

Both files must be present relative to the working directory when running
`train_cav_from_span.sh`. Using the shared scaler (rather than fitting one per
concept) ensures CAV scores are comparable across EC, GO, PFAM, PPI, and motif
CAVs.
