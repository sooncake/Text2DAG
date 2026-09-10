# Text2DAG patient embeddings, symptom classification, and graph evaluation

The preprocessing workflow creates one 768-dimensional mean text embedding per
represented patient and attaches the five SynSUM symptom labels. Separate
supervised workflows train the lightweight symptom classifier, including the
strict patient-level OOF graph experiment documented below.

## Setup

Use Python 3.10 or newer in a virtual environment, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run

From the project directory, with `gfs_sentence_mapping.csv` and `SynSUM.csv` in
that directory:

```bash
python prepare_patient_embeddings.py \
  --mapping-path gfs_sentence_mapping.csv \
  --source-path SynSUM.csv \
  --output-dir outputs/supervised_embeddings \
  --batch-size 32 \
  --device auto
```

`--device auto` selects CUDA when it is available and CPU otherwise. The first
run downloads `nomic-ai/modernbert-embed-base` through SentenceTransformers.

The output directory contains:

- `patient_mean_embeddings_with_labels.parquet`
- `patient_mean_embeddings_with_labels.csv`
- `patient_embeddings_and_labels.npz`
- `sentence_embeddings.parquet`

Run the lightweight tests, which do not download the model, with:

```bash
python -m unittest -v
```

The four yes/no symptoms are encoded as `0/1`. Fever is ordinal, following the
requested mapping: `none=0`, `low=1`, and `high=2`.

## Strict five-fold OOF classifier and graph experiment

`run_oof_graph_experiment.py` is a separate, leakage-safe extension of the
modern supervised workflow. It consumes the existing patient embedding NPZ,
uses one shared patient-level multilabel-stratified five-fold split, and trains
the 5%, 10%, 20%, and 100% conditions as nested fractions of each outer 80%
training pool. The architecture and training helpers live in
`modern_symptom_classifier.py` and preserve the modern notebook settings.

For every condition, each patient receives exactly one prediction from the one
outer-fold model that excluded that patient. The five held-out prediction sets
are concatenated before graph discovery. The old `all_dataset_predictions*`
files are never read by this workflow.

The repository and its upstream revision do not contain a prior downstream
causal-discovery implementation or graph metric functions. The new graph layer
is therefore isolated in `run_oof_graph_experiment.py` and fixes these settings
for the oracle and all four OOF conditions:

- implementation: `causal-learn==0.1.4.8`;
- algorithm: stable PC;
- conditional-independence test: `gsq`, causal-learn's likelihood-ratio G² test;
- significance level: `alpha=0.05`;
- collider rule: `uc_rule=0`, `uc_priority=2`;
- maximum conditioning depth: `max_k=None` (no added cap).

The reference graph must be a named binary adjacency CSV. Its first column
contains node names; the remaining square matrix uses rows as sources and
columns as targets. For example:

```csv
node,A,B,C
A,0,1,0
B,0,0,1
C,0,0,0
```

Run the complete experiment with:

```bash
python run_oof_graph_experiment.py \
  --embedding-npz outputs/supervised_embeddings/patient_embeddings_and_labels.npz \
  --structured-data SynSUM.csv \
  --reference-adjacency expert_dag_adjacency.csv \
  --output-dir outputs/oof_graph_experiment \
  --device auto
```

The strict default is exactly 10,000 unique patients. For each fold this yields
8,000 outer-training and 2,000 held-out patients, with nested training sizes of
400, 800, 1,600, and 8,000. The structured-data patient ID defaults to the
SynSUM column `Unnamed: 0`; change `--patient-id-column` if the source uses a
different explicit ID field.

The output directory includes:

- `outer_fold_assignments.csv` and
  `nested_training_subset_membership.csv`;
- `oof_predictions_005/010/020/100.csv` plus matching NPZ files;
- fold-level, complete-OOF, per-symptom, and inner-CV classifier metrics;
- the four reconstructed `graph_dataset_*.csv` files;
- oracle and OOF graph edge lists, metric adjacencies, and raw causal-learn
  endpoint matrices;
- `graph_metrics.csv`, `pc_learn_config.json`, and `experiment_metadata.json`.

`graph_metrics.csv` repeats the PC algorithm, G² test name, alpha, stability,
orientation-rule, and conditioning-depth settings on every condition row.

Graph matrices are explicitly aligned by node name before evaluation. The
isolated metric definition treats rows as sources and columns as targets,
expands an unoriented PC edge in both directions, calculates SHD as
off-diagonal binary Hamming distance (so reversal costs two), and calculates
correlation and precision/recall/F1 over the same ordered edge entries. These
choices and all PC settings are recorded in the metadata rather than inferred
from DataFrame order.

## Google Colab

For the complete strict OOF-to-graph experiment, open
`colab_oof_graph_experiment.ipynb` in Colab and run it from top to bottom. It
mounts Google Drive, clones/updates the experiment branch, installs the pinned
dependencies, builds or reuses the sentence mapping and patient embeddings,
trains the 5%/10%/20%/100% lightweight heads across the shared five outer
folds, displays symptom metrics, runs the oracle and four OOF PC-learn graph
conditions, and performs a final leakage/artifact audit. The notebook defaults
to the `codex/oof-graph-experiment` branch; change `REPOSITORY_BRANCH` to
`main` after the pull request is merged.

Place `SynSUM.csv` under `MyDrive/Text2DAG/inputs/` before running. The named
expert/reference adjacency `expert_dag_adjacency.csv` is versioned in this
repository and loaded automatically. All generated data, checkpoints, OOF
predictions, metrics, graphs, and metadata are written persistently beneath
`MyDrive/Text2DAG/`.

The supplied expert DAG contains 16 nodes and 35 directed edges from the
hand-specified reference. The reference term `dyspnea` is stored as `dysp` to
match the existing SynSUM symptom-label contract. Every remaining node name
must exactly match its corresponding `SynSUM.csv` column.

The older notebook below remains available for legacy and fixed-split
reproduction.

Open `colab_embedding_and_training.ipynb` in Google Colab for GPU embedding
generation and reproducible supervised training. Before running it, select a GPU
runtime and edit the clearly marked repository URL and Google Drive path cells.
The notebook defaults to `LEGACY_REPRODUCTION_MODE=True` and
`LEGACY_LEARNING_CURVE_MODE=True`. It trains six legacy supervised models with
seed 5 using 5%, 10%, 20%, 30%, 50%, and 100% of the fixed 80% outer training
pool. Every model is evaluated on the complete 100% dataset. These metrics include
training patients and are descriptive rather than held-out. The principal outputs
are `legacy_variablewise_f1.csv` and `legacy_macro_f1.csv`; fraction-level
checkpoints, histories, split IDs, detailed metrics, and predictions are also
saved. Everything is written under `legacy_reproduction/`, without reusing or
overwriting modern embeddings.

Set `LEGACY_LEARNING_CURVE_MODE=False` to run the original single legacy
20%-train-pool reproduction instead.

Set `LEGACY_REPRODUCTION_MODE=False` to run the existing modern learning-curve
workflow. Modern embeddings remain in `generated_sentence_embeddings`. Set
`FORCE_RECOMPUTE=True` only after changing the selected workflow's source text,
mapping, or embedding model.

The audit and known uncertainties are documented in
`LEGACY_REPRODUCTION_AUDIT.md`. The exact legacy implementation is in
`legacy_reproduction.py`; it is called by the notebook and can also be run as a
CLI.
