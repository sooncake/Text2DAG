# Patient embedding preprocessing

This project creates one 768-dimensional mean text embedding per represented
patient and attaches the five SynSUM symptom labels. It does not train a
supervised model.

## Setup

Use Python 3.10 or newer in a virtual environment, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run

From the project directory, with `subsentence_mapping.csv` and `SynSUM.csv` in
that directory:

```bash
python prepare_patient_embeddings.py \
  --mapping-path subsentence_mapping.csv \
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
- `subsentence_embeddings.parquet`

Run the lightweight tests, which do not download the model, with:

```bash
python -m unittest -v
```

The four yes/no symptoms are encoded as `0/1`. Fever is ordinal, following the
requested mapping: `none=0`, `low=1`, and `high=2`.

## Google Colab

Open `colab_embedding_and_training.ipynb` in Google Colab for GPU embedding
generation and reproducible supervised training. Before running it, select a GPU
runtime and edit the clearly marked repository URL and Google Drive path cells.
The notebook skips embedding generation when its NPZ output already exists,
unless `FORCE_RECOMPUTE` is enabled.
