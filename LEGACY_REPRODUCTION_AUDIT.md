# Legacy symptom-classifier reproduction audit

Audit source of truth: `symtoms_classifier_mean.ipynb`. Current pipeline:
`colab_embedding_and_training.ipynb` plus `prepare_patient_embeddings.py` and
`build_gfs_sentence_mapping.py`.

No old `.npy`, checkpoint, per-patient prediction, metric, or `SynSUM.csv` file
was available in the workspace at audit time. The historical notebook outputs
are therefore the only result reference currently available.

| Component | Legacy implementation | Current implementation | Required legacy-mode behavior |
|---|---|---|---|
| Source text | `SynSUM.csv["text"]`, source row order | Prebuilt sentence mapping joined by source ID | Read `SynSUM.csv["text"]` directly; row IDs `0..N-1` |
| Sentence splitter | Header/newline regex; literal `". "` split | Nearly equivalent per-document mapping builder with safer missing handling | Copy legacy list-of-documents behavior |
| Text cleaning | Lowercase; punctuation removal; whitespace collapse | Same normalization plus validation/filtering | Use legacy function directly |
| Tokenizer/model | Raw `AutoTokenizer` and `AutoModel`; max length 128 | `SentenceTransformer`; `search_document:` prefix | Raw Transformers; no prefix |
| Token-to-sentence pooling | `pooler_output`, else token 0 / CLS | Model-configured SentenceTransformer sentence pooling | Legacy pooler/CLS fallback |
| Sentence-to-patient pooling | Direct `mean(dim=1)` | Mean of valid mapped sentence vectors | Direct unmasked mean |
| Padded sentences | Zero ID/mask slots enter the mean | No padded slots enter patient mean | Preserve padded-slot influence |
| Embedding dimension | 768 | 768 | Assert 768 |
| Label order | dysp, cough, pain, fever, nasal | Same | Preserve exactly |
| Fever | none=0, low=1, high=2; no pre-BCE conversion | `fever > 0` | Pass raw target to BCE and warn on 2 |
| Outer split | Random unstratified 80/20, seed 5 | Multilabel-stratified 80/20, seed 42 | Random unstratified seed 5 |
| Training subset | Random 20% of 8,000 train rows (1,600) | Nested stratified orders and many fractions | Python-shuffled 1,600-row subset, seed 5 |
| Validation | Random 90/10 holdout of cache rows | Five-fold multilabel-stratified CV | Random holdout, seed 5 |
| Classifier | Linear(768,256), Linear(256,5) | Linear(768,128), ReLU, dropout, Linear(128,5) | Exact two-layer linear head |
| Activation/dropout/normalization | None | ReLU and dropout 0.25 | None |
| Feature standardization | None | Fold/subset z-score | None |
| Loss | Unweighted BCEWithLogitsLoss | BCE with `pos_weight` | Unweighted BCE |
| Optimizer | `AdamW(model.parameters(), lr=3e-5)` | AdamW, lr 1e-3, weight decay 1e-4 | Legacy call; implicit PyTorch default weight decay |
| Batch size | 32 | 32 | 32 |
| Maximum epochs | Executed call: 120 | 100 | 120 |
| Early stopping | Lowest mean validation batch loss; restore best | CV macro-F1; median epoch; full-subset retrain | Validation loss; patience 5; restore best |
| Random seeds | Python/NumPy/Torch/CUDA seed 5 | Base 42 and experiment seeds 42–46 | Seed 5 plus deterministic cuDNN controls |
| Threshold | Strict `probability > 0.5` | `>= 0.5` | Strict `> 0.5` |
| Evaluation | Test 2,000, then all 10,000; same CSV overwritten | Separate fixed-test and all-data outputs | Save both without overwrite; test is generalization scope |

## Differences most likely to explain the mismatch

1. The raw Transformers pooler/CLS path and unmasked padded-sentence mean can
   produce materially different patient vectors from SentenceTransformer mean
   pooling over valid, prefixed sentences.
2. Raw fever value `2` was passed into a binary BCE loss and compared directly
   with binary predictions in the legacy accuracy calculation.
3. The legacy experiment trained on a random 1,600-patient subset with a random
   validation holdout; the modern experiment uses stratification, CV, repeated
   seeds, and full-subset retraining.
4. Scaling, positive-class weights, learning rate, weight decay, nonlinear
   activation, dropout, and the early-stopping target all changed.
5. The shuffled training DataLoader was used while caching embeddings, so its
   row order and batch-local maximum sentence count affect both the embedding
   values and the later validation split.

## Reconstructed executed legacy path

- `train_embs.npy`: 1,600 rows × 768.
- `test_embs.npy`: 2,000 rows × 768.
- `all_embs.npy`: 10,000 rows × 768.
- Head training: 120 displayed validation-loss epochs; final displayed loss
  `0.1373`; patience 5 never triggered.
- Test output: exact-match accuracy `0.6535`; per-label accuracy
  `[0.934, 0.9175, 0.9075, 0.828, 0.966]`.
- Later all-data output: exact-match accuracy `0.6459`; per-label accuracy
  `[0.9424, 0.9121, 0.908, 0.8191, 0.9629]`; this later cell overwrote the same
  `predictions.csv` filename.

## Implementation and artifacts

`LEGACY_REPRODUCTION_MODE=True` is the notebook default. Workflow A calls
`legacy_reproduction.py` and writes only beneath `legacy_reproduction/`:

- `embeddings/legacy_patient_embeddings.npy` and matching labels/IDs;
- separate train-subset and test caches with labels and patient IDs;
- an RNG-state file allowing cached embeddings to retain the post-embedding
  legacy RNG stream for head training;
- `checkpoints/legacy_best_validation_loss_model.pt` and training history;
- `results/legacy_split_patient_ids.npz`;
- test/all per-patient CSV and NPZ predictions;
- test/all per-symptom CSV metrics;
- `legacy_reproduction_metadata.json` with package, CUDA, model revision,
  tokenizer revision, data, label, split, embedding, optimizer, and loss details;
- a generated run report.

Workflow B remains the current learning-curve experiment. Its `head_type` may be
set to `legacy_two_layer_linear`, but that changes only the architecture and is
not described as reproduction.

## Reproduction status and uncertainties

Status at implementation time: **not yet empirically reproduced**. The local
workspace lacks `SynSUM.csv`, old cached embeddings/predictions, and a configured
GPU runtime. A complete Colab run is required before comparing rows and results.

Exact probabilities may still depend on PyTorch, Transformers, tokenizers,
CUDA/cuDNN, GPU type, resolved Hugging Face model/tokenizer revisions, DataLoader
behavior, and the provenance of any reused cache. The run metadata records these
details. Exact patient IDs, raw labels, and binary predictions should match when
reference files are available; similar aggregate accuracy alone is insufficient.
