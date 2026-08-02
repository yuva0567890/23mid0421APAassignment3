# 23mid0421APAassignment3
 Table of Contents

- [Aim](#aim)
- [Datasets](#datasets)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [How to Run](#how-to-run)
- [Methodology](#methodology)
- [Models](#models)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Limitations](#limitations)
- [Reproducibility Notes](#reproducibility-notes)
- [License](#license)
- [Author](#author)

---

## Aim

To design, build, and evaluate a reproducible binary email spam-classification
benchmark across two independently sourced public email corpora, comparing a
probabilistic baseline (Naive Bayes family), a distance-based classifier
(K-Nearest Neighbours), a linear discriminative/margin baseline (Logistic
Regression, LinearSVC), and a trainable deep-learning sequence model
(Embedding + BiLSTM) — reporting both within-dataset and cross-dataset
(domain-transfer) performance with full uncertainty and error analysis.

## Datasets

| ID | Dataset | Task / Labels | Role |
|----|---------|----------------|------|
| D2 | [Enron-Spam](#) | Binary: legitimate (ham) vs spam | Primary training/evaluation corpus |
| D3 | [SpamAssassin Public Corpus](#) | Binary: legitimate (ham) vs spam | Second corpus + cross-dataset transfer target |

> **Add the exact dataset source URLs, version/download date, and license
> notes here before publishing.** Datasets are not committed to this
> repository — see [Setup](#setup) for how to obtain them.

Both datasets are normalized to a shared schema before use:

| Column | Type | Notes |
|---|---|---|
| `email_id` | string | Unique anonymized record ID |
| `subject` | string | Subject text (empty allowed) |
| `body` | string | Plain-text email body |
| `label` | string | `legitimate` / `spam` |
| `dataset_id` | string | `enron_spam` or `spamassassin` |

## Repository Structure

.
├── data/ # Raw/normalized CSVs (not committed — see Setup)
├── notebooks/
│ └── lab03_email_classification.ipynb
├── src/
│ ├── data_loader.py # Dataset loading, audit, schema validation
│ ├── pipelines.py # TF-IDF + classifier pipeline registry
│ ├── train_classical.py # CV, model selection, locked-test evaluation
│ ├── train_bilstm.py # Tokenizer, BiLSTM architecture, training loop
│ └── evaluate.py # Metrics, bootstrap CI, confusion matrices
├── outputs/
│ ├── models/ # Saved fitted pipelines / BiLSTM checkpoints
│ ├── cv_results_all_datasets.csv
│ ├── test_results.csv
│ └── split_manifest.json # Locked train/test IDs
├── reports/
│ └── MDI3003_Lab03_Report.docx
├── requirements.txt
└── README.md


## Setup

```bash
# Clone
git clone <your-repo-url>
cd <your-repo-name>

# Create environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

`requirements.txt`:

pandas
numpy
scipy
scikit-learn
matplotlib
joblib
tensorflow


### Getting the data

This repository does **not** commit raw email data. Download Enron-Spam and
SpamAssassin from their original sources (see [Datasets](#datasets)), place
the normalized CSVs under `data/`, matching the schema above, and update
`src/data_loader.py`'s `DATASETS` registry with the file paths if needed.

## How to Run

```bash
# 1. Audit datasets and create the locked train/test split manifest
python -m src.data_loader

# 2. Run training-only cross-validation across all classical classifiers
python -m src.train_classical --stage cv

# 3. Fit the selected classifier(s) and evaluate once on the locked test set
python -m src.train_classical --stage test

# 4. Train and evaluate the BiLSTM deep classifier
python -m src.train_bilstm

# 5. Generate metrics, confusion matrices, and plots
python -m src.evaluate
```

Or open `notebooks/lab03_email_classification.ipynb` and run top to bottom —
the notebook is designed to execute end-to-end without manual edits.

## Methodology

- **Feature representation (classical models):** TF-IDF (unigrams + bigrams,
  sublinear TF, `min_df=2`, `max_df=0.98`, `max_features=60000`), fitted
  **only inside each training fold** via a scikit-learn `Pipeline` to prevent
  vocabulary/IDF leakage.
- **Split protocol:** one stratified 80/20 train/test split per dataset,
  fixed random seed, split IDs saved to `split_manifest.json` and never
  altered after inspection.
- **Model selection:** ranked by mean macro F1 on **training-only** repeated
  stratified cross-validation — the locked test set is touched exactly once,
  after all model/hyperparameter decisions are finalized.
- **Cross-dataset transfer:** a model trained on D2's full training data is
  evaluated directly on D3's test data with no retraining, and vice versa
  (D2→D3, D3→D2), to quantify domain shift.
- **Deep model (BiLSTM):** Keras `Tokenizer` fitted on training text only,
  trainable embedding layer → Bidirectional LSTM → dense classification
  head, trained with early stopping and checkpointing on a validation subset
  carved out of the training partition.
- **Uncertainty:** repeated CV mean/SD on training folds, and a stratified
  bootstrap 95% confidence interval on the locked-test macro F1.

## Models

| Model | Type | Notes |
|---|---|---|
| Dummy (most-frequent) | Baseline | Performance floor |
| MultinomialNB | Classical | Sparse-text probabilistic baseline |
| ComplementNB | Classical | Imbalance-robust NB variant |
| Logistic Regression | Classical | Linear discriminative baseline |
| LinearSVC | Classical | Margin-based linear classifier |
| K-Nearest Neighbours | Classical | Distance-based, tuned over *k* |
| Embedding + BiLSTM | Deep learning | Trainable token embeddings, TensorFlow/Keras |

## Evaluation Metrics

- Accuracy, macro precision/recall/F1 (primary selection metric), weighted F1
- Per-class recall (legitimate-marked-as-spam errors tracked explicitly)
- Confusion matrices (raw counts and row-normalized)
- Repeated-CV mean ± SD (training folds only)
- Bootstrap 95% CI on locked-test macro F1
- Parameter count, training time, and inference latency (representation comparison)

## Results

Results are tracked in `outputs/cv_results_all_datasets.csv` and
`outputs/test_results.csv`, and summarized in
`reports/MDI3003_Lab03_Report.docx`.

| Dataset | Selected model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|---|
| D2 (Enron-Spam) | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| D3 (SpamAssassin) | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

> Populate this table (and the full report) after running the pipeline —
> see `outputs/test_results.csv` for the generated numbers.

## Limitations

- Scope is restricted to binary spam/ham classification on D2 and D3; no
  multiclass intent data or LLM-based draft generation is included here.
- Both corpora were collected some years ago; spam patterns evolve, so
  performance may not reflect current adversarial tactics.
- English-only text handling; no explicit multilingual or image-based spam
  handling.
- KNN on high-dimensional sparse TF-IDF vectors is computationally expensive
  at inference time and included for comparison, not as a production
  recommendation.
- The BiLSTM is trained on a relatively small, domain-specific corpus without
  large-scale pretraining, so it may not outperform strong linear baselines —
  this is treated as an expected, reportable outcome.
- Cross-dataset transfer is evaluated only between D2 and D3, not against a
  third unseen corpus or live production mail.

## Reproducibility Notes

- Fixed random seeds throughout (`RANDOM_STATE = 42`).
- Locked split IDs stored in `outputs/split_manifest.json`.
- All feature-fitting (TF-IDF vocabulary/IDF, BiLSTM tokenizer) occurs on
  training data only.
- No API keys, credentials, or raw personal/confidential email content are
  committed to this repository.
