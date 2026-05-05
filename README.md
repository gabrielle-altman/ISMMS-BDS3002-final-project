# Splice Site Prediction with Machine Learning

A machine learning project to predict donor and acceptor splice sites in human DNA sequences from the [HS3D dataset](http://www.sci.unisannio.it/docenti/rampone/) (Homo Sapiens Splice Sites Dataset). Built as a final project for BDS 3002: Machine Learning for Biomedical Data Science at the Icahn School of Medicine at Mount Sinai.

## Background

Splice sites are short DNA sequences at exon-intron boundaries that are essential for proper mRNA splicing. Mutations in these sites are linked to numerous genetic diseases including cystic fibrosis, beta-thalassemia, and spinal muscular atrophy. Accurate computational prediction of splice sites supports both disease diagnosis and the discovery of novel splicing biology.

This project trains and compares several model architectures to classify whether a given 140-nucleotide sequence contains a true splice site, with separate models for donor (5') and acceptor (3') sites.

## Dataset

- **Source:** Homo Sapiens Splice Sites Dataset (HS3D)
- **Donors:** 2,796 true / 271,937 false sequences (140 nt, GT at positions 70–71)
- **Acceptors:** 2,880 true / 329,374 false sequences (140 nt, AG at positions 69–70)
- **Bases:** A, C, G, T, and H (unknown)
- **Class imbalance:** roughly 100:1 false to true

The data is split 75/25 train/test with stratification on the label. The training set is further split into train/validation during model training.

## Models

Six architectures are implemented across separate scripts. All PyTorch models use `WeightedRandomSampler` to oversample the minority class, F1-macro for model selection, SGD with momentum (0.9) and weight decay (1e-4), `ReduceLROnPlateau`, and early stopping.

| Script | Model | Encoding | Notes |
|---|---|---|---|
| `random_forest.py` | Random Forest Classifier | One-hot, flattened | sklearn; with and without undersampling; extracts feature importances |
| `cnn.py` | 2-layer CNN | One-hot | Conv → ReLU → MaxPool → Dropout (×2) → Linear; exports first-layer filter weights for motif visualization |
| `lstm_unidirectional.py` | Unidirectional LSTM | Embedding | Embedding → LSTM → ReLU → FC → ReLU → FC. Defines the shared training loop, dataset, sampler, and early-stopping helpers used by all other PyTorch scripts. |
| `lstm_bidirectional.py` | Bidirectional LSTM | Embedding | Same as above with `bidirectional=True` |
| `cnn_bilstm.py` | CNN / BiLSTM | One-hot | Conv → Dropout → Conv → Dropout → BiLSTM → FC; exports filter weights |
| `cnn_bilstm_maxpool.py` | CNN / BiLSTM with MaxPool | One-hot | Conv → MaxPool → Dropout → Conv → MaxPool → Dropout → BiLSTM → FC; exports filter weights |

## Results summary

The Random Forest with undersampling and the CNN-based models achieved the strongest performance on the minority (true splice site) class. First-layer CNN filter weights were used to identify candidate motifs associated with splice site activity, which were then run through MEME / XSTREME for motif enrichment analysis. See `REvoLUtionary_finalreport.pdf` for full results, figures, and discussion.

## Repository structure

```
final_project_machine_learning/
├── README.md
├── REvoLUtionary_finalreport.pdf   # Full project report
├── preprocess.py                   # HS3D → donor/acceptor train/test TSVs
├── random_forest.py                # Random Forest models
├── cnn.py                          # Standalone 2-layer CNN
├── lstm_unidirectional.py          # Unidirectional LSTM (+ shared training utilities)
├── lstm_bidirectional.py           # Bidirectional LSTM
├── cnn_bilstm.py                   # CNN/BiLSTM (Table 4)
└── cnn_bilstm_maxpool.py           # CNN/BiLSTM with MaxPool (Table 5)
```

## Data setup

These scripts expect pre-split donor and acceptor TSVs with `seq` and `classification` columns. To produce them:

1. Download the HS3D dataset from the [original source](http://www.sci.unisannio.it/docenti/rampone/) and combine the donor and acceptor sequences into a single tab-separated file with at minimum a `seq` column and a `classification` column (one of `EI_true`, `EI_false`, `IE_true`, `IE_false`).
2. Run the preprocessing script:

```bash
python preprocess.py --input data/allSeqs.txt --output-dir data/
```

This produces `donors_train.txt`, `donors_test.txt`, `acceptors_train.txt`, and `acceptors_test.txt` using a 75/25 stratified split.

## Running a model

Each script takes hyperparameters as command-line arguments and looks for the train/test TSVs in `--data-dir` (default: `./data`). Plots are written to `--plot-dir` (default: `./plots`) and checkpoints / per-epoch metrics / filter-weight CSVs to `--results-dir` (default: `./results`).

Examples:

```bash
# Random Forest (no extra hyperparameters)
python random_forest.py

# Standalone CNN
python cnn.py my_run 0.001 5 32 15 6

# Unidirectional / Bidirectional LSTM
python lstm_unidirectional.py my_run 0.01 4 16 100 6
python lstm_bidirectional.py my_run 0.01 4 16 100 6

# CNN / BiLSTM (per-site filter counts; defaults match the report's sweep)
python cnn_bilstm.py my_run 0.01 4 100 6 \
    --num-filters-donor 128 --kernel-size-donor 9 \
    --num-filters-acceptor 64 --kernel-size-acceptor 9

# CNN / BiLSTM with MaxPool
python cnn_bilstm_maxpool.py my_run 0.01 4 100 6 \
    --num-filters-donor 128 --kernel-size-donor 9 \
    --num-filters-acceptor 128 --kernel-size-acceptor 7
```

Check the top of each script for full argument documentation.

## Dependencies

- Python 3.x
- PyTorch
- scikit-learn
- pandas, numpy
- matplotlib, seaborn

## Authors

Gabrielle Altman, Carina Seah, O'Jay Stewart — Icahn School of Medicine at Mount Sinai.