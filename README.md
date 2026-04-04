# Dual-Branch Local–Global Fusion for Text Emotion Recognition

**SC4001: Neural Networks and Deep Learning — Group Project B**  
Nanyang Technological University, AY2025–26 Semester 2

---

## Overview

This project proposes a **Dual-Branch Local–Global Fusion** architecture for text emotion recognition that explicitly disentangles word-level emotion signals from sentence-level semantic meaning.

**Key idea:** Emotion in text operates at two scales. Individual words carry emotional valence ("furious", "delighted"), while sentence-level structure can invert or amplify those signals ("I'm not happy" vs. "I'm not unhappy"). Our model uses two parallel branches:

- **Local Branch:** Multi-kernel 1D CNN over BERT token embeddings captures n-gram emotion patterns
- **Global Branch:** MLP on the [CLS] embedding captures sentence-level semantics
- **Gated Fusion:** A learned gate dynamically weights the two branches per sample

## Repository Structure

```
├── config.py              # All hyperparameters (single source of truth)
├── preprocess.py           # Data loading, cleaning, tokenization, DataLoaders
├── models.py               # All 6 model architectures
├── train.py                # Training pipeline (mixed precision, early stopping)
├── evaluate.py             # Test evaluation, error analysis, attention extraction
├── visualize.py            # All visualizations (confusion matrices, t-SNE, etc.)
├── run_experiments.py      # Master script — runs everything end-to-end
├── utils.py                # Utilities (losses, metrics, seeding)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Quick Start (Google Colab T4)

### 1. Setup

```python
# In a Colab cell:
!pip install torch torchvision transformers pandas scikit-learn matplotlib seaborn tqdm

# Upload project files to Colab, or clone from drive
# Then cd into the project directory
```

### 2. Prepare Datasets

**CROWDFLOWER** (~40K tweets, 13 emotions):
```python
# Option A: Download from Kaggle
!pip install kaggle
!kaggle datasets download -d pashupatigupta/emotion-detection-from-text
!unzip emotion-detection-from-text.zip -d data/
!mv data/tweet_emotions.csv data/crowdflower.csv

# Option B: Manual download from
# https://data.world/crowdflower/sentiment-analysis-in-text
# Save as data/crowdflower.csv
```

**WASSA2017** (~7K tweets, 4 emotions):
```python
# Auto-downloaded from GitHub by the code.
# If that fails, manually download from:
# https://github.com/vinayakumarr/WASSA-2017/tree/master/wassa
# Place files in data/wassa2017/
```

**GloVe Embeddings** (for TextCNN/BiLSTM baselines):
```python
# Optional — baselines will use random init if not available
!wget https://nlp.stanford.edu/data/glove.6B.zip
!unzip glove.6B.zip -d data/
# Only glove.6B.300d.txt is needed
```

### 3. Run Experiments

```bash
# Full run (all experiments, 3 seeds per model)
python run_experiments.py

# Quick test (2 epochs, 1 seed — verify everything works)
python run_experiments.py --quick

# Single dataset
python run_experiments.py --dataset wassa2017

# Skip slow non-BERT baselines
python run_experiments.py --skip-baselines

# Custom configuration
python run_experiments.py --epochs 8 --seeds 42 123 --batch-size 16
```

### 4. Expected Outputs

After running, find results in:
- `outputs/` — Metrics JSON files, error analysis
- `figures/` — All report visualizations (PNG)
- `checkpoints/` — Best model checkpoints
- `logs/` — Training logs

## Models Implemented

| # | Model | Description | Role |
|---|-------|-------------|------|
| 1 | TextCNN | Kim (2014) CNN + GloVe | Baseline (local only) |
| 2 | BiLSTM+Attention | Bidirectional LSTM with self-attention | Baseline (sequential) |
| 3 | Vanilla BERT | Standard [CLS] fine-tuning | Transformer baseline |
| 4 | BERT+CNN | BERT tokens → CNN (local branch only) | Ablation |
| 5 | BERT+[CLS] | BERT [CLS] → MLP (global branch only) | Ablation |
| 6 | **Dual-Branch** | **Local + Global + Gated Fusion** | **Proposed** |

## Experiments

1. **Baseline comparison:** All 6 models on both datasets
2. **Ablation study:** Gated vs. concat vs. average vs. local-only vs. global-only
3. **Kernel size analysis:** (2,3) vs. (3,4,5) vs. (2,3,4) vs. (2,3,4,5)
4. **Encoder comparison:** BERT-base vs. RoBERTa-base vs. DistilBERT

## Compute Requirements

- **GPU:** NVIDIA T4 (16 GB) is sufficient
- **Training time per model:** ~5-15 minutes per seed (BERT-based on T4)
- **Full experiment suite:** ~3-5 hours on T4
- **Quick test:** ~10-15 minutes

## Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Encoder LR | 2e-5 | Standard BERT fine-tuning rate |
| Head LR | 1e-3 | Higher for randomly initialized layers |
| Batch size | 32 | Fits T4 with fp16 |
| Max seq length | 128 | Sufficient for tweets |
| CNN kernels | (2, 3, 4) | Captures 2/3/4-gram patterns |
| Dropout | 0.3 | Applied in both branches |
| Loss | Focal (γ=2) | Handles class imbalance |
| Frozen layers | 8/12 | Preserves pretrained knowledge |
| Seeds | 42, 123, 456 | For mean ± std reporting |

## Notes

- All results reported as **mean ± std over 3 seeds**
- Primary metric: **Macro-F1** (robust to class imbalance)
- Mixed precision (fp16) enabled by default for T4 efficiency
- Early stopping (patience=3) on validation Macro-F1
