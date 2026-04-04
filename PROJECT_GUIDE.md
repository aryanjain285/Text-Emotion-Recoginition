# Project Guide — What's Going On and How Everything Fits Together

This document walks you through the entire project so you understand every piece before running it, presenting it, or writing about it in the report.

---

## The Big Picture

**Problem:** Emotion recognition in text. Given a tweet like "I can't believe they cancelled the concert, I'm devastated", predict the emotion: sadness.

**Why it's hard:** Existing approaches (CNN, RNN) either capture word-level patterns ("devastated" → sadness) OR sentence-level meaning, but not both explicitly. BERT captures both implicitly through attention, but standard fine-tuning ([CLS] → classifier) doesn't explicitly separate these two levels.

**Our solution:** A dual-branch architecture that:
1. Extracts word-level emotion signals via CNN on BERT token embeddings (**local branch**)
2. Extracts sentence-level meaning via [CLS] token MLP (**global branch**)
3. Combines them with a learned gate that decides "for this particular sentence, should I trust the word cues or the overall meaning more?"

---

## How the Model Works (Intuitive Explanation)

### Step 1: BERT Encoder (shared)
The input text goes through BERT, which produces:
- A 768-dimensional vector for EACH token (these capture contextualized word meanings)
- A special [CLS] vector that summarizes the entire sentence

### Step 2: Local Branch
Take all those per-token vectors and run 1D CNNs with different window sizes over them:
- Kernel size 2: looks at pairs of adjacent words ("not happy", "so angry")
- Kernel size 3: looks at triplets ("I am furious")
- Kernel size 4: looks at 4-word phrases ("could not be happier")

Then max-pool each: "what's the strongest emotion signal from any 2-word/3-word/4-word window?"

Result: a 384-dim vector capturing the strongest word-level emotion cues.

### Step 3: Global Branch
Take the [CLS] vector (sentence summary) and run it through a small 2-layer MLP.

Result: a 384-dim vector capturing sentence-level meaning (sarcasm, negation, overall tone).

### Step 4: Gated Fusion
Concatenate both vectors → feed through a sigmoid gate:
```
gate = σ(W · [local_features; global_features])
output = gate * local + (1 - gate) * global
```

If gate ≈ 1 → model trusts word-level cues (straightforward emotions like "I'm so happy!")
If gate ≈ 0 → model trusts sentence-level context (sarcasm, negation like "Oh great, another meeting")

The gate learns this automatically from data — you don't set it manually.

### Step 5: Classifier
A simple linear layer maps the 384-dim fused vector to emotion class probabilities.

---

## Why This Matters (For the Report)

The project handout literally says: "RNN and CNN capture local information and ignore global information." Our model directly addresses this by:

1. **Explicitly separating** local and global representations (not just hoping BERT handles it)
2. **Learning when to use which** via the gate mechanism
3. **Proving it works** through ablation studies (removing each component to measure its contribution)

The gate analysis visualization is the "wow factor" — it shows empirically that the model learns to lean on different branches for different emotion types.

---

## The Experiments (What Each One Proves)

### Experiment 1: Baseline Comparison
**Purpose:** Show our model beats existing approaches.

| Model | What it represents |
|-------|-------------------|
| TextCNN | Traditional CNN approach (no BERT, no global context) |
| BiLSTM+Attention | Sequential model with attention (implicit global) |
| Vanilla BERT | Standard fine-tuning (implicit both, no explicit separation) |
| BERT+CNN (local only) | Our local branch in isolation |
| BERT+[CLS] (global only) | Our global branch in isolation |
| **Dual-Branch (ours)** | **Both branches + gated fusion** |

**Expected result:** Dual-Branch > Vanilla BERT > BiLSTM > TextCNN

### Experiment 2: Ablation Study
**Purpose:** Prove each component contributes. "Is the gate actually useful, or does simple concatenation work just as well?"

| Config | What it tests |
|--------|--------------|
| Full (gated) | The complete model |
| Local only | Remove global branch entirely |
| Global only | Remove local branch entirely |
| Concat fusion | Keep both branches but replace gate with concatenation |
| Average fusion | Keep both branches but average instead of gating |

**Expected result:** Full gated > concat > average > either branch alone

### Experiment 3: Kernel Size Analysis
**Purpose:** Which n-gram lengths matter most for emotion detection?

Tests: (2,3), (3,4,5), (2,3,4), (2,3,4,5)

**Expected result:** (2,3,4) is optimal — bigrams and trigrams capture most emotion patterns.

### Experiment 4: Encoder Comparison
**Purpose:** Is our architecture encoder-agnostic? Does it work regardless of which pretrained model we use?

Tests: BERT-base, RoBERTa-base, DistilBERT

**Expected result:** Works with all three; RoBERTa may edge out BERT slightly.

---

## The Datasets

### CROWDFLOWER
- ~40K tweets labeled with 13 emotions
- We filter to top 6 for cleaner experiments
- Highly imbalanced (lots of "happiness", few "boredom")
- Noisier labels (crowdsourced)

### WASSA2017
- ~7K tweets with 4 emotions (anger, fear, joy, sadness)
- Cleaner but smaller → overfitting risk
- Moderately balanced

Using two datasets shows the model generalizes across different label spaces.

---

## Code Architecture (How the Files Connect)

```
config.py ─────────────────────────┐
  (all hyperparameters)             │
                                    ▼
preprocess.py ──────────────── run_experiments.py
  (data loading & tokenization)     │ (orchestrates everything)
                                    │
models.py ──────────────────────────┤
  (all 6 architectures)             │
                                    │
train.py ───────────────────────────┤
  (training loop)                   │
                                    │
evaluate.py ────────────────────────┤
  (test evaluation, error analysis) │
                                    │
visualize.py ───────────────────────┘
  (all plots & figures)

utils.py ─── imported by everything
  (losses, metrics, seeding)
```

**To run everything:** `python run_experiments.py`

That's it. One command runs all experiments, generates all figures, saves all metrics.

---

## Key Design Decisions (Why Things Are the Way They Are)

**Freezing 8/12 BERT layers:** The lower layers learn general language features (syntax, grammar). Only the top 4 layers need task-specific fine-tuning. This prevents overfitting on small datasets and speeds up training.

**Focal Loss instead of Cross-Entropy:** CROWDFLOWER is heavily imbalanced. Focal loss down-weights easy/frequent classes and focuses on hard/rare ones. γ=2 is the standard setting from the original paper.

**Mixed Precision (fp16):** Halves GPU memory usage and doubles throughput on T4. Essential for fitting BERT-base in 16GB GPU memory with reasonable batch sizes.

**3 random seeds:** Reporting mean ± std shows the result is robust, not a lucky seed. This is standard practice and graders look for it.

**Macro-F1 as primary metric:** Accuracy is misleading with imbalanced classes (predicting the majority class always gives high accuracy). Macro-F1 weights all classes equally regardless of size.

---

## Expected Timeline for Running

On a T4 GPU in Colab:

| Task | Time |
|------|------|
| Quick test (`--quick`) | ~10-15 min |
| TextCNN + BiLSTM baselines | ~15 min |
| BERT-based models (3 models × 3 seeds) | ~1.5 hours |
| Ablation study (2 extra configs × 3 seeds) | ~45 min |
| Kernel analysis (3 extra configs × 3 seeds) | ~45 min |
| Encoder comparison (2 extra encoders × 3 seeds) | ~1 hour |
| Visualizations + error analysis | ~5 min |
| **Total** | **~4-5 hours** |

**Recommendation:** Run `--quick` first to verify everything works. Then run the full suite overnight or across sessions.

---

## What to Put in the Report

The code generates everything you need:

1. **Figures (auto-generated in `figures/`):**
   - Confusion matrices (proposed vs. baseline)
   - Training curves (loss and F1 over epochs)
   - Per-class F1 bar charts
   - t-SNE of learned representations
   - Gate activation analysis
   - Attention heatmaps

2. **Tables (from `outputs/metrics_*.json`):**
   - Main results table (all models × both datasets)
   - Ablation results
   - Kernel analysis results
   - Encoder comparison results

3. **Error analysis (from `outputs/error_analysis_*.json`):**
   - Categories of misclassifications
   - Example errors with analysis

---

## Troubleshooting

**Colab session timeout:** The training saves checkpoints after each best epoch. If your session dies, you can re-run and it will reload from the last checkpoint. Training will restart but it's fast.

**Out of memory:** Reduce batch size: `--batch-size 16`. If still OOM, the code uses gradient accumulation automatically to maintain effective batch size.

**Dataset download fails:** Download manually from the URLs in the README and place in the `data/` directory. The code checks multiple filename patterns.

**RoBERTa tokenizer mismatch:** If you run encoder comparison, note that using BERT tokenizer with RoBERTa isn't ideal. For the best results, re-tokenize the data with each encoder's native tokenizer. The code handles this if you set `--encoder roberta-base` as a global override.
