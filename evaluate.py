"""
evaluate.py — Test set evaluation, error analysis, and feature extraction.

This module handles:
    1. Final evaluation on the held-out test set
    2. Error analysis (categorizing misclassifications)
    3. Feature extraction for visualizations (t-SNE, gate analysis)
    4. Attention weight extraction for heatmaps

Usage:
    from evaluate import evaluate_model_full, run_error_analysis
"""

import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict

from train import evaluate
from utils import compute_metrics, get_loss_fn

logger = logging.getLogger("project")


# ============================================================================
#  Full Test Evaluation
# ============================================================================

def evaluate_model_full(
    model: nn.Module,
    test_loader,
    cfg,
    num_classes: int,
    class_weights: Optional[torch.Tensor] = None,
    label_names: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
    collect_features: bool = True,
    collect_gate: bool = True,
) -> Dict:
    """
    Comprehensive evaluation on the test set.

    Collects metrics, predictions, feature vectors (for t-SNE),
    and gate values (for gate activation analysis).

    Args:
        model: Trained model.
        test_loader: Test DataLoader.
        cfg: ProjectConfig.
        num_classes: Number of classes.
        class_weights: For loss computation.
        label_names: Human-readable class names.
        device: Torch device.
        collect_features: Whether to collect feature vectors.
        collect_gate: Whether to collect gate values.

    Returns:
        Dictionary with all metrics, predictions, and collected features.
    """
    tcfg = cfg.train
    if device is None:
        device = torch.device(tcfg.device if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    loss_fn = get_loss_fn(
        tcfg.loss_fn, num_classes,
        class_weights=class_weights,
        focal_gamma=tcfg.focal_gamma,
        label_smoothing=tcfg.label_smoothing
    )

    # Determine if this model supports feature/gate extraction
    is_dual_branch = hasattr(model, "fusion_type")

    results = evaluate(
        model, test_loader, loss_fn, device,
        use_fp16=tcfg.fp16,
        return_features=collect_features and is_dual_branch,
        return_gate=collect_gate and is_dual_branch
    )

    # Log detailed results
    logger.info(f"\n{'='*60}")
    logger.info("TEST SET RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"  Loss:        {results['loss']:.4f}")
    logger.info(f"  Macro F1:    {results['macro_f1']:.4f}")
    logger.info(f"  Weighted F1: {results['weighted_f1']:.4f}")
    logger.info(f"  Accuracy:    {results['accuracy']:.4f}")

    if label_names:
        logger.info(f"\n  Per-class F1:")
        for i, name in enumerate(label_names):
            logger.info(
                f"    {name:15s}: P={results['precision_per_class'][i]:.3f} "
                f"R={results['recall_per_class'][i]:.3f} "
                f"F1={results['f1_per_class'][i]:.3f} "
                f"(n={results['support_per_class'][i]})"
            )

    logger.info(f"\n  Classification Report:\n{results['classification_report']}")

    return results


# ============================================================================
#  Error Analysis
# ============================================================================

def run_error_analysis(
    predictions: np.ndarray,
    labels: np.ndarray,
    texts: List[str],
    label_names: List[str],
    max_examples: int = 30,
    save_path: Optional[str] = None,
) -> Dict:
    """
    Analyze misclassified examples and categorize error types.

    Error categories:
        - ambiguous: Text could reasonably be multiple emotions
        - sarcasm_irony: Sarcastic/ironic tone causes misclassification
        - negation: Negation not properly handled ("not happy" → happy)
        - short_text: Very short text with insufficient signal
        - label_noise: Likely mislabeled example (annotator error)
        - confusion_pair: Commonly confused emotion pair

    Args:
        predictions: Model predictions (integer indices).
        labels: Ground truth (integer indices).
        texts: Original text strings.
        label_names: Class names.
        max_examples: Max misclassified examples to analyze.
        save_path: Optional path to save analysis JSON.

    Returns:
        Dictionary with error analysis results.
    """
    misclassified_idx = np.where(predictions != labels)[0]
    total_errors = len(misclassified_idx)
    total_samples = len(labels)

    logger.info(f"\nError Analysis: {total_errors}/{total_samples} "
                f"misclassified ({total_errors/total_samples*100:.1f}%)")

    # Analyze confusion pairs (which emotions get mixed up most)
    confusion_pairs = Counter()
    for idx in misclassified_idx:
        true_label = label_names[labels[idx]]
        pred_label = label_names[predictions[idx]]
        pair = f"{true_label} → {pred_label}"
        confusion_pairs[pair] += 1

    logger.info(f"\n  Top confusion pairs:")
    for pair, count in confusion_pairs.most_common(10):
        logger.info(f"    {pair}: {count}")

    # Sample misclassified examples for qualitative analysis
    sample_idx = np.random.choice(
        misclassified_idx,
        size=min(max_examples, total_errors),
        replace=False
    )

    # Simple heuristic categorization of errors
    error_examples = []
    category_counts = defaultdict(int)

    negation_words = {"not", "no", "never", "neither", "nor", "n't", "dont", "cant", "wont"}
    sarcasm_markers = {"yeah right", "oh great", "sure", "totally", "obviously",
                       "of course", "wow", "brilliant", "thanks a lot"}

    for idx in sample_idx:
        text = texts[idx]
        true_label = label_names[labels[idx]]
        pred_label = label_names[predictions[idx]]
        words = set(text.lower().split())

        # Heuristic categorization
        category = "other"
        if len(text.split()) <= 5:
            category = "short_text"
        elif words & negation_words:
            category = "negation"
        elif any(marker in text.lower() for marker in sarcasm_markers):
            category = "sarcasm_irony"
        elif len(text.split()) <= 10:
            category = "ambiguous"

        category_counts[category] += 1

        error_examples.append({
            "text": text,
            "true_label": true_label,
            "predicted_label": pred_label,
            "category": category,
        })

    logger.info(f"\n  Error categories (from {len(sample_idx)} samples):")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        logger.info(f"    {cat}: {count}")

    # Print some examples
    logger.info(f"\n  Sample misclassifications:")
    for ex in error_examples[:10]:
        logger.info(f"    [{ex['category']}] True: {ex['true_label']}, "
                     f"Pred: {ex['predicted_label']}")
        logger.info(f"      \"{ex['text'][:100]}\"")

    analysis = {
        "total_errors": total_errors,
        "total_samples": total_samples,
        "error_rate": total_errors / total_samples,
        "confusion_pairs": dict(confusion_pairs),
        "category_counts": dict(category_counts),
        "error_examples": error_examples,
    }

    if save_path:
        with open(save_path, "w") as f:
            json.dump(analysis, f, indent=2)
        logger.info(f"\n  Error analysis saved to: {save_path}")

    return analysis


# ============================================================================
#  Attention Weight Extraction
# ============================================================================

@torch.no_grad()
def extract_attention_weights(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    layer: int = -1,
) -> np.ndarray:
    """
    Extract BERT attention weights for visualization.

    Returns attention weights from a specified layer (default: last layer).
    These can be used to create attention heatmaps showing what words
    the model focuses on.

    Args:
        model: Model with a BERT encoder.
        input_ids: [1, seq_len] token indices (single example).
        attention_mask: [1, seq_len] mask.
        device: Torch device.
        layer: Which transformer layer's attention to extract (-1 = last).

    Returns:
        Attention weights of shape [num_heads, seq_len, seq_len].
    """
    model.eval()
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # Get the encoder (works for all BERT-based models in our codebase)
    encoder = getattr(model, "encoder", model)

    outputs = encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True
    )

    # outputs.attentions is a tuple of [batch, heads, seq, seq] per layer
    attentions = outputs.attentions
    attn_layer = attentions[layer]  # [1, heads, seq, seq]

    return attn_layer.squeeze(0).cpu().numpy()  # [heads, seq, seq]
