"""
utils.py — Shared utilities: seeding, logging, loss functions, metrics.

This module is imported by train.py, evaluate.py, and others.
"""

import os
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    f1_score, accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
#  Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seed for full reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms (slight perf hit but full reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
#  Logging
# ---------------------------------------------------------------------------

def setup_logger(log_dir: str, name: str = "project") -> logging.Logger:
    """Create a logger that writes to both console and file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Reset handlers to avoid duplicates

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
#  Loss Functions
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) for handling class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    When gamma > 0, the loss down-weights easy (well-classified) examples
    and focuses on hard (misclassified) examples. This is crucial for
    CROWDFLOWER's highly imbalanced emotion classes.

    Args:
        gamma: Focusing parameter. gamma=0 reduces to standard CE.
               gamma=2 is the recommended default.
        alpha: Per-class weights (tensor of shape [num_classes]).
               If None, all classes weighted equally.
        reduction: 'mean', 'sum', or 'none'.
    """
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean"
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model outputs, shape [batch_size, num_classes].
            targets: Ground truth labels, shape [batch_size] (integer indices).
        """
        probs = F.softmax(logits, dim=1)
        # Gather the probability of the true class for each sample
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
        p_t = (probs * targets_one_hot).sum(dim=1)  # shape: [batch_size]

        # Focal modulating factor
        focal_weight = (1.0 - p_t) ** self.gamma

        # Standard cross-entropy component
        ce_loss = F.cross_entropy(logits, targets, reduction="none")

        # Apply focal weight
        loss = focal_weight * ce_loss

        # Apply class-level alpha weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy with label smoothing to combat label noise.

    Instead of one-hot targets [0, 0, 1, 0], uses smoothed targets
    like [0.033, 0.033, 0.9, 0.033] with epsilon=0.1.
    Helps with noisy crowdsourced labels (CROWDFLOWER).
    """
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        # Smooth target distribution
        smooth_targets = torch.full_like(log_probs, self.smoothing / (self.num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)
        loss = (-smooth_targets * log_probs).sum(dim=1).mean()
        return loss


def get_loss_fn(
    loss_type: str,
    num_classes: int,
    class_weights: Optional[torch.Tensor] = None,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.1
) -> nn.Module:
    """Factory function for loss functions."""
    if loss_type == "focal":
        return FocalLoss(gamma=focal_gamma, alpha=class_weights)
    elif loss_type == "label_smoothing":
        return LabelSmoothingLoss(num_classes=num_classes, smoothing=label_smoothing)
    elif loss_type == "cross_entropy":
        if class_weights is not None:
            return nn.CrossEntropyLoss(weight=class_weights)
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_type}")


# ---------------------------------------------------------------------------
#  Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Optional[List[str]] = None
) -> Dict:
    """
    Compute all evaluation metrics for emotion classification.

    Returns:
        Dictionary with macro_f1, weighted_f1, accuracy,
        per-class metrics, and classification report string.
    """
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    report_str = classification_report(
        y_true, y_pred,
        target_names=label_names,
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred)

    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": accuracy,
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1_per_class,
        "support_per_class": support,
        "confusion_matrix": cm,
        "classification_report": report_str,
    }


# ---------------------------------------------------------------------------
#  Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Stop training when a monitored metric has stopped improving.

    Args:
        patience: Number of epochs with no improvement before stopping.
        metric_name: Name of the metric to monitor.
        mode: 'max' (higher is better) or 'min' (lower is better).
        min_delta: Minimum change to qualify as an improvement.
    """
    def __init__(
        self,
        patience: int = 3,
        metric_name: str = "macro_f1",
        mode: str = "max",
        min_delta: float = 0.0
    ):
        self.patience = patience
        self.metric_name = metric_name
        self.mode = mode
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """
        Update with current epoch score. Returns True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > (self.best_score + self.min_delta)
        else:
            improved = score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# ---------------------------------------------------------------------------
#  Misc Helpers
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count trainable and total parameters.

    Returns:
        (trainable_params, total_params)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def get_device(preferred: str = "cuda") -> torch.device:
    """Get the best available device."""
    if preferred == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device
