"""
train.py — Training pipeline for all models.

Handles:
    - Mixed-precision training (fp16) for GPU memory efficiency
    - Gradient accumulation for effective larger batch sizes
    - Early stopping based on validation macro-F1
    - Model checkpointing (saves best model)
    - Differential learning rates (lower LR for BERT, higher for heads)
    - Linear warmup + decay learning rate schedule
    - Per-epoch logging of loss and metrics

Usage:
    from train import train_model
    results = train_model(model, train_loader, val_loader, cfg, ...)
"""

import os
import time
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

# AMP compatibility: PyTorch 2.x moved GradScaler/autocast to torch.amp
try:
    from torch.amp import GradScaler, autocast
    _AMP_DEVICE_ARG = True  # New API requires device_type kwarg
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    _AMP_DEVICE_ARG = False  # Legacy API infers CUDA


def _autocast_ctx(device: torch.device):
    """Return the correct autocast context manager for the PyTorch version."""
    if _AMP_DEVICE_ARG:
        return autocast(device_type=device.type)
    return autocast()
from torch.optim.lr_scheduler import LambdaLR
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from utils import (
    compute_metrics, get_loss_fn, EarlyStopping,
    count_parameters, set_seed
)

logger = logging.getLogger("project")


# ============================================================================
#  Learning Rate Scheduler
# ============================================================================

def get_linear_warmup_decay_scheduler(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int
) -> LambdaLR:
    """
    Linear warmup followed by linear decay to 0.

    This is the standard schedule for BERT fine-tuning:
    - First 10% of steps: linearly increase LR from 0 to peak
    - Remaining 90%: linearly decrease LR from peak to 0

    This prevents destabilizing the pretrained weights during early training.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) /
            float(max(1, num_training_steps - num_warmup_steps))
        )
    return LambdaLR(optimizer, lr_lambda)


# ============================================================================
#  Optimizer with Differential Learning Rates
# ============================================================================

def build_optimizer(model: nn.Module, cfg) -> AdamW:
    """
    Create AdamW optimizer with differential learning rates.

    BERT/RoBERTa parameters get a lower LR (2e-5) because they are
    already pretrained — large updates would destroy learned representations.

    Classification heads get a higher LR (1e-3) because they are randomly
    initialized and need to learn faster.

    For non-BERT models (TextCNN, BiLSTM), all parameters get the head LR.
    """
    tcfg = cfg.train

    # Check if model has a BERT-like encoder
    has_encoder = hasattr(model, "encoder") and hasattr(model.encoder, "config")

    if has_encoder:
        # Separate encoder params from head params
        encoder_params = []
        head_params = []

        encoder_param_ids = set(id(p) for p in model.encoder.parameters())

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) in encoder_param_ids:
                encoder_params.append(param)
            else:
                head_params.append(param)

        param_groups = [
            {"params": encoder_params, "lr": tcfg.encoder_lr, "weight_decay": tcfg.weight_decay},
            {"params": head_params, "lr": tcfg.head_lr, "weight_decay": tcfg.weight_decay},
        ]
        logger.info(f"  Encoder params (lr={tcfg.encoder_lr}): "
                     f"{sum(p.numel() for p in encoder_params):,}")
        logger.info(f"  Head params (lr={tcfg.head_lr}): "
                     f"{sum(p.numel() for p in head_params):,}")
    else:
        # Non-BERT model: single parameter group
        param_groups = [
            {"params": [p for p in model.parameters() if p.requires_grad],
             "lr": tcfg.head_lr, "weight_decay": tcfg.weight_decay}
        ]

    optimizer = AdamW(param_groups, eps=1e-8)
    return optimizer


# ============================================================================
#  Single Epoch Training
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    scheduler,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: Optional[GradScaler],
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    use_fp16: bool = True
) -> Dict:
    """
    Train for one epoch. Returns average loss and predictions.

    Supports:
        - Mixed precision (fp16) via PyTorch AMP
        - Gradient accumulation for effective larger batches
        - Gradient clipping to prevent exploding gradients
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    num_batches = 0

    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc="  Training", leave=False)
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # Forward pass with optional mixed precision
        if use_fp16 and scaler is not None:
            with _autocast_ctx(device):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(logits, labels)
                loss = loss / gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
        else:
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)
            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1

        preds = logits.float().argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({"loss": f"{total_loss/num_batches:.4f}"})

    avg_loss = total_loss / num_batches
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))

    return {"loss": avg_loss, **metrics}


# ============================================================================
#  Evaluation
# ============================================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    loss_fn: nn.Module,
    device: torch.device,
    use_fp16: bool = True,
    return_features: bool = False,
    return_gate: bool = False
) -> Dict:
    """
    Evaluate model on a dataset. Returns loss, metrics, and optionally features.

    Args:
        return_features: If True, collect feature vectors for t-SNE visualization.
        return_gate: If True, collect gate values for analysis.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_local_feats = []
    all_global_feats = []
    all_fused_feats = []
    all_gate_values = []
    num_batches = 0

    for batch in tqdm(dataloader, desc="  Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        use_amp = use_fp16 and device.type == "cuda"

        if use_amp:
            with _autocast_ctx(device):
                if (return_features or return_gate) and hasattr(model, "fusion_type"):
                    logits, extras = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_features=return_features,
                        return_gate=return_gate
                    )
                else:
                    logits = model(input_ids=input_ids, attention_mask=attention_mask)
                    extras = {}
                loss = loss_fn(logits, labels)
        else:
            if (return_features or return_gate) and hasattr(model, "fusion_type"):
                logits, extras = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_features=return_features,
                    return_gate=return_gate
                )
            else:
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                extras = {}
            loss = loss_fn(logits, labels)

        total_loss += loss.item()
        num_batches += 1

        preds = logits.float().argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        if return_features and "fused_feat" in extras:
            all_local_feats.append(extras["local_feat"].cpu())
            all_global_feats.append(extras["global_feat"].cpu())
            all_fused_feats.append(extras["fused_feat"].cpu())
        if return_gate and "gate_values" in extras:
            all_gate_values.append(extras["gate_values"].cpu())

    avg_loss = total_loss / max(num_batches, 1)
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))

    result = {"loss": avg_loss, "predictions": np.array(all_preds),
              "labels": np.array(all_labels), **metrics}

    if all_fused_feats:
        result["local_features"] = torch.cat(all_local_feats, dim=0).numpy()
        result["global_features"] = torch.cat(all_global_feats, dim=0).numpy()
        result["fused_features"] = torch.cat(all_fused_feats, dim=0).numpy()
    if all_gate_values:
        result["gate_values"] = torch.cat(all_gate_values, dim=0).numpy()

    return result


# ============================================================================
#  Full Training Loop
# ============================================================================

def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    cfg,
    num_classes: int,
    class_weights: Optional[torch.Tensor] = None,
    label_names: Optional[List[str]] = None,
    save_name: str = "model",
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Complete training loop for a single model run.

    Orchestrates:
        1. Optimizer setup (differential LR)
        2. Scheduler setup (linear warmup + decay)
        3. Loss function (focal/CE/label-smoothing)
        4. Training loop with early stopping
        5. Checkpoint saving (best model on val macro-F1)
        6. Per-epoch metric logging

    Args:
        model: The model to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        cfg: ProjectConfig.
        num_classes: Number of classes.
        class_weights: Optional per-class weights for loss function.
        label_names: Class names for logging.
        save_name: Filename prefix for checkpoints.
        device: Torch device.

    Returns:
        Dictionary with training history and best metrics.
    """
    tcfg = cfg.train

    if device is None:
        device = torch.device(tcfg.device if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    trainable, total = count_parameters(model)
    logger.info(f"  Parameters: {trainable:,} trainable / {total:,} total")

    # ---- Setup ----
    optimizer = build_optimizer(model, cfg)

    total_steps = len(train_loader) * tcfg.epochs // tcfg.gradient_accumulation_steps
    warmup_steps = int(total_steps * tcfg.warmup_ratio)
    scheduler = get_linear_warmup_decay_scheduler(optimizer, warmup_steps, total_steps)

    loss_fn = get_loss_fn(
        tcfg.loss_fn, num_classes,
        class_weights=class_weights,
        focal_gamma=tcfg.focal_gamma,
        label_smoothing=tcfg.label_smoothing
    )

    if tcfg.fp16 and torch.cuda.is_available():
        scaler = GradScaler("cuda") if _AMP_DEVICE_ARG else GradScaler()
    else:
        scaler = None
    early_stopping = EarlyStopping(
        patience=tcfg.patience,
        metric_name=tcfg.metric_for_best,
        mode="max"
    )

    # ---- Training history ----
    history = {
        "train_loss": [], "val_loss": [],
        "train_macro_f1": [], "val_macro_f1": [],
        "train_accuracy": [], "val_accuracy": [],
    }
    best_val_f1 = 0.0
    best_epoch = 0
    checkpoint_path = os.path.join(cfg.paths.checkpoint_dir, f"{save_name}_best.pt")

    # ---- Training loop ----
    logger.info(f"  Training for up to {tcfg.epochs} epochs "
                f"(patience={tcfg.patience}, metric={tcfg.metric_for_best})")

    for epoch in range(tcfg.epochs):
        epoch_start = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, loss_fn,
            device, scaler, tcfg.max_grad_norm,
            tcfg.gradient_accumulation_steps, tcfg.fp16
        )

        # Validate
        val_metrics = evaluate(model, val_loader, loss_fn, device, tcfg.fp16)

        epoch_time = time.time() - epoch_start

        # Log
        logger.info(
            f"  Epoch {epoch+1}/{tcfg.epochs} ({epoch_time:.0f}s) | "
            f"Train Loss: {train_metrics['loss']:.4f} F1: {train_metrics['macro_f1']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} F1: {val_metrics['macro_f1']:.4f} "
            f"Acc: {val_metrics['accuracy']:.4f}"
        )

        # Update history
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_macro_f1"].append(train_metrics["macro_f1"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        # Save best model
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch + 1
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": best_val_f1,
                "val_accuracy": val_metrics["accuracy"],
            }, checkpoint_path)
            logger.info(f"  ★ New best model saved (F1: {best_val_f1:.4f})")

        # Early stopping check
        if early_stopping(val_metrics["macro_f1"]):
            logger.info(f"  Early stopping triggered at epoch {epoch+1}")
            break

    # ---- Load best model ----
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"  Loaded best model from epoch {best_epoch} "
                     f"(val F1: {best_val_f1:.4f})")

    return {
        "history": history,
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "checkpoint_path": checkpoint_path,
    }
