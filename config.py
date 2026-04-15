"""
config.py — Central configuration for the Dual-Branch Local–Global Fusion project.

All hyperparameters, paths, and experiment settings live here.
Nothing is hardcoded elsewhere. Change this file to adjust any experiment.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class PathConfig:
    """All filesystem paths."""
    base_dir: str = "."
    data_dir: str = "data"
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    figures_dir: str = "figures"

    def __post_init__(self):
        """Create directories if they don't exist."""
        for attr in ["data_dir", "output_dir", "checkpoint_dir", "log_dir", "figures_dir"]:
            path = os.path.join(self.base_dir, getattr(self, attr))
            setattr(self, attr, path)
            os.makedirs(path, exist_ok=True)


@dataclass
class DataConfig:
    """Dataset and preprocessing settings."""
    # Dataset selection: "crowdflower", "wassa2017", or "both"
    dataset: str = "both"

    # CROWDFLOWER settings
    # Top-N emotion classes to keep (merge/drop rare ones for cleaner experiments)
    crowdflower_top_k_emotions: int = 6

    # Tokenizer
    max_seq_length: int = 128
    tokenizer_name: str = "bert-base-uncased"

    # Train/val/test split ratios
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Data augmentation (synonym replacement probability)
    augment: bool = False
    augment_prob: float = 0.1


@dataclass
class ModelConfig:
    """Architecture hyperparameters for all models."""
    # --- Shared encoder ---
    encoder_name: str = "bert-base-uncased"  # or "roberta-base", "distilbert-base-uncased"
    hidden_size: int = 768       # BERT hidden dimension
    freeze_embeddings: bool = True
    freeze_n_layers: int = 8     # Freeze first N transformer layers (of 12)

    # --- Local branch (CNN) ---
    cnn_kernel_sizes: Tuple[int, ...] = (2, 3, 4)
    cnn_num_filters: int = 128   # Output channels per kernel
    local_feature_dim: int = 384  # = len(kernel_sizes) * cnn_num_filters

    # --- Global branch ([CLS] MLP) ---
    global_hidden_dim: int = 384
    global_feature_dim: int = 384

    # --- Fusion ---
    # "gated", "concat", "average", "bilinear", "attention"
    fusion_type: str = "gated"
    fused_dim: int = 384

    # --- Classifier ---
    dropout: float = 0.3

    # --- TextCNN baseline (non-BERT) ---
    vocab_size: int = 30522  # Will be set dynamically
    embedding_dim: int = 300  # GloVe dimension
    textcnn_num_filters: int = 128
    textcnn_kernel_sizes: Tuple[int, ...] = (2, 3, 4)

    # --- BiLSTM baseline ---
    lstm_hidden_size: int = 256
    lstm_num_layers: int = 2
    lstm_bidirectional: bool = True


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    # Optimizer
    optimizer: str = "adamw"
    encoder_lr: float = 2e-5     # Learning rate for BERT parameters
    head_lr: float = 5e-4        # Learning rate for classification heads
    weight_decay: float = 0.01

    # Scheduler
    warmup_ratio: float = 0.1    # Fraction of total steps for linear warmup
    scheduler: str = "linear_warmup_decay"

    # Training loop
    epochs: int = 12
    batch_size: int = 32
    gradient_accumulation_steps: int = 1  # Increase if OOM
    max_grad_norm: float = 1.0

    # Early stopping
    patience: int = 4            # Stop after N epochs without val improvement
    metric_for_best: str = "macro_f1"  # Metric to monitor

    # Loss
    loss_fn: str = "focal"       # "focal" or "cross_entropy" or "label_smoothing"
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1

    # Mixed precision
    fp16: bool = True

    # Reproducibility
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])

    # Device
    device: str = "cuda"  # Will fallback to cpu if unavailable


@dataclass
class ExperimentConfig:
    """Which experiments to run."""
    # Experiment 1: Baseline comparison
    run_baselines: bool = True
    baselines: List[str] = field(default_factory=lambda: [
        "textcnn",
        "bilstm_attention",
        "vanilla_bert",
        "bert_cnn_local",
        "bert_cls_global",
    ])

    # Experiment 2: Ablation study
    run_ablation: bool = True
    ablation_configs: List[str] = field(default_factory=lambda: [
        "full_gated",       # Full model with gated fusion
        "local_only",       # Only local branch
        "global_only",      # Only global branch
        "concat_fusion",    # Both branches, simple concatenation
        "average_fusion",   # Both branches, average
    ])

    # Experiment 3: CNN kernel size analysis
    run_kernel_analysis: bool = True
    kernel_configs: List[Tuple[int, ...]] = field(default_factory=lambda: [
        (2, 3),
        (3, 4, 5),
        (2, 3, 4),     # Default
        (2, 3, 4, 5),
    ])

    # Experiment 4: Encoder comparison
    run_encoder_comparison: bool = True
    encoders: List[str] = field(default_factory=lambda: [
        "bert-base-uncased",
        "roberta-base",
        "distilbert-base-uncased",
    ])


@dataclass
class ProjectConfig:
    """Master config combining all sub-configs."""
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


def get_config(**overrides) -> ProjectConfig:
    """
    Create a ProjectConfig with optional overrides.

    Usage:
        cfg = get_config()
        cfg = get_config(train={"epochs": 5, "batch_size": 16})
    """
    cfg = ProjectConfig()

    for section_name, section_overrides in overrides.items():
        if hasattr(cfg, section_name) and isinstance(section_overrides, dict):
            section = getattr(cfg, section_name)
            for key, value in section_overrides.items():
                if hasattr(section, key):
                    setattr(section, key, value)

    return cfg
