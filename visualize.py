"""
visualize.py — All visualizations for the report.

Generates:
    1. Confusion matrices (per dataset, best model vs baseline)
    2. Attention heatmaps (BERT attention on sample sentences)
    3. Gate activation analysis (distribution of gate values)
    4. t-SNE embeddings (fused representation colored by emotion)
    5. Training curves (loss and F1 over epochs)
    6. Per-class performance bar charts
    7. Results comparison table as figure

Usage:
    from visualize import (
        plot_confusion_matrix, plot_attention_heatmap,
        plot_gate_analysis, plot_tsne, plot_training_curves
    )
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (works in Colab and headless)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.manifold import TSNE
from typing import Dict, List, Optional, Tuple

# Consistent style
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "figure.figsize": (8, 6),
})


# ============================================================================
#  1. Confusion Matrix
# ============================================================================

def plot_confusion_matrix(
    cm: np.ndarray,
    label_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    normalize: bool = True,
    figsize: Tuple[int, int] = (8, 6)
) -> None:
    """
    Plot a confusion matrix heatmap.

    Args:
        cm: Confusion matrix array [num_classes, num_classes].
        label_names: Class names for axis labels.
        title: Plot title.
        save_path: Path to save the figure.
        normalize: If True, show percentages instead of counts.
        figsize: Figure size.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if normalize:
        # Normalize by true label (row-wise) to show recall percentages
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".1%"
        # Format annotations: show both percentage and count
        annot = np.array([
            [f"{cm_display[i,j]:.0%}\n({cm[i,j]})"
             for j in range(cm.shape[1])]
            for i in range(cm.shape[0])
        ])
    else:
        cm_display = cm
        fmt = "d"
        annot = cm.astype(str)

    sns.heatmap(
        cm_display,
        annot=annot if normalize else True,
        fmt="" if normalize else "d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
        linewidths=0.5,
        linecolor="gray"
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"  Saved: {save_path}")
    plt.close(fig)


# ============================================================================
#  2. Attention Heatmap
# ============================================================================

def plot_attention_heatmap(
    attention_weights: np.ndarray,
    tokens: List[str],
    title: str = "BERT Attention Weights",
    save_path: Optional[str] = None,
    head: int = 0,
    max_tokens: int = 40,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot BERT attention weights as a heatmap.

    Shows which tokens the model attends to for a given input.
    Useful for understanding if emotionally relevant words get
    higher attention weights.

    Args:
        attention_weights: Shape [num_heads, seq_len, seq_len].
        tokens: List of token strings.
        title: Plot title.
        save_path: Path to save.
        head: Which attention head to visualize (or -1 for mean).
        max_tokens: Maximum tokens to display.
    """
    tokens = tokens[:max_tokens]
    n = len(tokens)

    if head == -1:
        # Average over all heads
        attn = attention_weights.mean(axis=0)[:n, :n]
        head_label = "mean"
    else:
        attn = attention_weights[head][:n, :n]
        head_label = f"head {head}"

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        attn,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="YlOrRd",
        ax=ax,
        square=True
    )

    ax.set_title(f"{title} ({head_label})")
    ax.set_xlabel("Key (attended to)")
    ax.set_ylabel("Query (attending from)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_cls_attention(
    attention_weights: np.ndarray,
    tokens: List[str],
    title: str = "[CLS] Attention Distribution",
    save_path: Optional[str] = None,
    max_tokens: int = 40,
) -> None:
    """
    Bar chart of [CLS] token's attention to all other tokens.

    This directly shows what the model "looks at" when making
    the classification decision through the [CLS] representation.
    """
    tokens = tokens[:max_tokens]
    n = len(tokens)

    # Average [CLS] attention over all heads: row 0 (CLS query)
    cls_attn = attention_weights.mean(axis=0)[0, :n]

    fig, ax = plt.subplots(figsize=(max(8, n * 0.3), 4))

    colors = plt.cm.YlOrRd(cls_attn / cls_attn.max())
    bars = ax.bar(range(n), cls_attn, color=colors, edgecolor="gray", linewidth=0.5)

    ax.set_xticks(range(n))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Attention Weight")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"  Saved: {save_path}")
    plt.close(fig)


# ============================================================================
#  3. Gate Activation Analysis
# ============================================================================

def plot_gate_analysis(
    gate_values: np.ndarray,
    labels: np.ndarray,
    label_names: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5)
) -> None:
    """
    Analyze and visualize gate values from the gated fusion layer.

    Gate values near 1.0 → model relies on LOCAL (word-level) features.
    Gate values near 0.0 → model relies on GLOBAL (sentence-level) features.

    Creates two subplots:
        1. Overall distribution of mean gate values per sample
        2. Per-emotion distribution (box plot) — shows which emotions
           rely more on local vs. global context

    Args:
        gate_values: Shape [num_samples, feature_dim].
        labels: Shape [num_samples] integer labels.
        label_names: Class names.
        save_path: Path to save.
    """
    # Mean gate value per sample (across feature dimensions)
    mean_gates = gate_values.mean(axis=1)  # [num_samples]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # (a) Overall distribution
    axes[0].hist(mean_gates, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].axvline(x=0.5, color="red", linestyle="--", alpha=0.7, label="Equal weight")
    axes[0].set_xlabel("Mean Gate Value")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Gate Value Distribution\n(1=Local, 0=Global)")
    axes[0].legend()

    # (b) Per-emotion box plot
    gate_by_emotion = []
    emotion_labels_for_plot = []
    for i, name in enumerate(label_names):
        mask = labels == i
        if mask.sum() > 0:
            gate_by_emotion.append(mean_gates[mask])
            emotion_labels_for_plot.append(name)

    bp = axes[1].boxplot(
        gate_by_emotion,
        labels=emotion_labels_for_plot,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2)
    )

    colors = plt.cm.Set2(np.linspace(0, 1, len(emotion_labels_for_plot)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    axes[1].set_ylabel("Mean Gate Value")
    axes[1].set_title("Gate Values by Emotion")
    axes[1].axhline(y=0.5, color="red", linestyle="--", alpha=0.5)
    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"  Saved: {save_path}")
    plt.close(fig)


# ============================================================================
#  4. t-SNE Visualization
# ============================================================================

def plot_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    label_names: List[str],
    title: str = "t-SNE of Fused Representations",
    save_path: Optional[str] = None,
    perplexity: float = 30.0,
    max_samples: int = 2000,
    figsize: Tuple[int, int] = (8, 6)
) -> None:
    """
    t-SNE visualization of learned representations.

    Projects high-dimensional feature vectors to 2D to visualize
    how well the model separates different emotion classes.

    Good separability (distinct clusters) = model learned useful features.
    Overlapping clusters = emotions the model confuses.

    Args:
        features: Shape [num_samples, feature_dim].
        labels: Shape [num_samples] integer labels.
        label_names: Class names.
        title: Plot title.
        save_path: Path to save.
        perplexity: t-SNE perplexity parameter.
        max_samples: Subsample if dataset is large (t-SNE is O(n²)).
    """
    if len(features) > max_samples:
        idx = np.random.choice(len(features), max_samples, replace=False)
        features = features[idx]
        labels = labels[idx]

    print(f"  Running t-SNE on {len(features)} samples...")
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(features) - 1),
        random_state=42,
        n_iter=1000,
        init="pca",
        learning_rate="auto"
    )
    embeddings_2d = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=figsize)

    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label_idx in enumerate(unique_labels):
        mask = labels == label_idx
        name = label_names[label_idx] if label_idx < len(label_names) else str(label_idx)
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[i]],
            label=name,
            alpha=0.6,
            s=15,
            edgecolors="none"
        )

    ax.legend(fontsize=9, markerscale=2, loc="best")
    ax.set_title(title)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"  Saved: {save_path}")
    plt.close(fig)


# ============================================================================
#  5. Training Curves
# ============================================================================

def plot_training_curves(
    histories: Dict[str, Dict],
    title: str = "Training Curves",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> None:
    """
    Plot training/validation loss and F1 curves for multiple models.

    Overlays curves from different models for direct comparison.

    Args:
        histories: Dict of {model_name: history_dict}.
                   Each history_dict has keys: train_loss, val_loss,
                   train_macro_f1, val_macro_f1.
        title: Overall figure title.
        save_path: Path to save.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))

    for idx, (name, hist) in enumerate(histories.items()):
        epochs = range(1, len(hist["train_loss"]) + 1)
        color = colors[idx]

        # Loss curves
        axes[0].plot(epochs, hist["train_loss"], '--', color=color, alpha=0.5)
        axes[0].plot(epochs, hist["val_loss"], '-', color=color, label=name)

        # F1 curves
        axes[1].plot(epochs, hist["train_macro_f1"], '--', color=color, alpha=0.5)
        axes[1].plot(epochs, hist["val_macro_f1"], '-', color=color, label=name)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} — Loss (solid=val, dashed=train)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_title(f"{title} — Macro F1")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"  Saved: {save_path}")
    plt.close(fig)


# ============================================================================
#  6. Per-Class Performance Bar Chart
# ============================================================================

def plot_per_class_f1(
    results_dict: Dict[str, Dict],
    label_names: List[str],
    title: str = "Per-Class F1 by Model",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5)
) -> None:
    """
    Grouped bar chart comparing per-class F1 across models.

    Highlights which emotions each model handles well/poorly
    and whether the dual-branch model improves on specific emotions.

    Args:
        results_dict: Dict of {model_name: test_results}.
                      Each test_results has 'f1_per_class' array.
        label_names: Class names.
    """
    fig, ax = plt.subplots(figsize=figsize)

    n_classes = len(label_names)
    n_models = len(results_dict)
    bar_width = 0.8 / n_models
    x = np.arange(n_classes)

    colors = plt.cm.Set2(np.linspace(0, 1, n_models))

    for idx, (name, results) in enumerate(results_dict.items()):
        f1s = results["f1_per_class"][:n_classes]
        offset = (idx - n_models / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset, f1s, bar_width,
            label=name, color=colors[idx], edgecolor="white"
        )

    ax.set_xlabel("Emotion Class")
    ax.set_ylabel("F1 Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(label_names, rotation=30, ha="right")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"  Saved: {save_path}")
    plt.close(fig)


# ============================================================================
#  7. Results Summary Table (as figure)
# ============================================================================

def plot_results_table(
    results_dict: Dict[str, Dict],
    title: str = "Model Comparison",
    save_path: Optional[str] = None,
    metrics: List[str] = None,
    figsize: Tuple[int, int] = (10, 3)
) -> None:
    """
    Render the results comparison table as a publication-quality figure.

    Makes it easy to include in the report as an image.

    Args:
        results_dict: Dict of {model_name: test_results}.
        title: Table title.
        save_path: Path to save.
        metrics: Which metrics to include in columns.
    """
    if metrics is None:
        metrics = ["macro_f1", "weighted_f1", "accuracy"]

    # Build table data
    row_labels = []
    cell_data = []

    # Find best values for bolding
    best_vals = {m: -1 for m in metrics}
    for name, res in results_dict.items():
        for m in metrics:
            val = res.get(m, 0)
            if val > best_vals[m]:
                best_vals[m] = val

    for name, res in results_dict.items():
        row_labels.append(name)
        row = []
        for m in metrics:
            val = res.get(m, 0)
            if abs(val - best_vals[m]) < 1e-6:
                row.append(f"**{val:.4f}**")  # Best value marker
            else:
                row.append(f"{val:.4f}")
        cell_data.append(row)

    col_labels = [m.replace("_", " ").title() for m in metrics]

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    table = ax.table(
        cellText=cell_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Bold best values
    for i, row in enumerate(cell_data):
        for j, val in enumerate(row):
            if val.startswith("**"):
                table[i + 1, j].set_text_props(fontweight="bold")
                table[i + 1, j].get_text().set_text(val.replace("**", ""))

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"  Saved: {save_path}")
    plt.close(fig)


# ============================================================================
#  8. Master Visualization Runner
# ============================================================================

def generate_all_visualizations(
    all_results: Dict,
    all_histories: Dict,
    figures_dir: str,
    dataset_name: str,
    label_names: List[str],
) -> None:
    """
    Generate all figures for the report in a single call.

    Args:
        all_results: Dict of {model_name: test_results} for all models.
        all_histories: Dict of {model_name: training_history}.
        figures_dir: Directory to save figures.
        dataset_name: "crowdflower" or "wassa2017".
        label_names: Class names.
    """
    ds = dataset_name
    os.makedirs(figures_dir, exist_ok=True)

    print(f"\nGenerating visualizations for {ds}...")

    # 1. Confusion matrices for key models
    for model_name in ["vanilla_bert", "dual_branch"]:
        if model_name in all_results:
            res = all_results[model_name]
            plot_confusion_matrix(
                res["confusion_matrix"],
                label_names,
                title=f"{model_name} — {ds}",
                save_path=os.path.join(figures_dir, f"cm_{model_name}_{ds}.png")
            )

    # 2. Training curves (overlay all models)
    if all_histories:
        plot_training_curves(
            all_histories,
            title=f"Training Curves — {ds}",
            save_path=os.path.join(figures_dir, f"training_curves_{ds}.png")
        )

    # 3. Per-class F1 comparison
    if len(all_results) > 1:
        plot_per_class_f1(
            all_results,
            label_names,
            title=f"Per-Class F1 — {ds}",
            save_path=os.path.join(figures_dir, f"per_class_f1_{ds}.png")
        )

    # 4. Results table
    plot_results_table(
        all_results,
        title=f"Model Comparison — {ds}",
        save_path=os.path.join(figures_dir, f"results_table_{ds}.png")
    )

    # 5. t-SNE (if features available from dual branch)
    if "dual_branch" in all_results and "fused_features" in all_results["dual_branch"]:
        res = all_results["dual_branch"]
        plot_tsne(
            res["fused_features"],
            res["labels"],
            label_names,
            title=f"t-SNE of Dual-Branch Features — {ds}",
            save_path=os.path.join(figures_dir, f"tsne_dual_branch_{ds}.png")
        )

    # 6. Gate analysis (if available)
    if "dual_branch" in all_results and "gate_values" in all_results["dual_branch"]:
        res = all_results["dual_branch"]
        plot_gate_analysis(
            res["gate_values"],
            res["labels"],
            label_names,
            save_path=os.path.join(figures_dir, f"gate_analysis_{ds}.png")
        )

    print(f"  All visualizations saved to: {figures_dir}")
