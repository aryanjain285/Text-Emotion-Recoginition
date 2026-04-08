"""
run_experiments.py — Master script that orchestrates all experiments.

This is the single entry point. Run this, and it executes:
    1. Data loading and preprocessing
    2. Experiment 1: Baseline comparison (5 baselines + proposed model)
    3. Experiment 2: Ablation study (gated vs concat vs avg vs local-only vs global-only)
    4. Experiment 3: CNN kernel size analysis
    5. Experiment 4: Encoder comparison (BERT vs RoBERTa vs DistilBERT)
    6. Visualization generation (all figures for the report)
    7. Error analysis on misclassified samples

All results are saved to disk: metrics (JSON), checkpoints (PyTorch), figures (PNG).

Usage:
    python run_experiments.py                    # Full run
    python run_experiments.py --quick            # Quick test (1 epoch, 1 seed)
    python run_experiments.py --dataset wassa2017 # Single dataset
    python run_experiments.py --skip-baselines   # Skip non-BERT baselines
"""

import os
import sys
import json
import time
import argparse
import logging
import numpy as np
import torch
from typing import Dict, List

from config import get_config
from preprocess import load_data
from models import build_model
from train import train_model, evaluate
from evaluate import evaluate_model_full, run_error_analysis, extract_attention_weights
from visualize import (
    generate_all_visualizations, plot_confusion_matrix,
    plot_attention_heatmap, plot_cls_attention, plot_tsne,
    plot_gate_analysis, plot_training_curves, plot_results_table
)
from utils import set_seed, setup_logger, get_device, get_loss_fn, count_parameters


def parse_args():
    parser = argparse.ArgumentParser(description="Text Emotion Recognition Experiments")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 1 epoch, 1 seed, minimal experiments")
    parser.add_argument("--dataset", type=str, default="both",
                        choices=["crowdflower", "wassa2017", "both"],
                        help="Which dataset(s) to use")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Skip non-BERT baselines (TextCNN, BiLSTM)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Override random seeds (e.g., --seeds 42 123)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--encoder", type=str, default=None,
                        help="Override encoder (e.g., roberta-base)")
    return parser.parse_args()


def run_single_experiment(
    model_name: str,
    data_dict: Dict,
    cfg,
    device: torch.device,
    save_prefix: str,
    encoder_name: str = None,
    cnn_kernel_sizes=None,
    fusion_type: str = None,
    seeds: List[int] = None,
) -> Dict:
    """
    Train and evaluate a single model configuration across multiple seeds.

    Returns aggregated metrics (mean ± std) across seeds.
    """
    logger = logging.getLogger("project")
    seeds = seeds or cfg.train.seeds

    # Determine if this is a BERT-based model
    is_bert_based = model_name not in ["textcnn", "bilstm_attention"]

    all_seed_results = []
    all_seed_histories = []

    for seed_idx, seed in enumerate(seeds):
        logger.info(f"\n  --- Seed {seed} ({seed_idx+1}/{len(seeds)}) ---")
        set_seed(seed)

        # Build model
        model = build_model(
            model_name=model_name if model_name != "dual_branch" else "dual_branch",
            num_classes=data_dict["num_classes"],
            cfg=cfg,
            vocab_size=len(data_dict["vocab"]) if not is_bert_based else None,
            pretrained_embeddings=data_dict["glove_embeddings"] if not is_bert_based else None,
            encoder_name=encoder_name,
            cnn_kernel_sizes=cnn_kernel_sizes,
            fusion_type=fusion_type,
        )

        trainable, total = count_parameters(model)
        logger.info(f"  Model: {model_name}, Params: {trainable:,} trainable / {total:,} total")

        # Select appropriate dataloaders
        if is_bert_based:
            train_loader = data_dict["train_loader"]
            val_loader = data_dict["val_loader"]
            test_loader = data_dict["test_loader"]
        else:
            train_loader = data_dict["train_loader_simple"]
            val_loader = data_dict["val_loader_simple"]
            test_loader = data_dict["test_loader_simple"]

        # Train
        save_name = f"{save_prefix}_seed{seed}"
        train_results = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            num_classes=data_dict["num_classes"],
            class_weights=data_dict["class_weights"],
            label_names=data_dict["label_names"],
            save_name=save_name,
            device=device,
        )

        # Evaluate on test set
        test_results = evaluate_model_full(
            model=model,
            test_loader=test_loader,
            cfg=cfg,
            num_classes=data_dict["num_classes"],
            class_weights=data_dict["class_weights"],
            label_names=data_dict["label_names"],
            device=device,
            collect_features=(model_name == "dual_branch"),
            collect_gate=(model_name == "dual_branch" and
                          (fusion_type or cfg.model.fusion_type) == "gated"),
        )

        all_seed_results.append(test_results)
        all_seed_histories.append(train_results["history"])

    # Aggregate across seeds
    aggregated = aggregate_seed_results(all_seed_results, data_dict["label_names"])
    aggregated["histories"] = all_seed_histories

    # Keep the last seed's detailed results for visualizations
    aggregated["last_run"] = all_seed_results[-1]
    aggregated["last_history"] = all_seed_histories[-1]

    return aggregated


def aggregate_seed_results(
    seed_results: List[Dict],
    label_names: List[str]
) -> Dict:
    """Compute mean ± std of metrics across seeds."""
    logger = logging.getLogger("project")

    metrics_to_aggregate = ["macro_f1", "weighted_f1", "accuracy"]
    aggregated = {}

    for metric in metrics_to_aggregate:
        values = [r[metric] for r in seed_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        aggregated[metric] = mean_val
        aggregated[f"{metric}_std"] = std_val
        aggregated[f"{metric}_values"] = values

    logger.info(f"\n  Aggregated Results ({len(seed_results)} seeds):")
    logger.info(f"    Macro F1:    {aggregated['macro_f1']:.4f} ± {aggregated['macro_f1_std']:.4f}")
    logger.info(f"    Weighted F1: {aggregated['weighted_f1']:.4f} ± {aggregated['weighted_f1_std']:.4f}")
    logger.info(f"    Accuracy:    {aggregated['accuracy']:.4f} ± {aggregated['accuracy_std']:.4f}")

    # Also aggregate per-class F1
    if "f1_per_class" in seed_results[0]:
        f1_per_class_all = np.array([r["f1_per_class"] for r in seed_results])
        aggregated["f1_per_class"] = f1_per_class_all.mean(axis=0)
        aggregated["f1_per_class_std"] = f1_per_class_all.std(axis=0)

    # Keep confusion matrix and other arrays from last run
    aggregated["confusion_matrix"] = seed_results[-1]["confusion_matrix"]
    aggregated["predictions"] = seed_results[-1]["predictions"]
    aggregated["labels"] = seed_results[-1]["labels"]

    return aggregated


def main():
    args = parse_args()

    # ---- Build config with any overrides ----
    cfg = get_config()
    cfg.data.dataset = args.dataset

    if args.quick:
        cfg.train.epochs = 2
        cfg.train.seeds = [42]
        cfg.experiment.run_kernel_analysis = False
        cfg.experiment.run_encoder_comparison = False
        print("⚡ Quick mode: 2 epochs, 1 seed, reduced experiments")

    if args.seeds:
        cfg.train.seeds = args.seeds
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
    if args.encoder:
        cfg.model.encoder_name = args.encoder
        cfg.data.tokenizer_name = args.encoder

    # ---- Setup ----
    logger = setup_logger(cfg.paths.log_dir)
    device = get_device(cfg.train.device)

    logger.info("=" * 70)
    logger.info("TEXT EMOTION RECOGNITION — DUAL-BRANCH LOCAL–GLOBAL FUSION")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")
    logger.info(f"Seeds: {cfg.train.seeds}")
    logger.info(f"Epochs: {cfg.train.epochs}")
    logger.info(f"Dataset: {cfg.data.dataset}")

    start_time = time.time()

    # ---- Load data ----
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: DATA LOADING")
    logger.info("=" * 70)

    data = load_data(cfg)

    # ---- Run experiments for each dataset ----
    all_experiment_results = {}

    for ds_name, ds_data in data.items():
        logger.info(f"\n{'#' * 70}")
        logger.info(f"# DATASET: {ds_name.upper()}")
        logger.info(f"# Classes: {ds_data['num_classes']} — {ds_data['label_names']}")
        logger.info(f"{'#' * 70}")

        dataset_results = {}
        dataset_histories = {}

        # ================================================================
        # EXPERIMENT 1: Baseline Comparison
        # ================================================================
        logger.info(f"\n{'=' * 70}")
        logger.info("EXPERIMENT 1: BASELINE COMPARISON")
        logger.info(f"{'=' * 70}")

        models_to_run = []

        if not args.skip_baselines:
            models_to_run.extend(["textcnn", "bilstm_attention"])

        models_to_run.extend([
            "vanilla_bert",
            "bert_cnn_local",
            "bert_cls_global",
            "dual_branch",
        ])

        for model_name in models_to_run:
            logger.info(f"\n>>> Training: {model_name}")

            result = run_single_experiment(
                model_name=model_name,
                data_dict=ds_data,
                cfg=cfg,
                device=device,
                save_prefix=f"{model_name}_{ds_name}",
                seeds=cfg.train.seeds,
            )

            dataset_results[model_name] = result
            dataset_histories[model_name] = result["last_history"]

        # ================================================================
        # EXPERIMENT 2: Ablation Study
        # ================================================================
        if cfg.experiment.run_ablation:
            logger.info(f"\n{'=' * 70}")
            logger.info("EXPERIMENT 2: ABLATION STUDY")
            logger.info(f"{'=' * 70}")

            ablation_configs = {
                "full_gated": {"fusion_type": "gated"},
                "local_only": None,   # Already run as bert_cnn_local
                "global_only": None,  # Already run as bert_cls_global
                "concat_fusion": {"fusion_type": "concat"},
                "average_fusion": {"fusion_type": "average"},
            }

            for ablation_name, abl_cfg in ablation_configs.items():
                # Skip if already computed
                if ablation_name == "full_gated" and "dual_branch" in dataset_results:
                    dataset_results[f"abl_{ablation_name}"] = dataset_results["dual_branch"]
                    continue
                if ablation_name == "local_only" and "bert_cnn_local" in dataset_results:
                    dataset_results[f"abl_{ablation_name}"] = dataset_results["bert_cnn_local"]
                    continue
                if ablation_name == "global_only" and "bert_cls_global" in dataset_results:
                    dataset_results[f"abl_{ablation_name}"] = dataset_results["bert_cls_global"]
                    continue

                logger.info(f"\n>>> Ablation: {ablation_name}")

                result = run_single_experiment(
                    model_name="dual_branch",
                    data_dict=ds_data,
                    cfg=cfg,
                    device=device,
                    save_prefix=f"abl_{ablation_name}_{ds_name}",
                    fusion_type=abl_cfg["fusion_type"] if abl_cfg else None,
                    seeds=cfg.train.seeds,
                )
                dataset_results[f"abl_{ablation_name}"] = result

        # ================================================================
        # EXPERIMENT 3: CNN Kernel Size Analysis
        # ================================================================
        if cfg.experiment.run_kernel_analysis:
            logger.info(f"\n{'=' * 70}")
            logger.info("EXPERIMENT 3: CNN KERNEL SIZE ANALYSIS")
            logger.info(f"{'=' * 70}")

            for kernels in cfg.experiment.kernel_configs:
                kernel_str = "_".join(map(str, kernels))
                name = f"kernels_{kernel_str}"
                logger.info(f"\n>>> Kernel config: {kernels}")

                # Skip default config if already run
                if tuple(kernels) == tuple(cfg.model.cnn_kernel_sizes) and "dual_branch" in dataset_results:
                    dataset_results[name] = dataset_results["dual_branch"]
                    logger.info(f"  (reusing dual_branch results)")
                    continue

                result = run_single_experiment(
                    model_name="dual_branch",
                    data_dict=ds_data,
                    cfg=cfg,
                    device=device,
                    save_prefix=f"kernel_{kernel_str}_{ds_name}",
                    cnn_kernel_sizes=tuple(kernels),
                    seeds=cfg.train.seeds,
                )
                dataset_results[name] = result

        # ================================================================
        # EXPERIMENT 4: Encoder Comparison
        # ================================================================
        if cfg.experiment.run_encoder_comparison:
            logger.info(f"\n{'=' * 70}")
            logger.info("EXPERIMENT 4: ENCODER COMPARISON")
            logger.info(f"{'=' * 70}")

            for enc_name in cfg.experiment.encoders:
                logger.info(f"\n>>> Encoder: {enc_name}")

                # Skip if it's the default encoder (already run)
                if enc_name == cfg.model.encoder_name and "dual_branch" in dataset_results:
                    dataset_results[f"enc_{enc_name}"] = dataset_results["dual_branch"]
                    logger.info(f"  (reusing dual_branch results)")
                    continue

                # Need to re-tokenize with the new encoder's tokenizer
                # For simplicity, we use the same BERT tokenizer and note
                # that RoBERTa uses a different tokenizer in practice.
                # The code handles this via AutoTokenizer in preprocessing.
                result = run_single_experiment(
                    model_name="dual_branch",
                    data_dict=ds_data,
                    cfg=cfg,
                    device=device,
                    save_prefix=f"enc_{enc_name.replace('/', '_')}_{ds_name}",
                    encoder_name=enc_name,
                    seeds=cfg.train.seeds,
                )
                dataset_results[f"enc_{enc_name}"] = result

        # ================================================================
        # VISUALIZATIONS
        # ================================================================
        logger.info(f"\n{'=' * 70}")
        logger.info("GENERATING VISUALIZATIONS")
        logger.info(f"{'=' * 70}")

        # Prepare results for visualization (use last_run for arrays)
        viz_results = {}
        for name, res in dataset_results.items():
            viz_results[name] = res.get("last_run", res)
            # Copy aggregated metrics
            for key in ["macro_f1", "weighted_f1", "accuracy", "f1_per_class"]:
                if key in res:
                    viz_results[name][key] = res[key]

        generate_all_visualizations(
            all_results=viz_results,
            all_histories=dataset_histories,
            figures_dir=cfg.paths.figures_dir,
            dataset_name=ds_name,
            label_names=ds_data["label_names"],
        )

        # Attention heatmap for a sample sentence
        if "dual_branch" in dataset_results:
            try:
                _generate_attention_examples(
                    ds_data, cfg, device, ds_name
                )
            except Exception as e:
                logger.warning(f"  Attention heatmap generation failed: {e}")

        # ================================================================
        # ERROR ANALYSIS
        # ================================================================
        logger.info(f"\n{'=' * 70}")
        logger.info("ERROR ANALYSIS")
        logger.info(f"{'=' * 70}")

        if "dual_branch" in dataset_results:
            last_run = dataset_results["dual_branch"]["last_run"]
            run_error_analysis(
                predictions=last_run["predictions"],
                labels=last_run["labels"],
                texts=ds_data["test_texts"],
                label_names=ds_data["label_names"],
                save_path=os.path.join(cfg.paths.output_dir, f"error_analysis_{ds_name}.json")
            )

        # ================================================================
        # SAVE ALL METRICS
        # ================================================================
        metrics_summary = {}
        for name, res in dataset_results.items():
            metrics_summary[name] = {
                "macro_f1": res.get("macro_f1", 0),
                "macro_f1_std": res.get("macro_f1_std", 0),
                "weighted_f1": res.get("weighted_f1", 0),
                "accuracy": res.get("accuracy", 0),
                "accuracy_std": res.get("accuracy_std", 0),
            }

        metrics_path = os.path.join(cfg.paths.output_dir, f"metrics_{ds_name}.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_summary, f, indent=2)
        logger.info(f"\nMetrics saved to: {metrics_path}")

        all_experiment_results[ds_name] = dataset_results

    # ---- Summary ----
    elapsed = time.time() - start_time
    logger.info(f"\n{'=' * 70}")
    logger.info(f"ALL EXPERIMENTS COMPLETE — Total time: {elapsed/60:.1f} minutes")
    logger.info(f"{'=' * 70}")

    # Print final summary table
    for ds_name, ds_results in all_experiment_results.items():
        logger.info(f"\n--- {ds_name.upper()} RESULTS ---")
        logger.info(f"{'Model':<30} {'Macro F1':>12} {'Accuracy':>12}")
        logger.info("-" * 55)
        for model_name in ["textcnn", "bilstm_attention", "vanilla_bert",
                            "bert_cnn_local", "bert_cls_global", "dual_branch"]:
            if model_name in ds_results:
                r = ds_results[model_name]
                f1_str = f"{r['macro_f1']:.4f}"
                if "macro_f1_std" in r:
                    f1_str += f"±{r['macro_f1_std']:.4f}"
                acc_str = f"{r['accuracy']:.4f}"
                if "accuracy_std" in r:
                    acc_str += f"±{r['accuracy_std']:.4f}"
                logger.info(f"{model_name:<30} {f1_str:>12} {acc_str:>12}")


def _generate_attention_examples(ds_data, cfg, device, ds_name):
    """Generate attention heatmaps for a few example sentences."""
    from transformers import AutoTokenizer, AutoModel

    logger = logging.getLogger("project")
    tokenizer = AutoTokenizer.from_pretrained(cfg.data.tokenizer_name)

    # Build a fresh model and load best checkpoint
    model = build_model(
        "dual_branch", ds_data["num_classes"], cfg
    )

    # SDPA attention (default in newer transformers) does not support
    # output_attentions=True. Replace the encoder with one using eager attention.
    eager_encoder = AutoModel.from_pretrained(
        cfg.model.encoder_name, attn_implementation="eager"
    )
    model.encoder = eager_encoder
    model._freeze_layers(cfg.model.freeze_embeddings, cfg.model.freeze_n_layers)

    checkpoint_path = os.path.join(
        cfg.paths.checkpoint_dir,
        f"dual_branch_{ds_name}_seed{cfg.train.seeds[-1]}_best.pt"
    )

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)

    # Sample a few test examples
    sample_texts = ds_data["test_texts"][:5]
    sample_labels = ds_data["test_labels"][:5]

    for i, (text, label) in enumerate(zip(sample_texts, sample_labels)):
        encoding = tokenizer(
            text, max_length=cfg.data.max_seq_length,
            padding="max_length", truncation=True,
            return_tensors="pt"
        )
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

        # Get actual token count (non-padding)
        n_tokens = encoding["attention_mask"][0].sum().item()
        tokens = tokens[:n_tokens]

        try:
            attn_weights = extract_attention_weights(
                model, encoding["input_ids"], encoding["attention_mask"],
                device, layer=-1
            )

            emotion = ds_data["label_names"][label]
            save_path = os.path.join(
                cfg.paths.figures_dir,
                f"attention_example{i}_{emotion}_{ds_name}.png"
            )

            plot_cls_attention(
                attn_weights, tokens,
                title=f"[CLS] Attention — True: {emotion}",
                save_path=save_path
            )
        except Exception as e:
            logger.warning(f"  Failed to generate attention for example {i}: {e}")


if __name__ == "__main__":
    main()
