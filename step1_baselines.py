"""
step1_baselines.py — Train non-BERT baselines: TextCNN and BiLSTM+Attention.
Fast (~5 min on T4). Can be run by a teammate.

Usage:
    python step1_baselines.py                    # Both datasets
    python step1_baselines.py --dataset wassa2017  # Single dataset

Outputs:
    outputs/step1_baselines_{dataset}.json
    checkpoints/textcnn_*, checkpoints/bilstm_*
"""

import os, sys, json, time, logging
import numpy as np, torch
from config import get_config
from preprocess import load_data
from models import build_model
from train import train_model
from evaluate import evaluate_model_full
from utils import set_seed, setup_logger, get_device, count_parameters
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="both", choices=["crowdflower", "wassa2017", "both"])
    args = parser.parse_args()

    cfg = get_config()
    cfg.data.dataset = args.dataset
    logger = setup_logger(cfg.paths.log_dir, "step1")
    device = get_device(cfg.train.device)

    logger.info("=" * 60)
    logger.info("STEP 1: NON-BERT BASELINES (TextCNN + BiLSTM)")
    logger.info("=" * 60)

    data = load_data(cfg)
    all_results = {}

    for ds_name, ds_data in data.items():
        logger.info(f"\n{'#'*60}\n# DATASET: {ds_name.upper()}\n{'#'*60}")
        ds_results = {}

        for model_name in ["textcnn", "bilstm_attention"]:
            logger.info(f"\n>>> {model_name}")
            seed_metrics = []

            for seed in cfg.train.seeds:
                logger.info(f"  --- Seed {seed} ---")
                set_seed(seed)

                model = build_model(
                    model_name, ds_data["num_classes"], cfg,
                    vocab_size=len(ds_data["vocab"]),
                    pretrained_embeddings=ds_data["glove_embeddings"],
                )

                train_results = train_model(
                    model, ds_data["train_loader_simple"], ds_data["val_loader_simple"],
                    cfg, ds_data["num_classes"], ds_data["class_weights"],
                    ds_data["label_names"], f"{model_name}_{ds_name}_s{seed}", device,
                )

                test_results = evaluate_model_full(
                    model, ds_data["test_loader_simple"], cfg,
                    ds_data["num_classes"], ds_data["class_weights"],
                    ds_data["label_names"], device,
                )

                seed_metrics.append({
                    "seed": seed,
                    "macro_f1": float(test_results["macro_f1"]),
                    "weighted_f1": float(test_results["weighted_f1"]),
                    "accuracy": float(test_results["accuracy"]),
                    "f1_per_class": test_results["f1_per_class"].tolist(),
                    "history": train_results["history"],
                })

            # Aggregate
            f1s = [s["macro_f1"] for s in seed_metrics]
            accs = [s["accuracy"] for s in seed_metrics]
            ds_results[model_name] = {
                "macro_f1": float(np.mean(f1s)),
                "macro_f1_std": float(np.std(f1s)),
                "accuracy": float(np.mean(accs)),
                "accuracy_std": float(np.std(accs)),
                "seeds": seed_metrics,
            }
            logger.info(f"  {model_name}: F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}")

        all_results[ds_name] = ds_results

        # Save per-dataset
        out_path = os.path.join(cfg.paths.output_dir, f"step1_baselines_{ds_name}.json")
        with open(out_path, "w") as f:
            json.dump(ds_results, f, indent=2)
        logger.info(f"\nSaved: {out_path}")

    logger.info("\n✅ Step 1 complete.")

if __name__ == "__main__":
    main()
