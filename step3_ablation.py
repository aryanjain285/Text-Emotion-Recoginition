"""
step3_ablation.py — Ablation study: test fusion strategies.
Tests concat and average fusion (gated + local-only + global-only come from step2).

Usage:
    python step3_ablation.py
    python step3_ablation.py --dataset crowdflower

Outputs:
    outputs/step3_ablation_{dataset}.json

Time: ~40 min on T4 (2 fusion types × 2 datasets × 3 seeds)
"""

import os, json, logging
import numpy as np, torch
from config import get_config
from preprocess import load_data
from models import build_model
from train import train_model
from evaluate import evaluate_model_full
from utils import set_seed, setup_logger, get_device
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="both", choices=["crowdflower", "wassa2017", "both"])
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()

    cfg = get_config()
    cfg.data.dataset = args.dataset
    if args.seeds:
        cfg.train.seeds = args.seeds

    logger = setup_logger(cfg.paths.log_dir, "step3")
    device = get_device(cfg.train.device)

    logger.info("=" * 60)
    logger.info("STEP 3: ABLATION STUDY (Fusion Strategies)")
    logger.info("=" * 60)

    data = load_data(cfg)

    # Only test concat and average — gated is from step2 (dual_branch),
    # local-only is from step2 (bert_cnn_local), global-only is (bert_cls_global)
    fusion_types = ["concat", "average"]

    for ds_name, ds_data in data.items():
        logger.info(f"\n{'#'*60}\n# DATASET: {ds_name.upper()}\n{'#'*60}")
        ds_results = {}

        for fusion in fusion_types:
            name = f"dual_branch_{fusion}"
            logger.info(f"\n>>> Fusion: {fusion}")
            seed_metrics = []

            for seed in cfg.train.seeds:
                logger.info(f"  --- Seed {seed} ---")
                set_seed(seed)

                model = build_model(
                    "dual_branch", ds_data["num_classes"], cfg,
                    fusion_type=fusion,
                )

                train_results = train_model(
                    model, ds_data["train_loader"], ds_data["val_loader"],
                    cfg, ds_data["num_classes"], ds_data["class_weights"],
                    ds_data["label_names"], f"abl_{fusion}_{ds_name}_s{seed}", device,
                )

                test_results = evaluate_model_full(
                    model, ds_data["test_loader"], cfg,
                    ds_data["num_classes"], ds_data["class_weights"],
                    ds_data["label_names"], device,
                )

                seed_metrics.append({
                    "seed": seed,
                    "macro_f1": float(test_results["macro_f1"]),
                    "weighted_f1": float(test_results["weighted_f1"]),
                    "accuracy": float(test_results["accuracy"]),
                    "f1_per_class": test_results["f1_per_class"].tolist(),
                    "confusion_matrix": test_results["confusion_matrix"].tolist(),
                    "history": train_results["history"],
                })

            f1s = [s["macro_f1"] for s in seed_metrics]
            accs = [s["accuracy"] for s in seed_metrics]
            ds_results[name] = {
                "macro_f1": float(np.mean(f1s)),
                "macro_f1_std": float(np.std(f1s)),
                "accuracy": float(np.mean(accs)),
                "accuracy_std": float(np.std(accs)),
                "f1_per_class": np.mean([s["f1_per_class"] for s in seed_metrics], axis=0).tolist(),
                "seeds": seed_metrics,
            }
            logger.info(f"  {name}: F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}")

        out_path = os.path.join(cfg.paths.output_dir, f"step3_ablation_{ds_name}.json")
        with open(out_path, "w") as f:
            json.dump(ds_results, f, indent=2)
        logger.info(f"\nSaved: {out_path}")

    logger.info("\n✅ Step 3 complete.")

if __name__ == "__main__":
    main()
