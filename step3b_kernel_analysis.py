"""
step3b_kernel_analysis.py — Experiment 3: CNN kernel size analysis.
Tests which n-gram lengths are most informative for emotion detection.

Usage:
    python step3b_kernel_analysis.py
    python step3b_kernel_analysis.py --dataset crowdflower

Outputs:
    outputs/step3b_kernels_{dataset}.json

Time: ~1 hr on T4 (3 extra kernel configs × 2 datasets × 3 seeds)
      Default config (2,3,4) is reused from step2 results.
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

    logger = setup_logger(cfg.paths.log_dir, "step3b")
    device = get_device(cfg.train.device)

    logger.info("=" * 60)
    logger.info("STEP 3b: CNN KERNEL SIZE ANALYSIS")
    logger.info("=" * 60)

    data = load_data(cfg)

    kernel_configs = [
        (2, 3),
        (3, 4, 5),
        # (2, 3, 4) is the default — reused from step2 dual_branch results
        (2, 3, 4, 5),
    ]

    for ds_name, ds_data in data.items():
        logger.info(f"\n{'#'*60}\n# DATASET: {ds_name.upper()}\n{'#'*60}")
        ds_results = {}

        for kernels in kernel_configs:
            kernel_str = ",".join(map(str, kernels))
            name = f"kernels_({kernel_str})"
            logger.info(f"\n>>> Kernels: {kernels}")
            seed_metrics = []

            for seed in cfg.train.seeds:
                logger.info(f"  --- Seed {seed} ---")
                set_seed(seed)

                model = build_model(
                    "dual_branch", ds_data["num_classes"], cfg,
                    cnn_kernel_sizes=kernels,
                )

                train_results = train_model(
                    model, ds_data["train_loader"], ds_data["val_loader"],
                    cfg, ds_data["num_classes"], ds_data["class_weights"],
                    ds_data["label_names"], f"kern_{'_'.join(map(str,kernels))}_{ds_name}_s{seed}",
                    device,
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
                })

            f1s = [s["macro_f1"] for s in seed_metrics]
            accs = [s["accuracy"] for s in seed_metrics]
            ds_results[name] = {
                "kernels": list(kernels),
                "macro_f1": float(np.mean(f1s)),
                "macro_f1_std": float(np.std(f1s)),
                "accuracy": float(np.mean(accs)),
                "accuracy_std": float(np.std(accs)),
                "seeds": seed_metrics,
            }
            logger.info(f"  {name}: F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}")

        out_path = os.path.join(cfg.paths.output_dir, f"step3b_kernels_{ds_name}.json")
        with open(out_path, "w") as f:
            json.dump(ds_results, f, indent=2)
        logger.info(f"\nSaved: {out_path}")

        # Print summary table
        logger.info(f"\nKernel Analysis Summary — {ds_name}:")
        logger.info(f"{'Kernels':<20} {'Macro F1':>15}")
        logger.info("-" * 35)
        # Include default (2,3,4) from step2 if available
        step2_path = os.path.join(cfg.paths.output_dir, f"step2_bert_{ds_name}.json")
        if os.path.exists(step2_path):
            with open(step2_path) as f:
                step2 = json.load(f)
            if "dual_branch" in step2:
                r = step2["dual_branch"]
                logger.info(f"{'(2,3,4) [default]':<20} {r['macro_f1']:.4f}±{r.get('macro_f1_std',0):.4f}")
        for name, r in ds_results.items():
            logger.info(f"{name:<20} {r['macro_f1']:.4f}±{r['macro_f1_std']:.4f}")

    logger.info("\n✅ Step 3b complete.")

if __name__ == "__main__":
    main()
