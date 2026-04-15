"""
step2_bert_models.py — Train all BERT-based models including the proposed DualBranch.
This is the main experiment (~1.5 hrs on T4 for both datasets, 3 seeds).

Usage:
    python step2_bert_models.py                      # Both datasets
    python step2_bert_models.py --dataset crowdflower # Single dataset
    python step2_bert_models.py --seeds 42            # Single seed (fast test)

Outputs:
    outputs/step2_bert_{dataset}.json
    checkpoints/vanilla_bert_*, dual_branch_*, etc.
"""

import os, sys, json, time, logging
import numpy as np, torch
from config import get_config
from preprocess import load_data
from models import build_model
from train import train_model
from evaluate import evaluate_model_full, run_error_analysis
from utils import set_seed, setup_logger, get_device, count_parameters
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

    logger = setup_logger(cfg.paths.log_dir, "step2")
    device = get_device(cfg.train.device)

    logger.info("=" * 60)
    logger.info("STEP 2: BERT-BASED MODELS")
    logger.info(f"Seeds: {cfg.train.seeds}, Epochs: {cfg.train.epochs}")
    logger.info("=" * 60)

    data = load_data(cfg)
    all_results = {}

    bert_models = ["vanilla_bert", "bert_cnn_local", "bert_cls_global", "dual_branch"]

    for ds_name, ds_data in data.items():
        logger.info(f"\n{'#'*60}\n# DATASET: {ds_name.upper()} — {ds_data['num_classes']} classes")
        logger.info(f"# {ds_data['label_names']}\n{'#'*60}")
        ds_results = {}

        for model_name in bert_models:
            logger.info(f"\n>>> {model_name}")
            seed_metrics = []

            for seed in cfg.train.seeds:
                logger.info(f"  --- Seed {seed} ---")
                set_seed(seed)

                model = build_model(model_name, ds_data["num_classes"], cfg)

                train_results = train_model(
                    model, ds_data["train_loader"], ds_data["val_loader"],
                    cfg, ds_data["num_classes"], ds_data["class_weights"],
                    ds_data["label_names"], f"{model_name}_{ds_name}_s{seed}", device,
                )

                # Collect features/gate only for dual_branch
                is_dual = model_name == "dual_branch"
                test_results = evaluate_model_full(
                    model, ds_data["test_loader"], cfg,
                    ds_data["num_classes"], ds_data["class_weights"],
                    ds_data["label_names"], device,
                    collect_features=is_dual, collect_gate=is_dual,
                )

                entry = {
                    "seed": seed,
                    "macro_f1": float(test_results["macro_f1"]),
                    "weighted_f1": float(test_results["weighted_f1"]),
                    "accuracy": float(test_results["accuracy"]),
                    "f1_per_class": test_results["f1_per_class"].tolist(),
                    "precision_per_class": test_results["precision_per_class"].tolist(),
                    "recall_per_class": test_results["recall_per_class"].tolist(),
                    "confusion_matrix": test_results["confusion_matrix"].tolist(),
                    "history": train_results["history"],
                }
                seed_metrics.append(entry)

            # Aggregate
            f1s = [s["macro_f1"] for s in seed_metrics]
            accs = [s["accuracy"] for s in seed_metrics]
            ds_results[model_name] = {
                "macro_f1": float(np.mean(f1s)),
                "macro_f1_std": float(np.std(f1s)),
                "accuracy": float(np.mean(accs)),
                "accuracy_std": float(np.std(accs)),
                "f1_per_class": np.mean([s["f1_per_class"] for s in seed_metrics], axis=0).tolist(),
                "seeds": seed_metrics,
                "label_names": ds_data["label_names"],
            }
            logger.info(f"  {model_name}: F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}, Acc={np.mean(accs):.4f}")

        # Error analysis on dual_branch last seed
        if "dual_branch" in ds_results:
            logger.info("\n--- Error Analysis (dual_branch) ---")
            last_seed = ds_results["dual_branch"]["seeds"][-1]
            # Re-run to get predictions for error analysis
            set_seed(cfg.train.seeds[-1])
            model = build_model("dual_branch", ds_data["num_classes"], cfg)
            ckpt_path = os.path.join(cfg.paths.checkpoint_dir,
                f"dual_branch_{ds_name}_s{cfg.train.seeds[-1]}_best.pt")
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"])
            model.to(device)
            test_res = evaluate_model_full(
                model, ds_data["test_loader"], cfg,
                ds_data["num_classes"], ds_data["class_weights"],
                ds_data["label_names"], device,
            )
            run_error_analysis(
                test_res["predictions"], test_res["labels"],
                ds_data["test_texts"], ds_data["label_names"],
                save_path=os.path.join(cfg.paths.output_dir, f"error_analysis_{ds_name}.json"),
            )

        all_results[ds_name] = ds_results

        # Save per-dataset
        out_path = os.path.join(cfg.paths.output_dir, f"step2_bert_{ds_name}.json")
        with open(out_path, "w") as f:
            json.dump(ds_results, f, indent=2)
        logger.info(f"\nSaved: {out_path}")

    # Print final summary
    logger.info(f"\n{'='*60}\nFINAL SUMMARY\n{'='*60}")
    for ds_name, ds_res in all_results.items():
        logger.info(f"\n--- {ds_name.upper()} ---")
        logger.info(f"{'Model':<25} {'Macro F1':>15} {'Accuracy':>15}")
        logger.info("-" * 55)
        for m in bert_models:
            if m in ds_res:
                r = ds_res[m]
                logger.info(f"{m:<25} {r['macro_f1']:.4f}±{r['macro_f1_std']:.4f}   {r['accuracy']:.4f}±{r['accuracy_std']:.4f}")

    logger.info("\n✅ Step 2 complete.")

if __name__ == "__main__":
    main()
