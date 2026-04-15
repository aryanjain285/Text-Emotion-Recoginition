"""
step4_visualize.py — Generate all report figures from saved results.
Reads JSON outputs from steps 1-3 and produces all visualizations.

Run this AFTER steps 1-3 are complete. Takes ~1 minute.

Usage:
    python step4_visualize.py

Outputs:
    figures/  — All PNGs for the report
    outputs/report_tables.json — Formatted tables for copy-paste
"""

import os, json, logging
import numpy as np
import torch
from config import get_config
from preprocess import load_data
from train import evaluate
from evaluate import evaluate_model_full, extract_attention_weights
from models import build_model
from visualize import (
    plot_confusion_matrix, plot_training_curves, plot_per_class_f1,
    plot_results_table, plot_tsne, plot_gate_analysis, plot_cls_attention,
)
from utils import setup_logger, get_device, set_seed, get_loss_fn

logger_obj = None

def load_step_results(output_dir, dataset):
    """Load all step results for a dataset."""
    results = {}
    for step_file in ["step1_baselines", "step2_bert", "step3_ablation"]:
        path = os.path.join(output_dir, f"{step_file}_{dataset}.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
                results.update(data)
            print(f"  Loaded: {path} ({len(data)} models)")
        else:
            print(f"  Missing: {path} (skipped)")
    return results


def generate_figures(results, dataset, label_names, figures_dir):
    """Generate all figures for one dataset."""
    os.makedirs(figures_dir, exist_ok=True)

    # ---- 1. Confusion matrices ----
    for model_name in ["vanilla_bert", "dual_branch"]:
        if model_name in results and "seeds" in results[model_name]:
            last_seed = results[model_name]["seeds"][-1]
            if "confusion_matrix" in last_seed:
                cm = np.array(last_seed["confusion_matrix"])
                plot_confusion_matrix(
                    cm, label_names,
                    title=f"{model_name} — {dataset}",
                    save_path=os.path.join(figures_dir, f"cm_{model_name}_{dataset}.png"),
                )

    # ---- 2. Training curves ----
    histories = {}
    for model_name, res in results.items():
        if "seeds" in res and "history" in res["seeds"][-1]:
            histories[model_name] = res["seeds"][-1]["history"]

    if histories:
        # Split into baseline vs BERT for readability
        bert_histories = {k: v for k, v in histories.items()
                         if k not in ["textcnn", "bilstm_attention"]}
        if bert_histories:
            plot_training_curves(
                bert_histories,
                title=f"BERT Models — {dataset}",
                save_path=os.path.join(figures_dir, f"training_curves_bert_{dataset}.png"),
            )

        baseline_histories = {k: v for k, v in histories.items()
                             if k in ["textcnn", "bilstm_attention"]}
        if baseline_histories:
            plot_training_curves(
                baseline_histories,
                title=f"Non-BERT Baselines — {dataset}",
                save_path=os.path.join(figures_dir, f"training_curves_baselines_{dataset}.png"),
            )

    # ---- 3. Per-class F1 comparison ----
    f1_results = {}
    for model_name, res in results.items():
        if "f1_per_class" in res:
            f1_results[model_name] = {"f1_per_class": np.array(res["f1_per_class"])}

    if len(f1_results) > 1:
        plot_per_class_f1(
            f1_results, label_names,
            title=f"Per-Class F1 — {dataset}",
            save_path=os.path.join(figures_dir, f"per_class_f1_{dataset}.png"),
        )

    # ---- 4. Results table (main comparison) ----
    main_models = ["textcnn", "bilstm_attention", "vanilla_bert",
                   "bert_cnn_local", "bert_cls_global", "dual_branch"]
    table_results = {m: results[m] for m in main_models if m in results}
    if table_results:
        plot_results_table(
            table_results,
            title=f"Baseline Comparison — {dataset}",
            save_path=os.path.join(figures_dir, f"results_table_main_{dataset}.png"),
        )

    # ---- 5. Ablation table ----
    ablation_models = {
        "dual_branch (gated)": results.get("dual_branch"),
        "local only": results.get("bert_cnn_local"),
        "global only": results.get("bert_cls_global"),
        "concat fusion": results.get("dual_branch_concat"),
        "average fusion": results.get("dual_branch_average"),
    }
    ablation_results = {k: v for k, v in ablation_models.items() if v is not None}
    if len(ablation_results) > 1:
        plot_results_table(
            ablation_results,
            title=f"Ablation Study — {dataset}",
            save_path=os.path.join(figures_dir, f"results_table_ablation_{dataset}.png"),
        )


def generate_tsne_and_gate(cfg, dataset, ds_data, figures_dir, device):
    """Generate t-SNE and gate analysis from the best dual_branch checkpoint."""
    ckpt_path = os.path.join(cfg.paths.checkpoint_dir,
        f"dual_branch_{dataset}_s{cfg.train.seeds[-1]}_best.pt")

    if not os.path.exists(ckpt_path):
        print(f"  No checkpoint found at {ckpt_path}, skipping t-SNE/gate")
        return

    set_seed(cfg.train.seeds[-1])
    model = build_model("dual_branch", ds_data["num_classes"], cfg)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    loss_fn = get_loss_fn(
        cfg.train.loss_fn, ds_data["num_classes"],
        class_weights=ds_data["class_weights"],
        focal_gamma=cfg.train.focal_gamma,
    )

    test_results = evaluate(
        model, ds_data["test_loader"], loss_fn, device,
        use_fp16=cfg.train.fp16,
        return_features=True, return_gate=True,
    )

    # t-SNE
    if "fused_features" in test_results:
        plot_tsne(
            test_results["fused_features"], test_results["labels"],
            ds_data["label_names"],
            title=f"t-SNE — Dual-Branch — {dataset}",
            save_path=os.path.join(figures_dir, f"tsne_{dataset}.png"),
        )

    # Gate analysis
    if "gate_values" in test_results:
        plot_gate_analysis(
            test_results["gate_values"], test_results["labels"],
            ds_data["label_names"],
            save_path=os.path.join(figures_dir, f"gate_analysis_{dataset}.png"),
        )


def generate_attention_heatmaps(cfg, dataset, ds_data, figures_dir, device):
    """Generate attention heatmaps from the dual_branch checkpoint."""
    from transformers import AutoTokenizer, AutoModel

    ckpt_path = os.path.join(cfg.paths.checkpoint_dir,
        f"dual_branch_{dataset}_s{cfg.train.seeds[-1]}_best.pt")
    if not os.path.exists(ckpt_path):
        print(f"  No checkpoint found, skipping attention heatmaps")
        return

    tokenizer = AutoTokenizer.from_pretrained(cfg.data.tokenizer_name)

    # Build model with eager attention for output_attentions support
    model = build_model("dual_branch", ds_data["num_classes"], cfg)
    eager_encoder = AutoModel.from_pretrained(
        cfg.model.encoder_name, attn_implementation="eager"
    )
    model.encoder = eager_encoder
    model._freeze_layers(cfg.model.freeze_embeddings, cfg.model.freeze_n_layers)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    sample_texts = ds_data["test_texts"][:3]
    sample_labels = ds_data["test_labels"][:3]

    for i, (text, label) in enumerate(zip(sample_texts, sample_labels)):
        encoding = tokenizer(
            text, max_length=cfg.data.max_seq_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
        n_tokens = encoding["attention_mask"][0].sum().item()
        tokens = tokens[:n_tokens]

        try:
            attn = extract_attention_weights(
                model, encoding["input_ids"], encoding["attention_mask"],
                device, layer=-1,
            )
            emotion = ds_data["label_names"][label]
            plot_cls_attention(
                attn, tokens,
                title=f"[CLS] Attention — {emotion}",
                save_path=os.path.join(figures_dir, f"attn_{i}_{emotion}_{dataset}.png"),
            )
        except Exception as e:
            print(f"  Attention heatmap {i} failed: {e}")


def print_report_tables(all_results):
    """Print formatted tables for easy copy-paste into the report."""
    print("\n" + "=" * 70)
    print("TABLES FOR REPORT (copy-paste ready)")
    print("=" * 70)

    for dataset, results in all_results.items():
        print(f"\n--- {dataset.upper()} ---")

        # Main results table
        print(f"\n{'Model':<25} {'Macro-F1':>12} {'Weighted-F1':>12} {'Accuracy':>12}")
        print("-" * 65)
        for model in ["textcnn", "bilstm_attention", "vanilla_bert",
                       "bert_cnn_local", "bert_cls_global", "dual_branch"]:
            if model in results:
                r = results[model]
                f1 = f"{r['macro_f1']:.4f}±{r.get('macro_f1_std',0):.4f}"
                wf1 = f"{r.get('weighted_f1', r.get('macro_f1',0)):.4f}"
                acc = f"{r['accuracy']:.4f}±{r.get('accuracy_std',0):.4f}"
                print(f"{model:<25} {f1:>12} {wf1:>12} {acc:>12}")

        # Ablation table
        ablation = {
            "gated (ours)": results.get("dual_branch"),
            "local only": results.get("bert_cnn_local"),
            "global only": results.get("bert_cls_global"),
            "concat": results.get("dual_branch_concat"),
            "average": results.get("dual_branch_average"),
        }
        ablation = {k: v for k, v in ablation.items() if v}
        if len(ablation) > 2:
            print(f"\nAblation Study:")
            print(f"{'Config':<25} {'Macro-F1':>12} {'Accuracy':>12}")
            print("-" * 50)
            for name, r in ablation.items():
                f1 = f"{r['macro_f1']:.4f}±{r.get('macro_f1_std',0):.4f}"
                acc = f"{r['accuracy']:.4f}"
                print(f"{name:<25} {f1:>12} {acc:>12}")


def main():
    cfg = get_config()
    cfg.data.dataset = "both"
    global logger_obj
    logger_obj = setup_logger(cfg.paths.log_dir, "step4")
    device = get_device(cfg.train.device)

    print("=" * 60)
    print("STEP 4: GENERATING ALL VISUALIZATIONS")
    print("=" * 60)

    all_results = {}

    for dataset in ["crowdflower", "wassa2017"]:
        print(f"\n--- {dataset.upper()} ---")
        results = load_step_results(cfg.paths.output_dir, dataset)

        if not results:
            print(f"  No results found for {dataset}, skipping")
            continue

        all_results[dataset] = results

        # Get label names
        label_names = None
        for model_name, res in results.items():
            if "label_names" in res:
                label_names = res["label_names"]
                break

        if label_names is None:
            # Infer from dataset
            if dataset == "crowdflower":
                label_names = ['happiness', 'love', 'neutral', 'sadness', 'surprise', 'worry']
            else:
                label_names = ['anger', 'fear', 'joy', 'sadness']

        # Generate standard figures from JSON results
        generate_figures(results, dataset, label_names, cfg.paths.figures_dir)

        # Generate t-SNE and gate analysis from checkpoints
        try:
            data = load_data(get_config(data={"dataset": dataset}))
            ds_data = data[dataset]
            generate_tsne_and_gate(cfg, dataset, ds_data, cfg.paths.figures_dir, device)
            generate_attention_heatmaps(cfg, dataset, ds_data, cfg.paths.figures_dir, device)
        except Exception as e:
            print(f"  t-SNE/gate/attention generation failed: {e}")
            print(f"  (This is OK — figures from JSONs are already saved)")

    # Print report-ready tables
    print_report_tables(all_results)

    # Save consolidated results
    report_path = os.path.join(cfg.paths.output_dir, "all_results_consolidated.json")
    # Convert any numpy arrays to lists for JSON serialization
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    print(f"\nConsolidated results saved to: {report_path}")

    print("\n✅ Step 4 complete. All figures in: figures/")

if __name__ == "__main__":
    main()
