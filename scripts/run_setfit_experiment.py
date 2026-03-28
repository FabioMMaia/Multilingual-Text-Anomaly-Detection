"""
run_setfit_experiment.py
------------------------
Runs the SetFit fine-tuning + semi-supervised anomaly detection benchmark
for one dataset/encoder pair.

Pipeline
--------
1. Load parquets (texts, labels, pre-computed embeddings).
2. Label normal vs anomaly; adjust contamination to 5%.
3. Split data: test set + SetFit few-shot splits + downstream training pool.
4. Fine-tune a SetFit model on the few-shot split.
5. Re-encode the downstream/test sets with the fine-tuned model body.
6. Run run_semi_supervised_benchmark over n_known_list for each model ×
   embedding configuration (Original vs SetFit).
7. Save results CSV to <results_dir>/N_<k>/<dataset_short>.csv.
8. (Optional) save t-SNE comparison plot.

Usage (Colab, typical)
----------------------
    !python scripts/run_setfit_experiment.py \\
        --project_path "/content/drive/MyDrive/Projeto ML/2026/Master/Code/Multilingual-Text-Anomaly-Detection" \\
        --dataset 20_newsgroups \\
        --k_per_class 20

Usage (local)
-------------
    python scripts/run_setfit_experiment.py --project_path . --dataset 20_newsgroups
"""

import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="SetFit + semi-supervised AD experiment.")
    parser.add_argument(
        "--project_path", type=str, default=".",
        help="Root path of the project (where data/ and src/ live).",
    )
    parser.add_argument(
        "--dataset", type=str, default="20_newsgroups",
        help="Dataset short name (must match parquet filenames, e.g. 20_newsgroups).",
    )
    parser.add_argument(
        "--encoder", type=str,
        default="sentence-transformers/distiluse-base-multilingual-cased-v2",
        help="Encoder used to produce the pre-computed embeddings parquet.",
    )
    parser.add_argument(
        "--k_per_class", type=int, default=20,
        help="Few-shot budget per class for SetFit training.",
    )
    parser.add_argument(
        "--results_dir", type=str, default="data/setfit_results",
        help="Base directory for saving CSV results (relative to project_path).",
    )
    parser.add_argument(
        "--plots_dir", type=str, default="plots/tsne_embeddings_visualization",
        help="Base directory for saving t-SNE plots (relative to project_path).",
    )
    parser.add_argument(
        "--no_tsne", action="store_true",
        help="Skip the t-SNE comparison plot (faster).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    project_path = os.path.abspath(args.project_path)
    os.chdir(project_path)
    sys.path.insert(0, os.path.join(project_path, "src"))

    # ------------------------------------------------------------------ #
    # Imports (after sys.path is set)
    # ------------------------------------------------------------------ #
    import pandas as pd
    import numpy as np
    from setfit import SetFitModel

    from pipeline.anomaly_detection import (
        label_normal_vs_anomaly,
        adjust_contamination,
        pretty_print_data_info,
    )
    from pipeline.setfit_runner import (
        prepare_data_splits_for_anomaly_detection,
        train_and_evaluate_setfit_model,
        encode_with_setfit,
        build_n_known_list,
        run_semi_supervised_benchmark,
    )
    from models.MLP import MLPTF
    from deepod.models import DevNet, DeepSAD

    # ------------------------------------------------------------------ #
    # Config
    # ------------------------------------------------------------------ #
    dataset_short = args.dataset
    encoder_name = args.encoder
    encoder_short = encoder_name.split("/")[-1]
    K = args.k_per_class

    data_dir = os.path.join(project_path, "data")
    results_base = os.path.join(project_path, args.results_dir, f"N_{K}")
    plots_base = os.path.join(project_path, args.plots_dir, f"N_{K}")

    os.makedirs(results_base, exist_ok=True)

    MODEL_REGISTRY = {
        "DevNet": {
            "cls": DevNet,
            "kwargs": {"epochs": 20, "verbose": 0},
        },
        "DeepSAD": {
            "cls": DeepSAD,
            "kwargs": {"epochs": 20, "verbose": 0},
        },
        "MLP": {
            "cls": MLPTF,
            "kwargs": {
                "epochs": 20,
                "hidden_dims": (256, 128),
                "batch_size": 256,
                "verbose": 0,
            },
        },
    }

    # ------------------------------------------------------------------ #
    # 1. Load data
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print(f"Dataset : {dataset_short}")
    print(f"Encoder : {encoder_short}")
    print(f"k/class : {K}")
    print(f"{'='*60}\n")

    texts_df = pd.read_parquet(os.path.join(data_dir, f"texts_{dataset_short}.parquet"))
    labels_df = pd.read_parquet(os.path.join(data_dir, f"labels_{dataset_short}.parquet"))
    embeddings_df = pd.read_parquet(
        os.path.join(data_dir, f"embeddings_{dataset_short}_{encoder_short}.parquet")
    )

    # ------------------------------------------------------------------ #
    # 2. Label + contamination adjustment
    # ------------------------------------------------------------------ #
    labeled_df = label_normal_vs_anomaly(labels_df, as_df=True)

    texts, labels, embeddings = adjust_contamination(
        texts=texts_df.squeeze().tolist(),
        labels=labeled_df.squeeze().values,
        embeddings=embeddings_df.values,
        perc_anomalous=0.05,
    )
    pretty_print_data_info(texts, labels, embeddings)

    # ------------------------------------------------------------------ #
    # 3. Data splits
    # ------------------------------------------------------------------ #
    downstream_train, test_set, setfit_data, _ = prepare_data_splits_for_anomaly_detection(
        texts=texts,
        embeddings=embeddings,
        labels=labels,
        k_per_class=K,
        random_state=args.seed,
        verbose=True,
    )

    # ------------------------------------------------------------------ #
    # 4. SetFit fine-tuning
    # ------------------------------------------------------------------ #
    model = SetFitModel.from_pretrained(encoder_name)
    metrics, trainer = train_and_evaluate_setfit_model(
        model=model,
        setfit_data=setfit_data,
        test_set=test_set,
        seed=args.seed,
        verbose=True,
    )
    print("\nSetFit classification metrics:", metrics)

    # ------------------------------------------------------------------ #
    # 5. Re-encode with fine-tuned body
    # ------------------------------------------------------------------ #
    print("\nEncoding downstream_train and test_set with fine-tuned SetFit body...")
    encode_with_setfit(trainer, downstream_train, test_set)

    # ------------------------------------------------------------------ #
    # 6. (Optional) t-SNE comparison
    # ------------------------------------------------------------------ #
    if not args.no_tsne:
        from utils.visualization import plot_tsne_comparison
        plot_tsne_comparison(
            embeddings_a=downstream_train["embeddings"],
            embeddings_b=downstream_train["embeddings_sf"],
            labels=downstream_train["labels"],
            title_a="Original Embeddings",
            title_b=f"SetFit Embeddings (k={K})",
            dataset_short=dataset_short,
            encoder_short=encoder_short,
            save_dir=plots_base,
            stage="pre_vs_post_setfit",
        )

    # ------------------------------------------------------------------ #
    # 7. Semi-supervised benchmark
    # ------------------------------------------------------------------ #
    EMBEDDING_CONFIGS = {
        "Original": {
            "X_train": downstream_train["embeddings"],
            "X_test": test_set["embeddings"],
        },
        "SetFit": {
            "X_train": downstream_train["embeddings_sf"],
            "X_test": test_set["embeddings_sf"],
        },
    }

    total_anomalies = int(sum(1 for y in downstream_train["labels"] if y == 1))
    n_known_list = build_n_known_list(total_anomalies)
    print(f"\nn_known_list: {n_known_list}")

    all_results = []

    for model_name, model_cfg in MODEL_REGISTRY.items():
        for emb_name, emb_cfg in EMBEDDING_CONFIGS.items():
            method_name = (
                f"{model_name} (SetFit, k={K})"
                if emb_name == "SetFit"
                else f"{model_name} (Original)"
            )
            print(f"\nRunning: {method_name}")

            df = run_semi_supervised_benchmark(
                model_cls=model_cfg["cls"],
                model_kwargs=model_cfg["kwargs"],
                X_train=emb_cfg["X_train"],
                y_train=downstream_train["labels"],
                X_test=emb_cfg["X_test"],
                y_test=test_set["labels"],
                n_known_list=n_known_list,
                method_name=method_name,
                seed=args.seed,
                verbose=True,
            )

            df["model"] = model_name
            df["representation"] = emb_name
            df["k_per_class"] = K if emb_name == "SetFit" else 0
            df["dataset"] = dataset_short
            df["encoder"] = encoder_short
            all_results.append(df)

    # ------------------------------------------------------------------ #
    # 8. Save results
    # ------------------------------------------------------------------ #
    results_df = pd.concat(all_results, ignore_index=True)
    out_path = os.path.join(results_base, f"{dataset_short}.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\n✅ Results saved to: {out_path}")
    print(results_df.groupby(["model", "representation"])[["roc_auc", "pr_auc"]].mean().round(4))


if __name__ == "__main__":
    main()
