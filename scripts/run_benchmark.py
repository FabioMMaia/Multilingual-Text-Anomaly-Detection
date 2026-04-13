"""
run_benchmark.py
----------------
Runs the full unsupervised + semi-supervised anomaly detection benchmark
across all dataset/encoder combinations.

Usage (local):
    python scripts/run_benchmark.py --project_path .

Usage (Colab):
    !python scripts/run_benchmark.py \
        --project_path "/content/drive/MyDrive/Projeto ML/2025/AD/Multilingual-Text-Anomaly-Detection" \
        --results_dir "data/benchmark_results"
"""

import argparse
import os
import sys
import time
import pandas as pd
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark AD models across dataset/encoder pairs.")
    parser.add_argument("--project_path", type=str, default=".", help="Root path of the project.")
    parser.add_argument("--results_dir", type=str, default="data/benchmark_results", help="Where to save CSV results.")
    parser.add_argument("--contamination", type=float, nargs="+", default=[0.05],
                        help="Contamination levels for semi-supervised models (e.g. 0.01 0.05).")
    parser.add_argument("--n_rounds", type=int, default=1,
                        help="Number of random rounds per contamination level.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for deep models: 'cpu' or 'cuda'.")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to parquet files (texts/labels/embeddings). Defaults to <project_path>/data.")
    parser.add_argument("--dataset_filter", type=str, default=None,
                        help="Only run combos where dataset name contains this string (case-insensitive).")
    parser.add_argument("--encoder_filter", type=str, default=None,
                        help="Only run combos where encoder name contains this string (case-insensitive).")
    return parser.parse_args()


def main():
    args = parse_args()
    project_path = os.path.abspath(args.project_path)
    results_dir = os.path.join(project_path, args.results_dir)
    data_dir = os.path.abspath(args.data_dir) if args.data_dir else os.path.join(project_path, "data")

    sys.path.append(os.path.join(project_path, "src"))
    os.chdir(project_path)

    from pipeline.anomaly_detection import label_normal_vs_anomaly, adjust_contamination, pretty_print_data_info, split_data
    from pipeline.benchmark_runner import benchmark_unsupervised_models, benchmark_semisupervised_models
    from models.MLP import MLP

    from deepod.models import DeepSAD, DeepSVDD, DevNet
    from pyod.models.xgbod import XGBOD
    from pyod.models.lof import LOF
    from pyod.models.iforest import IForest
    from pyod.models.ocsvm import OCSVM
    from pyod.models.auto_encoder import AutoEncoder
    from pyod.models.vae import VAE
    from pyod.models.pca import PCA
    from pyod.models.hbos import HBOS

    # ------------------------------------------------------------------ #
    # Dataset / encoder config
    # ------------------------------------------------------------------ #
    # v5 dataset grid only (2×2): tweets_hs, 20_newsgroups, HateBR, wikinews
    # Single encoder (distiluse-v2) for fair comparison with the LLM pipeline.
    config = {
        "en": {
            "datasets": [
                "tweets-hate-speech-detection/tweets_hate_speech_detection",
                "SetFit/20_newsgroups",
            ],
            "encoders": [
                "sentence-transformers/distiluse-base-multilingual-cased-v2",
            ],
        },
        "pt": {
            "datasets": [
                "franciellevargas/HateBR",
                "wikinews",
            ],
            "encoders": [
                "sentence-transformers/distiluse-base-multilingual-cased-v2",
            ],
        },
    }

    # ------------------------------------------------------------------ #
    # Model groups
    # ------------------------------------------------------------------ #
    model_groups = {
        "semi": {
            "models": {
                "DevNet": lambda: DevNet(device=args.device),
                "DeepSAD": lambda: DeepSAD(epochs=100, rep_dim=128, device=args.device),
                "XGBOD": lambda: XGBOD(estimator_list=[LOF(), IForest()]),
                "MLP": lambda: MLP(),
            },
            "benchmark_fn": benchmark_semisupervised_models,
            "extra_args": {
                "contamination_levels": args.contamination,
                "n_rounds": args.n_rounds,
            },
            "wrap_model": lambda name, fn: {name: (fn, False)},
        },
        "unsupervised": {
            "models": {
                "IForest": lambda: IForest(),
                "LOF": lambda: LOF(),
                "DeepSVDD": lambda: DeepSVDD(epochs=100, rep_dim=128, device=args.device),
                "OCSVM": lambda: OCSVM(kernel="rbf", nu=0.05, gamma="scale"),
                "AutoEncoder": lambda: AutoEncoder(),
                "VAE": lambda: VAE(),
                "HBOS": lambda: HBOS(),
            },
            "benchmark_fn": benchmark_unsupervised_models,
            "extra_args": {},
            "wrap_model": lambda name, fn: {name: fn},
        },
    }

    # ------------------------------------------------------------------ #
    # Run benchmark
    # ------------------------------------------------------------------ #
    total_combinations = sum(
        len(cfg["datasets"]) * len(cfg["encoders"]) for cfg in config.values()
    )

    with tqdm(total=total_combinations, desc="Benchmarking dataset/encoder pairs") as pbar:
        for lang, lang_config in config.items():
            for dataset_name in lang_config["datasets"]:
                dataset_short = dataset_name.split("/")[-1]

                for encoder_name in lang_config["encoders"]:
                    encoder_short = encoder_name.split("/")[-1]
                    combo_name = f"{dataset_short}_{encoder_short}"

                    if args.dataset_filter and args.dataset_filter.lower() not in dataset_short.lower():
                        pbar.update(1)
                        continue
                    if args.encoder_filter and args.encoder_filter.lower() not in encoder_short.lower():
                        pbar.update(1)
                        continue

                    try:
                        print(f"\nProcessing: {combo_name}")
                        texts_df = pd.read_parquet(os.path.join(data_dir, f"texts_{dataset_short}.parquet"))
                        labels_df = pd.read_parquet(os.path.join(data_dir, f"labels_{dataset_short}.parquet"))
                        embeddings_df = pd.read_parquet(os.path.join(data_dir, f"embeddings_{dataset_short}_{encoder_short}.parquet"))

                        labeled_anomalies_df = label_normal_vs_anomaly(labels_df, as_df=True)

                        texts, labels, embeddings = adjust_contamination(
                            texts=texts_df.squeeze().tolist(),
                            labels=labeled_anomalies_df.squeeze().values,
                            embeddings=embeddings_df.values,
                            perc_anomalous=0.05,
                        )

                        pretty_print_data_info(texts, labels, embeddings)
                        x_train, x_test, y_train, y_test = split_data(embeddings, labels, random_state=42)

                        for group_name, group_info in model_groups.items():
                            models = group_info["models"]
                            benchmark_fn = group_info["benchmark_fn"]
                            extra_args = group_info["extra_args"]
                            wrap_model = group_info["wrap_model"]

                            for model_name, model_fn in models.items():
                                print(f"Running {group_name} model: {model_name} on {combo_name}")
                                start_time = time.time()

                                benchmark_fn(
                                    x_train, y_train, x_test, y_test,
                                    model_constructor=wrap_model(model_name, model_fn),
                                    dataset_name=combo_name,
                                    results_dir=results_dir,
                                    **extra_args,
                                )

                                print(f"✅ Finished {model_name} in {time.time() - start_time:.2f}s")

                    except Exception as e:
                        print(f"[❌ ERROR] Failed processing {combo_name}: {e}")

                    pbar.update(1)

    print(f"\n✅ Benchmark complete. Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
