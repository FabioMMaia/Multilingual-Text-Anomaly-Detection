"""
run_data_preparation.py
-----------------------
Encodes all datasets with all configured encoders and saves parquet files.

Usage (local):
    python scripts/run_data_preparation.py --project_path .

Usage (Colab):
    !python scripts/run_data_preparation.py \
        --project_path "/content/drive/MyDrive/Projeto ML/2025/AD/Multilingual-Text-Anomaly-Detection"
"""

import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Data preparation: encode datasets and save embeddings.")
    parser.add_argument(
        "--project_path",
        type=str,
        default=".",
        help="Root path of the project (where data/ folder will be created)."
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=60_000,
        help="Max number of samples per dataset."
    )
    parser.add_argument(
        "--sampling_strategy",
        type=str,
        default="stratified",
        choices=["stratified", "head"],
        help="Strategy to subsample large datasets."
    )
    parser.add_argument(
        "--encoder_filter",
        type=str,
        default=None,
        help=(
            "Only run encoders whose name contains this substring. "
            "E.g. 'distiluse-base-multilingual-cased-v2' to run only the Exp 3 encoder."
        )
    )
    parser.add_argument(
        "--dataset_filter",
        type=str,
        default=None,
        help=(
            "Only run datasets whose name contains this substring. "
            "E.g. 'told-br' to run only one dataset."
        )
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help=(
            "Directory where parquet files will be saved. "
            "Defaults to <project_path>/data. "
            "Use this on Colab to point to a shared Drive folder."
        )
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve project_path: if "." is passed and src/ is not there,
    # fall back to the directory containing this script.
    candidate = os.path.abspath(args.project_path)
    if not os.path.isdir(os.path.join(candidate, "src")):
        candidate = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    project_path = candidate

    # Add src to path before any local imports
    src_path = os.path.join(project_path, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from encoders.text_encoders import SentenceBERT, BERTimbau
    from pipeline.data_handler import process_dataset

    os.chdir(project_path)

    # ------------------------------------------------------------------ #
    # Dataset configs: (hf_name, subset, language, source)
    # ------------------------------------------------------------------ #
    datasets_to_process = [
        ("JAugusto97/told-br", "binary", "pt", "hf"),
        ("franciellevargas/HateBR", None, "pt", "hf"),
        ("tweets-hate-speech-detection/tweets_hate_speech_detection", None, "en", "hf"),
        ("SetFit/20_newsgroups", None, "en", "hf"),
        ("wikinews", "wikinews_dataset/wikinews_filtered.parquet", "pt", "wikinews_dump"),
        ("augustop/portuguese-tweets-for-sentiment-analysis", "NoThemeTweets.csv", "pt", "kaggle"),
        ("cardiffnlp/tweet_eval", "sentiment", "en", "hf"),
    ]

    # ------------------------------------------------------------------ #
    # Encoder configs
    # For Experiment 3 (LLM loop), only distiluse-v2 is needed.
    # The others are kept for reference (Exp 1/2 benchmark) but commented out.
    # ------------------------------------------------------------------ #
    encoder_configs = [
        ("sentence-transformers/distiluse-base-multilingual-cased-v2", SentenceBERT, "multi"),
        # ("sentence-transformers/distiluse-base-multilingual-cased-v1", SentenceBERT, "multi"),
        # ("FacebookAI/xlm-roberta-large", SentenceBERT, "multi"),
        # ("neuralmind/bert-base-portuguese-cased", BERTimbau, "pt"),
        # ("neuralmind/bert-large-portuguese-cased", BERTimbau, "pt"),
        # ("PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir", SentenceBERT, "pt"),
    ]

    # ------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------ #
    for dataset_name, subset, dataset_lang, source in datasets_to_process:
        if args.dataset_filter and args.dataset_filter not in dataset_name:
            continue
        for model_name, encoder_class, model_lang in encoder_configs:
            if args.encoder_filter and args.encoder_filter not in model_name:
                continue
            # Skip encoders that don't match the dataset language
            if model_lang != "multi" and model_lang != dataset_lang:
                continue
            try:
                process_dataset(
                    dataset_name,
                    subset,
                    model_name,
                    encoder_class,
                    source,
                    project_path=project_path,
                    data_dir=args.data_dir,
                    max_samples=args.max_samples,
                    sampling_strategy=args.sampling_strategy,
                )
            except Exception as e:
                print(f"❌ Failed for {dataset_name} with {model_name}: {e}")

    print("\n✅ Data preparation complete.")


if __name__ == "__main__":
    main()
