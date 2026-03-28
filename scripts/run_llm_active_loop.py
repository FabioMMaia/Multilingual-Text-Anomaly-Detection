"""
run_llm_active_loop.py
----------------------
LLM-guided anomaly detection pipeline with no human labels during training.

Pipeline
--------
1. Load parquets (texts, labels, pre-computed embeddings).
2. Label normal vs anomaly; adjust contamination to 5%.
3. Train/test split (ground-truth labels used ONLY for final evaluation).
4. Run DeepSVDD (unsupervised) on train embeddings -> anomaly scores.
5. Select n_llm_calls samples (random OR score-guided).
6. Query LLM -> get anomaly_score + reason per sample.
7. Convert LLM scores to binary labels via threshold.
8. Fine-tune SetFit on LLM-labeled samples.
9. Re-encode with SetFit embeddings.
10. Train DeepSAD on LLM labels + SetFit embeddings.
11. Evaluate on locked test set -> save results CSV.

Backends supported
------------------
  gemini    Google Gemini 2.0 Flash (cheapest API; set GEMINI_API_KEY env var)
  openai    OpenAI gpt-4o-mini (set OPENAI_API_KEY env var)
  llamacpp  Local GGUF model via llama-cpp-python (CPU, no GPU needed)
            Recommended: Qwen2.5-1.5B-Instruct-Q4_K_M.gguf (~1GB)

Usage (Colab, typical)
----------------------
    !python scripts/run_llm_active_loop.py \\
        --project_path "/content/drive/MyDrive/.../Multilingual-Text-Anomaly-Detection" \\
        --dataset told_br \\
        --backend gemini \\
        --strategy score_guided \\
        --n_llm_calls 100

Usage (local, no GPU, CPU model)
---------------------------------
    python scripts/run_llm_active_loop.py \\
        --project_path "." \\
        --dataset told_br \\
        --backend llamacpp \\
        --llamacpp_model_path /path/to/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf \\
        --strategy score_guided \\
        --n_llm_calls 50

Run all strategies and N values (example loop in bash/colab):
    for strategy in random score_guided; do
      for n in 50 100 200; do
        python scripts/run_llm_active_loop.py --dataset told_br --strategy $strategy --n_llm_calls $n
      done
    done
"""

import argparse
import os
import sys
import warnings


def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM-guided anomaly detection loop (no human labels in training)."
    )
    parser.add_argument(
        "--project_path", type=str, default=".",
        help="Root path of the project (where data/ and src/ live).",
    )
    parser.add_argument(
        "--dataset", type=str, default="told_br",
        choices=["told_br", "tweets_hs", "pt_tweets", "tweeteval", "20_newsgroups", "wikinews"],
        help="Dataset name (must match TASK_CONTEXT keys and parquet filenames).",
    )
    parser.add_argument(
        "--encoder", type=str,
        default="sentence-transformers/distiluse-base-multilingual-cased-v2",
        help="Encoder used to produce the pre-computed embeddings parquet.",
    )
    # LLM backend
    parser.add_argument(
        "--backend", type=str, default="groq",
        choices=["groq", "gemini", "openai", "llamacpp"],
        help="LLM backend to use for annotation.",
    )
    parser.add_argument(
        "--api_key", type=str, default=None,
        help="API key for gemini/openai (or set GEMINI_API_KEY / OPENAI_API_KEY env var).",
    )
    parser.add_argument(
        "--llm_model", type=str, default=None,
        help="Override default LLM model name (e.g. 'gemini-2.0-flash-lite' for even cheaper).",
    )
    parser.add_argument(
        "--llamacpp_model_path", type=str, default=None,
        help="Path to local GGUF model file (required for --backend llamacpp).",
    )
    parser.add_argument(
        "--llamacpp_threads", type=int, default=4,
        help="Number of CPU threads for llama-cpp-python inference.",
    )
    # Selection strategy
    parser.add_argument(
        "--strategy", type=str, default="score_guided",
        choices=["random", "score_guided"],
        help="Sample selection strategy for LLM annotation budget.",
    )
    parser.add_argument(
        "--n_llm_calls", type=int, default=100,
        help="LLM annotation budget (number of samples sent to LLM).",
    )
    parser.add_argument(
        "--anomaly_threshold", type=float, default=0.6,
        help="LLM anomaly_score >= threshold -> label as anomaly.",
    )
    parser.add_argument(
        "--min_anomalies", type=int, default=5,
        help="Minimum anomaly labels required to proceed (warns otherwise).",
    )
    # SetFit
    parser.add_argument(
        "--setfit_model", type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="SetFit base model for contrastive fine-tuning.",
    )
    # Device
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda"],
        help="Device for DeepSVDD/DeepSAD training. Use 'cuda' on Colab/GPU.",
    )
    # Data
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help=(
            "Directory containing the parquet files (texts_*, labels_*, embeddings_*). "
            "Defaults to {project_path}/data/. "
            "Use this when your parquets live elsewhere, e.g. a different Drive folder."
        ),
    )
    # Output
    parser.add_argument(
        "--results_dir", type=str, default="data/llm_results",
        help="Base directory for saving CSV results (relative to project_path).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed.",
    )
    parser.add_argument(
        "--contamination", type=float, default=0.05,
        help="Target anomaly contamination rate for training set.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    project_path = os.path.abspath(args.project_path)
    os.chdir(project_path)
    sys.path.insert(0, os.path.join(project_path, "src"))

    import numpy as np
    import pandas as pd
    from deepod.models import DeepSVDD, DeepSAD
    from sklearn.metrics import roc_auc_score, average_precision_score

    from pipeline.anomaly_detection import (
        label_normal_vs_anomaly,
        adjust_contamination,
    )
    from pipeline.llm_runner import LLMAnnotator, run_llm_active_loop

    # ------------------------------------------------------------------
    # Derived names
    # ------------------------------------------------------------------
    encoder_short = args.encoder.split("/")[-1]

    # Mapping from short dataset arg -> actual parquet filename prefix
    DATASET_FILE_MAP = {
        "told_br"        : "told-br",
        "tweets_hs"      : "tweets_hate_speech_detection",
        "pt_tweets"      : "portuguese-tweets-for-sentiment-analysis",
        "tweeteval"      : "tweet_eval",
        "20_newsgroups"  : "20_newsgroups",
        "wikinews"       : "wikinews",
    }
    dataset_file = DATASET_FILE_MAP.get(args.dataset, args.dataset)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    data_dir = os.path.abspath(args.data_dir) if args.data_dir else os.path.join(project_path, "data")
    texts_path = os.path.join(data_dir, f"texts_{dataset_file}.parquet")
    labels_path = os.path.join(data_dir, f"labels_{dataset_file}.parquet")
    emb_path = os.path.join(data_dir, f"embeddings_{dataset_file}_{encoder_short}.parquet")

    print(f"Loading data for '{args.dataset}'...")
    texts_df = pd.read_parquet(texts_path)
    labels_df = pd.read_parquet(labels_path)
    emb_df = pd.read_parquet(emb_path)

    texts = texts_df["text"].values
    embeddings = emb_df.values.astype("float32")

    # ------------------------------------------------------------------
    # Preprocessing: binary labels + contamination adjustment
    # ------------------------------------------------------------------
    binary_labels = label_normal_vs_anomaly(labels_df, verbose=True)
    texts, binary_labels, embeddings = adjust_contamination(
        texts, binary_labels, embeddings,
        perc_anomalous=args.contamination,
        random_state=args.seed,
    )
    texts = np.array(texts)

    print(f"Dataset size after contamination adjustment: {len(texts)} samples")
    print(f"Anomaly rate: {binary_labels.mean():.2%}")

    # ------------------------------------------------------------------
    # Init LLM annotator
    # ------------------------------------------------------------------
    llamacpp_kwargs = {}
    if args.backend == "llamacpp":
        if not args.llamacpp_model_path:
            raise ValueError("--llamacpp_model_path is required for --backend llamacpp")
        llamacpp_kwargs = {"n_threads": args.llamacpp_threads}
        model_arg = args.llamacpp_model_path
    else:
        model_arg = args.llm_model  # None -> use default

    annotator = LLMAnnotator(
        backend=args.backend,
        api_key=args.api_key,
        model=model_arg,
        llamacpp_kwargs=llamacpp_kwargs if args.backend == "llamacpp" else None,
    )

    # ------------------------------------------------------------------
    # Run pipeline
    # ------------------------------------------------------------------
    print(f"\nRunning LLM active loop: strategy={args.strategy}, n_llm_calls={args.n_llm_calls}")

    loop_result = run_llm_active_loop(
        texts=texts,
        embeddings=embeddings,
        dataset_name=args.dataset,
        annotator=annotator,
        unsup_model_cls=DeepSVDD,
        semisup_model_cls=DeepSAD,
        setfit_model_name=args.setfit_model,
        strategy=args.strategy,
        n_llm_calls=args.n_llm_calls,
        anomaly_score_threshold=args.anomaly_threshold,
        min_anomalies_required=args.min_anomalies,
        test_size=0.2,
        random_state=args.seed,
        device=args.device,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # Evaluate (ground-truth labels used here only)
    # ------------------------------------------------------------------
    test_idx = loop_result["test_idx"]
    y_test = binary_labels[test_idx]
    test_scores = loop_result["test_scores"]

    roc_auc = roc_auc_score(y_test, test_scores)
    pr_auc = average_precision_score(y_test, test_scores)

    print(f"\nResults — {args.dataset} | {args.strategy} | N={args.n_llm_calls}")
    print(f"  ROC-AUC : {roc_auc:.4f}")
    print(f"  PR-AUC  : {pr_auc:.4f}")
    print(f"  LLM labeled: {loop_result['n_llm_labeled']} valid "
          f"({loop_result['n_anomalies_found']} anomalies / {loop_result['n_normals_found']} normals)")
    print(f"  Parse errors: {loop_result['n_parse_errors']}")

    # ------------------------------------------------------------------
    # Save results (append mode — accumulates multiple seeds/runs)
    # ------------------------------------------------------------------
    out_dir = os.path.join(
        project_path,
        args.results_dir,
        args.strategy,
        f"N_{args.n_llm_calls}",
    )
    os.makedirs(out_dir, exist_ok=True)

    # Main metrics CSV — one row per run, appended
    metrics_row = {
        "dataset": args.dataset,
        "strategy": args.strategy,
        "n_llm_calls": args.n_llm_calls,
        "seed": args.seed,
        "n_llm_labeled": loop_result["n_llm_labeled"],
        "n_anomalies_found": loop_result["n_anomalies_found"],
        "n_normals_found": loop_result["n_normals_found"],
        "n_parse_errors": loop_result["n_parse_errors"],
        "setfit_skipped": loop_result["setfit_skipped"],
        "anomaly_threshold": args.anomaly_threshold,
        "roc_auc": round(roc_auc, 6),
        "pr_auc": round(pr_auc, 6),
        "backend": args.backend,
        "llm_model": annotator.model,
        "encoder": encoder_short,
        "device": args.device,
    }
    metrics_df = pd.DataFrame([metrics_row])
    metrics_path = os.path.join(out_dir, f"{args.dataset}.csv")
    write_header = not os.path.exists(metrics_path)
    metrics_df.to_csv(metrics_path, mode="a", header=write_header, index=False)
    print(f"\nMetrics {'created' if write_header else 'appended'} to: {metrics_path}")

    # LLM labels CSV — append per-run rows (for quality inspection)
    labels_path_out = os.path.join(out_dir, f"{args.dataset}_llm_labels.csv")
    write_header_labels = not os.path.exists(labels_path_out)
    loop_result["llm_labels_df"].to_csv(labels_path_out, mode="a", header=write_header_labels, index=False)
    print(f"LLM labels {'created' if write_header_labels else 'appended'} to: {labels_path_out}")


if __name__ == "__main__":
    main()
