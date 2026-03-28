"""
test_pipeline_integration.py
------------------------------
End-to-end integration test for the LLM-guided pipeline using synthetic data.
No real datasets needed — runs locally on CPU.

Tests:
  1. Synthetic embeddings + texts created
  2. DeepSVDD trains and scores samples
  3. LLM (Groq) annotates selected samples
  4. SetFit fine-tunes on LLM labels
  5. DeepSAD trains on SetFit embeddings + LLM labels
  6. ROC-AUC and PR-AUC computed
  7. CSVs saved to data/llm_results/integration_test/

Usage:
    python scripts/test_pipeline_integration.py
    python scripts/test_pipeline_integration.py --n_llm_calls 5 --backend groq
"""

import argparse
import os
import sys
import numpy as np


def make_synthetic_data(n_normal=200, n_anomaly=20, emb_dim=512, seed=42):
    """Create synthetic embeddings and texts that mimic the told_br distribution."""
    rng = np.random.default_rng(seed)
    # Normals: centered at origin with small variance
    X_normal = rng.normal(loc=0.0, scale=0.5, size=(n_normal, emb_dim)).astype("float32")
    # Anomalies: shifted mean + larger variance
    X_anomaly = rng.normal(loc=2.0, scale=1.0, size=(n_anomaly, emb_dim)).astype("float32")

    embeddings = np.vstack([X_normal, X_anomaly])
    labels = np.array([0] * n_normal + [1] * n_anomaly)

    # Shuffle
    idx = rng.permutation(len(labels))
    embeddings = embeddings[idx]
    labels = labels[idx]

    # Synthetic texts: 10 normal templates, 10 anomaly templates (told_br style)
    normal_texts = [
        "Hoje foi um dia cansativo mas consegui terminar o projeto.",
        "Fui ao mercado e comprei frutas para o almoço.",
        "O filme que assisti ontem foi muito interessante.",
        "Estou estudando para a prova de amanhã.",
        "A reunião foi produtiva e todos participaram bem.",
        "Minha família se reuniu para comemorar o aniversário.",
        "O tempo está ótimo para uma caminhada no parque.",
        "Aprendi uma receita nova e ficou deliciosa.",
        "O projeto final está quase pronto, falta só revisar.",
        "Tive uma conversa muito agradável com meu amigo.",
    ]
    anomaly_texts = [
        "Esse negro macumbeiro não merece nada mesmo, lixo da sociedade.",
        "Mulher não sabe dirigir, deveria ficar em casa.",
        "Judeu safado, sempre roubando todo mundo.",
        "Nordestino é tudo burro mesmo, não presta.",
        "Viado nojento, vai se tratar.",
        "Essa raça de vagabundo só serve pra causar problema.",
        "Odeio esse povo de favela, bando de ladrão.",
        "Porta-malas é o lugar certo pra preto.",
        "Gordo imundo, se cuida monstro.",
        "Deficiente não merece espaço no mercado de trabalho.",
    ]

    texts = []
    normal_count = 0
    anomaly_count = 0
    for label in labels:
        if label == 0:
            texts.append(normal_texts[normal_count % len(normal_texts)])
            normal_count += 1
        else:
            texts.append(anomaly_texts[anomaly_count % len(anomaly_texts)])
            anomaly_count += 1
    texts = np.array(texts)

    return texts, labels, embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_path", type=str, default=".")
    parser.add_argument("--backend", type=str, default="groq", choices=["groq", "gemini", "openai"])
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--n_llm_calls", type=int, default=10,
                        help="Number of LLM calls (keep small for cost/time).")
    parser.add_argument("--strategy", type=str, default="score_guided",
                        choices=["random", "score_guided"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_path = os.path.abspath(args.project_path)
    sys.path.insert(0, os.path.join(project_path, "src"))

    import pandas as pd
    from deepod.models import DeepSVDD, DeepSAD
    from sklearn.metrics import roc_auc_score, average_precision_score
    from pipeline.llm_runner import LLMAnnotator, run_llm_active_loop

    # ------------------------------------------------------------------
    # 1. Synthetic data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("INTEGRATION TEST — LLM-guided pipeline (synthetic data)")
    print("=" * 60)
    print("\n[DATA] Generating synthetic told_br-like data...")

    texts, labels, embeddings = make_synthetic_data(
        n_normal=200, n_anomaly=20, emb_dim=512, seed=args.seed
    )
    print(f"       {len(texts)} samples | {labels.sum()} anomalies ({labels.mean():.1%})")

    # ------------------------------------------------------------------
    # 2. LLM annotator
    # ------------------------------------------------------------------
    print(f"\n[LLM] Initialising {args.backend} annotator...")
    annotator = LLMAnnotator(backend=args.backend, api_key=args.api_key)
    print(f"      Model: {annotator.model}")

    # ------------------------------------------------------------------
    # 3. Pipeline
    # ------------------------------------------------------------------
    print(f"\n[RUN] strategy={args.strategy} | n_llm_calls={args.n_llm_calls} | seed={args.seed}")
    loop_result = run_llm_active_loop(
        texts=texts,
        embeddings=embeddings,
        labels=labels,
        dataset_name="told_br",
        annotator=annotator,
        unsup_model_cls=DeepSVDD,
        semisup_model_cls=DeepSAD,
        setfit_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        strategy=args.strategy,
        n_llm_calls=args.n_llm_calls,
        anomaly_score_threshold=0.6,
        min_anomalies_required=2,
        test_size=0.2,
        random_state=args.seed,
        device="cpu",
        verbose=True,
    )

    # ------------------------------------------------------------------
    # 4. Evaluate
    # ------------------------------------------------------------------
    test_idx = loop_result["test_idx"]
    y_test = labels[test_idx]
    test_scores = loop_result["test_scores"]

    roc_auc = roc_auc_score(y_test, test_scores)
    pr_auc = average_precision_score(y_test, test_scores)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  ROC-AUC : {roc_auc:.4f}")
    print(f"  PR-AUC  : {pr_auc:.4f}")
    print(f"  LLM labeled : {loop_result['n_llm_labeled']}"
          f"  (anomalies={loop_result['n_anomalies_found']} | normals={loop_result['n_normals_found']})")
    print(f"  Parse errors: {loop_result['n_parse_errors']}")

    # ------------------------------------------------------------------
    # 5. Save CSVs
    # ------------------------------------------------------------------
    out_dir = os.path.join(project_path, "data", "llm_results", "integration_test")
    os.makedirs(out_dir, exist_ok=True)

    metrics_row = {
        "dataset": "told_br_SYNTHETIC",
        "strategy": args.strategy,
        "n_llm_calls": args.n_llm_calls,
        "seed": args.seed,
        "n_llm_labeled": loop_result["n_llm_labeled"],
        "n_anomalies_found": loop_result["n_anomalies_found"],
        "n_normals_found": loop_result["n_normals_found"],
        "n_parse_errors": loop_result["n_parse_errors"],
        "anomaly_threshold": 0.6,
        "roc_auc": round(roc_auc, 6),
        "pr_auc": round(pr_auc, 6),
        "backend": args.backend,
        "llm_model": annotator.model,
        "device": "cpu",
    }
    metrics_path = os.path.join(out_dir, "metrics.csv")
    write_header = not os.path.exists(metrics_path)
    pd.DataFrame([metrics_row]).to_csv(metrics_path, mode="a", header=write_header, index=False)
    print(f"\nMetrics saved to: {metrics_path}")

    labels_path = os.path.join(out_dir, "llm_labels.csv")
    write_header_lbl = not os.path.exists(labels_path)
    loop_result["llm_labels_df"].to_csv(labels_path, mode="a", header=write_header_lbl, index=False)
    print(f"LLM labels saved to: {labels_path}")

    print("\nINTEGRATION TEST PASSED")


if __name__ == "__main__":
    main()
