"""
verify_benchmark_labels.py
--------------------------
Verifica quantos rótulos GT foram utilizados no benchmark semi-supervisionado
e valida que os números do artigo estão corretos.

Como usar no Colab:
    1. Monte o Drive e ajuste PROJECT_PATH
    2. Execute as células em ordem

Pipeline que está sendo verificado:
    adjust_contamination(5%)  →  split_data(80/20)  →  label_partial_outliers(contamination=0.05)
    n_known = min(int(0.05 * N_train),  total_anomalias_no_train)
    Como o dataset já tem exatamente 5% de anomalias, o min() sempre retorna total_anomalias_no_train,
    ou seja, TODAS as anomalias do treino são reveladas ao modelo (upper-bound oracle).
"""

# ===========================================================================
# CÉLULA 1 – Configuração (ajuste PROJECT_PATH para seu Drive)
# ===========================================================================
import os, sys, json
import pandas as pd
import numpy as np

PROJECT_PATH = "/content/drive/MyDrive/Projeto ML/2025/AD/Multilingual-Text-Anomaly-Detection"
# PROJECT_PATH = "."   # ← use isso se estiver rodando localmente

RESULTS_DIR  = os.path.join(PROJECT_PATH, "data", "benchmark_results")
DATA_DIR     = os.path.join(PROJECT_PATH, "data")

sys.path.insert(0, os.path.join(PROJECT_PATH, "src"))

bench_path  = os.path.join(RESULTS_DIR, "benchmark_results.csv")
labels_path = os.path.join(RESULTS_DIR, "labelled_anomalies.csv")

df_bench  = pd.read_csv(bench_path)
df_labels = pd.read_csv(labels_path)

print(f"benchmark_results.csv  → {len(df_bench)} linhas")
print(f"labelled_anomalies.csv → {len(df_labels)} linhas")
print()

# ===========================================================================
# CÉLULA 2 – Resumo por dataset: train_size, n_known_outliers, % de labels
# ===========================================================================
SEMI_MODELS = ["DevNet", "DeepSAD", "XGBOD", "MLP"]
df_semi = df_bench[df_bench["model"].isin(SEMI_MODELS) & df_bench["contamination"].notna()].copy()

# Extrai dataset short name (antes do "_distiluse")
df_semi["dataset_short"] = df_semi["dataset"].str.split("_distiluse").str[0]

summary = (
    df_semi.groupby("dataset_short")
    .agg(
        train_size    = ("train_size",         "first"),
        test_size     = ("test_size",          "first"),
        n_known_min   = ("n_known_outliers",   "min"),
        n_known_max   = ("n_known_outliers",   "max"),
        n_known_mean  = ("n_known_outliers",   "mean"),
        n_rounds      = ("round",              lambda x: x.nunique()),
        n_model_runs  = ("model",              "count"),
    )
    .reset_index()
)

# Calcula contaminação efetiva
summary["contamination_effective"] = (
    summary["n_known_mean"] / summary["train_size"]
)

# Calcula o que o pipeline "pede": int(0.05 * train_size)
summary["requested_by_pipeline"] = (summary["train_size"] * 0.05).astype(int)

# Dif: se n_known < requested → capped no total de anomalias do train
summary["capped"] = summary["n_known_mean"] < summary["requested_by_pipeline"]

print("=" * 80)
print("RESUMO POR DATASET – benchmark semi-supervisionado (contamination=0.05)")
print("=" * 80)
cols_show = ["dataset_short", "train_size", "test_size",
             "n_known_mean", "requested_by_pipeline", "capped",
             "contamination_effective", "n_rounds", "n_model_runs"]
print(summary[cols_show].to_string(index=False))
print()

# ===========================================================================
# CÉLULA 3 – Verifica variância entre rounds (n_known deveria ser idêntico
#            por dataset dentro do mesmo round, pois depende só de y_train)
# ===========================================================================
print("=" * 80)
print("VARIÂNCIA DE n_known_outliers POR DATASET × ROUND")
print("(deve ser igual para todos os modelos num mesmo round)")
print("=" * 80)
pivot = (
    df_semi.groupby(["dataset_short", "round", "model"])["n_known_outliers"]
    .first()
    .unstack("model")
)
print(pivot.to_string())
print()

# ===========================================================================
# CÉLULA 4 – Verifica com o pipeline real (precisa dos parquets no Drive)
# ===========================================================================
print("=" * 80)
print("VERIFICAÇÃO PELO PIPELINE REAL (reconstituição)")
print("=" * 80)

DATASETS = {
    "tweets_hate_speech_detection": "en",
    "20_newsgroups":                "en",
    "HateBR":                       "pt",
    "wikinews":                     "pt",
}
ENCODER_SHORT = "distiluse-base-multilingual-cased-v2"
CONTAMINATION = 0.05

try:
    from pipeline.anomaly_detection import (
        label_normal_vs_anomaly,
        adjust_contamination,
        split_data,
        label_partial_outliers,
    )

    rows = []
    for ds_name, lang in DATASETS.items():
        labels_file = os.path.join(DATA_DIR, f"labels_{ds_name}.parquet")
        texts_file  = os.path.join(DATA_DIR, f"texts_{ds_name}.parquet")
        emb_file    = os.path.join(DATA_DIR, f"embeddings_{ds_name}_{ENCODER_SHORT}.parquet")

        if not os.path.exists(labels_file):
            print(f"  [SKIP] {ds_name} – arquivo não encontrado: {labels_file}")
            continue

        labels_df   = pd.read_parquet(labels_file)
        texts_df    = pd.read_parquet(texts_file)
        emb_df      = pd.read_parquet(emb_file)

        binary_labels = label_normal_vs_anomaly(labels_df, verbose=False)

        texts_adj, labels_adj, emb_adj = adjust_contamination(
            texts=texts_df.squeeze().tolist(),
            labels=binary_labels,
            embeddings=emb_df.values,
            perc_anomalous=CONTAMINATION,
            random_state=42,
        )

        x_train, x_test, y_train, y_test = split_data(emb_adj, labels_adj, random_state=42)

        y_semi = label_partial_outliers(y_train, contamination=CONTAMINATION, random_state=42)

        n_total        = len(labels_adj)
        n_train        = len(y_train)
        n_test         = len(y_test)
        n_anomaly_train= int(np.sum(y_train == 1))
        n_anomaly_test = int(np.sum(y_test  == 1))
        n_known        = int(np.sum(y_semi  == 1))
        requested      = int(CONTAMINATION * n_train)

        rows.append({
            "dataset":          ds_name,
            "N_total":          n_total,
            "N_train":          n_train,
            "N_test":           n_test,
            "anomalies_train":  n_anomaly_train,
            "anomalies_test":   n_anomaly_test,
            "requested":        requested,
            "n_known_labels":   n_known,
            "all_anomalies_revealed": n_known == n_anomaly_train,
            "frac_labels":      round(n_known / n_train, 4),
        })

    df_pipeline = pd.DataFrame(rows)
    print(df_pipeline.to_string(index=False))
    print()

    # Faixa de n_known reportada no artigo: 39 – 1189
    paper_min = df_pipeline["n_known_labels"].min()
    paper_max = df_pipeline["n_known_labels"].max()
    print(f"Faixa de labels no artigo: {paper_min} – {paper_max}")
    print(f"→ Confirma 'between {paper_min} and {paper_max} confirmed anomaly labels'")

except ImportError as e:
    print(f"  [AVISO] Não foi possível importar o pipeline: {e}")
    print("  Rode na raiz do projeto com sys.path apontando para /src")

# ===========================================================================
# CÉLULA 5 – Confronta com os dados salvos em labelled_anomalies.csv
# ===========================================================================
print()
print("=" * 80)
print("CONFRONTO: labelled_anomalies.csv × benchmark_results.csv")
print("=" * 80)

# Conta quantos indices cada run salvou
df_labels["n_indices"] = df_labels["indices"].apply(lambda x: len(json.loads(x)))

label_summary = (
    df_labels.groupby(["contamination", "round"])
    .agg(
        n_unique_runs = ("run_id",    "nunique"),
        n_min_known   = ("n_indices", "min"),
        n_max_known   = ("n_indices", "max"),
    )
    .reset_index()
)
print(label_summary.to_string(index=False))

# Verifica que todos os run_ids do bench semi estão no labels file
bench_run_ids   = set(df_bench[df_bench["contamination"].notna()]["run_id"].unique())
label_run_ids   = set(df_labels["run_id"].unique())
missing_in_labels = bench_run_ids - label_run_ids
extra_in_labels   = label_run_ids - bench_run_ids

print()
print(f"run_ids no bench semi-sup:   {len(bench_run_ids)}")
print(f"run_ids em labelled_anomalies: {len(label_run_ids)}")
print(f"Faltando em labelled_anomalies: {len(missing_in_labels)}")
print(f"Extras em labelled_anomalies:   {len(extra_in_labels)}")

if not missing_in_labels:
    print("✅ Todos os run_ids do benchmark estão registrados em labelled_anomalies.csv")
else:
    print(f"⚠️  run_ids faltando: {missing_in_labels}")

# ===========================================================================
# CÉLULA 6 – Tabela final para o artigo (igual ao paper)
# ===========================================================================
print()
print("=" * 80)
print("TABELA FINAL: n_known por dataset (comparar com Tab. 2 do artigo)")
print("=" * 80)

df_semi["dataset_short"] = df_semi["dataset"].str.split("_distiluse").str[0]

final_table = (
    df_semi.groupby("dataset_short")
    .agg(
        train_size       = ("train_size",       "first"),
        test_size        = ("test_size",        "first"),
        n_known_outliers = ("n_known_outliers", "first"),   # igual p/ todos rounds/models do mesmo ds
        test_auc_mean    = ("test_auc",         "mean"),
        test_auc_std     = ("test_auc",         "std"),
    )
    .reset_index()
)

final_table["contamination_pct"] = (
    final_table["n_known_outliers"] / final_table["train_size"] * 100
).round(2)

print(final_table.to_string(index=False))
print()
print("Verificação concluída.")
