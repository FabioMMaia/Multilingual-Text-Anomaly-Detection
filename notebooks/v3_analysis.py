"""
v3_analysis.py — Visualizações do pilot v3 (48 runs)

Cada função responde uma pergunta analítica específica.
Execute em Colab ou Jupyter: `%run notebooks/v3_analysis.py` ou importe as funções.

Perguntas cobertas:
  Q1. 7B vs 14B — o modelo maior compensa o custo?
  Q2. Estratégia — random vs score_guided vs diversity
  Q3. N — N=50 é suficiente ou N=200 acrescenta?
  Q4. Dificuldade por dataset — panorama geral
  Q5. score_guided N=50 — quando e por que colapsa?
  Q6. LLM quality — agreement/precision/recall por config
  Q7. Custo vs benefício — tempo × ROC
  Q8. Ranking de configs — top configs por dataset
"""

import pandas as pd
import numpy as np
import glob
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no popup windows
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# ── palette ──────────────────────────────────────────────────────────────────
COLORS = {
    "qwen2.5-7b":  "#4C9BE8",
    "qwen2.5-14b": "#E8784C",
    "random":       "#5AB4AC",
    "score_guided": "#D8601E",
    "diversity":    "#8B5CF6",
    50:  "#60A5FA",
    200: "#1E40AF",
}
DATASET_ORDER = ["20_newsgroups", "tweets_hs", "wikinews", "told_br"]
DATASET_LABELS = {
    "20_newsgroups": "20news\n(EN, topic)",
    "tweets_hs":     "tweets_hs\n(EN, hate)",
    "wikinews":      "wikinews\n(PT, topic)",
    "told_br":       "told_br\n(PT, hate)",
}


# ── loader ────────────────────────────────────────────────────────────────────
def load_v3(results_dir="data/llm_results/v3"):
    files = glob.glob(f"{results_dir}/**/*.csv", recursive=True)
    files = [f for f in files if "llm_labels" not in f and "THIS_ONE" not in f]
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["model"] = df["llm_model"].str.extract(r"(qwen2\.5-\d+b)", expand=False)
    df["dataset"] = pd.Categorical(df["dataset"], categories=DATASET_ORDER, ordered=True)
    return df.sort_values(["dataset", "model", "strategy", "n_llm_calls"])


# ─────────────────────────────────────────────────────────────────────────────
# Q1 — 7B vs 14B
# ─────────────────────────────────────────────────────────────────────────────
def plot_q1_model_comparison(df):
    """
    Q1: O modelo 14B justifica o dobro de VRAM e tempo vs 7B?
    Heatmap de Δ ROC (14B − 7B) por dataset × config.
    """
    pivot = df.pivot_table(
        index=["strategy", "n_llm_calls"],
        columns=["dataset"],
        values="roc_auc",
        aggfunc=lambda x: x.values[0] if len(x) >= 1 else np.nan,
    )
    # Separate per model then diff
    p14 = df[df.model == "qwen2.5-14b"].pivot_table(
        index=["strategy", "n_llm_calls"], columns="dataset", values="roc_auc"
    )
    p7 = df[df.model == "qwen2.5-7b"].pivot_table(
        index=["strategy", "n_llm_calls"], columns="dataset", values="roc_auc"
    )
    delta = (p14 - p7).reindex(columns=DATASET_ORDER)
    delta.index = [f"{s} N={n}" for s, n in delta.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        delta, annot=True, fmt=".3f", center=0,
        cmap="RdBu_r", linewidths=0.5, ax=ax,
        xticklabels=[DATASET_LABELS[d] for d in DATASET_ORDER],
    )
    ax.set_title("Q1 — Δ ROC-AUC (14B − 7B)\nPositivo = 14B melhor", fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("Estratégia × N")
    plt.tight_layout()
    plt.savefig("data/llm_results/v3/q1_model_comparison.png", dpi=150)
    print("Saved: q1_model_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Q2 — Estratégia
# ─────────────────────────────────────────────────────────────────────────────
def plot_q2_strategy(df):
    """
    Q2: Qual estratégia de seleção funciona melhor?
    Barras agrupadas por dataset, cor = estratégia, facet = N.
    """
    # média dos dois modelos por config
    agg = df.groupby(["dataset", "strategy", "n_llm_calls"])["roc_auc"].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, n in zip(axes, [50, 200]):
        sub = agg[agg.n_llm_calls == n]
        x = np.arange(len(DATASET_ORDER))
        width = 0.25
        for i, strategy in enumerate(["random", "score_guided", "diversity"]):
            vals = [
                sub[(sub.dataset == ds) & (sub.strategy == strategy)]["roc_auc"].values
                for ds in DATASET_ORDER
            ]
            vals = [v[0] if len(v) else np.nan for v in vals]
            ax.bar(x + i * width, vals, width, label=strategy,
                   color=COLORS[strategy], alpha=0.85)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="chance")
        ax.set_title(f"N = {n}", fontsize=12)
        ax.set_xticks(x + width)
        ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER], fontsize=9)
        ax.set_ylabel("ROC-AUC (média 7B+14B)")
        ax.set_ylim(0.35, 1.0)
        ax.legend(fontsize=9)
    fig.suptitle("Q2 — Estratégia de seleção × Dataset", fontsize=14)
    plt.tight_layout()
    plt.savefig("data/llm_results/v3/q2_strategy.png", dpi=150)
    print("Saved: q2_strategy.png")


# ─────────────────────────────────────────────────────────────────────────────
# Q3 — N=50 vs N=200
# ─────────────────────────────────────────────────────────────────────────────
def plot_q3_n_budget(df):
    """
    Q3: N=50 é suficiente? Quanto N=200 adiciona?
    Scatter: ROC N=50 (eixo x) vs ROC N=200 (eixo y). Pontos acima da diagonal = N=200 melhor.
    """
    agg = df.groupby(["dataset", "model", "strategy", "n_llm_calls"])["roc_auc"].first().reset_index()
    p50  = agg[agg.n_llm_calls == 50].set_index(["dataset", "model", "strategy"])["roc_auc"]
    p200 = agg[agg.n_llm_calls == 200].set_index(["dataset", "model", "strategy"])["roc_auc"]
    combined = pd.DataFrame({"N50": p50, "N200": p200}).dropna().reset_index()

    fig, ax = plt.subplots(figsize=(7, 6))
    for ds in DATASET_ORDER:
        sub = combined[combined.dataset == ds]
        ax.scatter(sub.N50, sub.N200, label=DATASET_LABELS[ds], s=60, alpha=0.8)

    lims = [0.35, 1.0]
    ax.plot(lims, lims, "k--", linewidth=0.8, label="N50 = N200")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("ROC-AUC  N=50", fontsize=11)
    ax.set_ylabel("ROC-AUC  N=200", fontsize=11)
    ax.set_title("Q3 — N=50 vs N=200\nPontos acima da diagonal: N=200 melhora", fontsize=13)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("data/llm_results/v3/q3_n_budget.png", dpi=150)
    print("Saved: q3_n_budget.png")


# ─────────────────────────────────────────────────────────────────────────────
# Q4 — Dificuldade por dataset
# ─────────────────────────────────────────────────────────────────────────────
def plot_q4_dataset_difficulty(df):
    """
    Q4: Panorama geral — box plot de ROC por dataset, todos os configs.
    Mostra spread e mediana (robustez ao grid de configs).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [df[df.dataset == ds]["roc_auc"].values for ds in DATASET_ORDER]
    bp = ax.boxplot(data, patch_artist=True, medianprops=dict(color="black", linewidth=2))
    palette = ["#60A5FA", "#34D399", "#FBBF24", "#F87171"]
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="chance")
    ax.set_xticks(range(1, len(DATASET_ORDER) + 1))
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER])
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Q4 — Dificuldade por dataset\n(todos os 12 configs por dataset)", fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig("data/llm_results/v3/q4_dataset_difficulty.png", dpi=150)
    print("Saved: q4_dataset_difficulty.png")


# ─────────────────────────────────────────────────────────────────────────────
# Q5 — score_guided N=50 colapso
# ─────────────────────────────────────────────────────────────────────────────
def plot_q5_score_guided_collapse(df):
    """
    Q5: Por que score_guided N=50 colapsa?
    Compara n_anomalies_found e roc_auc para score_guided vs random, N=50.
    """
    sub = df[df.n_llm_calls == 50].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # anomalias encontradas
    ax = axes[0]
    agg = sub.groupby(["dataset", "strategy"])["n_anomalies_found"].mean().reset_index()
    x = np.arange(len(DATASET_ORDER))
    w = 0.28
    for i, strat in enumerate(["random", "score_guided", "diversity"]):
        vals = [agg[(agg.dataset == ds) & (agg.strategy == strat)]["n_anomalies_found"].values for ds in DATASET_ORDER]
        vals = [v[0] if len(v) else 0 for v in vals]
        ax.bar(x + i * w, vals, w, label=strat, color=COLORS[strat], alpha=0.85)
    ax.set_xticks(x + w)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER], fontsize=9)
    ax.set_ylabel("Anomalias encontradas (média 7B+14B)")
    ax.set_title("Anomalias rotuladas pelo LLM")
    ax.legend(fontsize=9)

    # ROC resultante
    ax = axes[1]
    agg2 = sub.groupby(["dataset", "strategy"])["roc_auc"].mean().reset_index()
    for i, strat in enumerate(["random", "score_guided", "diversity"]):
        vals = [agg2[(agg2.dataset == ds) & (agg2.strategy == strat)]["roc_auc"].values for ds in DATASET_ORDER]
        vals = [v[0] if len(v) else np.nan for v in vals]
        ax.bar(x + i * w, vals, w, label=strat, color=COLORS[strat], alpha=0.85)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x + w)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER], fontsize=9)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("ROC-AUC resultante")
    ax.legend(fontsize=9)
    ax.set_ylim(0.3, 1.0)

    fig.suptitle("Q5 — score_guided N=50: colapso de anomalias → ROC degrada", fontsize=13)
    plt.tight_layout()
    plt.savefig("data/llm_results/v3/q5_score_guided_collapse.png", dpi=150)
    print("Saved: q5_score_guided_collapse.png")


# ─────────────────────────────────────────────────────────────────────────────
# Q6 — Qualidade LLM
# ─────────────────────────────────────────────────────────────────────────────
def plot_q6_llm_quality(df):
    """
    Q6: O LLM anota corretamente? agreement / precision / recall por dataset e modelo.
    """
    agg = df.groupby(["dataset", "model"])[["llm_agreement", "llm_precision", "llm_recall"]].mean().reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    metrics = ["llm_agreement", "llm_precision", "llm_recall"]
    titles  = ["Agreement", "Precision", "Recall"]

    for ax, metric, title in zip(axes, metrics, titles):
        x = np.arange(len(DATASET_ORDER))
        w = 0.35
        for i, model in enumerate(["qwen2.5-7b", "qwen2.5-14b"]):
            vals = [
                agg[(agg.dataset == ds) & (agg.model == model)][metric].values
                for ds in DATASET_ORDER
            ]
            vals = [v[0] if len(v) else np.nan for v in vals]
            ax.bar(x + i * w, vals, w, label=model.replace("qwen2.5-", ""),
                   color=COLORS[model], alpha=0.85)
        ax.set_xticks(x + w / 2)
        ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER], fontsize=8)
        ax.set_title(title)
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=9)

    fig.suptitle("Q6 — Qualidade das anotações LLM por dataset e modelo", fontsize=13)
    plt.tight_layout()
    plt.savefig("data/llm_results/v3/q6_llm_quality.png", dpi=150)
    print("Saved: q6_llm_quality.png")


# ─────────────────────────────────────────────────────────────────────────────
# Q7 — Custo vs benefício (tempo × ROC)
# ─────────────────────────────────────────────────────────────────────────────
def plot_q7_cost_benefit(df):
    """
    Q7: ROC vs tempo de execução — vale gastar mais tempo?
    Scatter: elapsed_min (x) vs roc_auc (y), cor = dataset, forma = modelo.
    """
    sub = df.dropna(subset=["elapsed_seconds"]).copy()
    sub["elapsed_min"] = sub["elapsed_seconds"] / 60

    fig, ax = plt.subplots(figsize=(9, 6))
    markers = {"qwen2.5-7b": "o", "qwen2.5-14b": "s"}
    palette = {"20_newsgroups": "#60A5FA", "tweets_hs": "#34D399",
               "wikinews": "#FBBF24", "told_br": "#F87171"}

    for _, row in sub.iterrows():
        ax.scatter(
            row["elapsed_min"], row["roc_auc"],
            color=palette.get(str(row["dataset"]), "gray"),
            marker=markers.get(row["model"], "o"),
            s=70, alpha=0.75,
        )

    # legend patches
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    ds_patches = [mpatches.Patch(color=c, label=DATASET_LABELS[d]) for d, c in palette.items()]
    model_lines = [mlines.Line2D([], [], color="gray", marker=m, linestyle="None",
                                 label=m.replace("qwen2.5-", "Qwen ")) for m, m in markers.items()]
    ax.legend(handles=ds_patches + model_lines, fontsize=8, loc="lower right")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Tempo de execução (min)", fontsize=11)
    ax.set_ylabel("ROC-AUC", fontsize=11)
    ax.set_title("Q7 — Custo × Benefício: tempo vs ROC\n● = 7B  ■ = 14B", fontsize=13)
    plt.tight_layout()
    plt.savefig("data/llm_results/v3/q7_cost_benefit.png", dpi=150)
    print("Saved: q7_cost_benefit.png")


# ─────────────────────────────────────────────────────────────────────────────
# Q8 — Ranking de configs por dataset
# ─────────────────────────────────────────────────────────────────────────────
def plot_q8_top_configs(df):
    """
    Q8: Qual a melhor config (modelo × estratégia × N) por dataset?
    Heatmap de ROC para todas as 12 configs × 4 datasets.
    """
    df2 = df.copy()
    df2["config"] = df2["model"].str.replace("qwen2.5-", "") + " / " + df2["strategy"] + " / N=" + df2["n_llm_calls"].astype(str)

    pivot = df2.pivot_table(index="config", columns="dataset", values="roc_auc").reindex(columns=DATASET_ORDER)

    # sort by mean ROC
    pivot = pivot.assign(mean=pivot.mean(axis=1)).sort_values("mean", ascending=False).drop(columns="mean")

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(
        pivot, annot=True, fmt=".3f",
        cmap="YlOrRd", linewidths=0.4, ax=ax,
        xticklabels=[DATASET_LABELS[d] for d in DATASET_ORDER],
        vmin=0.45, vmax=0.96,
    )
    ax.set_title("Q8 — ROC-AUC por config × dataset\n(ordenado por média)", fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("Config (modelo / estratégia / N)")
    plt.tight_layout()
    plt.savefig("data/llm_results/v3/q8_top_configs.png", dpi=150)
    print("Saved: q8_top_configs.png")


# ─────────────────────────────────────────────────────────────────────────────
# PDF export — todas as perguntas num único arquivo
# ─────────────────────────────────────────────────────────────────────────────
def export_pdf(df, path="data/llm_results/v3/v3_analysis.pdf"):
    from matplotlib.backends.backend_pdf import PdfPages

    questions = [
        ("Q1 — 7B vs 14B", "O modelo 14B justifica o dobro de VRAM e tempo?\nHeatmap de Δ ROC (14B − 7B) por estratégia × dataset.", plot_q1_model_comparison),
        ("Q2 — Estratégia de seleção", "Qual estratégia funciona melhor: random, score_guided ou diversity?\nBarras por dataset, facetado por N.", plot_q2_strategy),
        ("Q3 — Budget de anotações", "N=50 é suficiente ou N=200 acrescenta?\nScatter N=50 vs N=200 — pontos acima da diagonal indicam ganho.", plot_q3_n_budget),
        ("Q4 — Dificuldade por dataset", "Quão difícil é cada dataset? Boxplot de ROC sobre todos os configs.", plot_q4_dataset_difficulty),
        ("Q5 — Colapso score_guided N=50", "Por que score_guided com N=50 colapsa?\nAnomalias rotuladas vs ROC resultante.", plot_q5_score_guided_collapse),
        ("Q6 — Qualidade das anotações LLM", "O LLM anota corretamente? Agreement, precision e recall por modelo e dataset.", plot_q6_llm_quality),
        ("Q7 — Custo × Benefício", "Vale gastar mais tempo? Scatter de tempo de execução vs ROC-AUC.", plot_q7_cost_benefit),
        ("Q8 — Ranking de configs", "Qual a melhor config por dataset? Heatmap de todas as 12 configs × 4 datasets.", plot_q8_top_configs),
    ]

    with PdfPages(path) as pdf:
        # ── capa ──
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor("#1E293B")
        fig.text(0.5, 0.62, "v3 Pilot — Analysis Report", ha="center", va="center",
                 fontsize=28, color="white", fontweight="bold")
        fig.text(0.5, 0.52, "4 datasets × 3 strategies × 2 N × 2 models  |  seed=42  |  48 runs",
                 ha="center", va="center", fontsize=13, color="#94A3B8")
        fig.text(0.5, 0.44, "Qwen 2.5 7B & 14B Q4_K_M  ·  distiluse-base-multilingual-cased-v2",
                 ha="center", va="center", fontsize=11, color="#64748B")

        toc_lines = [f"  {q}  —  {desc.splitlines()[0]}" for q, desc, _ in questions]
        fig.text(0.12, 0.30, "\n".join(toc_lines), ha="left", va="top",
                 fontsize=9, color="#CBD5E1", family="monospace")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── uma página por pergunta ──
        for q_title, q_desc, plot_fn in questions:
            # header page with question text
            fig_h = plt.figure(figsize=(11, 1.6))
            fig_h.patch.set_facecolor("#0F172A")
            fig_h.text(0.05, 0.72, q_title, ha="left", va="top",
                       fontsize=16, color="white", fontweight="bold")
            fig_h.text(0.05, 0.25, q_desc, ha="left", va="top",
                       fontsize=10, color="#94A3B8")
            pdf.savefig(fig_h, bbox_inches="tight")
            plt.close(fig_h)

            # actual plot — call function but intercept the figure
            plt.ioff()
            plot_fn(df)
            fig_plot = plt.gcf()
            pdf.savefig(fig_plot, bbox_inches="tight")
            plt.close(fig_plot)

        # ── metadata ──
        from datetime import datetime
        d = pdf.infodict()
        d["Title"] = "v3 Pilot Analysis"
        d["Author"] = "v3_analysis.py"
        d["CreationDate"] = datetime.now()

    print(f"PDF saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_v3()
    print(f"Loaded {len(df)} runs\n")

    export_pdf(df)
    print("\nDone. PDF saved to data/llm_results/v3/v3_analysis.pdf")
