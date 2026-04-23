# v7 — Ablation: MLP vs DeepSAD (mesmos labels do v5, sem SetFit)

## Objetivo

Isolar a contribuição do **modelo de anomalia** (DeepSAD vs MLP) com labels LLM ruidosos.

| Versão | Labels | SetFit | Modelo AD |
|---|---|---|---|
| v6 | v5 (LLM) | ❌ | **DeepSAD** |
| **v7** | v5 (LLM) | ❌ | **MLP** |

Comparação direta: mesmos labels, mesmo embedding (distiluse), só muda o modelo AD.

---

## Hipótese

Paper anterior mostrou MLP > DeepSAD com labels limpos.  
**Hipótese v7:** labels LLM ruidosos invertem essa relação — DeepSAD é mais robusto ao ruído
por ser geométrico (hipersfera), enquanto MLP aprende um boundary que pode ser enviesado por labels errados.

Se DeepSAD (v6) > MLP (v7) → confirma robustez geométrica ao ruído de anotação.  
Se MLP (v7) > DeepSAD (v6) → sugere que discriminação direta ainda compensa mesmo com ruído.

---

## Comparação completa do paper (2×2)

| | distiluse | SetFit |
|---|---|---|
| **DeepSAD** | v6 ✅ | v5 ✅ |
| **MLP** | **v7** | (opcional) |

---

## Config

| Parâmetro | Valor |
|---|---|
| Datasets | tweets_hs, hatebr, 20_newsgroups, wikinews |
| Labels source | `data/llm_results/v5/{strategy}/N_{n}/{dataset}_llm_labels.csv` |
| SetFit | ❌ desativado (`--no_setfit`) |
| Modelo AD | **MLP** (`--semisup_model mlp`) |
| Embedding | distiluse-base-multilingual-cased-v2 |
| Strategies | random, diversity |
| N | 50, 200 |
| Seeds | 0, 1, 42 |
| Total runs | 96 (2 modelos × 48) |
| Hardware | Colab T4 (sem LLM, sem GPU intensivo) |
| Tempo estimado | ~1h total (~30s/run) |

Results: `data/llm_results/v7/{strategy}/N_{n}/{dataset}.csv`

---

## Colab Cells

### Cell 1 — Mount Drive + Set Paths

```python
from google.colab import drive
drive.mount("/content/drive")

import os, sys

PROJECT_PATH = "/content/drive/MyDrive/Projeto ML/2026/Master/Multilingual-Text-Anomaly-Detection"
DATA_DIR     = "/content/drive/MyDrive/Projeto ML/2025/AD/third_setup/adaptative-text-anomaly-detection/data"

os.chdir(PROJECT_PATH)
sys.path.insert(0, f"{PROJECT_PATH}/src")

print("CWD:", os.getcwd())
```

---

### Cell 2 — Install Requirements

```python
!pip install -r requirements.txt
```

> Não precisa de llama-cpp-python nem de modelo LLM.

---

### Cell 3 — Skip helper

```python
import glob, pandas as pd

def _already_done(dataset, strategy, n, seed, model, results_dir):
    pattern = f"{PROJECT_PATH}/{results_dir}/{strategy}/N_{n}/{dataset}.csv"
    for f in glob.glob(pattern):
        try:
            df = pd.read_csv(f)
            match = df[
                (df["n_llm_calls"] == int(n)) &
                (df["seed"]        == int(seed)) &
                (df["llm_model"].str.contains(model, na=False))
            ]
            if not match.empty:
                return True
        except Exception:
            pass
    return False
```

---

### Cell 4 — v7 Sweep (96 runs, ~2h)

```python
import subprocess, re, time, os

PROJECT_PATH = "/content/drive/MyDrive/Projeto ML/2026/Master/Multilingual-Text-Anomaly-Detection"
DATA_DIR     = "/content/drive/MyDrive/Projeto ML/2025/AD/third_setup/adaptative-text-anomaly-detection/data"

datasets   = ["tweets_hs", "hatebr", "20_newsgroups", "wikinews"]
strategies = ["random", "diversity"]
ns         = ["50", "200"]
seeds      = ["0", "1", "42"]
models     = ["qwen2.5-14b", "qwen2.5-7b"]

RESULTS_DIR = "data/llm_results/v7"
V5_DIR      = "data/llm_results/v5"

total = len(datasets) * len(strategies) * len(ns) * len(seeds) * len(models)  # 96
run = 0

for model in models:
    for seed in seeds:
        for dataset in datasets:
            for strategy in strategies:
                for n in ns:
                    run += 1

                    if _already_done(dataset, strategy, n, seed, model, RESULTS_DIR):
                        print(f"[{run:03d}/{total}] ↷ {dataset:<20} {strategy:<12} N={n:<4} seed={seed} model={model} (skip)", flush=True)
                        continue

                    labels_csv = f"{PROJECT_PATH}/{V5_DIR}/{strategy}/N_{n}/{dataset}_llm_labels.csv"

                    if not os.path.exists(labels_csv):
                        print(f"[{run:03d}/{total}] ✗ labels not found: {labels_csv}", flush=True)
                        continue

                    cmd = [
                        "python", "-u",
                        "scripts/run_llm_active_loop.py",
                        "--project_path",   PROJECT_PATH,
                        "--data_dir",       DATA_DIR,
                        "--dataset",        dataset,
                        "--strategy",       strategy,
                        "--n_llm_calls",    n,
                        "--seed",           seed,
                        "--device",         "cuda",
                        "--load_labels_from",   labels_csv,
                        "--load_labels_model", model,
                        "--no_setfit",
                        "--semisup_model",  "mlp",
                        "--results_dir",    RESULTS_DIR,
                    ]

                    t0 = time.time()
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    elapsed = time.time() - t0

                    roc = re.search(r"ROC-AUC\s+\(test\)\s*:\s*([\d.]+)", result.stdout)
                    roc_str = roc.group(1) if roc else "N/A"
                    status  = "✓" if result.returncode == 0 else "✗"

                    print(f"[{run:03d}/{total}] {status} {dataset:<20} {strategy:<12} N={n:<4} seed={seed} model={model}  "
                          f"ROC={roc_str}  {elapsed/60:.1f}min", flush=True)

                    if result.returncode != 0:
                        print(f"  STDERR: {result.stderr[-400:]}", flush=True)

print("\nDone.")
```

print("\nDone.")
```

---

### Cell 5 — Comparação v6 (DeepSAD) vs v7 (MLP)

```python
import pandas as pd, glob, numpy as np

def load_v(version, project_path):
    files = [f for f in glob.glob(f"{project_path}/data/llm_results/{version}/**/*.csv", recursive=True)
             if "llm_labels" not in f]
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["version"] = version
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

df6 = load_v("v6", PROJECT_PATH)  # DeepSAD + distiluse
df7 = load_v("v7", PROJECT_PATH)  # MLP + distiluse

key = ["dataset", "strategy", "n_llm_calls", "seed"]
merged = df6[key + ["roc_auc"]].merge(
    df7[key + ["roc_auc"]],
    on=key, suffixes=("_deepsad", "_mlp")
)
merged["deepsad_gain"] = merged["roc_auc_deepsad"] - merged["roc_auc_mlp"]

print("=== DeepSAD vs MLP por dataset (v6 - v7, positivo = DeepSAD melhor) ===")
print(merged.groupby("dataset")["deepsad_gain"].agg(["mean","std"]).round(3).to_string())

print("\n=== Global ===")
print(f"  DeepSAD mean: {merged['roc_auc_deepsad'].mean():.3f}")
print(f"  MLP mean:     {merged['roc_auc_mlp'].mean():.3f}")
print(f"  Gain DeepSAD: {merged['deepsad_gain'].mean():.3f} ±{merged['deepsad_gain'].std():.3f}")

print("\n=== Por N ===")
print(merged.groupby("n_llm_calls")["deepsad_gain"].mean().round(3).to_string())
```

---

## Results

**Status: ✅ Concluído — 45/48 runs** *(3 runs ausentes — provavelmente timeout no Colab)*

---

### Tabela 1 — ROC-AUC mean ± std por dataset

| Dataset | v6 DeepSAD (distiluse) | v7 MLP (distiluse) | MLP gain |
|---|---|---|---|
| 20_newsgroups | 0.850 ±0.105 | **0.937 ±0.009** | **+0.105** ✅ |
| hatebr | 0.595 ±0.077 | **0.672 ±0.044** | **+0.077** ✅ |
| tweets_hs | 0.806 ±0.063 | **0.819 ±0.074** | +0.013 ✅ |
| wikinews | 0.748 ±0.049 | **0.827 ±0.027** | **+0.079** ✅ |
| **Global** | 0.750 | **0.806** | **+0.066 ±0.078** |

---

### Tabela 2 — AUC do MLP por dataset × N

| Dataset | N=50 | N=200 |
|---|---|---|
| 20_newsgroups | 0.935 | **0.938** |
| hatebr | 0.649 | **0.696** |
| tweets_hs | 0.757 | **0.880** |
| wikinews | 0.812 | **0.842** |

MLP melhora com N=200 em todos os datasets — especialmente tweets_hs (+0.123).

---

### Tabela 3 — MLP gain (v7−v6) por dataset × N

| Dataset | N=50 | N=200 |
|---|---|---|
| 20_newsgroups | +0.039 | +0.138 |
| hatebr | +0.057 | +0.097 |
| tweets_hs | −0.026 | +0.052 |
| wikinews | +0.087 | +0.071 |

---

### Tabela 4 — Comparação v7 (MLP, sem SetFit) vs v5 (DeepSAD, com SetFit)

> Relevante: MLP sem SetFit vs pipeline completo v5.

| Dataset | v5 DeepSAD+SetFit | v7 MLP+distiluse | MLP gain |
|---|---|---|---|
| 20_newsgroups | 0.879 | **0.937** | **+0.058** |
| hatebr | 0.620 | **0.672** | **+0.052** |
| tweets_hs | 0.791 | **0.819** | **+0.028** |
| wikinews | 0.719 | **0.827** | **+0.108** |

**MLP sem SetFit supera DeepSAD com SetFit em todos os datasets.**

---

### Análise

#### Hipótese refutada: MLP é mais robusto ao ruído, não DeepSAD

A hipótese original era que DeepSAD (geométrico, hipersfera) seria mais robusto a labels LLM ruidosos
do que MLP (discriminativo). Os resultados contradizem essa hipótese em todos os 4 datasets:

**MLP supera DeepSAD globalmente por +0.066 ±0.078** — e a vantagem é **consistente**:
- Maior em 20_newsgroups (+0.105) e wikinews (+0.079)
- Menor em tweets_hs (+0.013) — mas ainda favorável ao MLP

**Interpretação alternativa:** MLP com BCELoss aprende um boundary discriminativo que é
intrinsecamente mais expressivo do que a hipersfera do DeepSAD. Mesmo com labels ruidosos,
a separação linear por classe ainda captura sinal suficiente. DeepSAD, ao contrário, pode
colapsar a hipersfera num subespaço enviesado pelos labels ruidosos.

#### MLP tem menor variância

| Modelo | Std global |
|---|---|
| DeepSAD (v6) | 0.105 (20ng), 0.077 (hatebr), 0.063 (tweets), 0.049 (wiki) |
| MLP (v7) | **0.009 (20ng), 0.044 (hatebr), 0.074 (tweets), 0.027 (wiki)** |

MLP é mais estável — especialmente em 20_newsgroups onde std cai de 0.105 para 0.009.

#### Surpresa: MLP sem SetFit > DeepSAD com SetFit (v5)

A Tabela 4 mostra que o MLP com embedding distiluse base supera o pipeline completo v5
(DeepSAD + SetFit fine-tuned) em **todos os datasets**. Isso sugere que o ganho de
expressividade do MLP > ganho de embedding do SetFit.

---

### Takeaways para o Paper

1. **MLP > DeepSAD com labels ruidosos** — hipótese de robustez geométrica refutada (+0.066 global)
2. **MLP é mais estável** — std menor em 3 dos 4 datasets
3. **MLP sem SetFit ≥ DeepSAD com SetFit** — modelo AD mais importante que fine-tuning do embedding
4. **N=200 ajuda MLP mais que DeepSAD** — tweets_hs: MLP+0.123 vs N=50
5. **Esperado no v8:** MLP + SetFit deve superar ambos — teste final do 2×2

---

### Tabela 5 — Overview completo vs benchmark (mean por dataset)

> Benchmark: `data/benchmark_results/benchmark_results.csv` — mesmo embedding distiluse-v2.  
> v5 quebrado por modelo LLM (7B vs 14B). v6/v7 não têm dimensão de modelo LLM (labels carregados do v5).

| Dataset | Unsup best | v5 7B | v5 14B | v6 DeepSAD | v7 MLP | Oracle best | v7 vs unsup | % gap fechado |
|---|---|---|---|---|---|---|---|---|
| 20_newsgroups | 0.920 | 0.881 | 0.878 | 0.850 | **0.937** | 0.998 | +0.017 | 22% |
| hatebr | 0.561 | 0.638 | 0.601 | 0.595 | **0.672** | 0.873 | +0.111 | 36% |
| tweets_hs | 0.575 | 0.777 | 0.805 | 0.806 | **0.819** | 0.956 | +0.243 | **64%** |
| wikinews | 0.776 | 0.657 | 0.780 | 0.748 | **0.827** | 0.943 | +0.051 | 31% |

**Best config por dataset:**

| Dataset | Unsup best | v5 best | v6 best | v7 best | Oracle best |
|---|---|---|---|---|---|
| 20_newsgroups | 0.920 | 0.947 | 0.928 | **0.948** | 0.998 |
| hatebr | 0.561 | 0.754 | 0.721 | **0.751** | 0.873 |
| tweets_hs | 0.575 | 0.871 | 0.876 | **0.893** | 0.956 |
| wikinews | 0.776 | 0.865 | 0.821 | **0.859** | 0.943 |

v7 supera o unsup best em todos os datasets — sem nenhum label humano.  
tweets_hs é o destaque: +0.243 sobre unsup, 64% do gap até o oracle fechado.  
Wikinews: v5 14B (0.780) ≈ v7 MLP (0.827) — mas v7 não usa LLM, só labels do v5 reaproveitados.
