# v8 — Ablation: MLP + SetFit (mesmos labels do v5)

## Objetivo

Testar se SetFit melhora o MLP com labels LLM ruidosos.

| Versão | Labels | SetFit | Modelo AD |
|---|---|---|---|
| v7 | v5 (LLM) | ❌ | **MLP** |
| **v8** | v5 (LLM) | ✅ | **MLP** |

Comparação direta com v7: mesmos labels, mesmo embedding base (distiluse → SetFit fine-tuned), só muda se SetFit é usado.

---

## Posição no 2×2

| | sem SetFit | com SetFit |
|---|---|---|
| **DeepSAD** | v6 ✅ | v5 ✅ |
| **MLP** | v7 ✅ | **v8** |

---

## Hipótese

SetFit melhora o fine-tuning dos embeddings para o domínio específico de cada dataset.  
**Hipótese v8:** SetFit + MLP > distiluse + MLP (v7) — o embedding mais especializado compensa o ruído de anotação LLM.

Comparando v8 vs v7 → efeito do SetFit no MLP  
Comparando v8 vs v5 → efeito do MLP vs DeepSAD com SetFit

---

## Config

| Parâmetro | Valor |
|---|---|
| Datasets | tweets_hs, hatebr, 20_newsgroups, wikinews |
| Labels source | `data/llm_results/v5/{strategy}/N_{n}/{dataset}_llm_labels.csv` |
| SetFit | ✅ ativado |
| Modelo AD | **MLP** (`--semisup_model mlp`) |
| Embedding | distiluse-base-multilingual-cased-v2 → fine-tuned via SetFit |
| Strategies | random, diversity |
| N | 50, 200 |
| Seeds | 0, 1, 42 |
| Total runs | 48 (labels carregados do v5 — sem re-executar LLM) |
| Hardware | Colab T4 |
| Tempo estimado | ~2–3h total (~1–2min/run com SetFit) |

Results: `data/llm_results/v8/{strategy}/N_{n}/{dataset}.csv`

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

---

### Cell 3 — Skip helper

```python
import glob, pandas as pd

def _already_done(dataset, strategy, n, seed, results_dir):
    pattern = f"{PROJECT_PATH}/{results_dir}/{strategy}/N_{n}/{dataset}.csv"
    for f in glob.glob(pattern):
        try:
            df = pd.read_csv(f)
            match = df[
                (df["n_llm_calls"] == int(n)) &
                (df["seed"]        == int(seed))
            ]
            if not match.empty:
                return True
        except Exception:
            pass
    return False
```

---

### Cell 4 — v8 Sweep (96 runs)

```python
import subprocess, re, time, os

PROJECT_PATH = "/content/drive/MyDrive/Projeto ML/2026/Master/Multilingual-Text-Anomaly-Detection"
DATA_DIR     = "/content/drive/MyDrive/Projeto ML/2025/AD/third_setup/adaptative-text-anomaly-detection/data"

datasets   = ["tweets_hs", "hatebr", "20_newsgroups", "wikinews"]
strategies = ["random", "diversity"]
ns         = ["50", "200"]
seeds      = ["0", "1", "42"]

RESULTS_DIR = "data/llm_results/v8"
V5_DIR      = "data/llm_results/v5"

total = len(datasets) * len(strategies) * len(ns) * len(seeds)  # 96
run = 0

for seed in seeds:
    for dataset in datasets:
        for strategy in strategies:
            for n in ns:
                run += 1

                if _already_done(dataset, strategy, n, seed, RESULTS_DIR):
                    print(f"[{run:03d}/{total}] ↷ {dataset:<20} {strategy:<12} N={n:<4} seed={seed} (skip)", flush=True)
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
                    "--load_labels_from", labels_csv,
                    # sem --no_setfit → SetFit ativo
                    "--semisup_model",  "mlp",
                    "--results_dir",    RESULTS_DIR,
                ]

                t0 = time.time()
                result = subprocess.run(cmd, capture_output=True, text=True)
                elapsed = time.time() - t0

                roc = re.search(r"ROC-AUC\s+\(test\)\s*:\s*([\d.]+)", result.stdout)
                roc_str = roc.group(1) if roc else "N/A"
                status  = "✓" if result.returncode == 0 else "✗"

                print(f"[{run:03d}/{total}] {status} {dataset:<20} {strategy:<12} N={n:<4} seed={seed}  "
                      f"ROC={roc_str}  {elapsed/60:.1f}min", flush=True)

                if result.returncode != 0:
                    print(f"  STDERR: {result.stderr[-400:]}", flush=True)

print("\nDone.")
```

---

### Cell 5 — Análise completa 2×2

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

# v5 = DeepSAD + SetFit | v6 = DeepSAD + distiluse | v7 = MLP + distiluse | v8 = MLP + SetFit
df5 = load_v("v5", PROJECT_PATH)
df6 = load_v("v6", PROJECT_PATH)
df7 = load_v("v7", PROJECT_PATH)
df8 = load_v("v8", PROJECT_PATH)

key = ["dataset", "strategy", "n_llm_calls", "seed"]

# ── Tabela 2×2 global ──
print("=== Tabela 2×2 — mean ROC-AUC ===")
print(f"              | sem SetFit | com SetFit")
print(f"  DeepSAD     |   {df6['roc_auc'].mean():.3f}    |   {df5['roc_auc'].mean():.3f}")
print(f"  MLP         |   {df7['roc_auc'].mean():.3f}    |   {df8['roc_auc'].mean():.3f}")

# ── Efeito SetFit no MLP (v8 - v7) ──
m = df7[key + ["roc_auc"]].merge(df8[key + ["roc_auc"]], on=key, suffixes=("_v7", "_v8"))
m["setfit_gain"] = m["roc_auc_v8"] - m["roc_auc_v7"]

print("\n=== Efeito SetFit no MLP (v8 - v7) por dataset ===")
print(m.groupby("dataset")["setfit_gain"].agg(["mean","std"]).round(3).to_string())
print(f"\nGlobal: {m['setfit_gain'].mean():.3f} ±{m['setfit_gain'].std():.3f}")

# ── MLP vs DeepSAD com SetFit (v8 - v5) ──
m2 = df5[key + ["roc_auc"]].merge(df8[key + ["roc_auc"]], on=key, suffixes=("_v5", "_v8"))
m2["mlp_gain"] = m2["roc_auc_v8"] - m2["roc_auc_v5"]

print("\n=== MLP vs DeepSAD com SetFit (v8 - v5) por dataset ===")
print(m2.groupby("dataset")["mlp_gain"].agg(["mean","std"]).round(3).to_string())
print(f"\nGlobal: {m2['mlp_gain'].mean():.3f} ±{m2['mlp_gain'].std():.3f}")
```

---

## Results

**48/48 runs completos.** `SetFit skipped` em runs onde o LLM não encontrou anomalias suficientes para fine-tuning.

---

### Tabela 1 — v8 mean AUC por dataset

| Dataset | Unsup (best) | v7 MLP | **v8 MLP+SF** | Oracle | Δ v8−v7 | % gap fechado (v8) |
|---|---|---|---|---|---|---|
| 20_newsgroups | 0.920 | 0.937 | **0.937** | 0.998 | +0.000 | 22% |
| hatebr | 0.561 | 0.672 | **0.731** | 0.873 | +0.059 | 54% |
| tweets_hs | 0.575 | 0.819 | **0.829** | 0.956 | +0.010 | 67% |
| wikinews | 0.776 | 0.827 | **0.778** | 0.943 | −0.049 | 1% |
| **Global** | **0.708** | **0.814** | **0.819** | **0.942** | **+0.005** | — |

> SetFit ajuda em hatebr (+0.059) e tweets_hs (+0.010), não muda 20_newsgroups (sempre skipped) e **regride em wikinews** (−0.049).

---

### Tabela 2 — 2×2 completa (mean AUC)

|  | sem SetFit | com SetFit |
|---|---|---|
| **DeepSAD** | 0.750 (v6) | 0.752 (v5) |
| **MLP** | 0.814 (v7) | **0.819 (v8)** |

> MLP > DeepSAD independentemente do SetFit. O SetFit dá ganho marginal ao MLP (+0.005 global).

Por dataset:

| Dataset | DS+SF (v5) | DS (v6) | MLP (v7) | MLP+SF (v8) |
|---|---|---|---|---|
| 20_newsgroups | 0.879 | 0.850 | **0.937** | **0.937** |
| hatebr | 0.620 | 0.595 | 0.672 | **0.731** |
| tweets_hs | 0.791 | 0.806 | 0.819 | **0.829** |
| wikinews | 0.719 | 0.748 | **0.827** | 0.778 |

---

### Tabela 3 — Efeito do SetFit no MLP (v8 − v7, paired)

| Dataset | Δ mean | Δ std |
|---|---|---|
| 20_newsgroups | +0.000 | ±0.000 |
| hatebr | +0.059 | ±0.082 |
| tweets_hs | +0.010 | ±0.038 |
| wikinews | −0.049 | ±0.099 |
| **Global** | **+0.005** | **±0.078** |

---

### Tabela 4 — MLP+SF vs DeepSAD+SF (v8 − v5, paired)

| Dataset | Δ mean | Δ std |
|---|---|---|
| 20_newsgroups | +0.058 | ±0.080 |
| hatebr | +0.111 | ±0.107 |
| tweets_hs | +0.038 | ±0.061 |
| wikinews | +0.060 | ±0.111 |
| **Global** | **+0.067** | **±0.093** |

> MLP+SF superior a DeepSAD+SF em todos os datasets (+0.067 global) — confirma resultado de v7.

---

### Tabela 5 — SetFit skipped: quando o fine-tuning acontece

| Dataset | N | SetFit ran | SetFit skipped | AUC ran | AUC skipped |
|---|---|---|---|---|---|
| 20_newsgroups | 50 | 0 | 6 | — | 0.936 |
| 20_newsgroups | 200 | 0 | 6 | — | 0.939 |
| hatebr | 50 | 1 | 5 | 0.719 | 0.650 |
| hatebr | 200 | 6 | 0 | 0.801 | — |
| tweets_hs | 50 | 3 | 3 | 0.848 | 0.725 |
| tweets_hs | 200 | 6 | 0 | 0.872 | — |
| wikinews | 50 | 4 | 2 | 0.727 | 0.822 |
| wikinews | 200 | 6 | 0 | 0.798 | — |

> SetFit é skipped quando o LLM não encontra anomalias suficientes. Para 20_newsgroups (~0.8 anomalias/run), nunca há exemplos positivos suficientes. Para outros datasets, N=200 garante que SetFit rode.

---

### Analysis

**Q1: SetFit melhora o MLP?**  
Condicionalmente. Melhora hatebr (+0.059) e levemente tweets_hs (+0.010), enquanto regride em wikinews (−0.049) e não tem efeito em 20_newsgroups (sempre skipped). Global: +0.005 — ganho marginal.

**Q2: Por que 20_newsgroups não se beneficia do SetFit?**  
Com apenas ~0.8 anomalias por run (LLM não consegue identificar), o SetFit não tem exemplos positivos suficientes para fine-tuning. É skipped em 100% dos runs.

**Q3: Por que wikinews regride com SetFit?**  
Wikinews tem ~15 anomalias por run (N=200), mas o SetFit com poucas amostras pode fazer overfitting no embedding. O wikinews sem SetFit já era o melhor resultado de v7 (0.827 — 31% gap), e o fine-tuning com poucos exemplos ruidosos degrada o embedding.

**Q4: Qual a melhor configuração final?**  
MLP + SetFit (v8) é o melhor **globally** (0.819 vs oracle 0.942), mas por dataset:
- 20_newsgroups: v7 = v8 (SetFit irrelevante)
- hatebr: **v8** claramente melhor
- tweets_hs: **v8** ligeiramente melhor
- wikinews: **v7** melhor (SetFit agride)

---

### Takeaways v8

1. **MLP > DeepSAD em todos os datasets**, com ou sem SetFit — confirma v7.
2. **SetFit ajuda quando há anomalias LLM suficientes** (hatebr N=200: +0.150 vs skipped).
3. **SetFit é inútil para 20_newsgroups** — LLM raramente encontra anomalias nesse dataset.
4. **SetFit pode regredir** quando fine-tuned com labels muito ruidosos em datasets com menos separabilidade (wikinews).
5. **Melhor configuração global: MLP + SetFit (v8)** com 0.819 AUC médio — fecha 22–67% do gap unsup→oracle.
6. **Gap wikinews colapsa com SetFit** (31% → 1%): o embedding fine-tuned piora a separabilidade nesse dataset.
