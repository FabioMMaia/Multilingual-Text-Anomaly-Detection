# v6 — Ablation: sem SetFit (mesmos labels do v5)

## Objetivo

Isolar a contribuição do **SetFit** no pipeline.

v5 testou: `LLM labels + SetFit embeddings + DeepSAD`  
v6 testa:  `LLM labels + distiluse embeddings + DeepSAD` (sem fine-tuning)

Usando **os mesmos labels do v5** (`_llm_labels.csv`), o v6 é uma ablation limpa:
mesma anotação LLM, mesmo DeepSAD, só muda o espaço de embedding.

---

## Comparação final do paper

| Condição | LLM labels | SetFit | Fonte |
|---|---|---|---|
| Unsupervised (IForest, DeepSVDD...) | ❌ | ❌ | benchmark |
| **v6** (ablation) | ✅ (v5 labels) | ❌ | este experimento |
| **v5** (pipeline completo) | ✅ | ✅ | v5 |
| Oracle (DeepSAD + ground truth) | ground truth | ❌ | benchmark |

**Pergunta central:** `v5 - v6 = ganho do SetFit`

---

## Config

| Parâmetro | Valor |
|---|---|
| Datasets | tweets_hs, hatebr, 20_newsgroups, wikinews |
| Labels source | `data/llm_results/v5/{strategy}/N_{n}/{dataset}_llm_labels.csv` |
| SetFit | **desativado** (`--no_setfit`) |
| Embedding | distiluse-base-multilingual-cased-v2 (pré-computado) |
| Modelo AD | DeepSAD (mesma config do v5) |
| Strategies | random, diversity |
| N | 50, 200 |
| Seeds | 0, 1, 42 |
| Total runs | 96 |
| Hardware | Colab T4 (sem LLM — só DeepSAD) |
| Tempo estimado | ~2h total (~1-2 min/run) |

Results: `data/llm_results/v6/{strategy}/N_{n}/{dataset}.csv`

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

> Não precisa instalar llama-cpp-python nem baixar modelo — o v6 não chama LLM.

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

### Cell 4 — v6 Sweep (96 runs, ~2h)

```python
import subprocess, re, time, glob
import pandas as pd

PROJECT_PATH = "/content/drive/MyDrive/Projeto ML/2026/Master/Multilingual-Text-Anomaly-Detection"
DATA_DIR     = "/content/drive/MyDrive/Projeto ML/2025/AD/third_setup/adaptative-text-anomaly-detection/data"

datasets   = ["tweets_hs", "hatebr", "20_newsgroups", "wikinews"]
strategies = ["random", "diversity"]
ns         = ["50", "200"]
seeds      = ["0", "1", "42"]

RESULTS_DIR = "data/llm_results/v6"
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
                    "--backend",        "llamacpp",        # necessário pelo argparse, mas não é chamado
                    "--load_labels_from", labels_csv,
                    "--no_setfit",
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

### Cell 5 — Comparação v5 vs v6 (ganho do SetFit)

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

df5 = load_v("v5", PROJECT_PATH)
df6 = load_v("v6", PROJECT_PATH)

# Merge por chave comum
key = ["dataset", "strategy", "n_llm_calls", "seed"]
merged = df5[key + ["roc_auc"]].merge(
    df6[key + ["roc_auc"]],
    on=key, suffixes=("_v5", "_v6")
)
merged["setfit_gain"] = merged["roc_auc_v5"] - merged["roc_auc_v6"]

# Média por dataset
print("=== Ganho do SetFit por dataset (v5 - v6) ===")
gain = merged.groupby("dataset")["setfit_gain"].agg(["mean", "std"]).round(3)
print(gain.to_string())

print("\n=== Ganho global ===")
print(f"  mean={merged['setfit_gain'].mean():.3f}  std={merged['setfit_gain'].std():.3f}")

# Por N
print("\n=== Ganho por N ===")
print(merged.groupby("n_llm_calls")["setfit_gain"].mean().round(3).to_string())
```

---

## Results

**Status: ✅ Concluído — 48/48 runs (3 seeds × 4 datasets × 2 strategies × 2 N)**

> v6 é uma versão do sweep sem o eixo de modelo LLM (labels carregados do v5 — sem 7B/14B), por isso 48 runs.

---

### Tabela 1 — ROC-AUC mean ± std por dataset (colapsado sobre strategy + N)

| Dataset | v5 (SetFit) | v6 (sem SetFit) | SetFit gain |
|---|---|---|---|
| 20_newsgroups | 0.879 ±0.079 | 0.850 ±0.105 | **+0.029** ✅ |
| hatebr | 0.620 ±0.089 | 0.595 ±0.077 | **+0.025** ✅ |
| tweets_hs | 0.791 ±0.063 | 0.806 ±0.063 | −0.014 ❌ |
| wikinews | 0.719 ±0.085 | 0.748 ±0.049 | −0.030 ❌ |
| **Global** | **0.752** | **0.750** | **+0.002 ±0.064** |

---

### Tabela 2 — Ganho do SetFit por N

| N | SetFit gain mean | std |
|---|---|---|
| 50 | −0.005 | 0.060 |
| 200 | +0.010 | 0.067 |

---

### Tabela 3 — Ganho do SetFit por strategy

| Strategy | SetFit gain mean | std |
|---|---|---|
| diversity | −0.005 | 0.061 |
| random | +0.010 | 0.066 |

---

### Tabela 4 — Ganho do SetFit por dataset × N

| Dataset | N=50 | N=200 |
|---|---|---|
| 20_newsgroups | +0.026 | +0.032 |
| hatebr | −0.001 | +0.050 |
| tweets_hs | −0.018 | −0.011 |
| wikinews | −0.026 | −0.033 |

---

### Análise

#### SetFit: efeito condicional, não universal

O ganho global de SetFit é **+0.002 ±0.064** — estatisticamente insignificante.
O desvio padrão elevado (0.064) revela que o efeito é **altamente variável**:

- **Ajuda** em datasets EN com sinal semântico limpo: 20_newsgroups (+0.029), hatebr (+0.025 — notável para dataset PT difícil)
- **Prejudica** em datasets onde distiluse já é bom: tweets_hs (−0.014), wikinews (−0.030)

**Hipótese explicativa:** SetFit fine-tunes o embedding nos labels LLM. Quando esses labels são ruidosos _e_ o embedding base já captura bem a estrutura do dado (tweets, wikinews multilingual), o fine-tuning propaga o ruído para o espaço de embedding — piorando o DeepSAD.

#### Efeito do N no SetFit

N=200 amplifica tanto os ganhos quanto as perdas (std maior que N=50).
A única configuração onde N=200 ajuda claramente: **hatebr +0.050** — o maior ganho isolado do SetFit.
Hipótese: mais labels → SetFit converge melhor mesmo com ruído, quando há sinal cultural suficiente.

#### Efeito da strategy

Random (gain=+0.010) supera diversity (−0.005) com SetFit.
Consistency com v5: random já era melhor no sweep principal.

---

### Takeaways para o Paper

1. **SetFit é condicional:** ajuda em EN/sinal forte (+0.029 20ng, +0.025 hatebr), prejudica onde embedding base já é suficiente
2. **Globalmente negligível:** +0.002 — não justifica custo computacional em todos os datasets
3. **Exceção: hatebr N=200 +0.050** — único caso onde SetFit claramente compensa
4. **Wikinews sem SetFit é melhor (−0.030)** — distiluse multilingual já cobre o espaço semântico
5. **Std elevado (0.064)** — SetFit introduz variância adicional, especialmente com labels ruidosos
