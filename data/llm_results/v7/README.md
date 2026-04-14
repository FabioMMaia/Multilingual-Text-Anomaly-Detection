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
| Total runs | 96 |
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

### Cell 4 — v7 Sweep (96 runs, ~1h)

```python
import subprocess, re, time, os

PROJECT_PATH = "/content/drive/MyDrive/Projeto ML/2026/Master/Multilingual-Text-Anomaly-Detection"
DATA_DIR     = "/content/drive/MyDrive/Projeto ML/2025/AD/third_setup/adaptative-text-anomaly-detection/data"

datasets   = ["tweets_hs", "hatebr", "20_newsgroups", "wikinews"]
strategies = ["random", "diversity"]
ns         = ["50", "200"]
seeds      = ["0", "1", "42"]

RESULTS_DIR = "data/llm_results/v7"
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

                print(f"[{run:03d}/{total}] {status} {dataset:<20} {strategy:<12} N={n:<4} seed={seed}  "
                      f"ROC={roc_str}  {elapsed/60:.1f}min", flush=True)

                if result.returncode != 0:
                    print(f"  STDERR: {result.stderr[-400:]}", flush=True)

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

*Não iniciado.*

| Runs | Status |
|---|---|
| 96/96 | ⏳ |
