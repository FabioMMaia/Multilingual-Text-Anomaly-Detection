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
| Total runs | 96 |
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

*Não iniciado.*

| Runs | Status |
|---|---|
| 0/96 | ⏳ |
