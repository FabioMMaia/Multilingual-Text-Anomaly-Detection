# v5 — Full Production Experiment 🏁

## Overview

**Goal:** Final publishable experiment — 96 runs with 3 seeds for confidence intervals.  
Grid: **4 datasets × 2 strategies × 2 N × 2 models × 3 seeds = 96 runs**

> All decisions locked in v4. See `data/llm_results/v4/README.md` for full rationale.

---

## Dataset Grid (2×2)

|                | English         | Portuguese |
|----------------|-----------------|------------|
| Hate Detection | tweets_hs       | hatebr     |
| Topic Class.   | 20_newsgroups   | wikinews   |

---

## Config

| Parameter  | Value                                        |
|------------|----------------------------------------------|
| Datasets   | tweets_hs, hatebr, 20_newsgroups, wikinews   |
| LLM Models | Qwen 2.5 7B Q4 + Qwen 2.5 14B Q4 (llamacpp) |
| Strategies | random, diversity                            |
| N          | 50, 200                                      |
| Seeds      | 0, 1, 42                                     |
| Threshold  | 0.5                                          |
| Encoder    | distiluse-base-multilingual-cased-v2         |
| Hardware   | Colab T4                                     |

Results: `data/llm_results/v5/{strategy}/N_{n}/{dataset}.csv`

---

## Execution Plan

### Phase 0 — Pre-flight (~15 min, before any LLM run)

```bash
# 1. Run benchmark baselines (all 4 datasets, distiluse-v2)
python scripts/run_benchmark.py \
    --project_path $PROJECT_PATH \
    --data_dir $DATA_DIR \
    --results_dir data/benchmark_results \
    --device cuda

# 2. Compute sep_ratio for all 4 datasets (Cell 0b below)
# 3. Confirm all parquets exist in DATA_DIR
```

### Phase 1 — 7B full sweep (48 runs, ~3.5h)

```python
MODEL = "qwen2.5-7b" ; seeds = ["0", "1", "42"] ; ns = ["50", "200"]
```

### Phase 2 — 14B N=50 (24 runs, ~1.3h)

```python
MODEL = "qwen2.5-14b" ; ns = ["50"]
```

### Phase 3 — 14B N=200 (24 runs, ~4.2h)

```python
MODEL = "qwen2.5-14b" ; ns = ["200"]
```

**Total: ~9.5h. Session drop → resume safely (skip logic in Cell 6).**

> Tempos medidos empiricamente no v3 (T4, llamacpp, distiluse-v2).
> `random` e `diversity` **não treinam DeepSVDD** (skip adicionado no v5 — poupa ~30–60s/run).

| Model | N=50 avg | N=200 avg | 48 runs est. |
|-------|----------|-----------|--------------|
| 7B    | 2.3 min  | 6.7 min   | ~3.6h        |
| 14B   | 3.5 min  | 11.3 min  | ~5.9h        |

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

### Cell 2 — Install llama-cpp-python (CUDA 12.2)

```python
!nvcc --version

!pip install llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122 \
    --no-cache-dir
```

---

### Cell 3 — Install Project Requirements

```python
!pip install -r requirements.txt
```

---

### Cell 4 — Download Qwen 7B (one-time, ~4 GB)

```python
MODEL_DIR = f"{PROJECT_PATH}/models"
os.makedirs(MODEL_DIR, exist_ok=True)

BASE  = "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main"
SHARD = "qwen2.5-7b-instruct-q4_k_m.gguf"

import subprocess
subprocess.run(["wget", "-q", "--show-progress", f"{BASE}/{SHARD}", "-O", f"{MODEL_DIR}/{SHARD}"])
print("Done.")
```

---

### Cell 5 — Download Qwen 14B (one-time, ~9 GB, 3 shards)

```python
MODEL_DIR = f"{PROJECT_PATH}/models"
os.makedirs(MODEL_DIR, exist_ok=True)

BASE   = "https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF/resolve/main"
SHARDS = [
    "qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf",
    "qwen2.5-14b-instruct-q4_k_m-00002-of-00003.gguf",
    "qwen2.5-14b-instruct-q4_k_m-00003-of-00003.gguf",
]

import subprocess
for shard in SHARDS:
    print(f"Downloading {shard} …")
    subprocess.run(["wget", "-q", "--show-progress", f"{BASE}/{shard}", "-P", MODEL_DIR])
print("Done.")
```

---

### Cell 6 — v5 Sweep (7B + 14B, full 96 runs)

```python
import subprocess, re, time, threading, glob
import pandas as pd

PROJECT_PATH = "/content/drive/MyDrive/Projeto ML/2026/Master/Multilingual-Text-Anomaly-Detection"
DATA_DIR     = "/content/drive/MyDrive/Projeto ML/2025/AD/third_setup/adaptative-text-anomaly-detection/data"

datasets   = ["tweets_hs", "hatebr", "20_newsgroups", "wikinews"]
strategies = ["random", "diversity"]
ns         = ["50", "200"]
seeds      = ["0", "1", "42"]
models     = ["qwen2.5-7b", "qwen2.5-14b"]   # 7B first (~3.6h), then 14B (~5.9h)

RESULTS_DIR = "data/llm_results/v5"

# ── Keep-alive ────────────────────────────────────────────────────────────────
def _keepalive(stop_event, interval=240):
    while not stop_event.wait(interval):
        try:
            _ = glob.glob(f"{PROJECT_PATH}/{RESULTS_DIR}/**/*.csv", recursive=True)
        except Exception:
            pass

_stop = threading.Event()
threading.Thread(target=_keepalive, args=(_stop,), daemon=True).start()

# ── Skip helper ───────────────────────────────────────────────────────────────
def _already_done(dataset, strategy, n, seed, model, results_dir):
    pattern = f"{PROJECT_PATH}/{results_dir}/{strategy}/N_{n}/{dataset}.csv"
    for f in glob.glob(pattern):
        try:
            df = pd.read_csv(f)
            model_tag = model.split("/")[-1]
            match = df[
                (df["n_llm_calls"] == int(n)) &
                (df["seed"]        == int(seed)) &
                (df["llm_model"].str.contains(model_tag, na=False))
            ]
            if not match.empty:
                return True
        except Exception:
            pass
    return False

# ── Sweep ─────────────────────────────────────────────────────────────────────
total = len(models) * len(datasets) * len(strategies) * len(ns) * len(seeds)  # 96
run   = 0
consecutive_errors = 0
MAX_CONSECUTIVE_ERRORS = 5  # stop early if model is broken (e.g. OOM)

try:
    for model in models:
        print(f"\n{'='*60}")
        print(f"  MODEL: {model}")
        print(f"{'='*60}\n")
        consecutive_errors = 0  # reset between models
        for seed in seeds:
            for dataset in datasets:
                for strategy in strategies:
                    for n in ns:
                        run += 1

                        if _already_done(dataset, strategy, n, seed, model, RESULTS_DIR):
                            print(f"[{run:03d}/{total}] ↷ {model:<20} {dataset:<20} {strategy:<12} N={n:<4} seed={seed} (skip)", flush=True)
                            consecutive_errors = 0
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
                            "--backend",        "llamacpp",
                            "--llamacpp_model", model,
                            "--results_dir",    RESULTS_DIR,
                        ]
                        t0 = time.time()
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        elapsed = time.time() - t0

                        roc = re.search(r"ROC-AUC\s+\(test\)\s*:\s*([\d.]+)", result.stdout)
                        roc_str = roc.group(1) if roc else "N/A"
                        status  = "✓" if result.returncode == 0 else "✗"

                        print(f"[{run:03d}/{total}] {status} {model:<20} {dataset:<20} {strategy:<12} N={n:<4} seed={seed}  "
                              f"ROC={roc_str}  {elapsed/60:.1f}min", flush=True)

                        if result.returncode != 0:
                            consecutive_errors += 1
                            print(f"  STDERR: {result.stderr[-400:]}")
                            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                                print(f"\n⚠️  {MAX_CONSECUTIVE_ERRORS} consecutive errors — stopping sweep for {model}.")
                                break
                        else:
                            consecutive_errors = 0
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                continue
            break
finally:
    _stop.set()
    print("\nDone.")
```

---

### Cell 7 — Results Summary (mean ± std over seeds)

```python
import pandas as pd, glob, numpy as np

files = glob.glob(f"{PROJECT_PATH}/data/llm_results/v5/**/*.csv", recursive=True)
metrics_files = [f for f in files if "llm_labels" not in f]

dfs = []
for f in metrics_files:
    try:
        dfs.append(pd.read_csv(f, engine="python", on_bad_lines="skip"))
    except Exception as e:
        print(f"Error in {f}: {e}")

df = pd.concat(dfs, ignore_index=True)

df["model"] = (
    df["llm_model"].str.split("/").str[-1]
    .str.replace(r"-instruct-q4_k_m.*\.gguf", "", regex=True)
)

summary = (
    df.groupby(["dataset", "model", "strategy", "n_llm_calls"])["roc_auc"]
    .agg(["mean", "std", "count"])
    .reset_index()
    .rename(columns={"mean": "auc_mean", "std": "auc_std", "count": "n_seeds"})
    .sort_values(["dataset", "model", "strategy", "n_llm_calls"])
)

pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
display(summary)
```

---

## Results

**Status: ✅ Concluído — 96/96 runs (3 seeds × 4 datasets × 2 strategies × 2 N × 2 modelos)**

| Phase | Runs | Status |
|-------|------|--------|
| Phase 1 — 7B full sweep (seeds 0/1/42) | 48 | ✅ |
| Phase 2 — 14B N=50 | 24 | ✅ |
| Phase 3 — 14B N=200 | 24 | ✅ |

> Timings reais medidos no Colab T4: 7B N=50 ~1.9 min, N=200 ~7.0 min; 14B N=50 ~3.6 min, N=200 ~13.2 min.

---

### Tabela 1 — ROC-AUC mean ± std por configuração (3 seeds)

| Dataset | Strategy | N | 7B AUC | 14B AUC |
|---|---|---|---|---|
| 20_newsgroups | diversity | 50 | **0.939 ±0.011** | **0.939 ±0.011** |
| 20_newsgroups | diversity | 200 | 0.820 ±0.120 | 0.786 ±0.141 |
| 20_newsgroups | random | 50 | 0.904 ±0.029 | 0.925 ±0.011 |
| 20_newsgroups | random | 200 | 0.860 ±0.041 | 0.863 ±0.037 |
| hatebr | diversity | 50 | 0.590 ±0.072 | 0.566 ±0.093 |
| hatebr | diversity | 200 | 0.615 ±0.083 | 0.576 ±0.158 |
| hatebr | random | 50 | 0.624 ±0.083 | 0.582 ±0.072 |
| hatebr | random | 200 | **0.724 ±0.026** | **0.681 ±0.011** |
| tweets_hs | diversity | 50 | 0.725 ±0.062 | 0.751 ±0.081 |
| tweets_hs | diversity | 200 | 0.823 ±0.038 | **0.832 ±0.022** |
| tweets_hs | random | 50 | 0.769 ±0.111 | 0.815 ±0.051 |
| tweets_hs | random | 200 | 0.793 ±0.020 | 0.823 ±0.035 |
| wikinews | diversity | 50 | 0.620 ±0.040 | 0.777 ±0.024 |
| wikinews | diversity | 200 | 0.691 ±0.077 | 0.755 ±0.042 |
| wikinews | random | 50 | 0.625 ±0.082 | 0.773 ±0.080 |
| wikinews | random | 200 | 0.693 ±0.060 | **0.815 ±0.014** |

---

### Tabela 2 — Média global por dataset × modelo (colapsado sobre strategy + N)

| Dataset | 7B mean | 7B std | 14B mean | 14B std | 14B > 7B? |
|---|---|---|---|---|---|
| 20_newsgroups | **0.881** | 0.073 | 0.878 | 0.089 | ≈ empate |
| tweets_hs | 0.777 | 0.069 | **0.805** | 0.056 | +0.028 |
| wikinews | 0.657 | 0.068 | **0.780** | 0.046 | +0.123 |
| hatebr | **0.638** | 0.080 | 0.601 | 0.097 | 7B melhor |

---

### Tabela 3 — Comparação com Baselines (ROC-AUC)

> Baselines usam embeddings distiluse-v2 + ground truth labels (oracle) ou sem labels (unsup).  
> v5 usa labels LLM — **nenhum label humano no treino**.

#### 3a — Baselines não-supervisionados (sem labels)

| Dataset | IForest | LOF | DeepSVDD | OCSVM | AutoEncoder | VAE | HBOS | **Best unsup** |
|---|---|---|---|---|---|---|---|---|
| 20_newsgroups | 0.858 | 0.856 | 0.702 | 0.770 | 0.902 | 0.920 | 0.917 | **0.920** |
| hatebr | 0.382 | 0.561 | 0.515 | 0.372 | 0.474 | 0.381 | 0.377 | **0.561** |
| tweets_hs | 0.536 | 0.498 | 0.548 | 0.553 | 0.575 | 0.526 | 0.540 | **0.575** |
| wikinews | 0.697 | 0.676 | 0.591 | 0.666 | 0.757 | 0.776 | 0.743 | **0.776** |

#### 3b — Oracle semi-supervisionado (com ground truth labels — teto teórico)

| Dataset | DeepSAD | DevNet | MLP | XGBOD | **Best oracle** |
|---|---|---|---|---|---|
| 20_newsgroups | 0.998 | 0.980 | 0.998 | 0.996 | **0.998** |
| hatebr | 0.770 | 0.783 | 0.863 | 0.873 | **0.873** |
| tweets_hs | 0.896 | 0.709 | 0.956 | 0.945 | **0.956** |
| wikinews | 0.864 | 0.812 | 0.929 | 0.943 | **0.943** |

#### 3c — v5 pipeline vs baselines (melhor config por dataset)

| Dataset | Best unsup (sem labels) | v5 best (LLM labels) | Oracle best (GT labels) | v5 gap to unsup | v5 gap to oracle |
|---|---|---|---|---|---|
| 20_newsgroups | 0.920 | **0.939** | 0.998 | **+0.019** ✅ | −0.059 |
| hatebr | 0.561 | **0.724** | 0.873 | **+0.163** ✅ | −0.149 |
| tweets_hs | 0.575 | **0.832** | 0.956 | **+0.257** ✅ | −0.124 |
| wikinews | 0.776 | **0.815** | 0.943 | **+0.039** ✅ | −0.128 |

**v5 supera todos os baselines não-supervisionados em todos os datasets**, usando apenas labels gerados por LLM — sem nenhum label humano no treino.

---

### Análise Macro

#### Q1 — 7B vs 14B: vale a pena?

| Modelo | Mean AUC | Std | N runs |
|---|---|---|---|
| 7B | 0.738 | 0.121 | 48 |
| 14B | **0.766** | 0.126 | 48 |

14B é marginalmente melhor (+0.028 global), mas a vantagem é **concentrada em wikinews (+0.123)**.
Em 20_newsgroups os modelos são **idênticos** (mesmo score em 5 configurações — saturação da tarefa).
Em hatebr, **7B supera 14B** (+0.037), sugerindo que capacidade de modelo não compensa alinhamento cultural.

**Conclusão:** 7B é custo-benefício superior para hate speech. 14B compensa em tarefas com sinal semântico mais rico (tópico multilingual/wikinews).

---

#### Q2 — N=50 vs N=200: annotation budget importa?

| N | Mean AUC | Std |
|---|---|---|
| 50 | 0.745 | 0.141 |
| 200 | **0.759** | 0.104 |

Globalmente, N=200 é ligeiramente melhor (+0.014) e **mais estável** (std menor).
O efeito varia por dataset:

| Dataset | N=50 | N=200 | Δ |
|---|---|---|---|
| 20_newsgroups | **0.927** | 0.832 | **−0.094** |
| tweets_hs | 0.765 | **0.818** | +0.053 |
| wikinews | 0.699 | **0.738** | +0.040 |
| hatebr | 0.590 | **0.649** | +0.059 |

**Inversão em 20_newsgroups:** N=50 é melhor (+0.094) — o dataset tem sinal forte o suficiente
para que menos anotações LLM produzam labels mais precisos (menos ruído acumulado).
Para hate speech multilingual (hatebr), N=200 é essencial: +0.059 e sobe até 0.724.

---

#### Q3 — random vs diversity: estratégia de seleção importa?

| Strategy | Mean AUC | Std |
|---|---|---|
| random | **0.767** | 0.111 |
| diversity | 0.738 | 0.135 |

Random supera diversity globalmente (+0.029). Interação com N:

| Strategy | N=50 | N=200 |
|---|---|---|
| diversity | 0.738 | 0.737 |
| random | 0.752 | **0.781** |

Diversity não melhora com N=200, enquanto random ganha +0.029 extra.
Hipótese: diversity com N pequeno cobre regiões normais demais (afasta o LLM do boundary anômalo).

---

#### Q4 — Dificuldade dos datasets

| Dataset | Mean AUC | Std | Best unsup | Δ vs unsup |
|---|---|---|---|---|
| 20_newsgroups | **0.879** | 0.079 | 0.920 | −0.041 (N=50 já supera: 0.927) |
| tweets_hs | 0.791 | 0.063 | 0.575 | **+0.216** |
| wikinews | 0.719 | 0.085 | 0.776 | −0.057 (best config: 0.815) |
| hatebr | 0.620 | 0.089 | 0.561 | **+0.059** (best: 0.724) |

---

#### Q5 — Melhor configuração por dataset

| Dataset | Config | AUC |
|---|---|---|
| 20_newsgroups | diversity / N=50 / 14B | **0.939 ±0.011** |
| tweets_hs | diversity / N=200 / 14B | **0.832 ±0.022** |
| wikinews | random / N=200 / 14B | **0.815 ±0.014** |
| hatebr | random / N=200 / 7B | **0.724 ±0.026** |

---

### Takeaways para o Paper

1. **v5 supera todos os baselines não-supervisionados em todos os datasets** — sem nenhum label humano
2. **Gap ao oracle:** 6–15 pp abaixo do teto com ground truth — razoável dado o ruído de anotação LLM
3. **N=50 já é competitivo** em datasets com sinal forte (20_newsgroups: 0.927) → annotation-efficiency
4. **7B vs 14B:** 7B suficiente para hate speech; 14B compensa em tópico multilingual (wikinews)
5. **Hatebr:** gap cultural LLM (EN-centric) — mitigável com N=200 (0.59 → 0.72)
6. **random > diversity** globalmente — estratégia simples é suficiente

---

### Tabela 1 — ROC-AUC mean ± std por configuração (3 seeds)

| Dataset | Strategy | N | 7B AUC | 14B AUC |
|---|---|---|---|---|
| 20_newsgroups | diversity | 50 | **0.939 ±0.011** | **0.939 ±0.011** |
| 20_newsgroups | diversity | 200 | 0.820 ±0.120 | 0.786 ±0.141 |
| 20_newsgroups | random | 50 | 0.904 ±0.029 | 0.925 ±0.011 |
| 20_newsgroups | random | 200 | 0.860 ±0.041 | 0.863 ±0.037 |
| hatebr | diversity | 50 | 0.590 ±0.072 | 0.566 ±0.093 |
| hatebr | diversity | 200 | 0.615 ±0.083 | 0.576 ±0.158 |
| hatebr | random | 50 | 0.624 ±0.083 | 0.582 ±0.072 |
| hatebr | random | 200 | **0.724 ±0.026** | **0.681 ±0.011** |
| tweets_hs | diversity | 50 | 0.725 ±0.062 | 0.751 ±0.081 |
| tweets_hs | diversity | 200 | 0.823 ±0.038 | **0.832 ±0.022** |
| tweets_hs | random | 50 | 0.769 ±0.111 | 0.815 ±0.051 |
| tweets_hs | random | 200 | 0.793 ±0.020 | 0.823 ±0.035 |
| wikinews | diversity | 50 | 0.620 ±0.040 | 0.777 ±0.024 |
| wikinews | diversity | 200 | 0.691 ±0.077 | 0.755 ±0.042 |
| wikinews | random | 50 | 0.625 ±0.082 | 0.773 ±0.080 |
| wikinews | random | 200 | 0.693 ±0.060 | **0.815 ±0.014** |

---

### Tabela 2 — Média global por dataset × modelo (colapsado sobre strategy + N)

| Dataset | 7B mean | 7B std | 14B mean | 14B std | 14B > 7B? |
|---|---|---|---|---|---|
| 20_newsgroups | **0.881** | 0.073 | 0.878 | 0.089 | ≈ empate |
| tweets_hs | 0.777 | 0.069 | **0.805** | 0.056 | +0.028 |
| wikinews | 0.657 | 0.068 | **0.780** | 0.046 | +0.123 |
| hatebr | **0.638** | 0.080 | 0.601 | 0.097 | 7B melhor |

---

### Análise Macro

#### Q1 — 7B vs 14B: vale a pena?

| Modelo | Mean AUC | Std | N runs |
|---|---|---|---|
| 7B | 0.738 | 0.121 | 48 |
| 14B | **0.766** | 0.126 | 48 |

14B é marginalmente melhor (+0.028 global), mas a vantagem é **concentrada em wikinews (+0.123)**.
Em 20_newsgroups os modelos são **idênticos** (mesmo score em 5 configurações — saturação da tarefa).
Em hatebr, **7B supera 14B** (+0.037), sugerindo que capacidade de modelo não compensa alinhamento cultural.

**Conclusão:** 7B é custo-benefício superior para hate speech. 14B compensa em tarefas com sinal semântico mais rico (tópico multilingual/wikinews).

---

#### Q2 — N=50 vs N=200: annotation budget importa?

| N | Mean AUC | Std |
|---|---|---|
| 50 | 0.745 | 0.141 |
| 200 | **0.759** | 0.104 |

Globalmente, N=200 é ligeiramente melhor (+0.014) e **mais estável** (std menor).
O efeito varia por dataset:

| Dataset | N=50 | N=200 | Δ |
|---|---|---|---|
| 20_newsgroups | **0.927** | 0.832 | **−0.094** |
| tweets_hs | 0.765 | **0.818** | +0.053 |
| wikinews | 0.699 | **0.738** | +0.040 |
| hatebr | 0.590 | **0.649** | +0.059 |

**Inversão em 20_newsgroups:** N=50 é melhor (+0.094) — o dataset tem sinal forte o suficiente
para que menos anotações LLM produzam labels mais precisos (menos ruído acumulado).
Para hate speech multilingual (hatebr), N=200 é essencial: +0.059 e sobe até 0.724.

---

#### Q3 — random vs diversity: estratégia de seleção importa?

| Strategy | Mean AUC | Std |
|---|---|---|
| random | **0.767** | 0.111 |
| diversity | 0.738 | 0.135 |

Random supera diversity globalmente (+0.029). Interação com N:

| Strategy | N=50 | N=200 |
|---|---|---|
| diversity | 0.738 | 0.737 |
| random | 0.752 | **0.781** |

Diversity não melhora com N=200, enquanto random ganha +0.029 extra.
Hipótese: diversity com N pequeno cobre regiões normais demais (afasta o LLM do boundary anômalo).

---

#### Q4 — Dificuldade dos datasets

| Dataset | Mean AUC | Std | Interpretação |
|---|---|---|---|
| 20_newsgroups | **0.879** | 0.079 | Tópico EN — sinal semântico forte |
| tweets_hs | 0.791 | 0.063 | Hate EN — LLM calibrado culturalmente |
| wikinews | 0.719 | 0.085 | Tópico multilingual (PT/EN mix) |
| hatebr | 0.620 | 0.089 | Hate PT — gap cultural LLM (EN-centric) |

Hate speech em português (hatebr) é o caso mais difícil — o LLM produz labels ruidosos
em texto informal PT, o que propaga ruído para o DeepSAD. É um achado publicável
(não uma falha do pipeline, mas uma limitação identificada e quantificada).

---

#### Q5 — Melhor configuração por dataset

| Dataset | Config | AUC |
|---|---|---|
| 20_newsgroups | diversity / N=50 / 14B | **0.939 ±0.011** |
| tweets_hs | diversity / N=200 / 14B | **0.832 ±0.022** |
| wikinews | random / N=200 / 14B | **0.815 ±0.014** |
| hatebr | random / N=200 / 7B | **0.724 ±0.026** |

---

### Takeaways para o Paper

1. **Pipeline funciona:** AUC ≥ 0.80 em 3 de 4 datasets com configuração ótima (sem acesso a labels de treino)
2. **N=50 já é competitivo** em datasets com sinal forte → argumento de annotation-efficiency
3. **7B vs 14B:** 7B é suficiente para hate speech; 14B compensa em tarefas com maior complexidade semântica multilingual
4. **Hatebr outlier:** não é falha do pipeline — é evidência de gap cultural LLM (EN-centric) em hate speech PT; N=200 mitiga (0.59 → 0.72)
5. **random > diversity** globalmente: estratégia simples é suficiente; diversity adiciona variância sem ganho médio

---

### Eficiência de Labels — Pipeline vs Oracle

> O oracle semi-supervisionado usa **~5% do treino com ground truth labels** (anomalias reais confirmadas).
> O pipeline LLM anota N amostras — mas a maioria é rotulada como **normal** pelo LLM.
> Abaixo: quantas anomalias cada abordagem realmente passa ao modelo AD.

| Dataset | Oracle (GT, ~5% treino) | Pipeline N=50 (LLM anom. média) | Pipeline N=200 (LLM anom. média) | Razão oracle/N=200 |
|---|---|---|---|---|
| 20_newsgroups | **39** | ~0.2 | ~0.8 | **~49×** |
| hatebr | **140** | ~2.2 | ~7.8 | **~18×** |
| tweets_hs | **1189** | ~4.8 | ~15.3 | **~78×** |
| wikinews | **233** | ~4.1 | ~15.1 | **~15×** |

O oracle recebe entre **15× e 78× mais anomalias confirmadas** do que o pipeline N=200.
Apesar disso, o pipeline (v7 best config) chega a:

| Dataset | v7 MLP (best) | Oracle best | Gap |
|---|---|---|---|
| 20_newsgroups | 0.948 | 0.998 | −0.050 |
| hatebr | 0.751 | 0.873 | −0.122 |
| tweets_hs | 0.893 | 0.956 | −0.063 |
| wikinews | 0.859 | 0.943 | −0.084 |

**A comparação não é simétrica — favorece o oracle.** O pipeline opera com informação de anomalia ordens de magnitude menor, sem qualquer label humano.
