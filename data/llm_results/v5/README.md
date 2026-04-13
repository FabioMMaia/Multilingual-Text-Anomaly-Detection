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

try:
    for model in models:
        print(f"\n{'='*60}")
        print(f"  MODEL: {model}")
        print(f"{'='*60}\n")
        for seed in seeds:
            for dataset in datasets:
                for strategy in strategies:
                    for n in ns:
                        run += 1

                        if _already_done(dataset, strategy, n, seed, model, RESULTS_DIR):
                            print(f"[{run:03d}/{total}] ↷ {model:<20} {dataset:<20} {strategy:<12} N={n:<4} seed={seed} (skip)", flush=True)
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
                            print("  STDERR:", result.stderr[-400:])
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

*Not yet started.*

| Phase | Runs | Status |
|-------|------|--------|
| Phase 0 — Benchmark + sep_ratio pre-flight | — | ⏳ |
| Phase 1 — 7B full sweep (seeds 0/1/42) | 48 | ⏳ |
| Phase 2 — 14B N=50 | 24 | ⏳ |
| Phase 3 — 14B N=200 | 24 | ⏳ |
