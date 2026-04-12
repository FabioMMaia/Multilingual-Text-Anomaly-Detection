# v4 — Full Experiment (3 Seeds + Revised told_br Prompt)

## Overview

**Goal:** Full production experiment — 3 seeds for publishable confidence intervals.  
Grid: 4 datasets × 3 strategies × 2 N × 2 models × 3 seeds = **144 runs**

**New in v4 vs v3:**

- 3 seeds (0, 1, 42) — mandatory for confidence intervals
- **Revised told_br prompt (v2):** task redefined as *toxic/offensive language detection*
  covering insult, obscene, LGBTQphobia, misogyny, racism, xenophobia (aligns with ToLD-Br annotation schema)
  vs. v3 prompt which only targeted identity-based hate speech (missed ~60% of told_br positives)
- All other prompts unchanged (tweets_hs, 20_newsgroups, wikinews)

**Dataset grid (2×2):**

|                | English         | Portuguese |
|----------------|-----------------|------------|
| Hate Detection | tweets_hs       | told_br    |
| Topic Class.   | 20_newsgroups   | wikinews   |

---

## Config

| Parameter | Value |
|-----------|-------|
| Datasets  | tweets_hs, told_br, 20_newsgroups, wikinews |
| Models    | Qwen 2.5 7B Q4 + Qwen 2.5 14B Q4 (llamacpp) |
| Strategies | random, score_guided, diversity |
| N         | 50, 200 |
| Seeds     | 0, 1, 42 |
| Threshold | 0.45 |
| told_br Prompt | v2 (toxicity: insult+obscene+LGBTQ+misogyny+racism+xenophobia) |
| Other Prompts | v1 (unchanged) |
| Encoder   | distiluse-base-multilingual-cased-v2 |
| Hardware  | Colab T4 |

Results saved under `data/llm_results/v4/{strategy}/N_{n}/{dataset}.csv`.

---

## Execution Plan

### Phase 0 — told_br prompt validation (before full run)

**2 runs only**: random, N=50+200, 7B, seed=42  
Expected time: ~8 min on T4.  
**Gate criteria:** AUC > 0.58 → proceed with full v4. If still ~0.51 → reassess prompt.

To run, use Cell 6 with:
```python
datasets   = ["told_br"]
strategies = ["random"]
ns         = ["50", "200"]
seeds      = ["42"]
MODEL      = "qwen2.5-7b"
```

### Phase 1 — 7B full sweep (72 runs, ~6.5h)

All datasets, all strategies, both N, seeds 0+1+42.

```python
datasets   = ["tweets_hs", "told_br", "20_newsgroups", "wikinews"]
strategies = ["random", "score_guided", "diversity"]
ns         = ["50", "200"]
seeds      = ["0", "1", "42"]
MODEL      = "qwen2.5-7b"
```

### Phase 2 — 14B N=50 (18 runs × 3 seeds = 54 total calls... wait: 4 × 3 × 1 × 3 = 36 runs, ~2h)

```python
ns         = ["50"]
MODEL      = "qwen2.5-14b"
```

### Phase 3 — 14B N=200 (36 runs, ~6.3h — split across sessions if needed)

```python
ns         = ["200"]
MODEL      = "qwen2.5-14b"
```

**Total 14B: 72 runs (~8.3h).  Grand total: ~15h across 3-4 Colab sessions.**

---

## Expected Time on T4

| Model | N=50 avg | N=200 avg | 72 runs total |
|-------|----------|-----------|---------------|
| 14B   | ~3.3 min | ~10.6 min | ~500 min      |
| 7B    | ~2 min   | ~6 min    | ~290 min      |

---

## Colab Cells

### Cell 1 — Mount Drive & Environment

```python
from google.colab import drive
drive.mount('/content/drive')

import os, sys

PROJECT_PATH = "/content/drive/MyDrive/Projeto ML/2026/Master/Multilingual-Text-Anomaly-Detection"
DATA_DIR     = "/content/drive/MyDrive/Projeto ML/2025/AD/third_setup/adaptative-text-anomaly-detection/data"

os.chdir(PROJECT_PATH)
sys.path.insert(0, os.path.join(PROJECT_PATH, "src"))
```

---

### Cell 2 — Install llama-cpp-python (CUDA)

```python
!nvcc --version  # confirm CUDA version (should be 12.x on T4)

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

### Cell 6 — v4 Sweep

```python
import subprocess, re, time, threading, glob
import pandas as pd

PROJECT_PATH = "/content/drive/MyDrive/Projeto ML/2026/Master/Multilingual-Text-Anomaly-Detection"
DATA_DIR     = "/content/drive/MyDrive/Projeto ML/2025/AD/third_setup/adaptative-text-anomaly-detection/data"

# ── Configure per phase (see Execution Plan above) ──
datasets   = ["tweets_hs", "told_br", "20_newsgroups", "wikinews"]
strategies = ["random", "score_guided", "diversity"]
ns         = ["50", "200"]
seeds      = ["0", "1", "42"]

MODEL      = "qwen2.5-7b"   # change to "qwen2.5-14b" for 14B pass

RESULTS_DIR = "data/llm_results/v4"

# ── Keep-alive: pings Drive every 4 min to prevent Colab idle timeout ──
def _keepalive(stop_event, interval=240):
    while not stop_event.wait(interval):
        try:
            _ = glob.glob(f"{PROJECT_PATH}/{RESULTS_DIR}/**/*.csv", recursive=True)
        except Exception:
            pass

_stop = threading.Event()
threading.Thread(target=_keepalive, args=(_stop,), daemon=True).start()

# ── Skip helper: returns True if this exact config is already saved ──
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

total = len(datasets) * len(strategies) * len(ns) * len(seeds)
run   = 0

try:
    for seed in seeds:
        for dataset in datasets:
            for strategy in strategies:
                for n in ns:
                    run += 1

                    if _already_done(dataset, strategy, n, seed, MODEL, RESULTS_DIR):
                        print(f"[{run:03d}/{total}] ↷ {dataset:<30} {strategy:<15} N={n:<4} seed={seed} (already done)", flush=True)
                        continue

                    cmd = [
                        "python", "-u",
                        "scripts/run_llm_active_loop.py",
                        "--project_path", PROJECT_PATH,
                        "--data_dir",     DATA_DIR,
                        "--dataset",      dataset,
                        "--strategy",     strategy,
                        "--n_llm_calls",  n,
                        "--seed",         seed,
                        "--device",       "cuda",
                        "--backend",      "llamacpp",
                        "--llamacpp_model", MODEL,
                        "--results_dir",  RESULTS_DIR,
                    ]
                    t0 = time.time()
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    elapsed = time.time() - t0

                    roc = re.search(r"ROC-AUC\s+\(test\)\s*:\s*([\d.]+)", result.stdout)
                    roc_str = roc.group(1) if roc else "N/A"

                    status = "✓" if result.returncode == 0 else "✗"
                    print(f"[{run:03d}/{total}] {status} {dataset:<30} {strategy:<15} N={n:<4} seed={seed}  "
                          f"ROC={roc_str}  {elapsed/60:.1f}min", flush=True)

                    if result.returncode != 0:
                        print("  STDERR:", result.stderr[-300:])
finally:
    _stop.set()
```

---

### Cell 7 — Quick Results Check

```python
import pandas as pd, glob, numpy as np

files = glob.glob(f"{PROJECT_PATH}/data/llm_results/v4/**/*.csv", recursive=True)
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

# Mean ± std over seeds
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

*In progress.*

| Phase | Runs | Status |
|-------|------|--------|
| Phase 0 — told_br validation (7B, random, N=50+200, seed=42) | 2 | ⏳ |
| Phase 1 — 7B full sweep | 72 | ⏳ |
| Phase 2 — 14B N=50 | 36 | ⏳ |
| Phase 3 — 14B N=200 | 36 | ⏳ |
