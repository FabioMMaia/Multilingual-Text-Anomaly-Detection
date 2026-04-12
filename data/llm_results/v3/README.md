# v3 — Pilot Experiment

## Overview

**Goal:** Validate the final experiment grid (4 datasets × 2 models × 3 strategies × 2 N-values × seed=42 = **48 runs**)  
before scaling to 3 seeds in v3-full.

**New in v3 vs v2:**

- Added `diversity` sampling strategy (KMeans++ — maximises embedding-space coverage)
- Dropped SA datasets (pt_tweets, tweeteval) — frequency-based anomaly is semantically incoherent
- Both Qwen 7B and 14B tested head-to-head on all 4 datasets
- N reduced to {50, 200} only (N=100/150 removed — not informative)

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
| Seed      | 42 |
| Threshold | 0.45 |
| Prompt    | v1 (unchanged from v1/v2) |
| Encoder   | distiluse-base-multilingual-cased-v2 |
| Hardware  | Colab T4 |

Results are saved under `data/llm_results/v3/{strategy}/N_{n}/{dataset}.csv`.

---

## Hypotheses to Validate

| Hypothesis | How to check |
|---|---|
| Diversity beats random in tweets_hs | ROC diversity N=50 > ROC random N=50 |
| 14B does NOT consistently beat 7B | Compare mean ROC per model per dataset |
| TC datasets are easy regardless of N | ROC for 20_newsgroups & wikinews stable across N=50/200 |
| told_br stays ~0.49 (structural issue) | Confirm before investing 3 seeds |
| N=50 is enough (or better than N=200) | Cost-benefit comparison by N |

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

> `DATA_DIR` points to the folder containing the parquet files (`texts_*`, `labels_*`, `embeddings_*`).  
> If parquets are already inside this repo set `DATA_DIR = f"{PROJECT_PATH}/data"`.

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

BASE = "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main"
SHARD = "qwen2.5-7b-instruct-q4_k_m.gguf"

import subprocess
result = subprocess.run(
    ["wget", "-q", "--show-progress", f"{BASE}/{SHARD}", "-O", f"{MODEL_DIR}/{SHARD}"]
)
print("Done:", result.returncode)
```

---

### Cell 5 — Download Qwen 14B (one-time, ~9 GB, 3 shards)

```python
MODEL_DIR = f"{PROJECT_PATH}/models"
os.makedirs(MODEL_DIR, exist_ok=True)

BASE = "https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF/resolve/main"
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

> **Tip:** Download to Drive once and reuse across Colab sessions. The files persist at `{PROJECT_PATH}/models/`.

---

### Cell 6 — v3 Pilot Sweep (48 runs)

Run Qwen **14B** first (slower — better to start with); re-run swapping to 7B.

```python
import subprocess, re, time, threading, glob
import pandas as pd

PROJECT_PATH = "/content/drive/MyDrive/Projeto ML/2026/Master/Multilingual-Text-Anomaly-Detection"
DATA_DIR     = "/content/drive/MyDrive/Projeto ML/2025/AD/third_setup/adaptative-text-anomaly-detection/data"

datasets   = ["tweets_hs", "told_br", "20_newsgroups", "wikinews"]
strategies = ["random", "score_guided", "diversity"]
ns         = ["50", "200"]
seed       = "42"

# ── Change this to "qwen2.5-7b" for the 7B pass ──
MODEL = "qwen2.5-14b"

# ── Keep-alive: pings drive every 4 min to prevent Colab idle timeout ──
def _keepalive(stop_event, interval=240):
    while not stop_event.wait(interval):
        try:
            _ = glob.glob(f"{PROJECT_PATH}/data/llm_results/v3/**/*.csv", recursive=True)
        except Exception:
            pass

_stop = threading.Event()
threading.Thread(target=_keepalive, args=(_stop,), daemon=True).start()

# ── Skip helper: returns True if this config is already saved in the CSV ──
def _already_done(dataset, strategy, n, seed, model, results_dir):
    pattern = f"{PROJECT_PATH}/{results_dir}/{strategy}/N_{n}/{dataset}.csv"
    for f in glob.glob(pattern):
        try:
            df = pd.read_csv(f)
            model_tag = model.split("/")[-1]
            match = df[
                (df["n_llm_calls"] == int(n)) &
                (df["seed"] == int(seed)) &
                (df["llm_model"].str.contains(model_tag, na=False))
            ]
            if not match.empty:
                return True
        except Exception:
            pass
    return False

total = len(datasets) * len(strategies) * len(ns)
run   = 0

try:
    for dataset in datasets:
        for strategy in strategies:
            for n in ns:
                run += 1

                if _already_done(dataset, strategy, n, seed, MODEL, "data/llm_results/v3"):
                    print(f"[{run:02d}/{total}] ↷ {dataset:<30} {strategy:<15} N={n:<4} (already done)", flush=True)
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
                    "--results_dir",  "data/llm_results/v3",
                ]
                t0 = time.time()
                result = subprocess.run(cmd, capture_output=True, text=True)
                elapsed = time.time() - t0

                roc = re.search(r"ROC-AUC\s+\(test\)\s*:\s*([\d.]+)", result.stdout)
                roc_str = roc.group(1) if roc else "N/A"

                status = "✓" if result.returncode == 0 else "✗"
                print(f"[{run:02d}/{total}] {status} {dataset:<30} {strategy:<15} N={n:<4} "
                      f"ROC={roc_str}  {elapsed/60:.1f}min", flush=True)

                if result.returncode != 0:
                    print("  STDERR:", result.stderr[-300:])
finally:
    _stop.set()  # stop keep-alive thread when sweep finishes or crashes
```

**Expected time on T4:**

| Model | N=50 avg | N=200 avg | 24 runs total |
|-------|----------|-----------|---------------|
| 14B   | ~3.3 min | ~10.6 min | ~170 min      |
| 7B    | ~2 min   | ~6 min    | ~95 min       |

> **Note:** If the session drops mid-sweep, simply re-run Cell 6 — already-completed runs are skipped automatically via `_already_done()`. The keep-alive thread pings Drive every 4 min to suppress Colab's idle-timeout warning.

---

### Cell 7 — Quick Results Check

```python
import pandas as pd, glob

files = glob.glob(f"{PROJECT_PATH}/data/llm_results/v3/**/*.csv", recursive=True)
metrics_files = [f for f in files if "llm_labels" not in f]

dfs = []
for f in metrics_files:
    try:
        dfs.append(pd.read_csv(f, engine="python", on_bad_lines="skip"))
    except Exception as e:
        print(f"Error in {f}: {e}")

df = pd.concat(dfs, ignore_index=True)

# Clean model name
df["model"] = (
    df["llm_model"].str.split("/").str[-1]
    .str.replace(r"-instruct-q4_k_m.*\.gguf", "", regex=True)
)

cols = [
    "dataset", "model", "strategy", "n_llm_calls", "seed",
    "roc_auc", "pr_auc",
    "llm_agreement", "llm_precision", "llm_recall",
    "n_anomalies_found", "n_normals_found", "n_parse_errors",
]
cols = [c for c in cols if c in df.columns]

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
pd.set_option("display.float_format", "{:.4f}".format)

display(
    df[cols]
    .sort_values(["dataset", "model", "strategy", "n_llm_calls"])
    .reset_index(drop=True)
)
```

---

## Results

> Fill in after pilot runs complete.

### tweets_hs

| model | strategy | N | ROC-AUC | PR-AUC | agreement | precision | recall |
|-------|----------|---|---------|--------|-----------|-----------|--------|
| — | — | — | — | — | — | — | — |

### told_br

| model | strategy | N | ROC-AUC | PR-AUC | agreement | precision | recall |
|-------|----------|---|---------|--------|-----------|-----------|--------|
| — | — | — | — | — | — | — | — |

### 20_newsgroups

| model | strategy | N | ROC-AUC | PR-AUC | agreement | precision | recall |
|-------|----------|---|---------|--------|-----------|-----------|--------|
| — | — | — | — | — | — | — | — |

### wikinews

| model | strategy | N | ROC-AUC | PR-AUC | agreement | precision | recall |
|-------|----------|---|---------|--------|-----------|-----------|--------|
| — | — | — | — | — | — | — | — |

---

## Notes

- **API keys:** Never hardcode in notebooks. Use `os.environ["OPENAI_API_KEY"] = ...` in a private Colab cell or Colab Secrets.
- **Model shards:** The 14B model needs all 3 shards present before llamacpp can load it.
- **Results dir:** All runs pass `--results_dir data/llm_results/v3` so outputs land here, separate from v0/v1/v2.
- **Seed reproducibility:** Seed=42 is fixed for the pilot. Seeds 0, 1, 42 will be used in v3-full (only for configs validated here).
