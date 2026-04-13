
# v4 — HateBR Calibration Phase ✅

## Overview

**Goal:** Calibration phase — fix the 2×2 dataset grid and lock all methodological decisions before the final production experiment (v5).

**Status: COMPLETE. All decisions made. Grid locked. → Go to v5.**

---

## Final Dataset Grid (2×2)

|                | English         | Portuguese      |
|----------------|-----------------|-----------------|
| Hate Detection | tweets_hs       | **hatebr** ←new |
| Topic Class.   | 20_newsgroups   | wikinews        |

---

## Decisions & Takeaways

### ❌ told_br — dropped
- Oracle AUC (GT labels used as LLM proxy): **0.67**
- With revised toxicity prompt (7B, random): ROC=**0.49** (N=50) / **0.52** (N=200) — no improvement
- Root cause: distiluse-v2 does not separate PT-BR toxic tweets geometrically. All encoders in benchmark gave 0.38–0.59. Structural flaw — not fixable with prompts.
- **Paper treatment:** Not cited. Reserved for future work ("LLM annotation quality on noisy datasets").

### ✅ hatebr — added (`franciellevargas/HateBR`)
- Expert-annotated Instagram comments on Brazilian politicians, 7,000 samples
- Columns: `comentario` → `text`, `label_final` → `label`
- Oracle run N=50: ROC=**0.6638** (SetFit skipped — only 3 anomalies found)
- Oracle run N=200: ROC=**0.7254** (SetFit active, 12 anomalies, agreement=92%)
- **Surpasses told_br oracle (0.67) with noisy LLM labels** → strong positive signal

### ❌ score_guided — dropped
- Performance in v3 (seed=42): worse than random in **7/8 configs** (worst: tweets_hs N=50, Δ=−0.40)
- Root cause: score_guided prioritises uncertain samples → selects hard negatives, not anomalies
- v3 seed=42 data cited as ablation evidence in paper.
- v5 strategies: `random` + `diversity` only.

### ✅ 7B + 14B — both kept, smaller/larger models discarded
- v3 pilot (seed=42): 7B=0.8708 vs 14B=0.8663 (tweets_hs random N=50), 7B=0.8509 vs 14B=0.8467 (diversity N=200)
- 20_newsgroups: essentially tied across all configs
- **Paper argument:** "annotation quality does not scale trivially with model size — 7B is sufficient and ~40% cheaper"
- Running both provides the model size ablation.
- **Qwen 3B discarded:** no prior evidence that annotation quality holds; would add +48 runs with no guaranteed return. Optional spot check post-v5 if a Colab session is available.
- **Q2 quantization discarded:** degrades reasoning more than Q4, especially on text classification.
- **32B+ infeasible:** T4 has 16GB VRAM.
- **Final decision: 7B Q4 + 14B Q4. Do not revisit.**

### ✅ N={50, 200} — locked, 100/150 discarded
- N=100/150 tested in v2, removed in v3: intermediate results with no qualitative difference
- N=50 = minimal annotation budget (~5% of training data); N=200 = generous budget (~20%)
- Captures the endpoints of the cost-benefit curve — sufficient for the paper's argument
- Adding N=100/150 would double the runs (96→192) without changing the conclusion
- In v3, tweets_hs N=50 already achieved ROC=0.87 — no "elbow" in the curve to discover
- **Final decision: N={50, 200}. Do not revisit.**

### ✅ `separation_ratio` — embedding separability metric
- Implemented in `src/utils/metrics.py`
- Formula: `sep_ratio = mean_cosine_dist(A→N) / mean_cosine_dist(A→A)`
  - **< 1** → anomalies cluster away from normals → easy task
  - **> 1** → anomalies blend into normal space → hard task
- **Paper use:** report in Table 1 alongside baseline AUC — becomes an explanatory variable for why some datasets are harder:
  - 20_newsgroups, wikinews: low sep_ratio → unsupervised models already perform well
  - tweets_hs, hatebr: higher sep_ratio → LLM guidance gives bigger lift
  - told_br (dropped): very high sep_ratio → structural limit confirmed empirically
- Compute in v5 Phase 0 (Cell 0b)

### ✅ distiluse-base-multilingual-cased-v2 — encoder confirmed
- Empirical evidence from `data/stil_data/benchmark_results (4).csv` (unsupervised models, mean AUC):

| dataset       | distiluse-v2 | xlm-roberta-large | bert-base-PT |
|---------------|:------------:|:-----------------:|:------------:|
| 20_newsgroups | 0.846        | 0.688             | —            |
| tweets_hs     | **0.539**    | 0.415             | —            |
| wikinews      | 0.701        | 0.678             | 0.755        |

- xlm-roberta-large underperforms on 2/3 datasets despite being a much larger model.
- bert-base-PT wins on wikinews but is monolingual — cannot handle EN datasets.
- **distiluse-v2 is the best multilingual option across the 2×2 grid.**
- **Paper argument:** "We evaluated multiple encoders and selected distiluse-v2 as it consistently outperformed alternatives. Encoder selection is orthogonal to our main contribution — all methods use the same encoder for a fair comparison."

### 📁 Folder structure locked
- LLM results: `data/llm_results/v5/`
- Benchmark results: `data/benchmark_results/` (written by `run_benchmark.py`)
- Parquets (texts/labels/embeddings): `DATA_DIR` (external Drive path)

---

## Phase Status

| Phase | Runs | Status |
|-------|------|--------|
| Phase 0a — told_br validation (7B, random, N=50+200, seed=42) | 2 | ✅ ROC=0.49/0.52 → dropped |
| Phase 0b — hatebr validation (7B, random, N=50+200, seed=42) | 2 | ✅ ROC=0.66/0.725 → added |

**Phase 0 is the only required phase for v4. Cells below are kept as reference for an optional single-seed pilot sweep.**

---

## Next Steps → v5

1. **Run benchmark** (all 4 datasets, distiluse-v2, Colab GPU, ~15 min):
```bash
python scripts/run_benchmark.py \
    --project_path $PROJECT_PATH \
    --data_dir $DATA_DIR \
    --results_dir data/benchmark_results \
    --device cuda
```

2. **Run LLM v5** (96 runs, ~10h, 2–3 Colab sessions) → see `data/llm_results/v5/README.md`

3. **Compile results** — benchmark_results.csv + v5 CSVs → paper tables

---

## Reference Cells (optional single-seed pilot sweep)

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

# ── v4 calibration: hatebr only, seed=42, 7B ──
# For full 3-seed production experiment see v5/README.md
datasets   = ["tweets_hs", "hatebr", "20_newsgroups", "wikinews"]  # told_br replaced by hatebr
strategies = ["random", "diversity"]   # score_guided dropped
ns         = ["50", "200"]
seeds      = ["42"]  # single seed for calibration; 3 seeds (0,1,42) go in v5

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

total = len(datasets) * len(strategies) * len(ns) * len(seeds)  # 4×2×2×1 = 16 (calibration)
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

## Reference Cells (optional pilot — not required for v5)

Cells below are kept as reference for running a single-seed pilot sweep on all 4 datasets.  
**Skip these and go directly to v5 if you are ready for the full run.**
