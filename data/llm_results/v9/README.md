# v9 — Hotfix: 20 Newsgroups LLM Annotation (corrected prompt)

> **After this run completes, migrate results into v5 and re-run v6/v7/v8 normally.**
> v9 is a temporary staging area — the final consolidated source is always v5/v6/v7/v8.

## Why This Exists

The LLM prompt for `20_newsgroups` in `src/pipeline/llm_runner.py` was wrong in v5–v8.
It told the LLM "comp.graphics = anomaly" but the actual dataset setup is:

- **Normal (label 0):** `rec.sport.hockey` — 600 posts (most frequent class → picked by `label_normal_vs_anomaly()`)
- **Anomaly (label 1):** all other 19 newsgroups merged, subsampled to 5%

Because comp.graphics is only ~1/19 of the anomaly pool, the LLM flagged ≈1 post as
anomalous per run. SetFit never activated. v5–v8 results for 20_newsgroups are invalid.

The corrected prompt (already applied to `src/pipeline/llm_runner.py`):
- `d_task`: decide if a post belongs to rec.sport.hockey (normal, 0.0) or any other topic (anomalous, 1.0)
- `c_normal`: posts about ice hockey — game scores, player trades, NHL news, equipment, strategy...
- `c_anomaly`: any topic other than ice hockey: computers, science, politics, religion, autos, space...

## Plan (3 steps)

**Step 1 — Run v9 (this README)**
Re-run LLM annotation for 20_newsgroups only, saving to `data/llm_results/v9/`.
This is identical to the v5 sweep Cell 6, just restricted to one dataset.

**Step 2 — Migrate v9 into v5**
Delete the old 20_newsgroups files from v5 and copy the v9 files in their place.
See the migration cell below (Cell 7).

**Step 3 — Re-run v6, v7, v8 for 20_newsgroups**
v6/v7/v8 read their labels from v5. With the v9 files now in v5, just re-run those
sweeps filtered to `--dataset 20_newsgroups`. Their existing README cells already
have the correct commands — just change `datasets = ["20_newsgroups"]` and run.

---

## Final version map (after migration)

| Version | AD Model | SetFit | Dataset scope |
|---------|----------|--------|---------------|
| **v5** | DeepSAD + SetFit | ✅ | all 4 datasets (20_news patched from v9) |
| **v6** | DeepSAD, no SetFit | ❌ | all 4 datasets |
| **v7** | MLP, no SetFit | ❌ | all 4 datasets |
| **v8** | MLP + SetFit | ✅ | all 4 datasets |

After migration: `python scripts/_consolidate_results.py --project_path .` just loads v5/v6/v7/v8.

---

## Folder Structure

```
data/llm_results/
├── v9/               ← LLM annotation + DeepSAD + SetFit  (20_newsgroups only)
│   ├── random/
│   │   ├── N_50/     → 20_newsgroups.csv, 20_newsgroups_llm_labels.csv
│   │   └── N_200/    → 20_newsgroups.csv, 20_newsgroups_llm_labels.csv
│   └── diversity/
│       ├── N_50/     → 20_newsgroups.csv, 20_newsgroups_llm_labels.csv
│       └── N_200/    → 20_newsgroups.csv, 20_newsgroups_llm_labels.csv
├── v9_deepsad/       ← DeepSAD, no SetFit, reads v9 labels
│   └── (same layout, only 20_newsgroups.csv, no _llm_labels files)
├── v9_mlp/           ← MLP, no SetFit, reads v9 labels
│   └── (same layout)
└── v9_mlp_sf/        ← MLP + SetFit, reads v9 labels
    └── (same layout)
```

---

## Config

| Parameter  | Value                                        |
|------------|----------------------------------------------|
| Dataset    | **20_newsgroups only**                       |
| LLM Models | Qwen 2.5 7B Q4 + Qwen 2.5 14B Q4 (llamacpp) |
| Strategies | random, diversity                            |
| N          | 50, 200                                      |
| Seeds      | 0, 1, 42                                     |
| Threshold  | 0.5                                          |
| Encoder    | distiluse-base-multilingual-cased-v2         |
| Hardware   | Colab T4                                     |
| Results    | `data/llm_results/v9/{strategy}/N_{n}/20_newsgroups.csv` |

**Total LLM annotation runs:** 2 models × 2 strategies × 2 N × 3 seeds = **24 runs**
**Time estimate:** 7B ~1.5h + 14B ~3h = ~4.5h total

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

### Cell 4 — (Optional) Download Models if not already on Drive

```python
MODEL_DIR = f"{PROJECT_PATH}/models"
os.makedirs(MODEL_DIR, exist_ok=True)

# 7B (single file, ~4 GB)
import subprocess
subprocess.run(["wget", "-q", "--show-progress",
    "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf",
    "-O", f"{MODEL_DIR}/qwen2.5-7b-instruct-q4_k_m.gguf"])

# 14B (3 shards, ~9 GB total)
for shard in [
    "qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf",
    "qwen2.5-14b-instruct-q4_k_m-00002-of-00003.gguf",
    "qwen2.5-14b-instruct-q4_k_m-00003-of-00003.gguf",
]:
    subprocess.run(["wget", "-q", "--show-progress",
        f"https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF/resolve/main/{shard}",
        "-P", MODEL_DIR])
print("Done.")
```

---

### Cell 5 — Skip Helper

```python
import glob, pandas as pd

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
```

---

### Cell 6 — v9 LLM Annotation Sweep (24 runs, ~4.5h)

Runs the LLM annotation + DeepSAD + SetFit for 20_newsgroups with the corrected prompt.
Results go to `data/llm_results/v9/`.

```python
import subprocess, re, time, threading, glob

PROJECT_PATH = "/content/drive/MyDrive/Projeto ML/2026/Master/Multilingual-Text-Anomaly-Detection"
DATA_DIR     = "/content/drive/MyDrive/Projeto ML/2025/AD/third_setup/adaptative-text-anomaly-detection/data"

strategies = ["random", "diversity"]
ns         = ["50", "200"]
seeds      = ["0", "1", "42"]
models     = ["qwen2.5-7b", "qwen2.5-14b"]

RESULTS_DIR = "data/llm_results/v9"

# Keep-alive (prevents Colab disconnection)
import threading
def _keepalive(stop_event, interval=240):
    while not stop_event.wait(interval):
        try: _ = glob.glob(f"{PROJECT_PATH}/{RESULTS_DIR}/**/*.csv", recursive=True)
        except: pass
_stop = threading.Event()
threading.Thread(target=_keepalive, args=(_stop,), daemon=True).start()

total = len(models) * len(strategies) * len(ns) * len(seeds)  # 24
run = 0

try:
    for model in models:
        print(f"\n{'='*60}\n  MODEL: {model}\n{'='*60}\n")
        for seed in seeds:
            for strategy in strategies:
                for n in ns:
                    run += 1

                    if _already_done("20_newsgroups", strategy, n, seed, model, RESULTS_DIR):
                        print(f"[{run:02d}/{total}] ↷ {strategy:<12} N={n:<4} seed={seed} model={model} (skip)", flush=True)
                        continue

                    cmd = [
                        "python", "-u",
                        "scripts/run_llm_active_loop.py",
                        "--project_path",  PROJECT_PATH,
                        "--data_dir",      DATA_DIR,
                        "--dataset",       "20_newsgroups",
                        "--strategy",      strategy,
                        "--n_llm_calls",   n,
                        "--seed",          seed,
                        "--device",        "cuda",
                        "--backend",       "llamacpp",
                        "--llamacpp_model", model,
                        "--results_dir",   RESULTS_DIR,
                    ]

                    t0 = time.time()
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    elapsed = time.time() - t0

                    roc = re.search(r"ROC-AUC\s+\(test\)\s*:\s*([\d.]+)", result.stdout)
                    roc_str = roc.group(1) if roc else "N/A"
                    status  = "✓" if result.returncode == 0 else "✗"

                    print(f"[{run:02d}/{total}] {status} {strategy:<12} N={n:<4} seed={seed} model={model}  "
                          f"ROC={roc_str}  {elapsed/60:.1f}min", flush=True)

                    if result.returncode != 0:
                        print(f"  STDERR: {result.stderr[-600:]}", flush=True)
finally:
    _stop.set()
    print("\nDone.")
```

---

### Cell 7 — Migrate v9 → v5 (run locally after Cell 6 completes)

Delete the old (wrong-prompt) 20_newsgroups files from v5 and replace with v9 results.

```python
import shutil, glob, os

PROJECT_PATH = "/content/drive/MyDrive/Projeto ML/2026/Master/Multilingual-Text-Anomaly-Detection"

for strategy in ["random", "diversity"]:
    for n in ["50", "200"]:
        src_dir = f"{PROJECT_PATH}/data/llm_results/v9/{strategy}/N_{n}"
        dst_dir = f"{PROJECT_PATH}/data/llm_results/v5/{strategy}/N_{n}"

        for fname in ["20_newsgroups.csv", "20_newsgroups_llm_labels.csv"]:
            src = f"{src_dir}/{fname}"
            dst = f"{dst_dir}/{fname}"
            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"  ✓ copied {strategy}/N_{n}/{fname}")
            else:
                print(f"  ✗ not found: {src}")

print("Migration done. v5 now has corrected 20_newsgroups files.")
```

---

### Cell 8 — Re-run v6 for 20_newsgroups only (~15 min)

Open `data/llm_results/v6/README.md` Cell 4, change:
```python
datasets = ["20_newsgroups"]
```
and run it. The skip logic will ignore already-done rows for the other datasets.

---

### Cell 9 — Re-run v7 for 20_newsgroups only (~15 min)

Same as above using `data/llm_results/v7/README.md` Cell 4.

---

### Cell 10 — Re-run v8 for 20_newsgroups only (~15–30 min)

Same as above using `data/llm_results/v8/README.md` Cell 4.

---

### Cell 11 — Consolidate and print paper tables (run locally)

```bash
python scripts/_consolidate_results.py --project_path .
```

---

### (old) Cell 7 — v9_deepsad: DeepSAD without SetFit (reads v9 labels, ~15 min)

```python
import subprocess, re, time, os

PROJECT_PATH = "/content/drive/MyDrive/Projeto ML/2026/Master/Multilingual-Text-Anomaly-Detection"
DATA_DIR     = "/content/drive/MyDrive/Projeto ML/2025/AD/third_setup/adaptative-text-anomaly-detection/data"

strategies = ["random", "diversity"]
ns         = ["50", "200"]
seeds      = ["0", "1", "42"]
models     = ["qwen2.5-7b", "qwen2.5-14b"]

RESULTS_DIR = "data/llm_results/v9_deepsad"
V9_DIR      = "data/llm_results/v9"

total = len(models) * len(strategies) * len(ns) * len(seeds)  # 24
run = 0

for model in models:
    for seed in seeds:
        for strategy in strategies:
            for n in ns:
                run += 1

                if _already_done("20_newsgroups", strategy, n, seed, model, RESULTS_DIR):
                    print(f"[{run:02d}/{total}] ↷ {strategy:<12} N={n:<4} seed={seed} model={model} (skip)", flush=True)
                    continue

                labels_csv = f"{PROJECT_PATH}/{V9_DIR}/{strategy}/N_{n}/20_newsgroups_llm_labels.csv"
                if not os.path.exists(labels_csv):
                    print(f"[{run:02d}/{total}] ✗ labels not found: {labels_csv}", flush=True)
                    continue

                cmd = [
                    "python", "-u",
                    "scripts/run_llm_active_loop.py",
                    "--project_path",    PROJECT_PATH,
                    "--data_dir",        DATA_DIR,
                    "--dataset",         "20_newsgroups",
                    "--strategy",        strategy,
                    "--n_llm_calls",     n,
                    "--seed",            seed,
                    "--device",          "cuda",
                    "--backend",         "llamacpp",
                    "--load_labels_from",   labels_csv,
                    "--load_labels_model",  model,
                    "--no_setfit",
                    "--results_dir",     RESULTS_DIR,
                ]

                t0 = time.time()
                result = subprocess.run(cmd, capture_output=True, text=True)
                elapsed = time.time() - t0

                roc = re.search(r"ROC-AUC\s+\(test\)\s*:\s*([\d.]+)", result.stdout)
                roc_str = roc.group(1) if roc else "N/A"
                status  = "✓" if result.returncode == 0 else "✗"

                print(f"[{run:02d}/{total}] {status} {strategy:<12} N={n:<4} seed={seed} model={model}  "
                      f"ROC={roc_str}  {elapsed/60:.1f}min", flush=True)

                if result.returncode != 0:
                    print(f"  STDERR: {result.stderr[-600:]}", flush=True)

print("\nDone.")
```

---

### Cell 8 — v9_mlp: MLP without SetFit (reads v9 labels, ~15 min)

> These cells (old Cell 8/9 for v9_mlp/v9_mlp_sf) are no longer needed.
> Use the migration approach instead (Cells 7–10 above).

---

### Cell 12 — Quick Sanity Check (run after Cell 6 completes, before migrating)

```python
import pandas as pd, glob

files = glob.glob(f"{PROJECT_PATH}/data/llm_results/v9/**/*.csv", recursive=True)
metrics = [f for f in files if "llm_labels" not in f]

df = pd.concat([pd.read_csv(f) for f in metrics], ignore_index=True)
print("Runs completed:", len(df))
print("\nMean n_anomalies_found by model x N:")
print(df.groupby(["llm_model", "n_llm_calls"])["n_anomalies_found"].mean().round(1))
print("\nSetFit skipped rate:")
print(df.groupby(["llm_model", "n_llm_calls"])["setfit_skipped"].value_counts())
print("\nMean ROC-AUC by model x N:")
print(df.groupby(["llm_model", "n_llm_calls"])["roc_auc"].mean().round(4))
```

**Expected sanity check output:**
- `n_anomalies_found`: should be **much higher** than v5's ~0–2 (expect 5–15+ at N=50, 15–50+ at N=200)
- `setfit_skipped`: should be `False` for many runs at N=200 (activation threshold = 5 anomalies)
- `roc_auc`: may vary — the interesting question is whether SetFit now helps

---

## After Running: Next Steps

1. Migrate v9 → v5 (Cell 7)
2. Re-run v6, v7, v8 for 20_newsgroups (Cells 8–10)
3. Run consolidation script (Cell 11): `python scripts/_consolidate_results.py --project_path .`
4. Update `puplication/STIL2026-.../main.tex` with new 20 Newsgroups numbers
5. See `docs/FIX_20NEWSGROUPS_RERUN.md` for the full paper-update checklist
