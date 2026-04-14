# Multilingual Text Anomaly Detection

Research repository investigating text anomaly detection in low-supervision multilingual scenarios — from classical benchmarks to LLM-guided annotation without any human labels.

> **Current direction:** See [GUIDELINE.md](GUIDELINE.md)

---

## Research Story

### Motivation

Classical anomaly detection is fully unsupervised — no labels required. Semi-supervised models (DeepSAD, MLP) are far more powerful but require ground-truth labels. **Can we replace human labels with LLM-generated annotations and still beat unsupervised baselines?**

---

### Experiment Line

#### Baseline — Unsupervised + Oracle benchmark

Ran 11 models (IForest, LOF, DeepSVDD, OCSVM, AutoEncoder, VAE, HBOS, DevNet, DeepSAD, MLP, XGBOD) on 4 datasets with `distiluse-base-multilingual-cased-v2` embeddings.

| Dataset | Best unsup (no labels) | Oracle best (GT labels) | Gap |
|---|---|---|---|
| 20_newsgroups | 0.920 (VAE) | 0.998 (MLP) | 0.078 |
| hatebr | 0.561 (LOF) | 0.873 (XGBOD) | 0.312 |
| tweets_hs | 0.575 (AutoEncoder) | 0.956 (MLP) | 0.381 |
| wikinews | 0.776 (VAE) | 0.943 (XGBOD) | 0.167 |

→ [`data/benchmark_results/`](data/benchmark_results/)

---

#### v5 — LLM active loop (main pipeline)

LLM (Qwen 7B or 14B) annotates N samples selected by a strategy (random or diversity). SetFit fine-tunes the embedding on LLM labels. DeepSAD uses the fine-tuned embedding.

**Grid:** 4 datasets × 2 strategies × 2 N (50/200) × 2 models (7B/14B) × 3 seeds = **96 runs**

| Dataset | Best config | AUC | vs unsup |
|---|---|---|---|
| 20_newsgroups | diversity / N=50 / 14B | **0.947** | +0.027 ✅ |
| hatebr | random / N=200 / 7B | **0.754** | +0.193 ✅ |
| tweets_hs | diversity / N=200 / 14B | **0.871** | +0.296 ✅ |
| wikinews | random / N=200 / 14B | **0.865** | +0.089 ✅ |

**Pipeline supera todos os baselines não-supervisionados em todos os datasets — sem nenhum label humano.**

→ [`data/llm_results/v5/`](data/llm_results/v5/README.md)

---

#### v6 — Ablation: sem SetFit (isola contribuição do fine-tuning)

Mesmos labels do v5, mesmo DeepSAD. Só remove o SetFit — usa distiluse direto.

**SetFit gain global: +0.002 ±0.064** — efeito condicional:

| Dataset | SetFit gain | Conclusão |
|---|---|---|
| 20_newsgroups | +0.029 | SetFit ajuda — sinal EN forte |
| hatebr | +0.025 (+0.050 @ N=200) | SetFit ajuda — mais anotações compensam |
| tweets_hs | −0.014 | SetFit prejudica — distiluse já é bom |
| wikinews | −0.030 | SetFit prejudica — multilingual forte no base |

**Conclusão:** SetFit é dataset-specific. A alta variância (±0.064) indica instabilidade sobre labels ruidosos.

→ [`data/llm_results/v6/`](data/llm_results/v6/README.md)

---

#### v7 — Ablation: MLP vs DeepSAD (isola contribuição do modelo AD)

Mesmos labels do v5, sem SetFit (distiluse direto). Troca DeepSAD por MLP (PyTorch, BCELoss).

**Hipótese:** DeepSAD (geométrico) seria mais robusto a labels ruidosos. **Refutada.**

| Dataset | v6 DeepSAD | v7 MLP | MLP gain | % gap unsup→oracle fechado |
|---|---|---|---|---|
| 20_newsgroups | 0.850 | **0.937** | +0.105 | 22% |
| hatebr | 0.595 | **0.672** | +0.077 | 36% |
| tweets_hs | 0.806 | **0.819** | +0.013 | **64%** |
| wikinews | 0.748 | **0.827** | +0.079 | 31% |

MLP supera DeepSAD em todos os datasets (+0.066 global). MLP sem SetFit supera até DeepSAD com SetFit (v5).

**Conclusão:** modelo AD é mais determinante que o fine-tuning do embedding.

→ [`data/llm_results/v7/`](data/llm_results/v7/README.md)

---

#### v8 — MLP + SetFit (pipeline completo com MLP) ✅

MLP + SetFit fecha o 2×2. SetFit ajuda condicionalmente: +0.059 em hatebr, +0.010 em tweets_hs, sem efeito em 20_newsgroups (LLM não encontra anomalias suficientes para fine-tuning), e **regride em wikinews** (−0.049 — overfitting com labels ruidosos).

| | sem SetFit | com SetFit |
|---|---|---|
| **DeepSAD** | v6 ✅ | v5 ✅ |
| **MLP** | v7 ✅ | **v8 ✅** |

**Melhor configuração global: MLP + SetFit** com AUC médio 0.819 vs oracle 0.942 (fecha 22–67% do gap).

→ [`data/llm_results/v8/`](data/llm_results/v8/README.md)

---

### Tabela Final — 2×2 completa (mean ROC-AUC)

| Dataset | Unsup | DS+SF (v5) | DS (v6) | MLP (v7) | **MLP+SF (v8)** | Oracle |
|---|---|---|---|---|---|---|
| 20_newsgroups | 0.920 | 0.879 | 0.850 | 0.937 | **0.937** | 0.998 |
| hatebr | 0.561 | 0.620 | 0.595 | 0.672 | **0.731** | 0.873 |
| tweets_hs | 0.575 | 0.791 | 0.806 | 0.819 | **0.829** | 0.956 |
| wikinews | 0.776 | 0.719 | 0.748 | **0.827** | 0.778 | 0.943 |
| **Global** | **0.708** | **0.752** | **0.750** | **0.814** | **0.819** | **0.942** |

---

### Key Takeaways

1. **LLM labels > sem labels:** pipeline supera unsupervised em todos os datasets (tweets_hs +0.296, hatebr +0.193)
2. **MLP > DeepSAD com labels ruidosos** (+0.066 global) — hipótese de robustez geométrica refutada
3. **SetFit é condicional:** ajuda quando há anomalias LLM suficientes (hatebr N=200 +0.15), sem efeito para datasets com poucas anomalias (20_newsgroups), pode regredir com labels muito ruidosos (wikinews −0.049)
4. **Modelo AD > fine-tuning:** MLP sem SetFit bate DeepSAD com SetFit em todos os datasets
5. **tweets_hs é o caso mais forte:** 67% do gap unsup→oracle fechado com labels LLM (v8)
6. **7B suficiente para hate speech; 14B compensa em tópico multilingual** (wikinews: 7B=0.657 vs 14B=0.780)
7. **Eficiência de labels extrema:** oracle usa 15×–78× mais anomalias confirmadas que o pipeline N=200 — e mesmo assim o gap de AUC é de apenas 5–12 pp
8. **Melhor configuração final: MLP + SetFit (v8)** — 0.819 global, fecha 22–67% do gap unsup→oracle por dataset

---

### Label Efficiency — Pipeline vs Oracle

| Dataset | Oracle anomalias (GT) | Pipeline N=200 (LLM anom. média) | Razão | Gap AUC |
|---|---|---|---|---|
| 20_newsgroups | 39 | ~1 | **~49×** | −0.050 |
| hatebr | 140 | ~8 | **~18×** | −0.122 |
| tweets_hs | 1189 | ~15 | **~78×** | −0.063 |
| wikinews | 233 | ~15 | **~15×** | −0.084 |

O oracle recebe dezenas a centenas de vezes mais anomalias reais confirmadas, com labels humanos perfeitos. O pipeline usa apenas anotações LLM ruidosas de N=200 amostras — e ainda assim fecha 64% do gap em tweets_hs.

---

## Documentation

| File | Description |
|---|---|
| [GUIDELINE.md](GUIDELINE.md) | Current research direction, proposed pipeline, and next steps |
| [docs/BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md) | Results from the STIL 2025 paper (6 datasets, 12 models, 6 encoders) |
| [docs/SYNTHESIS.md](docs/SYNTHESIS.md) | Cross-experiment synthesis and findings |
| [docs/SETFIT_RESULTS.md](docs/SETFIT_RESULTS.md) | SetFit fine-tuning pipeline and experiment results (k=20, k=40) |

---

## Repository Structure

```text
.
├── GUIDELINE.md                 # Current research strategy
├── data/                        # Parquet files: texts, labels, embeddings; LLM results
│   └── llm_results/             # CSVs from run_llm_active_loop.py
├── experiments_results/         # CSV results from benchmark runs
├── docs/
│   ├── BENCHMARK_RESULTS.md     # STIL 2025 paper results
│   ├── SYNTHESIS.md             # Cross-experiment findings
│   └── SETFIT_RESULTS.md        # SetFit pipeline + results
├── notebooks/
│   ├── data_preparation.ipynb             # Download datasets, generate embeddings
│   ├── experiments_benchmarks.ipynb       # Exploratory benchmark runs
│   └── Setfit_complete_pipeline_v0.ipynb  # SetFit AD pipeline (Colab)
├── scripts/
│   ├── run_data_preparation.py      # Encode datasets → save parquet files
│   ├── run_benchmark.py             # Unsupervised + semi-supervised benchmark
│   ├── run_setfit_experiment.py     # SetFit fine-tuning + AD benchmark
│   ├── run_llm_active_loop.py       # ★ Exp 3: LLM-guided AD pipeline (no human labels)
│   ├── test_llm_annotator.py        # Unit test: LLM annotator (API or mock)
│   └── test_pipeline_integration.py # Integration test: full pipeline on synthetic data
├── src/
│   ├── encoders/
│   │   └── text_encoders.py         # SentenceBERT, BERTimbau wrappers
│   ├── models/
│   │   └── MLP.py                   # MLP (sklearn) + MLPTF (TF/Keras)
│   ├── pipeline/
│   │   ├── anomaly_detection.py     # label, split, contamination adjustment
│   │   ├── benchmark_runner.py      # evaluate_model, benchmark loops
│   │   ├── data_handler.py          # process_dataset (download → parquet)
│   │   ├── llm_runner.py            # LLMAnnotator + run_llm_active_loop()
│   │   └── setfit_runner.py         # SetFit splits, training, benchmark
│   └── utils/
│       └── visualization.py         # AUC curves, t-SNE, benchmark plots
├── .env                         # API keys (GROQ_API_KEY, etc.) — not committed
├── requirements.txt
└── README.md
```

---

## Setup & Running

### 1 · Install dependencies

```bash
pip install -r requirements.txt
```

### 2 · API key (Groq — free)

Create a free account at [console.groq.com](https://console.groq.com) and get an API key.  
Create a `.env` file in the project root:

```
GROQ_API_KEY=gsk_...
```

### 3 · Data: embeddings parquet files

The pipeline reads three parquet files per dataset from `data/`:

```
data/texts_{dataset}.parquet
data/labels_{dataset}.parquet
data/embeddings_{dataset}_distiluse-base-multilingual-cased-v2.parquet
```

**Option A — already have parquets (e.g. from Colab Drive):**  
Copy the parquet files to the `data/` folder in this project. File names must match the pattern above.

**Option B — generate from scratch:**

```bash
python scripts/run_data_preparation.py --project_path .
```

This downloads all 6 datasets and encodes them with `distiluse-base-multilingual-cased-v2`.  
Expected time: ~30–60 min per dataset (GPU not required).

### 4 · Run Experiment 3 — LLM-guided pipeline

**Local (PowerShell, CPU):**

```bash
# Single run
python scripts/run_llm_active_loop.py \
    --dataset told_br \
    --strategy score_guided \
    --n_llm_calls 50 \
    --device cpu

# Full experiment matrix
foreach ($dataset in @("told_br","tweets_hs","20_newsgroups")) {
  foreach ($strategy in @("random","score_guided")) {
    foreach ($n in @(50,100,200)) {
      foreach ($seed in @(42,0,1,2,3)) {
        python scripts/run_llm_active_loop.py `
          --dataset $dataset --strategy $strategy `
          --n_llm_calls $n --seed $seed --device cpu
      }
    }
  }
}
```

**Google Colab (notebook cell):**

```python
# Cell 1 — mount Drive and set up environment
from google.colab import drive
drive.mount('/content/drive')

import os, sys
PROJECT_PATH = "/content/drive/MyDrive/Projeto ML/2026/Master/Multilingual-Text-Anomaly-Detection"
DATA_DIR     = PROJECT_PATH + "/data"  # change if parquets live elsewhere, e.g.:
# DATA_DIR   = "/content/drive/MyDrive/Projeto ML/2025/AD/third_setup/data"
os.chdir(PROJECT_PATH)
sys.path.insert(0, os.path.join(PROJECT_PATH, "src"))

# Set API key (do not commit this)
os.environ["GROQ_API_KEY"] = "gsk_..."  # paste your Groq key here
```

```python
# Cell 2 — install dependencies (first run only)
!pip install -q -r requirements.txt
```

---

#### Option A — Groq (cloud, free tier)

Requires a free API key from [console.groq.com](https://console.groq.com). Limit: **100k tokens/day** (~3 datasets × N=50 per day).

```python
# Cell 3a — set Groq key (Cell 1 already done)
os.environ["GROQ_API_KEY"] = "gsk_..."  # paste your key
```

```python
# Cell 4a — single run (Groq)
!python scripts/run_llm_active_loop.py \
    --project_path "{PROJECT_PATH}" \
    --data_dir     "{DATA_DIR}" \
    --dataset told_br \
    --strategy score_guided \
    --n_llm_calls 50 \
    --device cuda
# default backend is groq, default model is llama-3.3-70b-versatile
```

---

#### Option B — Local model via llama-cpp-python (no API limits)

Runs a quantized open-source model (GGUF) entirely on the Colab GPU. No token limits, no API key needed.

**Recommended model:** `qwen2.5-7b` — multilingual (PT/EN/etc), ~4.7 GB VRAM on T4.

```python
# Cell 3b — install llama-cpp-python with CUDA (first run only, ~3 min)
!CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir -q
```

```python
# Cell 4b — download model to Drive (first run only, ~5 min, saved permanently)
MODEL_DIR = f"{PROJECT_PATH}/models"
!mkdir -p "{MODEL_DIR}"
!wget -q --show-progress \
    "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf" \
    -O "{MODEL_DIR}/qwen2.5-7b-instruct-q4_k_m.gguf"
```

```python
# Cell 5b — single run (local model)
!python scripts/run_llm_active_loop.py \
    --project_path "{PROJECT_PATH}" \
    --data_dir     "{DATA_DIR}" \
    --dataset told_br \
    --strategy score_guided \
    --n_llm_calls 50 \
    --backend llamacpp \
    --llamacpp_model qwen2.5-7b \
    --device cuda
```

**Available model tags** (auto-resolved to `{project_path}/models/`):

| `--llamacpp_model` | Model | VRAM |
|---|---|---|
| `qwen2.5-7b` *(default)* | Qwen2.5-7B-Instruct Q4_K_M | ~4.7 GB |
| `qwen2.5-3b` | Qwen2.5-3B-Instruct Q4_K_M | ~2.3 GB |
| `qwen2.5-1.5b` | Qwen2.5-1.5B-Instruct Q4_K_M | ~1.1 GB |
| `mistral-7b` | Mistral-7B-Instruct-v0.2 Q4_K_M | ~4.7 GB |

> You can also pass an explicit path: `--llamacpp_model /path/to/model.gguf`

---

#### Full experiment sweep (Python loop — works with both backends)

```python
# Cell — full sweep
import subprocess

datasets   = ["told_br", "tweets_hs", "20_newsgroups", "wikinews", "pt_tweets", "tweeteval"]
strategies = ["random", "score_guided"]

for dataset in datasets:
    for strategy in strategies:
        cmd = [
            "python", "-u", "scripts/run_llm_active_loop.py",
            "--project_path", PROJECT_PATH,
            "--data_dir",     DATA_DIR,
            "--dataset",      dataset,
            "--strategy",     strategy,
            "--n_llm_calls",  "50",
            "--seed",         "42",
            "--device",       "cuda",
            # --- local model (comment out to use Groq) ---
            "--backend",      "llamacpp",
            "--llamacpp_model", "qwen2.5-7b",
        ]
        print(f"\n>>> {dataset} | {strategy} | N=50", flush=True)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end="", flush=True)
        proc.wait()
```

> **Colab tip:** results are saved incrementally to `data/llm_results/` on Drive — if the runtime disconnects mid-run, already-completed runs are preserved (CSV append mode). Just re-run the loop; completed entries will be duplicated but are easy to deduplicate by `(dataset, strategy, n_llm_calls, seed)`.


Results are saved incrementally to `data/llm_results/{strategy}/N_{n}/{dataset}.csv`.

### 5 · Integration test (no API key needed)

```bash
python scripts/test_pipeline_integration.py --n_llm_calls 20
```

### Dataset name → parquet file mapping

| `--dataset` | `texts_*.parquet` | Notebook `dataset_short` |
|---|---|---|
| `told_br` | `texts_told_br.parquet` | `told-br` |
| `tweets_hs` | `texts_tweets_hs.parquet` | `tweets_hate_speech_detection` |
| `20_newsgroups` | `texts_20_newsgroups.parquet` | `20_newsgroups` |
| `wikinews` | `texts_wikinews.parquet` | `wikinews` |
| `pt_tweets` | `texts_pt_tweets.parquet` | `portuguese-tweets-for-sentiment-analysis` |
| `tweeteval` | `texts_tweeteval.parquet` | `tweet_eval` |

> **Note:** parquet files from the old Colab notebook (`Setfit_complete_pipeline_v0`) use the same format and encoder — they are directly compatible with `run_llm_active_loop.py`.

---

## Datasets

| # | Dataset | Source | Task | Lang | Normal class | Anomaly class | Original distribution | Used in experiments |
|---|---|---|---|---|---|---|---|---|
| 1 | TweetEval | `cardiffnlp/tweet_eval` (sentiment) | SA | EN | Neutral + Positive | Negative | Neg: 11377 · Neu: 21043 · Pos: 27479 | All Neutral+Positive kept · Negative downsampled to 5% |
| 2 | TOLD-Br | `JAugusto97/told-br` (binary) | HS | PT | Non-toxic (0) | Toxic/offensive (1) | Normal: 11745 · Toxic: 9255 | All normal kept · Toxic downsampled to 5% |
| 3 | Tweets HS | `tweets-hate-speech-detection` | HS | EN | No hate speech (0) | Hate speech (1) | Normal: 29720 · Hate: 2242 | All normal kept · Hate downsampled to 5% (already ~7%) |
| 4 | 20 Newsgroups | `SetFit/20_newsgroups` | TC | EN | comp.graphics (class 1) | All other 19 topics | ~balanced across 20 classes | comp.graphics kept · others downsampled to 5% |
| 5 | WikiNews | `wikinews_dataset` (filtered) | TC | PT | Politics (Política) | Other sections | Politics: 5829 · Others: 4752 | All politics kept · others downsampled to 5% |
| 6 | PT Tweets | `augustop/portuguese-tweets-for-sentiment-analysis` | SA | PT | Negative (0) | Positive (1) | Negative: 39911 · Positive: 20089 | All negative kept · positive downsampled to 5% |

TC = Topic Classification · SA = Sentiment Analysis · HS = Hate Speech

> **Preprocessing:** all experiments apply `label_normal_vs_anomaly` (most frequent class = normal) followed by `adjust_contamination` (anomalies downsampled to 5% of the training set). The original class distribution is shown for reference — the actual training data always has ~95% normal / 5% anomaly.
>
> **Anomaly criterion note:** for HS datasets the anomaly class is semantically distinct (hate speech). For SA datasets (TweetEval, PT Tweets) it is purely frequency-based — the LLM must be explicitly told the labeling criterion, as positivity/negativity is not inherently "anomalous".

---

## Citation

```bibtex
@inproceedings{Maia2025LearningFew,
  title={Learning with Few: A Comparative Study of Multilingual Text Anomaly Detection},
  author={Fabio Masaracchia Maia and Anna Helena Reali Costa},
  booktitle={Proceedings of STIL 2025},
  year={2025}
}
```

## Acknowledgments
This study was financed in part by the Coordenação de Aperfeiçoamento de Pessoal de Nível Superior – Brasil (CAPES) – Finance Code 001.  
The authors also thank the National Council for Scientific and Technological Development (CNPq), grant #312360/2023-1.
