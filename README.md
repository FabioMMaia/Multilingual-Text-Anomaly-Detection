# Multilingual Text Anomaly Detection

Research repository investigating text anomaly detection in low-supervision multilingual scenarios — from classical benchmarks to LLM-guided annotation.

> **Current direction:** See [GUIDELINE.md](GUIDELINE.md)

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
PROJECT_PATH = "/content/drive/MyDrive/Projeto ML/2026/Master/Code/Multilingual-Text-Anomaly-Detection"
os.chdir(PROJECT_PATH)
sys.path.insert(0, os.path.join(PROJECT_PATH, "src"))

# Set API key (do not commit this)
os.environ["GROQ_API_KEY"] = "gsk_..."  # paste your Groq key here
```

```python
# Cell 2 — install dependencies (first run only)
!pip install -q -r requirements.txt
```

```python
# Cell 3 — single run
!python scripts/run_llm_active_loop.py \
    --project_path "{PROJECT_PATH}" \
    --dataset told_br \
    --strategy score_guided \
    --n_llm_calls 50 \
    --device cuda
```

```python
# Cell 4 — full experiment matrix (bash loop)
for dataset in told_br tweets_hs 20_newsgroups wikinews pt_tweets tweeteval; do
  for strategy in random score_guided; do
    for n in 50 100 200; do
      for seed in 42 0 1 2 3; do
        python scripts/run_llm_active_loop.py \
          --project_path "{PROJECT_PATH}" \
          --dataset $dataset --strategy $strategy \
          --n_llm_calls $n --seed $seed --device cuda
      done
    done
  done
done
```

> **Colab tip:** results are saved incrementally to `data/llm_results/` on Drive — if the runtime disconnects mid-run, already-completed seeds are preserved and the loop can be restarted from where it left off (CSV is in append mode).
```

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
