# Fix & Re-run: 20 Newsgroups Prompt-Corpus Mismatch

## What Went Wrong

The `20_newsgroups` prompt in `src/pipeline/llm_runner.py` was inherited from a prior
experiment setup that treated **comp.graphics** as the anomaly class. In the current
experiment, the dataset is prepared by `label_normal_vs_anomaly()`, which automatically
picks the **most frequent class** as normal.

The actual label distribution of the SetFit/20_newsgroups dataset is:

| Label | Newsgroup              | Count |
|-------|------------------------|-------|
| 10    | rec.sport.hockey       | 600   ← **most frequent → normal (label 0)** |
| 15    | soc.religion.christian | 599   |
| 9     | rec.sport.baseball     | 597   |
| 7     | rec.autos              | 594   |
| ...   | (14 more groups)       | ...   |
| 18    | talk.politics.misc     | 465   |
| 0     | alt.atheism            | 480   |
| 19    | talk.religion.misc     | 377   |

**Normal class (label 0):** `rec.sport.hockey` (600 posts)  
**Anomaly class (label 1):** all other 19 newsgroups merged and subsampled to 5%

### The mismatch

The prompt said `comp.graphics = anomalous`, but comp.graphics is just **1 of 19**
anomaly classes. With N=200, you'd expect ~0.5 comp.graphics posts per sample → the
LLM flags ≈1 anomaly per run. SetFit never activates (threshold = 5). The reported
results for 20 Newsgroups are therefore:

- `MLP` and `MLP+SF` are **identical** (no SetFit fine-tuning ever happened)
- `DeepSAD` results also degrade from this because of near-zero positive labels
- The "21% gap recovery" figure is based on essentially **no annotation signal**

---

## Step 1 — Fix the Prompt in `llm_runner.py`

**File:** `src/pipeline/llm_runner.py`

Replace the `20_newsgroups` entry in the `TASK_CONTEXT` dict (around line 160):

```python
# BEFORE (wrong — targets comp.graphics which is only 1 of 19 anomaly classes)
"20_newsgroups": {
    "description": (
        "English newsgroup posts. "
        "Your task: decide if a post belongs to the comp.graphics newsgroup (anomalous, score 1.0) "
        "or to any other newsgroup topic (normal, score 0.0)."
    ),
    "normal_description": (
        "Posts about any topic other than computer graphics: "
        "sports, hockey, religion, politics, science, medicine, automobiles, space, electronics, "
        "history, philosophy, or any other non-graphics subject."
    ),
    "anomaly_criterion": (
        "Posts about computer graphics: image formats (GIF, JPEG, PNG), 3D rendering, raytracing, "
        "image processing, graphics software, display hardware, or related technical graphics topics. "
        "Score 1.0 if the post is about computer graphics. Score 0.0 for any other topic."
    ),
},

# AFTER (correct — normal = rec.sport.hockey, anomaly = everything else)
"20_newsgroups": {
    "description": (
        "English newsgroup posts. "
        "Your task: decide if a post belongs to the rec.sport.hockey newsgroup (normal, score 0.0) "
        "or to any other newsgroup topic (anomalous, score 1.0)."
    ),
    "normal_description": (
        "Posts about ice hockey: game scores, player trades, team standings, NHL news and commentary, "
        "hockey equipment, rules, strategy, playoff discussions, or any other ice hockey topic."
    ),
    "anomaly_criterion": (
        "Posts about any topic other than ice hockey: computers and software, science, politics, "
        "religion, automobiles, space, medicine, history, philosophy, sports other than hockey, "
        "or any non-hockey subject. Score 1.0 for any non-hockey post."
    ),
},
```

---

## Step 2 — Re-run LLM Annotation (v5, 20_newsgroups only)

The v5 labels are the source consumed by v6 (DeepSAD+SF), v7 (MLP), and v8 (MLP+SF).
All must be re-run, but only for `20_newsgroups`.

**v5 annotation runs** (on Colab T4, Qwen 7B and 14B):

```bash
PROJECT_PATH="/content/drive/MyDrive/.../Multilingual-Text-Anomaly-Detection"

for MODEL_PATH in \
    "/path/to/qwen2.5-7b-instruct-q4_k_m.gguf" \
    "/path/to/qwen2.5-14b-instruct-q4_k_m.gguf"; do
  for STRATEGY in random diversity; do
    for N in 50 200; do
      for SEED in 0 1 42; do
        python scripts/run_llm_active_loop.py \
          --project_path "$PROJECT_PATH" \
          --dataset 20_newsgroups \
          --backend llamacpp \
          --llamacpp_model_path "$MODEL_PATH" \
          --strategy "$STRATEGY" \
          --n_llm_calls $N \
          --seed $SEED \
          --results_dir data/llm_results/v5
      done
    done
  done
done
```

Total v5 runs for 20_newsgroups: 2 models × 2 strategies × 2 N × 3 seeds = **24 runs**

> **Note:** The existing v5 results for the other 3 datasets (tweets_hs, hatebr, wikinews)
> do NOT need to be re-run — only overwrite/replace the 20_newsgroups files.

---

## Step 3 — Re-run AD Models (v6, v7, v8, 20_newsgroups only)

After v5 produces new labels, re-run the downstream AD experiments.

Check the README files in `data/llm_results/v6/`, `v7/`, `v8/` for the exact
script invocations used in the original runs, and repeat them filtering to
`--dataset 20_newsgroups`.

Typical pattern (see `scripts/run_llm_active_loop.py` with appropriate `--ad_model` flag):

```bash
# v6 = DeepSAD + SetFit (reads v5 labels)
# v7 = MLP, no SetFit  (reads v5 labels)
# v8 = MLP + SetFit    (reads v5 labels)
# Check each version's README for exact invocation
```

---

## Step 4 — Update the Paper (`main.tex`)

After collecting new results for 20 Newsgroups, update these locations:

### 4a. Table 2 (tab:main-results) — four cells to update

| Row | Column | Old value | Expected direction |
|-----|--------|-----------|--------------------|
| DeepSAD (Qwen 7B)       | 20 News | 0.881 ± .073 | likely ↑ (more positive labels now) |
| DeepSAD (Qwen 14B)      | 20 News | 0.878 ± .089 | likely ↑ |
| MLP (Qwen 7B)           | 20 News | 0.936 ± .008 | may ↑↓ (std likely ↑ too) |
| MLP (Qwen 14B)†         | 20 News | 0.936 ± .007 | may ↑↓ |

Also update **Best unsup**, **Best (GT)**, and **Gap↑** for 20 News if they change
(they shouldn't — those are benchmark values, not LLM-dependent).

### 4b. Tab:setfit-effect — row "20 Newsgroups"

Currently `Ran/6 = 0, Δ AUC = ---` for both models.
With the correct prompt, SetFit may now activate (≥5 anomalies flagged).
Update Ran/6 and Δ AUC accordingly.

### 4c. Tab:prompts (Appendix A.2)

Update the 20 Newsgroups prompt fields to match the corrected prompt:

| Field | New content |
|-------|------------|
| $d_{\text{task}}$ | English newsgroup posts. Decide if a post belongs to the rec.sport.hockey newsgroup (normal, 0.0) or to any other newsgroup topic (anomalous, 1.0). |
| $c_{\text{normal}}$ | Posts about ice hockey: game scores, player trades, team standings, NHL news, hockey equipment, rules, strategy, or any ice hockey topic. |
| $c_{\text{anomaly}}$ | Posts about any topic other than ice hockey: computers, science, politics, religion, automobiles, space, medicine, sports other than hockey, etc. |

### 4d. Discussion paragraphs

- **"Task misalignment" paragraph (§4.2):** Remove the explanation about the prompt-corpus
  mismatch (that was a bug, not a finding). If SetFit now activates on 20 Newsgroups,
  update accordingly. If it still doesn't activate, find the real reason (e.g., hockey
  posts are highly separable so there's nothing close to the boundary, or the LLM still
  under-flags anomalies).

- **"Unsupervised ceiling" paragraph (§4.1):** Currently says the encoder separates
  hockey from the anomaly pool well. This likely remains true; just verify the AUC values.

- **"LLM capacity" paragraph:** "both sizes saturate at the same level" — verify if
  this still holds with correct labels.

- **Introduction/Conclusion gap recovery stats:** `21%` gap recovery for 20 Newsgroups
  and `19--77×` label disadvantage figures will need updating if results change.

---

## Step 5 — Sanity Checks Before Finalising

After re-running, verify:

1. **LLM now flags >1 anomaly per run** for 20_newsgroups (expected: many non-hockey
   posts should score 1.0, since the LLM can easily tell "this is not a hockey post").

2. **SetFit activation rate** — at N=200 at least some runs should cross the threshold
   of 5. Check `Ran/6` for both 7B and 14B.

3. **AUC stability** — the corrected pipeline should produce lower variance on 20 News
   (currently std = 0.007--0.089 depending on model; with more labels the variance
   should decrease).

4. **Benchmark baseline unchanged** — the unsupervised results (0.920 Best unsup) and
   ground-truth upper bound (0.998) for 20 News are encoder-based and independent of
   the LLM prompt; they should not change.

---

## Summary of Files to Modify

| File | Change |
|------|--------|
| `src/pipeline/llm_runner.py` | Fix `20_newsgroups` prompt (Step 1) |
| `data/llm_results/v5/random/N_50/20_newsgroups_*.csv` | Overwrite with new LLM labels |
| `data/llm_results/v5/random/N_200/20_newsgroups_*.csv` | Overwrite with new LLM labels |
| `data/llm_results/v5/diversity/N_50/20_newsgroups_*.csv` | Overwrite with new LLM labels |
| `data/llm_results/v5/diversity/N_200/20_newsgroups_*.csv` | Overwrite with new LLM labels |
| `data/llm_results/v6/.../20_newsgroups.csv` | Overwrite with new AD results |
| `data/llm_results/v7/.../20_newsgroups.csv` | Overwrite with new AD results |
| `data/llm_results/v8/.../20_newsgroups.csv` | Overwrite with new AD results |
| `puplication/STIL2026-.../main.tex` | Update tables and text (Step 4) |
