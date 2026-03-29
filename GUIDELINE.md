# LLM-Guided Anomalous Text Detection — Research Guideline

## Overview

This project investigates whether **large language models can replace human annotators** in semi-supervised text anomaly detection — enabling good detection performance with zero human labeling effort.

The approach combines:
- Pre-trained sentence embeddings (fixed: `distiluse-base-multilingual-cased-v2`)
- Classical anomaly detection models (unsupervised + semi-supervised)
- **LLM as annotator** — queries an LLM to label selected samples, replacing the human

> **Reference implementations that can used:**
> - `src/codigos_exemplo/AD-LLM/` — zero-shot LLM detection + structured JSON output (Yang et al., 2025, ACL)
> - `src/codigos_exemplo/LLM-OOD/` — LLM fine-tuning (LoRA) for OOD embedding extraction

---

## Project Evolution

Three experiments build on each other:

| # | Experiment | Status | Key Finding |
|---|---|---|---|
| 1 | **STIL 2025 Benchmark** | Done | Random human labels dramatically improve AD; TOLD-Br is the hardest dataset (AUC ~0.45 unsup) |
| 2 | **SetFit Fine-tuning** | Done | Contrastive fine-tuning has diminishing returns; bottleneck is label *informativeness*, not quantity |
| 3 | **LLM Annotation Loop** | Next | Replace human with LLM; test random vs score-guided sample selection |

Both prior experiments confirm: **unsupervised alone is never sufficient** (best: 20ng ~0.80, worst: TOLD-Br ~0.45). The next step is replacing the human annotator entirely.

---

## Core Research Question

> **"Can we achieve satisfactory anomaly detection without any human labels, using an LLM as the annotator?"**

A secondary question:
> **"Does intelligent (score-guided) sample selection amplify the value of LLM annotation compared to random selection?"**

---

## Proposed Pipeline (Experiment 3)

```text
INPUT: pre-computed embeddings (distiluse-v2, fixed)
       raw texts (train set, no labels used during training)
       locked test set (ground-truth labels for evaluation only)

Step 1 — Unsupervised AD — DeepSVDD
     |  trains on ALL train embeddings (no labels)
     |  produces an anomaly score for each training sample

Step 2 — Sample Selection  [parameter: --n_llm_calls N]
     |  N = budget of LLM annotation calls (e.g. 50, 100, 200)
     +-- Strategy A: Random       — pick N samples at random
     +-- Strategy B: Score-guided — pick the N samples with the
                                    highest DeepSVDD anomaly scores

Step 3 — LLM Annotation
     |  each of the N selected texts is sent to the LLM
     |  LLM receives: text + dataset-specific anomaly criterion
     |  LLM returns: {"anomaly_score": 0.0–1.0, "reason": "..."}
     |  threshold (>= 0.6) converts to binary label: 1=anomaly, 0=normal

Step 4 — SetFit Fine-tuning  [on the N LLM-labeled samples]
     |  contrastive training on LLM-labeled pairs (balanced)
     |  produces a fine-tuned sentence encoder
     |  re-encodes the entire train + test set with the new encoder
     |  (skipped if < 3 samples per class → original embeddings kept)

Step 5 — Semi-supervised AD — DeepSAD
     |  trains on SetFit embeddings
     |  uses all N valid LLM labels as supervision signal
     |  produces anomaly scores for the test set

Step 6 — Evaluate (test set, ground truth)
     AUC-ROC, Average Precision
     Compared against: unsupervised baseline (Step 1)
     Reference ceiling: STIL 2025 human-labeled results
```

**How `n_llm_calls` affects the result:**  
More N → more LLM labels → SetFit and DeepSAD have more signal.  
With `score_guided`, those N samples are already enriched for anomalies, so fewer calls are needed to get enough anomaly labels.  
With `random`, N must be larger to guarantee enough anomaly examples (rare class at ~5%).

---

## Task Definition for LLM Annotation

This is the most sensitive design decision in Experiment 3. **The LLM cannot infer what "anomaly" means without explicit context** — the concept is dataset-specific and would be ambiguous or wrong if left to the model's generic priors.

### What the LLM needs to know (per dataset)

| Dataset | Normal class | Anomaly class | Key challenge for LLM |
|---|---|---|---|
| TOLD-Br | Non-hate speech | Hate speech (PT) | Implicit hate, irony, slang in Brazilian Portuguese |
| Tweets HS | Non-hate speech | Hate speech (EN) | Implicit hate, coded language |
| PT Tweets | Negative sentiment | Positive sentiment (PT) | **Anomaly is frequency-based, not semantic** — positive tweets are the minority; LLM has no natural reason to flag positivity as anomalous |
| TweetEval | Neutral/Positive | Negative sentiment (EN) | Sarcasm, understated negativity |
| 20 Newsgroups | comp.graphics posts | Posts from other topics | Off-topic detection, not quality |
| WikiNews | Politics news (PT) | News from other sections | Topic boundary, not quality |

> **Critical distinction — semantic vs. frequency-based anomaly:**
> In HS datasets (TOLD-Br, Tweets HS), the anomaly criterion is *both* frequency-based (hate is minority) and semantically meaningful (hate is distinct). The LLM's semantic judgment naturally aligns with the label.
>
> In PT Tweets, the anomaly is *only* frequency-based — positive sentiment is the minority class, but it is not semantically "wrong" or unusual. The LLM must be explicitly told to treat positive sentiment as anomalous, which goes against its priors. This makes PT Tweets a stress test for task definition quality.

### Prompt design principles

Following **AD-LLM Setting 2** (Yang et al., 2025), each prompt must include:

1. **Task description** — what the dataset is about and what the normal class represents
2. **Anomaly criterion** — the single most critical component: an explicit, unambiguous definition of what counts as anomalous *in this dataset*, independent of any general notion of "bad" or "unusual" text
3. **Output format** — structured JSON: `{"anomaly_score": float, "reason": str}`

> **The anomaly criterion is not optional.** Without it, the LLM falls back on generic priors (toxicity, low quality, off-topic content) that may be completely misaligned with the dataset's labeling criterion. This is especially dangerous for frequency-based anomalies like PT Tweets, where the LLM must suppress its natural judgment and follow a counter-intuitive criterion.

### Per-dataset anomaly criterion (to be included verbatim in prompts)

| Dataset | Anomaly criterion for prompt |
|---|---|
| TOLD-Br | "Texts that contain hate speech, discrimination, or offensive language targeting individuals or groups, including implicit, ironic, or coded forms common in Brazilian Portuguese." |
| Tweets HS | "Texts that contain hate speech or discriminatory language targeting individuals or groups based on identity characteristics." |
| PT Tweets | "Texts that express **positive sentiment** — this is the minority class in this dataset and is treated as anomalous regardless of whether positivity is natural or expected." |
| TweetEval | "Texts that express **negative sentiment**, including sarcasm and implicit negativity." |
| 20 Newsgroups | "Texts that belong to **any topic other than computer graphics** (comp.graphics). The anomaly is purely topical — quality, style, or language are irrelevant." |
| WikiNews | "Texts that belong to **any news section other than politics**. The anomaly is topical — any non-politics article is anomalous regardless of quality." |

Example prompt structure:
```
You are evaluating text samples from a [DATASET DESCRIPTION].

The anomaly criterion for this task is strictly defined as:
[ANOMALY CRITERION — use the exact definition above]

Rate the following text on a scale from 0.0 (clearly normal / not anomalous)
to 1.0 (clearly anomalous according to the criterion above).

Text: "[SAMPLE]"

Respond only in JSON: {"anomaly_score": <float>, "reason": "<brief explanation>"}
```

> **Note:** the `reason` field serves as a quality check — if the LLM's reasoning does not reference the correct criterion, the score is likely unreliable regardless of its value.

### Using `anomaly_score` vs binary label

The LLM returns a continuous score — this is intentional. Rather than forcing a binary choice (which increases noise for borderline cases), we:
- Use threshold `>= 0.6` as default for "anomaly"
- Report the score distribution to detect systematic LLM bias
- Consider soft-label variants as a sensitivity analysis

---

## Label Management

This is a key design difference from the SetFit experiment.

**In SetFit (Experiment 2):** labels were balanced by construction — exactly k normal + k anomaly samples, selected from a pre-labeled pool. The same k×2 samples served both SetFit fine-tuning and AD supervision, with a strict test set locked from the start.

**In the LLM experiment (Experiment 3):** the LLM produces labels, but with important caveats:

| Issue | Description | Mitigation |
|---|---|---|
| **Label imbalance** | Score-guided selection biases toward anomalies (high-score samples); random is more balanced | Report actual label distribution per run |
| **Label noise** | LLM may mislabel ambiguous samples (especially hate speech with irony/slang) | Use `anomaly_score` threshold (e.g. >= 0.6 = anomaly) rather than binary forced choice |
| **No guarantee of both classes** | With small N, LLM may return all-normal or all-anomaly labels | Require minimum 1 sample per class; skip run if not satisfied |
| **Data leakage** | Same test set used for final evaluation; LLM never sees test samples | Test split fixed before any LLM call, identical to STIL/SetFit protocol |

**Label budget comparison across experiments:**

| Experiment | Labels | Source | Balance |
|---|---|---|---|
| STIL 2025 | ~5% of train set | Human (ground truth) | Natural (imbalanced) |
| SetFit | 2×k (k=20 or 40) | Human (ground truth) | Forced balanced |
| LLM (Exp. 3) | N (50–200) | LLM | Random: ~natural; Score-guided: anomaly-biased |

> **Implication for interpretation:** if score-guided outperforms random, part of the gain may come from a higher anomaly ratio in the labeled set (not just better sample selection). This should be reported and discussed.

---

## Experimental Design

### Conditions (per dataset)

| Condition | Labels | Cost |
|---|---|---|
| Unsupervised baseline | 0 | Free |
| LLM random (N = 50, 100, 200) | N LLM calls | Low |
| LLM score-guided (N = 50, 100, 200) | N LLM calls | Low |
| Human labels ceiling | N human labels | High (from STIL) |

### Datasets — 3-tier priority

| Tier | Datasets | Unsup AUC | Rationale |
|---|---|---|---|
| Primary | TOLD-Br (PT/HS), Tweets HS (EN/HS) | ~0.45–0.49 | Unsup fails — LLM has most value; semantically complex |
| Secondary | PT Tweets (PT/SA), TweetEval (EN/SA) | ~0.48–0.57 | Moderate difficulty; validates pipeline |
| Control | 20 Newsgroups (EN/TC), WikiNews (PT/TC) | ~0.72–0.80 | Unsup already reasonable; confirms robustness |

> Control datasets are critical: if LLM labels help 20ng/WikiNews (easy) **and** TOLD-Br (hard), the pipeline is robust. If LLM only helps the easy ones, the hard task has a structural ceiling.

### Fixed parameters
- Encoder: `distiluse-base-multilingual-cased-v2`
- AD models: **MLP** (most stable) + **DevNet** (most sensitive to label quality)
- N per condition: 50, 100, 200 LLM calls
- Seeds: 5 per condition
- LLM: **Qwen 2.5 7B Instruct** (see model selection rationale below)

### LLM annotator model selection

The LLM annotator is a **fixed parameter** across all experiments — model choice is a design decision, not a variable of interest. The selection criterion was: best multilingual performance in the ~7B parameter class (VRAM-constrained to ≤ 5 GB for local inference on Colab T4).

| Model | Multilingual coverage | Portuguese quality | JSON instruction-following | VRAM (Q4_K_M) | Notes |
|---|---|---|---|---|---|
| **Qwen 2.5 7B Instruct** ✅ | 29 languages incl. PT | ★★★★ | ★★★★ | ~4.7 GB | Best multilingual 7B on Open LLM Leaderboard (2024) |
| Llama 3.1 8B Instruct | EN/DE/FR/IT/PT/HI/ES/TH | ★★★☆ | ★★★☆ | ~5.0 GB | Official PT support; weaker than Qwen on non-EN benchmarks |
| Mistral 7B Instruct v0.2 | EN-focused | ★★☆☆ | ★★★☆ | ~4.7 GB | No explicit multilingual training; poor PT performance |

**Rationale:** Qwen 2.5 7B was selected because it demonstrates the strongest performance on multilingual benchmarks (including Portuguese) among open-source models in the 7B parameter class, as evaluated by the HuggingFace Open LLM Leaderboard. Its training includes explicit multilingual data across 29 languages, with Portuguese represented. Llama 3.1 8B officially supports Portuguese but ranks lower on multilingual classification tasks in this parameter range. Mistral 7B was excluded due to predominantly English training.

> **Citation:** HuggingFace Open LLM Leaderboard. Available at: https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard. Accessed March 2026.

Comparing LLM model variants (1.5B vs 3B vs 7B vs 70B) is outside the scope of this work and is noted as future work.


### Success criteria
- **Positive:** LLM-labeled semi-supervised > unsupervised baseline on TOLD-Br and/or Tweets HS
- **Stronger:** score-guided > random selection with same N
- **Negative (also publishable):** LLM does not improve — confirms structural limits of frequency-based anomaly criterion for hate speech

---

## Key Hypotheses

1. LLM labels elevate semi-supervised AD above unsupervised — **with zero human effort**
2. Score-guided selection is more cost-efficient: same AUC with fewer LLM calls
3. Gain is larger for semantically complex tasks (HS > SA > TC)
4. For easy tasks (20ng, WikiNews), both strategies converge — pipeline is adaptive, not artificially inflated

---

## Validation Strategy

**Split:** 80% train / 20% test, fixed seed per run. No validation set.

**Rationale:** no hyperparameter is tuned using ground-truth labels during training. `anomaly_threshold=0.6` is fixed a priori. Adding a val set would (a) reduce train further at 5% contamination, and (b) create implicit leakage if used to tune threshold per dataset.

**Multiple seeds (5 per condition)** serve as a pseudo-CV — mean ± std AUC per condition is the reported metric.

**SetFit balancing:** the LLM labels passed to SetFit are balanced by subsampling the majority class to `min(n_anomalies, n_normals)` — matching the Exp 2 protocol. DeepSAD receives all valid LLM labels (imbalanced is fine for SAD loss). If either class has < 3 samples, SetFit is skipped and original embeddings are used — this is logged as `setfit_skipped=True` in the results CSV.

**Loop type:** one-shot (not incremental). DeepSVDD scores are computed once on the full training set; samples are selected in a single batch. Incremental refinement is noted as future work.

**Known risks:**
- LLM mislabels ambiguous samples (hate speech with irony/slang in PT) — mitigated by using continuous `anomaly_score` with threshold rather than forced binary
- Score-guided selection creates anomaly-biased label sets — reported explicitly via `n_anomalies_found` column
- Small N with random strategy may not yield enough anomaly labels for SetFit (`setfit_skipped` flag catches this)
- Error propagation: noisy LLM labels → noisy SetFit embeddings → degraded DeepSAD — this is the central empirical question

---

## Codebase Status

### Scripts

| Script | Status | Description |
|---|---|---|
| `scripts/run_data_preparation.py` | Done | Encode all datasets -> save parquets |
| `scripts/run_benchmark.py` | Done | Unsupervised + semi-supervised benchmark |
| `scripts/run_setfit_experiment.py` | Done | SetFit fine-tuning + AD benchmark |
| `scripts/run_llm_active_loop.py` | Next | LLM annotation pipeline (Experiment 3) |

### Source modules

```
src/
+-- encoders/
|   +-- text_encoders.py          # SentenceBERT, BERTimbau wrappers
+-- models/
|   +-- MLP.py                    # MLP (sklearn) + MLPTF (TF/Keras)
+-- pipeline/
|   +-- anomaly_detection.py      # label_normal_vs_anomaly, split_data, adjust_contamination
|   +-- benchmark_runner.py       # evaluate_model, benchmark_unsupervised/semisupervised
|   +-- setfit_runner.py          # SetFit splits, training, benchmark (Experiment 2)
|   +-- data_handler.py           # Dataset loading helpers
+-- utils/
    +-- visualization.py          # AUC curves, t-SNE, benchmark plots
```

### Documentation

```
GUIDELINE.md                      # <- this file
docs/
+-- BENCHMARK_RESULTS.md          # STIL 2025 full results + analysis
+-- SYNTHESIS.md                  # Cross-experiment synthesis
+-- SETFIT_RESULTS.md             # SetFit pipeline + k=20/k=40 results
```

---

## Next Steps

- [x] **`scripts/run_llm_active_loop.py`** — implemented. See `src/pipeline/llm_runner.py`.
  - Backends: Gemini 2.0 Flash (default), OpenAI gpt-4o-mini, local GGUF via llama-cpp-python
  - For local CPU dev: `Qwen2.5-1.5B-Instruct-Q4_K_M.gguf` (~1GB, ~2 tok/s on CPU)
- [ ] Run experiments:
  1. Load embeddings + texts from parquet
  2. Run unsupervised model -> get anomaly scores
  3. Select N samples (random OR score-guided)
  4. Query LLM -> get `anomaly_score` + `reason` per sample
  5. Threshold -> binary labels
  6. Train MLP / DevNet with LLM labels
  7. Evaluate on test set -> save CSV with `selection_strategy` column
- [ ] Run experiments: N in {50, 100, 200} x strategy in {random, score-guided} x 6 datasets
- [ ] Compare: unsupervised baseline vs LLM-random vs LLM-score-guided vs human ceiling
- [ ] Analyze cost-efficiency curve: N LLM calls -> AUC gain

---

## References

- Maia & Costa (2024) — Semi-supervised anomaly detection in Portuguese
- Maia & Costa (2025, STIL) — *Learning with Few: A Comparative Study of Multilingual Text Anomaly Detection*
- Yang et al. (2025, ACL) — AD-LLM: zero-shot LLM anomaly detection with structured output
- Pang et al. (2021) — Deep anomaly detection survey


