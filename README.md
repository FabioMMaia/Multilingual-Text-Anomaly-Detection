# Learning with Few: A Comparative Study of Multilingual Text Anomaly Detection

This repository contains the code, processed data, and scripts used in the study **"Learning with Few: A Comparative Study of Multilingual Text Anomaly Detection"**, which investigates how different models, representation strategies, and supervision levels impact anomaly detection in text, with a focus on multilingual and few-shot scenarios.

## Study Overview
Anomaly detection in textual data is critical for applications such as content moderation, fraud detection, and risk monitoring.  
However, the semantic complexity of language and the scarcity of labeled anomalies make this task challenging.  

This study presents a **comprehensive benchmark** that:
- Evaluates **unsupervised** and **semi-supervised** models (including deep learning approaches).
- Compares **multilingual** and **Portuguese-specific** sentence embeddings.
- Uses **six datasets** (three in Portuguese and three in English) spanning topic classification, sentiment analysis, and hate speech detection.
- Examines the effect of **low supervision** (1–5% labeled anomalies) on model performance.

**Key findings:**
- Even a small amount of labeled anomalies can significantly boost performance.
- Models such as **DevNet**, **XGBOD**, and **DeepSAD** showed the most robustness.
- Representation choice has a notable impact, varying with language and task complexity.

---

## Repository Structure

```text
.
├── data/ # Text, embeddings, and labels
├── experiments_results/  # Metrics and logs from experiments
├── notebooks/             # Jupyter notebooks for running the workflow
│   ├── data_preparation.ipynb      # Download datasets and generate embeddings
│   └── experiments_benchmarks.ipynb # Run anomaly detection benchmarksExperiment runners
├── src/
│   ├── encoders/
│   ├── models/
│   ├── pipeline/
│   └── utils/
├── .gitignore
├── README.md
└── requirements.txt

```

## Models Evaluated

**Unsupervised**  
- OCSVM  
- IForest  
- LOF  
- HBOS  
- KDE  
- AutoEncoder (AE)  
- Variational AutoEncoder (VAE)  
- DeepSVDD  

**Semi-supervised**  
- DeepSAD  
- DevNet  
- XGBOD  
- MLP (feed-forward)  

---

## Embeddings Models

| Short Name    | Model Name                                      | Type            | Language     | Dim |
|---------------|-------------------------------------------------|----------------|--------------|-----|
| distiluse-v1  | distiluse-base-multilingual-cased-v1             | Sentence-BERT  | Multilingual | 512 |
| distiluse-v2  | distiluse-base-multilingual-cased-v2             | Sentence-BERT  | Multilingual | 512 |
| XLM-RoBERTa   | xlm-roberta-large                               | Transformer    | Multilingual | 1024|
| BERT-base-PT  | bert-base-portuguese-cased                      | Transformer    | Portuguese   | 768 |
| BERT-large-PT | bert-large-portuguese-cased                     | Transformer    | Portuguese   | 1024|
| Serafim       | serafim-100m-portuguese-pt                      | Transformer    | Portuguese   | 768 |

---

## Datasets

| Dataset               | Task | Lang | Normal Class     | Anomaly Class |
|-----------------------|------|------|------------------|---------------|
| WikiNews (pt)         | TC   | pt   | Politics         | Others        |
| 20 Newsgroups         | TC   | en   | comp.graphics    | Others        |
| Portuguese Tweets     | SA   | pt   | Negative         | Positive      |
| TweetEval             | SA   | en   | Neutral/Positive | Negative      |
| TOLD-Br               | HS   | pt   | Non-hate         | Hate          |
| Hate Speech Tweets    | HS   | en   | Non-hate         | Hate          |

**Legend:**  
- TC = Topic Classification  
- SA = Sentiment Analysis  
- HS = Hate Speech Detection  

## Citation
If you use this repository, please cite:
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
