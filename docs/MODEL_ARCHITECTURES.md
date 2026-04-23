# Model Architectures

Details of the AD models used in the LLM pipeline experiments (v5–v8).

---

## DeepSAD (semi-supervised, via DeepOD)

Instantiated as `DeepSAD(epochs=100, rep_dim=128, device='cuda')`.

Internal network (`MLPnet`) — 3 linear blocks, **no bias**:

| Layer | Input → Output | Activation |
|---|---|---|
| LinearBlock 0 | `input_dim` → 100 | ReLU |
| LinearBlock 1 | 100 → 50 | ReLU |
| LinearBlock 2 | 50 → 1 | Identity |

- `input_dim` = 512 (distiluse) · 768 (SetFit/MiniLM) · 1024 (XLM-RoBERTa)
- **Training objective**: hypersphere minimization (SVDD loss); known anomalies are pushed outside the sphere
- Known normals passed as label `0`, known anomalies as label `1`, unlabeled as label `-1`
- 100 epochs, `rep_dim=128`

---

## MLP (supervised binary classifier, custom)

Source: [src/models/MLP.py](../src/models/MLP.py)  
Instantiated with defaults: `hidden_dims=(128, 64), lr=1e-3, epochs=50, batch_size=256`.

| Layer | Output dim | Activation |
|---|---|---|
| Linear | 128 | ReLU |
| Linear | 64 | ReLU |
| Linear | 1 | Sigmoid |

- **Loss**: `BCELoss` · **Optimizer**: `Adam(lr=1e-3)`
- `decision_function` returns the sigmoid output (probability of anomaly)
- Drop-in replacement for DeepSAD: same `fit(X, y)` / `decision_function(X)` API

---

## DeepSVDD (unsupervised, via DeepOD)

Instantiated as `DeepSVDD(epochs=100, rep_dim=128)`.

Same `MLPnet` as DeepSAD but used **without labels** — only normal samples define the hypersphere.

| Layer | Input → Output | Activation |
|---|---|---|
| LinearBlock 0 | `input_dim` → 100 | ReLU |
| LinearBlock 1 | 100 → 50 | ReLU |
| LinearBlock 2 | 50 → 128 (`rep_dim`) | Identity |

Used in the v5 pipeline (Step 2) to produce anomaly scores for score-guided sample selection.

---

## Embedding dimensions per encoder

| Encoder | Short name | dim |
|---|---|---|
| distiluse-base-multilingual-cased-v2 | distiluse-v2 | 512 |
| paraphrase-multilingual-MiniLM-L12-v2 (SetFit) | MiniLM | 384 |
| xlm-roberta-large | XLM-RoBERTa | 1024 |
