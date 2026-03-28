# SetFit Fine-Tuning for Anomaly Detection

Fine-tuning do encoder `distiluse-base-multilingual-cased-v2` via **SetFit** usando
`k` amostras por classe (normal + anomalia). Os embeddings resultantes substituem
os originais como entrada para os modelos semi-supervisionados.

**Código:** [`src/pipeline/setfit_runner.py`](../../src/pipeline/setfit_runner.py)  
**Script de execução:** [`scripts/run_setfit_experiment.py`](../../scripts/run_setfit_experiment.py)

---

## 1. Pipeline

### 1.1 Splits de dados

```
Dataset completo (N amostras)
│
│  prepare_data_splits_for_anomaly_detection()
│  train_test_split(test_size=0.2, stratified)
│
├───────────────────────────────────────────────────────────┐
│                                                           │
▼                                                           ▼
texts_train / embs_train / labels_train            ┌──── test_set ─────┐
           (80%×N)                                 │ texts, embeddings │
│                                                  │ labels            │
│  few-shot extraction: k amostras/classe          │                   │
│                                                  │ TRANCADO.         │
├─────────────────────────┬─────────────────────   │ Nunca entra em    │
│                         │                        │ nenhum treino.    │
▼                         ▼                        └───────────────────┘
Few-shot (2×k)        remaining_downstream
│                     texts, embeddings, labels
│  split(0.3)         (80%×N − 2×k)
├─────────┐           [disponível mas não usado
▼         ▼            no benchmark padrão]
setfit_data
  ["train_dataset"]   ← SetFitTrainer.train()
  ["eval_dataset"]    ← monitora loss (não influencia AD)
          │
          ▼
    train_and_evaluate_setfit_model()
          │
          ▼
    trainer  (encoder fine-tunado)
          │
          ├─► encode_with_setfit(trainer, downstream_train, test_set)
          │
          │   downstream_train["embeddings_sf"]  ← re-encoded com SetFit
          │   test_set["embeddings_sf"]           ← re-encoded com SetFit


                 downstream_train
          ┌──────────────────────────────────────────────┐
          │  texts          : list of strings            │
          │  labels         : list (0=normal, 1=anomaly) │
          │  embeddings     : Original (distiluse)       │
          │  embeddings_sf  : SetFit fine-tunado         │
          │                                              │
          │  = pool inteiro (80%×N),                     │
          │    INCLUINDO os 2×k few-shot                 │
          └──────────────────────────────────────────────┘
```

> **Por que os k samples entram no `downstream_train`?**  
> Os k labels representam o conhecimento disponível (do LLM ou de humanos).
> Faz sentido usá-los tanto para melhorar o encoder (SetFit) quanto como
> sinal de supervisão para o modelo de AD. O `remaining_downstream` existe
> para análises onde se quer excluir esse overlap.

### 1.2 Loop semi-supervisionado

```
══════════  LOOP SEMI-SUPERVISIONADO  ══════════
           run_semi_supervised_benchmark()

Para cada modelo  in {DevNet, DeepSAD, MLP}:
Para cada embedding in {Original, SetFit}:
  X_train = downstream_train["embeddings"]     (Original)
          | downstream_train["embeddings_sf"]  (SetFit)
  X_test  = test_set["embeddings"]
          | test_set["embeddings_sf"]
  y_train = downstream_train["labels"]   ← verdadeiro, mas NÃO passado ao modelo

  Para cada n in n_known_list = [5, 10, 20, 50, 100, 200, 500]:
  │
  │  Sorteia n índices aleatórios entre as anomalias de y_train
  │
  │  y_semi                    = zeros(len(downstream_train))
  │  y_semi[n_índices]         = 1   ← única supervisão visível ao modelo
  │
  │  clf.fit(X_train, y_semi)        ← vê o pool inteiro, labels parciais
  │
  │  scores = clf.decision_function(X_test)  ← nunca viu X_test antes
  │
  └─► roc_auc(y_test, scores)
      pr_auc (y_test, scores)
            │
            ▼
      results_df  — uma linha por (modelo, embedding, n_known)
      salvo em data/setfit_results/N_{k}/{dataset}.csv
```

### 1.3 Variáveis de saída de `prepare_data_splits_for_anomaly_detection`

| Variável | Conteúdo | Usado em |
|---|---|---|
| `downstream_train` | Pool de treino completo (inclui os k few-shot) | `clf.fit()` nos modelos de AD |
| `test_set` | 20% isolado no início | Avaliação final (roc_auc, pr_auc) |
| `setfit_data` | `train_dataset` + `eval_dataset` dos k few-shot | `SetFitTrainer.train()` |
| `remaining_downstream` | Pool de treino sem os k few-shot | Análises exploratórias |

### 1.4 Números concretos (k=20, N=5000, contaminação=5%)

```
N = 5 000  (95% normal = 4 750, 5% anomalia = 250)

Test set         → 1 000 amostras  (950 normais  +  50 anomalias)
Training pool    → 4 000 amostras  (3 800 normais + 200 anomalias)

Few-shot (2×k)   →    40 amostras  (20 normais + 20 anomalias)
  SetFit train   →    28 amostras  (14 normais + 14 anomalias)
  SetFit eval    →    12 amostras  ( 6 normais +  6 anomalias)

downstream_train →  4 000 amostras (inclui as 40 few-shot)
remaining pool   →  3 960 amostras (exclui as 40 few-shot)

n_known_list     →  [5, 10, 20, 50, 100, 200]
                    (500 > 200 anomalias disponíveis → dropado)
```

---

## 2. Setup experimental

| Parâmetro | Valor |
|---|---|
| Encoder base | `distiluse-base-multilingual-cased-v2` |
| k per class testados | `20`, `40` |
| Modelos AD | DevNet, DeepSAD, MLP |
| Métrica principal | ROC-AUC no test set |
| n_known_list | [5, 10, 20, 50, 100, 200, 500] |
| Contaminação ajustada | 5% |

---

## 3. Resultados — k=20

### 20 Newsgroups (EN — Topic Classification)

| Modelo | n=5 Orig | n=5 SF | n=20 Orig | n=20 SF |
|--------|----------|--------|-----------|---------|
| DevNet  | 0.092 | **0.994** | 0.242 | **0.994** |
| DeepSAD | 0.832 | **0.940** | 0.986 | 0.963 |
| MLP     | 0.964 | 0.974 | **0.999** | 0.988 |

SetFit é **crítico para DevNet** — sem ele, DevNet com poucos rótulos é praticamente inútil (AUC ~0.09).

### Portuguese Tweets (PT — Sentiment Analysis)

| Modelo | n=5 Orig | n=5 SF | n=50 Orig | n=50 SF |
|--------|----------|--------|-----------|---------|
| DevNet  | 0.477 | **0.982** | 0.485 | **0.955** |
| DeepSAD | 0.574 | **1.000** | 0.715 | **1.000** |
| MLP     | 0.726 | **1.000** | 0.832 | **1.000** |

Ganho mais expressivo: AUC = 1.000 com apenas n=5. Sentimento em PT é altamente separável após fine-tuning contrastivo.

### WikiNews (PT — Topic Classification)

| Modelo | n=5 Orig | n=5 SF | n=50 Orig | n=50 SF |
|--------|----------|--------|-----------|---------|
| DevNet  | 0.357 | **0.892** | 0.429 | **0.918** |
| DeepSAD | 0.610 | **0.890** | 0.735 | **0.943** |
| MLP     | 0.846 | **0.945** | 0.891 | **0.947** |

Ganhos consistentes em todos os modelos. SetFit eleva o piso de performance.

### Tweets Hate Speech (EN — Hate Speech Detection)

| Modelo | n=5 Orig | n=5 SF | n=50 Orig | n=50 SF |
|--------|----------|--------|-----------|---------|
| DevNet  | 0.605 | **0.752** | 0.622 | **0.779** |
| DeepSAD | 0.555 | **0.830** | 0.667 | **0.857** |
| MLP     | 0.762 | **0.847** | 0.807 | 0.863 |

Ganhos relevantes no regime de escassez extrema (n ≤ 20).

### TOLD-Br (PT — Hate Speech Detection)

| Modelo | n=5 Orig | n=5 SF | n=100 Orig | n=100 SF |
|--------|----------|--------|------------|---------|
| DevNet  | 0.517 | 0.469 | 0.522 | 0.472 |
| DeepSAD | 0.521 | **0.573** | 0.559 | **0.606** |
| MLP     | 0.530 | 0.536 | **0.666** | **0.692** |

⚠️ **Resultado mais fraco.** SetFit não ajuda DevNet e oferece ganhos marginais para MLP/DeepSAD. AUC próximo de 0.5 para todos os métodos.

### TweetEval (EN — Sentiment Analysis)

| Modelo | n=5 Orig | n=5 SF | n=200 Orig | n=200 SF |
|--------|----------|--------|------------|---------|
| DevNet  | 0.528 | 0.555 | 0.558 | 0.583 |
| DeepSAD | 0.498 | 0.479 | 0.619 | **0.704** |
| MLP     | 0.517 | **0.579** | **0.680** | **0.708** |

Ganhos moderados e crescentes com n. Não resolve o problema com poucos rótulos.

---

## 4. Resultados — k=40

### 20 Newsgroups (EN — Topic Classification)

| Modelo | n=5 Orig | n=5 SF | n=20 Orig | n=20 SF |
|--------|----------|--------|-----------|---------|
| DevNet  | 0.092 | **0.992** | 0.242 | **0.988** |
| DeepSAD | 0.832 | **0.953** | 0.986 | 0.952 |
| MLP     | 0.964 | **0.984** | **0.999** | 0.990 |

Equivalente ao k=20 — teto já atingido.

### Portuguese Tweets (PT — Sentiment Analysis)

| Modelo | n=5 Orig | n=5 SF | n=50 Orig | n=50 SF |
|--------|----------|--------|-----------|---------|
| DevNet  | 0.477 | **0.985** | 0.485 | **0.957** |
| DeepSAD | 0.574 | **1.000** | 0.715 | **1.000** |
| MLP     | 0.726 | **1.000** | 0.832 | **1.000** |

Teto atingido com k=20 já. DevNet tem ganho marginal no roc_auc mas colapso severo no pr_auc (0.576 vs 0.978 em k=20 para n=5).

### WikiNews (PT — Topic Classification)

| Modelo | n=5 Orig | n=5 SF | n=50 Orig | n=50 SF |
|--------|----------|--------|-----------|---------|
| DevNet  | 0.357 | 0.811 | 0.429 | 0.823 |
| DeepSAD | 0.610 | 0.764 | 0.735 | **0.922** |
| MLP     | 0.846 | 0.905 | 0.891 | **0.923** |

⚠️ k=40 é **pior** que k=20 em quase todos os pontos. DevNet sofre colapso grave (n=200: 0.729→0.096).

### Tweets Hate Speech (EN — Hate Speech Detection)

| Modelo | n=5 Orig | n=5 SF | n=50 Orig | n=50 SF |
|--------|----------|--------|-----------|---------|
| DevNet  | 0.605 | 0.775 | 0.622 | **0.846** |
| DeepSAD | 0.555 | **0.904** | 0.667 | **0.902** |
| MLP     | 0.762 | 0.808 | 0.807 | **0.912** |

DeepSAD com k=40 é o destaque (+7pp sobre k=20). DevNet entra em colapso em vários pontos (n=10: 0.182, n=20: 0.166).

### TOLD-Br (PT — Hate Speech Detection)

| Modelo | n=5 Orig | n=5 SF | n=100 Orig | n=100 SF |
|--------|----------|--------|------------|---------|
| DevNet  | 0.517 | **0.608** | 0.522 | **0.587** |
| DeepSAD | 0.521 | 0.562 | 0.559 | **0.609** |
| MLP     | 0.530 | **0.621** | **0.666** | **0.671** |

🔑 **Primeiro resultado positivo para TOLD-Br.** DevNet passa de 0.469 para **0.608** em n=5. Os valores ainda são modestos (~0.6), mas demonstram que quantidade importa — e levanta a hipótese de que **qualidade dos rótulos** pode ser o fator mais limitante.

### TweetEval (EN — Sentiment Analysis)

| Modelo | n=5 Orig | n=5 SF | n=200 Orig | n=200 SF |
|--------|----------|--------|------------|---------|
| DevNet  | 0.528 | **0.609** | 0.558 | **0.617** |
| DeepSAD | 0.498 | 0.456 | 0.619 | 0.691 |
| MLP     | 0.517 | **0.598** | **0.680** | 0.702 |

Ganhos marginais e mistos. Nenhuma mudança estrutural.

---

## 5. Comparação k=20 vs k=40

### Delta médio (SetFit − Original) por dataset

| Dataset | Task | Lang | Δ k=20 | Δ k=40 | Tendência |
|---------|------|------|--------|--------|-----------|
| 20 Newsgroups | TC | EN | +0.298 | +0.290 | ≈ Equivalente (teto atingido) |
| Portuguese Tweets | SA | PT | +0.287 | +0.301 | ≈ Equivalente (teto atingido) |
| WikiNews | TC | PT | +0.217 | +0.155 | ⚠️ k=40 pior — possível overfitting |
| Tweets HS | HS | EN | +0.121 | +0.071 | ⚠️ k=40 pior em média (DevNet instável) |
| TweetEval | SA | EN | +0.025 | +0.040 | ≈ Leve melhora com k=40 |
| **TOLD-Br** | **HS** | **PT** | **-0.002** | **+0.057** | **✅ k=40 reverte o déficit** |

### Delta médio por modelo

| Modelo | Δ k=20 | Δ k=40 | Interpretação |
|--------|--------|--------|---------------|
| DevNet  | +0.238 | +0.193 | k=40 introduz instabilidade em alguns datasets |
| DeepSAD | +0.133 | +0.155 | k=40 leve melhora — mais estável que DevNet |
| MLP     | +0.063 | +0.071 | Mais robusto, pouco afetado pela mudança de k |

### TOLD-Br: delta (SetFit − Original) por modelo e n_known

| Modelo | k | n=5 | n=10 | n=20 | n=50 | n=100 | n=200 |
|--------|---|-----|------|------|------|-------|-------|
| DevNet | 20 | -0.048 | -0.062 | -0.047 | -0.061 | -0.050 | -0.040 |
| DevNet | **40** | **+0.091** | **+0.075** | **+0.070** | **+0.085** | **+0.065** | **+0.043** |
| DeepSAD | 20 | +0.052 | +0.042 | +0.045 | -0.035 | +0.047 | +0.026 |
| DeepSAD | **40** | +0.042 | **+0.119** | **+0.063** | **+0.026** | **+0.041** | **+0.071** |
| MLP | 20 | +0.007 | -0.004 | +0.005 | +0.013 | +0.026 | +0.043 |
| MLP | **40** | **+0.092** | **+0.038** | **+0.009** | **+0.057** | **+0.003** | **+0.038** |

---

## 6. Takeaways

### ✅ Onde SetFit claramente ajuda

| Condição | Observação |
|---|---|
| **Regime de escassez extrema** (n ≤ 20) | Ganhos maiores — SetFit compensa falta de rótulos com melhor separação |
| **Tarefas com sinal lexical forte** (TC, SA em PT) | Fine-tuning contrastivo altamente efetivo |
| **DevNet** | Modelo mais sensível à qualidade do embedding — sem SetFit pode colapsar |
| **Portuguese Tweets e WikiNews** | Casos de sucesso mais expressivos |

### ⚠️ Onde SetFit não resolve

| Condição | Observação |
|---|---|
| **TOLD-Br com k=20** | AUC abaixo do baseline para DevNet — fine-tuning contraproducente |
| **TweetEval com n pequeno** | Ganhos marginais no início |
| **DevNet com k=40 em datasets saturados** | Instabilidade e colapso de pr_auc |

### Instabilidade do DevNet

DevNet é o mais sensível à geometria do espaço de embedding:
- **20 Newsgroups / PT Tweets**: funciona perfeitamente com SetFit
- **WikiNews k=40**: colapso severo em n=200 (0.729→0.096)
- **Tweets HS k=40**: colapsos em n=10, n=20, n=200
- **TOLD-Br k=20**: consistentemente abaixo do baseline sem SetFit

SetFit é útil quando alinha com a fronteira de anomalia, mas frágil quando o fine-tuning cria geometrias que confundem o boundary learning do DevNet.

### Resumo executivo

| Dataset | Task | Lang | k=20 | k=40 | Destaque |
|---------|------|------|------|------|---------|
| Portuguese Tweets | SA | PT | ✅ Muito | ✅ Muito | AUC = 1.0 com n=5 |
| 20 Newsgroups | TC | EN | ✅ Muito | ✅ Muito | DevNet ressuscitado — saturado |
| WikiNews | TC | PT | ✅ Muito | ⚠️ Moderado | k=40 introduz instabilidade |
| Tweets HS | HS | EN | ✅ Moderado | ✅ Moderado | DeepSAD k=40 melhor |
| TweetEval | SA | EN | 〰️ Moderado | 〰️ Moderado | Melhora só com n grande |
| **TOLD-Br** | **HS** | **PT** | **❌ Fraco** | **〰️ Fraco/Moderado** | **k=40 reverte déficit** |

---

## 7. Implicações para o próximo experimento

O padrão de TOLD-Br é o mais relevante para a tese:
- Com k=20 e rótulos **aleatórios**, SetFit não ajuda (DevNet fica abaixo do baseline)
- Com k=40 e rótulos **aleatórios**, há melhora — mas AUC máximo ~0.68
- Isso sugere que **a qualidade dos rótulos**, não apenas a quantidade, é o fator limitante

**Hipótese:**
> Com k=40 e rótulos fornecidos por LLM (raciocínio explícito + contexto cultural),
> os mesmos k amostras devem produzir embeddings SetFit mais discriminativos —
> superando o teto do rótulo aleatório.

```
SetFit (k=40, rótulos aleatórios) → AUC ~0.62–0.68 em TOLD-Br
         ↓
LLM rotula as mesmas k amostras (anomaly_score + reasoning)
         ↓
SetFit (k=40, rótulos LLM) → ?
         ↓
Compara: AUC com rótulos humanos vs. LLM vs. aleatório
```

**Aberto para investigação:**
- [ ] SetFit com k > 40 — os resultados de TOLD-Br continuam melhorando?
- [ ] SetFit com outros encoders (XLM-RoBERTa, BERT-large-PT) — o problema é o encoder ou o fine-tuning?
- [ ] **Impacto de rótulos LLM vs. aleatórios** ← próximo experimento principal
- [ ] Análise qualitativa: quais exemplos o LLM classifica diferente do rótulo original?
