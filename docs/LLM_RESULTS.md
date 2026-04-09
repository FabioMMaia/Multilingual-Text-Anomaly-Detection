# Exp 3 — LLM-Guided Anomaly Detection: Resultados e Takeaways

> **Configuração experimental:** Qwen2.5-7B-Instruct Q4_K_M e Qwen2.5-14B-Instruct Q4_K_M
> via `llamacpp`, Colab T4 GPU. Encoder: `distiluse-base-multilingual-cased-v2`.
> Threshold de anomalia: 0.6. Seed: 42. Estratégias: `random` e `score_guided`.
> Orçamento N ∈ {50, 100, 150, 200}. SetFit roda apenas se LLM encontrar ≥ 8 amostras de cada classe.

---

## 1. Resultado por Dataset — N=200, random (melhor configuração geral)

| Dataset | Tarefa | Idioma | Modelo | n_anomalias_LLM | SetFit rodou? | ROC-AUC | LLM Recall | LLM Precision |
|---|---|---|---|---|---|---|---|---|
| **20_newsgroups** | TC | EN | 7B | 1 / 200 | ❌ | **0.9025** | 0.111 | 1.000 |
| **20_newsgroups** | TC | EN | 14B | 2 / 200 | ❌ | **0.8825** | 0.222 | 1.000 |
| **wikinews** | TC | PT | 7B | 21 / 200 | ✅ | 0.7280 | 0.071 | 0.048 |
| **wikinews** | TC | PT | 14B | 10 / 200 | ✅ | **0.8305** | 0.500 | 0.700 |
| **pt_tweets** | SA | PT | 7B | 25 / 200 | ✅ | 0.7383 | 0.700 | 0.280 |
| **pt_tweets** | SA | PT | 14B | 9 / 200 | ✅ | 0.6326 | 0.500 | 0.556 |
| **tweeteval** | SA | EN | 7B | 72 / 200 | ✅ | 0.6270 | 0.846 | 0.153 |
| **tweeteval** | SA | EN | 14B | 78 / 200 | ✅ | 0.7008 | 0.923 | 0.154 |
| **told_br** | HD | PT | 7B | 7 / 200 | ❌ | 0.4346 | 0.200 | — |
| **told_br** | HD | PT | 14B | 1 / 200 | ❌ | 0.4662 | 0.000 | — |
| **tweets_hs** | HD | EN | 7B | **8** / 200 | ✅ | **0.8051** | 0.375 | — |
| **tweets_hs** | HD | EN | 14B | 3 / 200 | ❌ | 0.6214 | 0.125 | — |

> TC = Topic Classification · SA = Sentiment Analysis · HD = Hate Speech Detection

---

## 2. Padrão geral: o pipeline degrada para não-supervisionado

Em **~90% dos runs**, o flag `setfit_skipped=True` é ativado porque o LLM encontra
menos de 8 amostras de anomalia no orçamento anotado. Consequência direta:
**o resultado reportado como "Exp 3" é na prática o DeepSVDD puro com embeddings originais**,
sem contribuição do LLM ou do SetFit.

```
Total de runs (todos datasets, todos N, ambas estratégias):
  ├── SetFit rodou: ~12% dos runs
  └── SetFit pulado: ~88% dos runs

Breakdown por tarefa:
  TC (20ng)      → SetFit rodou: 0%   (LLM correto mas conservador demais)
  TC (wikinews)  → SetFit rodou: ~60% (LLM inconsistente por N)
  SA (pt_tweets) → SetFit rodou: ~70%
  SA (tweeteval) → SetFit rodou: ~90% (LLM sobre-anota demais)
  HD (told_br)   → SetFit rodou: 0%   (LLM cego para hate speech)
  HD (tweets_hs) → SetFit rodou: ~10% (1 run bem-sucedido em N=200 7B)
```

---

## 3. Análise por tipo de tarefa

### 3.1 Topic Classification — 20_newsgroups (EN)

**Achado principal: o modelo não-supervisionado JÁ RESOLVE o problema.**

O LLM encontra quase nada (0–2 anomalias em 200 amostras), mas o ROC-AUC é o mais
alto de todos os datasets: **0.87–0.96**. Isso indica que o encoder `distiluse-v2`
captura separabilidade semântica de tópicos de forma suficientemente boa para
que o DeepSVDD (puro, sem rótulos) identifique `comp.graphics` como outlier.

→ O pipeline LLM-guided é **desnecessário** para topic classification com separabilidade
  semântica clara. O ganho viria apenas de refinamento marginal, não de correção estrutural.

### 3.2 Topic Classification — wikinews (PT)

**Achado principal: o 14B é muito superior ao 7B neste dataset.**

| N | Modelo | n_anomalias | SetFit | ROC-AUC | Recall | Precision |
|---|---|---|---|---|---|---|
| 50 | 7B | 11 / 50 | ✅ | 0.698 | 0.00 | 0.00 |
| 50 | 14B | 1 / 50 | ❌ | 0.739 | 0.33 | 1.00 |
| 100 | 7B | 11 / 100 | ✅ | 0.844 | 0.20 | 0.09 |
| 100 | 14B | 2 / 100 | ❌ | 0.622 | 0.00 | 0.00 |
| 200 | 7B | 21 / 200 | ✅ | 0.728 | 0.07 | 0.05 |
| 200 | 14B | 10 / 200 | ✅ | **0.831** | 0.50 | 0.70 |

O 7B confunde artigos políticos relacionados à tecnologia/saúde com anomalias (precision ~5%).
O 14B é mais conservador e preciso, mas inconsistente por N. Score_guided atinge 0.847 com 14B.

→ Wikinews é o único dataset TC em que o LLM agrega valor — com o modelo certo (14B) e N suficiente.

### 3.3 Sentiment Analysis — pt_tweets (PT)

**Achado principal: LLM tem recall alto mas precision baixa → labels ruidosos.**

- 7B: recall ≈ 1.0 mas precision ≈ 0.11–0.28 → rotula positivo em excesso
- 14B: mais balanceado (recall 0.50–0.67, precision 0.36–0.79)
- ROC-AUC máximo: 0.822 com score_guided 7B N=200

O label noise do 7B contamina o SetFit, resultando em embeddings mal calibrados para DeepSAD.
O score_guided ajuda aqui porque concentra a anotação em amostras com scores extremos.

### 3.4 Sentiment Analysis — tweeteval (EN)

**Achado principal: o LLM sobre-detecta anomalias a ponto de quebrar o pipeline.**

- O LLM rotula 60–78% das amostras como "anômalas" (sentimento ≠ neutro)
- Isso é matematicamente correto — mas inútil: com 5% de contaminação real no dataset,
  o LLM produz ~15x mais "anomalias" do que existem de verdade
- Precision ≈ 0.07–0.15 → labels são essencialmente ruído
- ROC-AUC estagna em 0.63–0.70 **apesar** do SetFit rodar

→ Quando o critério de anomalia é cognitivamente trivial para o LLM (qualquer emoção),
  `anomaly_score_threshold=0.6` é muito baixo. Precisaria de 0.85+ para este dataset.

### 3.5 Hate Speech Detection — told_br (PT)

**Achado principal: falha sistemática — LLM é cego para hate speech em português.**

- Em todos os runs (N=50/100/150/200, ambas estratégias, ambos modelos):
  - LLM rotula 0–7 amostras como hate speech em 200 → SetFit nunca roda
  - LLM Agreement ≈ 95% → ilusório: o LLM prevê tudo como "normal"
  - LLM Recall ≈ 0.0 em quase todos os runs
  - ROC-AUC ≈ 0.43–0.50 → performance aleatória, igual ao baseline

- **Hipótese de causa raiz:** hate speech implícito/irônico em português não dispara
  o threshold 0.6 do Qwen. O modelo foi pré-treinado majoritariamente em inglês;
  discurso de ódio codificado em PT-BR (gírias, insinuações, double meaning) não é
  reconhecido como suficientemente "anômalo".

### 3.6 Hate Speech Detection — tweets_hs (EN)

**Achado principal: uma única configuração funciona; as demais falham.**

| N | Estratégia | Modelo | n_anomalias | SetFit | ROC-AUC | Recall |
|---|---|---|---|---|---|---|
| 50 | random | 7B | 2 | ❌ | 0.585 | 0.00 |
| 50 | random | 14B | 0 | ❌ | 0.552 | 0.00 |
| 100 | random | 7B | 1 | ❌ | 0.579 | 0.00 |
| 100 | random | 14B | 0 | ❌ | 0.575 | 0.00 |
| 150 | random | 7B | 7 | ❌ | 0.738 | 0.42 |
| 150 | random | 14B | 6 | ❌ | 0.746 | 0.33 |
| **200** | **random** | **7B** | **8** | **✅** | **0.8051** | **0.375** |
| 200 | random | 14B | 3 | ❌ | 0.621 | 0.13 |
| 200 | score_guided | 7B | 7 | ❌ | 0.569 | 0.14 |
| 200 | score_guided | 14B | 3 | ❌ | 0.492 | 0.00 |

- O único run bem-sucedido (7B, N=200, random) encontrou exatamente 8 anomalias — no limite mínimo.
- O score_guided é pior que random: o DeepSVDD não sabe identificar hate speech nos embeddings,
  então amostras com score alto não são mais prováveis de ser hate speech.
- tweet_hs é em inglês → Qwen tem mais sensibilidade do que em PT-BR, mas ainda é marginal.

---

## 4. Takeaways Transversais

### T1 — A estratégia score_guided é neutra ou prejudicial para hate speech

O pressuposto do score_guided é que o DeepSVDD já tem alguma capacidade de ranquear
anomalias — e o LLM apenas confirma/refuta. Mas para hate speech, o encoder multilingual
não separa hate de non-hate em distância euclidiana. O score_guided desperdiça o orçamento
N em amostras que o DeepSVDD considera "extremas" por razões não relacionadas ao hate.

**Implicação:** Para HD, a estratégia random é igual ou melhor.

### T2 — Threshold 0.6 está errado para HD — e provavelmente certo para SA

| Tarefa | Threshold ideal estimado | Problema atual |
|---|---|---|
| HD (told_br, tweets_hs) | 0.4–0.5 | Sub-detecção: LLM nunca chega a 0.6 |
| TC (wikinews) | 0.5–0.6 | Adequado com 14B |
| SA (pt_tweets) | 0.6 | OK, precision aceitável |
| SA (tweeteval) | 0.80–0.85 | Super-detecção: LLM sempre passa de 0.6 |

### T3 — 14B ≠ melhor em todos os contextos

- 14B é mais preciso e conservador: melhor para wikinews (PT, política estruturada)
- 14B é mais conservador e pior para HD: encontra ainda menos anomalias que o 7B
- 7B é mais "trigger-happy": melhor recall em SA e HD, mas pior precision em TC

**Implicação:** A escolha de modelo deve ser task-aware, não simplesmente "maior = melhor".

### T4 — O pipeline LLM-guided SÓ funciona quando o LLM entende o critério

Resultados positivos (SetFit rodou + ROC > 0.75):

| Dataset | Config | ROC-AUC | Condição de sucesso |
|---|---|---|---|
| 20_newsgroups | qualquer | 0.88–0.96 | Não precisa de LLM — DeepSVDD resolve |
| wikinews | 14B, N=200, score_guided | 0.847 | Tarefa taxonomicamente clara no 14B |
| pt_tweets | 7B, N=200, score_guided | 0.822 | Sentiment claro, recall alto |
| tweets_hs | 7B, N=200, random | 0.805 | Único caso marginal — borderline |
| told_br | — | ~0.49 | Fracasso sistemático |

### T5 — told_br requer re-engenharia antes de re-rodar

Não é questão de hiperparâmetro: o LLM não está entendendo o que é hate speech em PT-BR.
Possíveis intervenções:
1. **Baixar threshold para 0.4**: captura hate implícito que hoje fica em 0.4–0.59
2. **Reescrever o prompt** incluindo exemplos explícitos de hate codificado em PT-BR
3. **Usar modelo fine-tunado para HD em PT** (ex: via Groq com modelo já calibrado)
4. **Usar few-shot no prompt** com 2–3 exemplos de hate/não-hate

### T6 — N=200 é o mínimo viável para HD; N=50–100 são insuficientes

Com contaminação de 5% e N=200, esperamos ~10 amostras anômalas — mas o LLM encontra
apenas 1–8. O orçamento de anotação precisaria ser ajustado para garantir que
probabilisticamente encontremos ≥ 8 anomalias:

```
P(n_anomalias ≥ 8 | N, contaminação=5%, threshold) = f(LLM_recall)

Se LLM_recall ≈ 0.4 e contaminação=5%:
  N=200 → E[anomalias] = 200 × 0.05 × 0.4 = 4   ← ainda insuficiente
  N=500 → E[anomalias] = 500 × 0.05 × 0.4 = 10  ← marginal suficiente
```

Para HD com recall baixo, precisaríamos N ≈ 300–500 para garantir o mínimo.

---

## 5. Diagnóstico por dataset (síntese)

| Dataset | Diagnóstico principal | Ação recomendada |
|---|---|---|
| **20_newsgroups** | DeepSVDD resolve, LLM desnecessário | Usar como baseline unsupervised forte |
| **wikinews** | Funciona com 14B+N=200; instável | Re-rodar com N=300, threshold=0.55 |
| **pt_tweets** | Funciona mas com precision baixa | Score_guided + 14B + threshold=0.65 |
| **tweeteval** | LLM sobre-anota demais | Elevar threshold para 0.80 ou redesenhar critério |
| **told_br** | LLM cego para hate PT-BR | Reformular prompt + threshold=0.4 + few-shot |
| **tweets_hs** | 1 caso de sucesso; muito frágil | N=300, threshold=0.45, reforçar prompt |

---

## 6. Implicação para a tese

Os resultados revelam que o pipeline LLM-guided não é um método geral para anomaly detection
textual — ele é **condicionalmente eficaz** dependendo de:

1. Se o LLM é capaz de reconhecer o critério de anomalia no idioma/domain do dataset
2. Se o threshold de anotação é adequado à dificuldade cognitiva da tarefa para o LLM
3. Se o N é grande o suficiente para compensar o recall do LLM com a taxa de contaminação real

A tese pode ser reposicionada não como "o pipeline funciona para todos", mas como análise de
**quando e por que o LLM como anotador substituto falha** — o que é igualmente relevante
do ponto de vista científico, especialmente para hate speech em português.

---

*Dados coletados em April 2026. Backend: llamacpp, Colab T4.*
*Código: [`src/pipeline/llm_runner.py`](../src/pipeline/llm_runner.py)*
*Script: [`scripts/run_llm_active_loop.py`](../scripts/run_llm_active_loop.py)*
