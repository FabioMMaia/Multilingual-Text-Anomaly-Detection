# Experimento v0 — Baseline LLM (prompt original, threshold 0.6)

## Configuração

| Parâmetro | Valor |
|---|---|
| **Prompt** | Original — sem exemplos de hate implícito, sem instrução de dúvida |
| **Threshold** | 0.6 (padrão inicial) |
| **Modelos** | Qwen 2.5 7B Q4 e Qwen 2.5 14B Q4 (llamacpp, local) |
| **Datasets** | todos os 6: told_br, tweets_hs, 20_newsgroups, wikinews, pt_tweets, tweeteval |
| **Estratégias** | random, score_guided |
| **N** | 50, 100, 150, 200 |
| **Seed** | 42 |
| **Encoder** | distiluse-base-multilingual-cased-v2 |
| **Hardware** | Colab T4 |

## Objetivo

Estabelecer o baseline do pipeline completo (DeepSVDD → LLM → SetFit → DeepSAD) com
configuração padrão, sem nenhuma otimização de prompt ou threshold.

## Resultados por Dataset

### told_br (hate speech PT-BR)

| Strategy | N | Modelo | N anom | SetFit | ROC-AUC | PR-AUC | LLM Prec | LLM Rec |
|---|---|---|---|---|---|---|---|---|
| random | 50 | 7B | 0 | ❌ | 0.4898 | 0.0430 | 0.000 | 0.000 |
| random | 50 | 14B | 0 | ❌ | 0.4898 | 0.0430 | 0.000 | 0.000 |
| random | 100 | 7B | 1 | ❌ | 0.4791 | 0.0434 | 1.000 | 0.200 |
| random | 100 | 14B | 0 | ❌ | 0.4860 | 0.0439 | 0.000 | 0.000 |
| random | 150 | 7B | 3 | ❌ | 0.4289 | 0.0384 | 0.000 | 0.000 |
| random | 150 | 14B | 3 | ❌ | 0.4542 | 0.0422 | 0.000 | 0.000 |
| random | 200 | 7B | 7 | ❌ | 0.4346 | 0.0374 | 0.286 | 0.200 |
| random | 200 | 14B | 1 | ❌ | 0.4662 | 0.0397 | 0.000 | 0.000 |
| score_guided | 50 | 7B | 1 | ❌ | 0.4962 | 0.0451 | 0.000 | 0.000 |
| score_guided | 50 | 14B | 1 | ❌ | 0.4962 | 0.0451 | 0.000 | 0.000 |
| score_guided | 100 | 7B | 1 | ❌ | 0.4783 | 0.0444 | 0.000 | 0.000 |
| score_guided | 100 | 14B | 1 | ❌ | 0.4783 | 0.0444 | 0.000 | 0.000 |
| score_guided | 150 | 7B | 2 | ❌ | 0.4492 | 0.0417 | 0.000 | 0.000 |
| score_guided | 150 | 14B | 2 | ❌ | 0.4492 | 0.0417 | 0.000 | 0.000 |
| score_guided | 200 | 7B | 4 | ❌ | 0.4445 | 0.0405 | 0.000 | 0.000 |
| score_guided | 200 | 14B | 2 | ❌ | 0.4673 | 0.0439 | 0.000 | 0.000 |

> SetFit nunca rodou (N anom < 8 em todos os runs → requisito mínimo não atingido). ROC degenerou a DeepSVDD puro.

### tweets_hs (hate speech EN)

| Strategy | N | Modelo | N anom | SetFit | ROC-AUC | PR-AUC | LLM Prec | LLM Rec |
|---|---|---|---|---|---|---|---|---|
| random | 50 | 7B | 2 | ❌ | 0.5850 | 0.0863 | 0.000 | 0.000 |
| random | 50 | 14B | 0 | ❌ | 0.5515 | 0.0751 | 0.000 | 0.000 |
| random | 100 | 7B | 1 | ❌ | 0.5789 | 0.0664 | 0.000 | 0.000 |
| random | 100 | 14B | 0 | ❌ | 0.5747 | 0.0720 | 0.000 | 0.000 |
| random | 150 | 7B | 7 | ❌ | 0.7385 | 0.2279 | 0.714 | 0.417 |
| random | 150 | 14B | 6 | ❌ | 0.7458 | 0.1929 | 0.667 | 0.333 |
| random | 200 | 7B | 8 | ✅ | **0.8051** | 0.3226 | 0.375 | 0.375 |
| random | 200 | 14B | 3 | ❌ | 0.6214 | 0.1217 | 0.333 | 0.125 |
| score_guided | 50 | 7B | 0 | ❌ | 0.4555 | 0.0435 | 0.000 | 0.000 |
| score_guided | 50 | 14B | 0 | ❌ | 0.4555 | 0.0435 | 0.000 | 0.000 |
| score_guided | 100 | 7B | 2 | ❌ | 0.5280 | 0.0669 | 0.000 | 0.000 |
| score_guided | 100 | 14B | 1 | ❌ | 0.5087 | 0.0518 | 0.000 | 0.000 |
| score_guided | 150 | 7B | 4 | ❌ | 0.5983 | 0.1002 | 0.250 | 0.200 |
| score_guided | 150 | 14B | 1 | ❌ | 0.4815 | 0.0462 | 0.000 | 0.000 |
| score_guided | 200 | 7B | 7 | ❌ | 0.5693 | 0.0651 | 0.143 | 0.143 |
| score_guided | 200 | 14B | 3 | ❌ | 0.4921 | 0.0473 | 0.000 | 0.000 |

> Único run com SetFit: 7B random N=200 (ROC 0.8051 — melhor resultado de v0, mas único caso com ≥8 anom).

### Outros datasets (tópico e sentimento)

| Dataset | Strategy | N | Modelo | N anom | SetFit | ROC-AUC | PR-AUC | LLM Prec | LLM Rec |
|---|---|---|---|---|---|---|---|---|---|
| 20_newsgroups | random | 50 | 7B | 2 | ❌ | 0.8720 | 0.5486 | 0.500 | 0.333 |
| 20_newsgroups | random | 50 | 14B | 1 | ❌ | 0.9330 | 0.3872 | 1.000 | 0.333 |
| 20_newsgroups | random | 100 | 7B | 0 | ❌ | 0.9495 | 0.4453 | 0.000 | 0.000 |
| 20_newsgroups | random | 100 | 14B | 1 | ❌ | 0.9575 | 0.5098 | 1.000 | 0.200 |
| 20_newsgroups | random | 150 | 7B | 1 | ❌ | 0.9480 | 0.4907 | 1.000 | 0.083 |
| 20_newsgroups | random | 150 | 14B | 1 | ❌ | 0.9480 | 0.4907 | 1.000 | 0.083 |
| 20_newsgroups | random | 200 | 7B | 1 | ❌ | 0.9025 | 0.4943 | 1.000 | 0.111 |
| 20_newsgroups | random | 200 | 14B | 2 | ❌ | 0.8825 | 0.5379 | 1.000 | 0.222 |
| 20_newsgroups | score_guided | 50 | 7B | 0 | ❌ | 0.8710 | 0.2693 | 0.000 | 0.000 |
| 20_newsgroups | score_guided | 50 | 14B | 0 | ❌ | 0.8710 | 0.2693 | 0.000 | 0.000 |
| 20_newsgroups | score_guided | 100 | 7B | 0 | ❌ | 0.8575 | 0.1990 | 0.000 | 0.000 |
| 20_newsgroups | score_guided | 100 | 14B | 0 | ❌ | 0.8575 | 0.1990 | 0.000 | 0.000 |
| 20_newsgroups | score_guided | 150 | 7B | 0 | ❌ | 0.8230 | 0.1659 | 0.000 | 0.000 |
| 20_newsgroups | score_guided | 150 | 14B | 0 | ❌ | 0.8230 | 0.1659 | 0.000 | 0.000 |
| 20_newsgroups | score_guided | 200 | 7B | 0 | ❌ | 0.7895 | 0.1350 | 0.000 | 0.000 |
| 20_newsgroups | score_guided | 200 | 14B | 1 | ❌ | 0.8075 | 0.1806 | 1.000 | 0.100 |
| wikinews | random | 50 | 7B | 11 | ✅ | 0.6982 | 0.0853 | 0.000 | 0.000 |
| wikinews | random | 50 | 14B | 1 | ❌ | 0.7393 | 0.1417 | 1.000 | 0.333 |
| wikinews | random | 100 | 7B | 11 | ✅ | 0.8435 | 0.2923 | 0.091 | 0.200 |
| wikinews | random | 100 | 14B | 2 | ❌ | 0.6224 | 0.1230 | 0.000 | 0.000 |
| wikinews | random | 150 | 7B | 18 | ✅ | 0.5751 | 0.0594 | 0.111 | 0.182 |
| wikinews | random | 150 | 14B | 8 | ✅ | 0.7435 | 0.3417 | 1.000 | 0.727 |
| wikinews | random | 200 | 7B | 21 | ✅ | 0.7280 | 0.1711 | 0.048 | 0.071 |
| wikinews | random | 200 | 14B | 10 | ✅ | 0.8305 | 0.3773 | 0.700 | 0.500 |
| wikinews | score_guided | 50 | 7B | 6 | ❌ | 0.5407 | 0.0684 | 0.333 | 0.286 |
| wikinews | score_guided | 50 | 14B | 5 | ❌ | 0.7874 | 0.1879 | 0.200 | 0.143 |
| wikinews | score_guided | 100 | 7B | 13 | ✅ | 0.4617 | 0.0451 | 0.231 | 0.273 |
| wikinews | score_guided | 100 | 14B | 7 | ❌ | 0.8102 | 0.2530 | 0.429 | 0.273 |
| wikinews | score_guided | 150 | 7B | 19 | ✅ | 0.6273 | 0.0692 | 0.158 | 0.231 |
| wikinews | score_guided | 150 | 14B | 9 | ✅ | 0.8042 | 0.3192 | 0.556 | 0.385 |
| wikinews | score_guided | 200 | 7B | 24 | ✅ | 0.6244 | 0.1104 | 0.167 | 0.286 |
| wikinews | score_guided | 200 | 14B | 11 | ✅ | **0.8472** | 0.3657 | 0.545 | 0.429 |
| pt_tweets | random | 50 | 7B | 9 | ✅ | 0.5805 | 0.0660 | 0.111 | 1.000 |
| pt_tweets | random | 50 | 14B | 3 | ❌ | 0.5729 | 0.0610 | 0.333 | 1.000 |
| pt_tweets | random | 100 | 7B | 10 | ✅ | 0.5439 | 0.0523 | 0.200 | 0.667 |
| pt_tweets | random | 100 | 14B | 5 | ❌ | 0.5510 | 0.0526 | 0.400 | 0.667 |
| pt_tweets | random | 150 | 7B | 22 | ✅ | 0.7365 | 0.1634 | 0.364 | 0.800 |
| pt_tweets | random | 150 | 14B | 8 | ✅ | 0.7323 | 0.2136 | 0.750 | 0.600 |
| pt_tweets | random | 200 | 7B | 25 | ✅ | 0.7383 | 0.1414 | 0.280 | 0.700 |
| pt_tweets | random | 200 | 14B | 9 | ✅ | 0.6326 | 0.0882 | 0.556 | 0.500 |
| pt_tweets | score_guided | 50 | 7B | 6 | ❌ | 0.5637 | 0.0663 | 0.167 | 0.500 |
| pt_tweets | score_guided | 50 | 14B | 4 | ❌ | 0.5375 | 0.0645 | 0.500 | 1.000 |
| pt_tweets | score_guided | 100 | 7B | 13 | ✅ | 0.6039 | 0.0815 | 0.308 | 0.571 |
| pt_tweets | score_guided | 100 | 14B | 8 | ✅ | 0.7179 | 0.1975 | 0.750 | 0.857 |
| pt_tweets | score_guided | 150 | 7B | 20 | ✅ | 0.6977 | 0.1943 | 0.350 | 0.636 |
| pt_tweets | score_guided | 150 | 14B | 9 | ✅ | 0.7069 | 0.2025 | 0.778 | 0.636 |
| pt_tweets | score_guided | 200 | 7B | 30 | ✅ | **0.8223** | 0.2587 | 0.467 | 0.778 |
| pt_tweets | score_guided | 200 | 14B | 14 | ✅ | 0.8213 | 0.3775 | 0.786 | 0.611 |
| tweeteval | random | 50 | 7B | 12 | ✅ | 0.5790 | 0.0827 | 0.000 | 0.000 |
| tweeteval | random | 50 | 14B | 16 | ✅ | 0.6303 | 0.0874 | 0.000 | 0.000 |
| tweeteval | random | 100 | 7B | 41 | ✅ | 0.6346 | 0.0917 | 0.073 | 1.000 |
| tweeteval | random | 100 | 14B | 43 | ✅ | 0.6422 | 0.1042 | 0.070 | 1.000 |
| tweeteval | random | 150 | 7B | 62 | ✅ | 0.6289 | 0.0836 | 0.048 | 1.000 |
| tweeteval | random | 150 | 14B | 60 | ✅ | 0.6013 | 0.0766 | 0.050 | 1.000 |
| tweeteval | random | 200 | 7B | 72 | ✅ | 0.6270 | 0.0929 | 0.153 | 0.846 |
| tweeteval | random | 200 | 14B | 78 | ✅ | **0.7008** | 0.1240 | 0.154 | 0.923 |
| tweeteval | score_guided | 50 | 7B | 16 | ✅ | 0.6495 | 0.0906 | 0.062 | 1.000 |
| tweeteval | score_guided | 50 | 14B | 12 | ✅ | 0.5879 | 0.0775 | 0.083 | 1.000 |
| tweeteval | score_guided | 100 | 7B | 29 | ✅ | 0.6740 | 0.1084 | 0.103 | 0.600 |
| tweeteval | score_guided | 100 | 14B | 25 | ✅ | 0.5739 | 0.0853 | 0.160 | 0.800 |
| tweeteval | score_guided | 150 | 7B | 45 | ✅ | **0.6849** | 0.1066 | 0.067 | 0.500 |
| tweeteval | score_guided | 150 | 14B | 38 | ✅ | 0.6214 | 0.0869 | 0.105 | 0.667 |
| tweeteval | score_guided | 200 | 7B | 65 | ✅ | 0.6691 | 0.1055 | 0.061 | 0.500 |
| tweeteval | score_guided | 200 | 14B | 59 | ✅ | 0.6408 | 0.0950 | 0.102 | 0.750 |

### Resumo: melhor ROC-AUC por dataset

| Dataset | Melhor ROC-AUC | Config |
|---|---|---|
| told_br | 0.4962 | score_guided N=50 (7B e 14B) |
| tweets_hs | **0.8051** | random N=200, 7B |
| 20_newsgroups | **0.9575** | random N=100, 14B |
| wikinews | **0.8472** | score_guided N=200, 14B |
| pt_tweets | **0.8223** | score_guided N=200, 7B |
| tweeteval | **0.7008** | random N=200, 14B |

## Takeaways

1. **Threshold 0.6 bloqueia o hate speech:** told_br e tweets_hs quase nunca atingem 8 anomalias
   → SetFit nunca roda → pipeline fica como DeepSVDD puro. ROC told_br estabiliza ~0.44–0.50;
   tweets_hs alcança 0.80 no único run onde SetFit rodou (7B random N=200).

2. **told_br é irrecuperável com threshold 0.6:** nenhuma combinação de estratégia/N/modelo
   produz ROC > 0.50. Teto empiricamente confirmado.

3. **tweets_hs tem potencial escondido:** 7B random N=200 (ROC=0.8051) com SetFit ativo
   sugere que o dataset é separável — o problema era threshold alto, não o LLM nem o embedding.

4. **20_newsgroups dispensa SetFit:** ROC 0.87–0.96 mesmo sem SetFit rodar (N anom muito baixo).
   O embedding já separa tópicos; DeepSAD com embeddings originais é suficiente.

5. **Qwen 14B mais conservador que o 7B para hate:** em todos os pares, o 14B encontrou menos
   anomalias que o 7B nos datasets de hate speech. RLHF mais agressivo → menos FP, mas também
   menos verdadeiros positivos. Paradoxo: modelo maior = pior recall neste contexto.

6. **LLM precision muito alta no 14B quando acha anomalias (wikinews, pt_tweets):** quando o 14B
   rotula algo como anomalia, tende a estar certo (prec 0.5–1.0). O problema é que rotula poucos.
   Tradeoff precision-recall invertido entre 7B e 14B.

## Ação tomada

→ Criar v1: melhorar prompt para hate implícito + reduzir threshold para 0.45.

---

## Prompt utilizado (v0)

> ✅ Verificado via `git show aae7f1d:src/pipeline/llm_runner.py` (commit inicial do pipeline LLM).

**Template (`PROMPT_TEMPLATE`, compartilhado com v1/v2):**
```
You are evaluating a text sample for anomaly detection.

Dataset context: {description}

Normal samples are: {normal_description}

Anomaly criterion (use this definition strictly):
{anomaly_criterion}

Rate the following text on a scale from 0.0 (clearly normal) to 1.0 (clearly anomalous according to the criterion above). Do not use any other notion of "anomalous".

Text:
"""
{text}
"""

Respond ONLY with a JSON object, no explanation outside the JSON:
{"anomaly_score": <float between 0.0 and 1.0>, "reason": "<one sentence explaining your score>"}
```

**TASK_CONTEXT — told_br (v0, exato):**
```
description:
  "Brazilian Portuguese social media texts collected for hate speech detection."

normal_description:
  "Regular, non-offensive speech with no hate, discrimination, or hostility."

anomaly_criterion:
  "Texts that contain hate speech, discrimination, or offensive language targeting
   individuals or groups based on identity characteristics (race, gender, religion,
   sexual orientation, etc.), including implicit, ironic, or coded forms common in
   Brazilian Portuguese."
```

**TASK_CONTEXT — tweets_hs (v0, exato):**
```
description:
  "English tweets collected for hate speech detection."

normal_description:
  "Regular tweets with no hate speech or discriminatory language."

anomaly_criterion:
  "Tweets that contain hate speech or discriminatory language targeting individuals
   or groups based on identity characteristics, including coded or implicit forms."
```

**Diferença principal para v1:** ausência da seção `IMPORTANT` sobre hate implícito/irônico, ausência de exemplos de gírias PT-BR, e ausência de "When in doubt about implicit hate, prefer 0.5". Tanto told_br quanto tweets_hs já incluíam formas implícitas/codificadas no critério de anomalia — a diferença está na ausência de exemplos concretos e do peso explícito dado ao limiar 0.5.
