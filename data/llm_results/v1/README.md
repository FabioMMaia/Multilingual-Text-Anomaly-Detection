# Experimento v1 — Prompt melhorado, threshold 0.45

## Configuração

| Parâmetro | Valor |
|---|---|
| **Prompt** | v1 — adicionados exemplos de hate implícito/irônico PT-BR; instrução "prefira 0.5 em caso de dúvida"; descrição de normal mais explícita |
| **Threshold** | 0.45 (reduzido de 0.6) |
| **Modelo** | Qwen 2.5 7B Q4 (llamacpp, local) |
| **Datasets** | told_br (foco HD); demais datasets não re-testados |
| **Estratégias** | random, score_guided |
| **N** | 50, 100, 150, 200 |
| **Seed** | 42 |
| **Encoder** | distiluse-base-multilingual-cased-v2 |
| **Hardware** | Colab T4 |

## Objetivo

Verificar se prompt melhorado + threshold menor resolve o problema de SetFit sempre ser pulado
em told_br (hate speech PT-BR).

## Resultados

| Strategy | N | Anomalias LLM | SetFit | LLM Precision | LLM Recall | ROC-AUC |
|---|---|---|---|---|---|---|
| random | 50 | ~3 | ❌ | baixa | baixa | ~0.49 |
| random | 100 | ~8–12 | ⚠️ borda | baixa | baixa | ~0.50 |
| random | 150 | ~15–23 | ✅ | 0.13–0.25 | ~0.20 | ~0.51 |
| random | 200 | ~20–30 | ✅ | 0.13–0.33 | ~0.20 | ~0.49–0.53 |
| score_guided | 50 | ~2 | ❌ | — | — | ~0.49 |
| score_guided | 100 | ~5 | ❌ | — | — | ~0.56 ← melhor |
| score_guided | 150 | ~18 | ✅ | baixa | baixa | ~0.50 |
| score_guided | 200 | ~23 | ✅ | baixa | baixa | ~0.49 |

## Takeaways

1. **Prompt v1 + threshold 0.45 resolveu o SetFit skipped:** SetFit passou a rodar para N≥150,
   confirmando que o problema de v0 era threshold alto + prompt sem exemplos de hate implícito.

2. **SetFit rodando não melhorou o ROC:** precisão do LLM (0.13–0.33) é baixa demais para que
   pares contrastivos do SetFit sejam informativos. O fine-tuning com labels ruins degrada o
   embedding em vez de melhorá-lo.

3. **score_guided N=100 foi o melhor resultado (ROC 0.56):** SetFit foi *pulado* neste caso.
   Isso sugere que o DeepSAD com embeddings originais + poucos labels corretos supera o
   DeepSAD com embeddings SetFit ruidosos. Paradoxo: SetFit atrapalha.

4. **Análise qualitativa do told_br N=200:** LLM comete FPs em gírias PT-BR neutras
   ("muie", "mina", "vagabundo", "capeta") — confunde linguagem informal com ódio.
   Além disso, o ground truth tem ruído: ~30% das amostras GT=1 não apresentam hate
   speech visível, sugerindo anotação subjetiva/disagreement entre anotadores humanos.

5. **Hipótese central identificada:** o problema estrutural em told_br é duplo:
   (a) hate speech implícito culturalmente codificado que zero-shot não captura, e
   (b) ruído no ground truth que limita o teto teórico de qualquer modelo.

6. **Conclusão de v1:** Qwen 7B com prompt melhorado não é suficiente para told_br.
   Próximos passos: testar modelo maior via API e testar tweets_hs (EN, hate mais explícito).

## Ação tomada

→ Criar v2: testar GPT-4o-mini (OpenAI) em told_br e Qwen 14B em tweets_hs.
→ Expandir análise para tweets_hs onde hate speech é mais explícito e ground truth mais limpo.

---

## Prompt utilizado (v1) — told_br e tweets_hs

**Template (idêntico ao v0 e v2):**
```
You are evaluating a text sample for anomaly detection.

Dataset context: {description}

Normal samples are: {normal_description}

Anomaly criterion (use this definition strictly):
{anomaly_criterion}

Rate the following text on a scale from 0.0 (clearly normal) to 1.0 (clearly anomalous
according to the criterion above). Do not use any other notion of "anomalous".

Text:
"""
{text}
"""

Respond ONLY with a JSON object, no explanation outside the JSON:
{"anomaly_score": <float between 0.0 and 1.0>, "reason": "<one sentence explaining your score>"}
```

**TASK_CONTEXT — told_br (v1, texto exato ainda no código):**
```
description:
  "Brazilian Portuguese social media texts.
   Your task: decide if a text is HATE SPEECH (anomalous, score 1.0) or not (normal, score 0.0)."

normal_description:
  "Any text that does NOT target people based on group identity:
   profanity, personal insults between individuals, arguments, crude humor,
   strong opinions, complaints, offensive language with no identity group target — all normal (score 0.0)."

anomaly_criterion:
  "Text that attacks, degrades, or dehumanizes people because of their race, ethnicity, gender,
   sexual orientation, religion, nationality, or disability.
   IMPORTANT — hate speech in Brazilian Portuguese is often implicit or coded:
   score >= 0.5 for ironic statements that demean a group ('essa gente é assim mesmo'),
   stereotypes presented as facts ('mulheres não sabem dirigir'),
   language that normalizes prejudice against an identity group,
   and slurs or coded terms targeting a group even if softened.
   Score 0.0 only when the text targets a specific individual with no group-identity dimension,
   or contains no attack at all. When in doubt about implicit hate, prefer 0.5."
```

**TASK_CONTEXT — tweets_hs (v1, texto exato ainda no código):**
```
description:
  "English tweets.
   Your task: decide if a tweet contains HATE SPEECH (anomalous, score 1.0) or not (normal, score 0.0)."

normal_description:
  "Tweets that do NOT attack people based on group identity:
   opinions, arguments, profanity, personal conflicts between individuals,
   offensive language with no identity group target — all normal (score 0.0)."

anomaly_criterion:
  "Tweets that attack, degrade, or dehumanize people because of their race, ethnicity, gender,
   sexual orientation, religion, nationality, or disability.
   IMPORTANT — hate speech is often implicit or coded:
   score >= 0.5 for tweets using irony or sarcasm to mock a group,
   derogatory generalizations about a group presented as facts,
   coded language or slurs (even softened or misspelled) targeting an identity group,
   and content that dehumanizes or calls for exclusion of a group.
   Score 0.0 only when the tweet targets a specific individual with no group-identity dimension,
   or contains no attack at all. When in doubt about implicit hate, prefer 0.5."
```
