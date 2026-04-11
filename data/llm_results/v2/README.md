# Experimento v2 — Multi-modelo, foco em tweets_hs + diagnóstico told_br

## Configuração

| Parâmetro | Valor |
|---|---|
| **Prompt** | v1 (mesmo de v1 — hate implícito, threshold 0.45) |
| **Threshold** | 0.45 |
| **Modelos** | Qwen 2.5 14B Q4 (llamacpp, local) + GPT-4o-mini (OpenAI API) |
| **Datasets** | tweets_hs (grade completa); told_br (teste pontual GPT-4o-mini) |
| **Estratégias** | random, score_guided |
| **N** | 50, 100, 150, 200 |
| **Seed** | 42 |
| **Encoder** | distiluse-base-multilingual-cased-v2 |
| **Hardware** | Colab T4 (local) / API (OpenAI) |

## Objetivo

1. **Testar tweets_hs** com Qwen 14B — hate speech em inglês é mais explícito, esperamos
   precision/recall melhores que told_br.
2. **Testar told_br** com GPT-4o-mini — verificar se o melhor modelo zero-shot do mercado
   resolve o problema de precision baixa em hate speech PT-BR.
3. **Estabelecer contrastes** para a hipótese da tese: separabilidade do embedding como
   pré-condição para LLM annotation funcionar.

## Resultados parciais (em andamento)

### tweets_hs — Qwen 2.5 14B (llamacpp)

| Strategy | N | Anomalias LLM | SetFit | LLM Precision | LLM Recall | ROC-AUC |
|---|---|---|---|---|---|---|
| random | 50 | 7 | ❌ (7 < 8) | 28.6% | **100%** | **0.8663** ← melhor do projeto |
| random | 100 | — | — | — | — | — |
| random | 150 | — | — | — | — | — |
| random | 200 | — | — | — | — | — |
| score_guided | 50 | — | — | — | — | — |
| score_guided | 100 | — | — | — | — | — |
| score_guided | 150 | — | — | — | — | — |
| score_guided | 200 | — | — | — | — | — |

### told_br — GPT-4o-mini (OpenAI API)

| Strategy | N | Anomalias LLM | SetFit | LLM Precision | LLM Recall | ROC-AUC |
|---|---|---|---|---|---|---|
| score_guided | 200 | 8 | ✅ (8×2) | 12.5% | 14.3% | 0.4928 |

## Takeaways preliminares

1. **tweets_hs N=50 random: ROC 0.8663 — melhor resultado de todo o projeto:**
   Qwen 14B encontrou 7 anomalies com recall 100% (encontrou todos os hates reais nas 50
   amostras). SetFit foi pulado (7 < 8). O pipeline sem SetFit superou todas as
   configurações anteriores por larga margem.

2. **Recall 100% com precision 28.6% é suficiente para DeepSAD:**
   Os FPs (5 em 7) não prejudicaram o modelo semi-supervisionado — DeepSAD é robusto
   a ruído de labels quando o recall é alto. O sinal correto "este é hate" prevalece.

3. **GPT-4o-mini falhou em told_br (precision 12.5%, recall 14.3%, ROC 0.49):**
   O melhor modelo zero-shot comercial não consegue anotar hate speech PT-BR implícito
   com precisão suficiente. A falha não é do modelo mas da tarefa: hate codificado
   culturalmente em PT-BR requer conhecimento que zero-shot não captura.

4. **SetFit com 8×2 amostras = ruído, não sinal:**
   told_br GPT-4o-mini rodou SetFit com apenas 8 positivos → ROC 0.49 (pior que random).
   tweets_hs sem SetFit → ROC 0.87. Isso confirma: SetFit só ajuda quando LLM precision
   for alta o suficiente para criar pares contrastivos informativos.

5. **Hipótese de separabilidade emergindo:**
   tweets_hs (EN, hate explícito) → embedding separa bem → LLM recall alto → ROC alto.
   told_br (PT-BR, hate implícito) → embedding não separa → LLM confuso → ROC ~0.49.
   Próximo passo: medir Silhouette Score com ground truth para quantificar essa separabilidade
   e usá-la como preditor a priori do sucesso do pipeline.

6. **Gemini 2.5 Flash-Lite: descartado por limite de 20 req/dia** (não os 1.500 documentados).
   Groq 70B: descartado por limite de 1.000 req/dia (TPM 12K → 8s delay).
   GPT-4o-mini: viável (~$0.027 por run de N=200), sem rate limit relevante.

## Estado atual

- [ ] tweets_hs grade completa (Qwen 14B) — **em execução**
- [x] told_br N=200 score_guided (GPT-4o-mini) — concluído
- [ ] Silhouette Score diagnosis — planejado
- [ ] tweets_hs com GPT-4o-mini — opcional (custo ~$0.11 para N=200)

## Ação planejada

→ Analisar grade completa tweets_hs ao terminar.
→ Implementar Silhouette Score no pipeline para medir separabilidade por dataset.
→ Se tweets_hs N>50 confirmar ROC alto → expandir para múltiplos seeds.
→ Discutir com orientador: HateXplain como 3º dataset (EN, IAA mais alto que tweets_hs).

---

## Prompt utilizado (v2)

> Mesmo prompt de v1 — nenhuma alteração de TASK_CONTEXT ou template entre v1 e v2.
> O que mudou foi o modelo (Qwen 14B local e GPT-4o-mini API) e o dataset foco (tweets_hs).
> Ver prompt completo em [v1/README.md](../v1/README.md#prompt-utilizado-v1----told_br-e-tweets_hs).
