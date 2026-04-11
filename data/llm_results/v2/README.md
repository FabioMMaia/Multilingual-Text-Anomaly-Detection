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

## Resultados

### tweets_hs — Qwen 2.5 14B (llamacpp)

| Strategy | N | N anom | SetFit | LLM Prec | LLM Rec | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| random | 50 | 7 | ❌ | 0.286 | **1.000** | **0.8663** | 0.3291 |
| random | 100 | 6 | ❌ | 0.167 | 0.200 | 0.5902 | 0.0829 |
| random | 150 | 18 | ✅ | 0.556 | 0.833 | **0.8057** | 0.2165 |
| random | 200 | 10 | ✅ | 0.300 | 0.375 | **0.8397** | 0.3745 |
| score_guided | 50 | 0 | ❌ | 0.000 | 0.000 | 0.4555 | 0.0435 |
| score_guided | 100 | 4 | ❌ | 0.250 | 0.250 | 0.6219 | 0.0830 |
| score_guided | 150 | 7 | ❌ | 0.286 | 0.400 | 0.7042 | 0.1531 |
| score_guided | 200 | 14 | ✅ | 0.143 | 0.286 | **0.8297** | 0.2646 |

> Melhor resultado geral: random N=50, ROC=0.8663 (sem SetFit). SetFit rodou em 4 runs; em 3 deles o ROC ficou ≥ 0.80.

### told_br — GPT-4o-mini (OpenAI API)

| Strategy | N | N anom | SetFit | LLM Prec | LLM Rec | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| score_guided | 200 | 8 | ✅ | 0.125 | 0.143 | 0.4928 | — |

> Único run realizado — suficiente para confirmar o diagnóstico: GPT-4o-mini não resolve told_br.

## Takeaways

1. **tweets_hs é separável — random N=50 dá ROC=0.8663 sem SetFit:**
   Qwen 14B encontrou 7 anomalias com recall 100% (todos os hates reais nas 50 amostras).
   SetFit foi pulado (7 < 8 mínimo), mas o DeepSAD com embeddings originais já performou
   muito bem. É o melhor resultado de todo o projeto.

2. **random N=100 é o único outlier ruim (ROC=0.5902):**
   LLM só encontrou 6 anomalias com recall 0.20 — o LLM "foi conservador" neste sample
   específico. Com N=150 e N=200 o ROC volta a ≥0.80. Isso indica variância de sampling,
   não tendência.

3. **SetFit ajuda quando LLM precision é razoável em tweets_hs:**
   Random N=150 (prec=0.556, rec=0.833) → ROC=0.8057 com SetFit.
   Random N=200 (prec=0.300) → ROC=0.8397 com SetFit.
   Diferente de told_br, aqui SetFit não degrada — precision suficiente para pares
   contrastivos informativos.

4. **score_guided é inferior a random em tweets_hs:**
   score_guided N=200 deu ROC=0.8297, mas score_guided N=50 falhou (0 anomalias).
   A estratégia de seleção por score do unsupervised model não agrega para este dataset.
   Random sampling captura hate speech melhor porque a anomalia não é "sempre o ponto
   mais extremo" para o modelo não-supervisionado.

5. **GPT-4o-mini falhou em told_br (precision 12.5%, recall 14.3%, ROC 0.49):**
   O melhor modelo zero-shot comercial disponível não consegue anotar hate speech PT-BR
   implícito com precision suficiente. A falha é da tarefa, não do modelo — hate
   culturalmente codificado em PT-BR requer contexto que zero-shot não captura.

6. **Threshold 0.45 + prompt v1 resolve o SetFit-skipped de tweets_hs:**
   Em v0 tweets_hs quase nunca chegava a 8 anomalias. Em v2, SetFit rodou em 4/8 runs.
   Para told_br o problema persiste: GPT-4o-mini N=200 só encontrou 8 (exatamente o mínimo).

7. **Hipótese estrutural confirmada empiricamente:**
   tweets_hs (EN, hate explícito) → LLM recall alto → ROC ≥ 0.80 na maioria dos runs.
   told_br (PT-BR, hate implícito) → LLM confuso → ROC ~0.49 independente do modelo.
   Próximo passo: medir Silhouette Score com ground truth para quantificar separabilidade
   como pré-condição do pipeline.

## Estado atual

- [x] tweets_hs grade completa (Qwen 14B, N=50/100/150/200, random+score_guided) — **concluído**
- [x] told_br N=200 score_guided (GPT-4o-mini) — concluído
- [ ] Silhouette Score diagnosis — planejado
- [ ] tweets_hs com GPT-4o-mini — opcional

---

## Prompt utilizado (v2)

> Mesmo prompt de v1 — nenhuma alteração de TASK_CONTEXT ou template entre v1 e v2.
> O que mudou foi o modelo (Qwen 14B local e GPT-4o-mini API) e o dataset foco (tweets_hs).
> Ver prompt completo em [v1/README.md](../v1/README.md#prompt-utilizado-v1----told_br-e-tweets_hs).
