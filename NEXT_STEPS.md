# Next Steps — Rumo ao Experimento Final

## Contexto

Decisões tomadas após análise de v0/v1/v2:

- **4 datasets fixados** (grade 2×2): tweets_hs, told_br, 20_newsgroups, wikinews
- **SA descartado** (pt_tweets, tweeteval): anomalia frequency-based sem coerência semântica com o pipeline
- **Baseline:** DeepSVDD puro (já implícito nos dados de v0 — runs com n_anomalies=0)
- **Upperbound de referência:** SetFit com GT labels k=40 (SETFIT_RESULTS.md) — citado como contexto, não como comparação direta

```
Grade 2×2:
              English         Portuguese
Hate Det.   tweets_hs  ←→   told_br
Topic Cls.  20_newsgr. ←→   wikinews
```

---

## Passo 1 — Implementar estratégia `diversity` (k-means++)

**Arquivo:** `src/pipeline/llm_runner.py` → função `select_samples`

**O que fazer:**
- Adicionar parâmetro `embeddings: np.ndarray` à função
- Adicionar `elif strategy == "diversity"` com KMeans(init="k-means++")
- Para cada cluster: selecionar o ponto mais próximo do centroide
- Atualizar o `Literal` de `strategy` e propagar `embeddings` no `run_llm_active_loop`
- Atualizar `run_llm_active_loop.py` para aceitar `--strategy diversity`

**Estimativa:** ~30 min

**Por que k-means++:** random captura o espaço por sorte; score_guided tem viés do DeepSVDD (ruído em HD); diversity garante cobertura máxima do espaço de embedding em N amostras.

---

## Passo 2 — v3 Pilot (1 seed = 42)

**Objetivo:** validar hipóteses antes de multiplicar seeds. Identificar quais configs valem o full.

| Parâmetro | Valor |
|---|---|
| Datasets | tweets_hs, told_br, 20_newsgroups, wikinews |
| Modelos | Qwen 2.5 7B Q4 + Qwen 2.5 14B Q4 (llamacpp) |
| Estratégias | random, score_guided, diversity |
| N | 50, 200 |
| Seed | 42 |
| Threshold | 0.45 |
| Prompt | v1 (mesmo de v1/v2) |
| Encoder | distiluse-base-multilingual-cased-v2 |
| Hardware | Colab T4 |

**Total de runs:** 4 × 2 × 3 × 2 = **48 runs (~2.5h Colab)**

**Resultados salvos em:** `data/llm_results/v3/`

### Hipóteses a validar no pilot

| Hipótese | Como verificar |
|---|---|
| Diversity supera random em tweets_hs | ROC diversity N=50 > ROC random N=50 |
| 14B não supera 7B sistematicamente | Comparar ROC médio 7B vs 14B por dataset |
| TC datasets são fáceis independente de N | ROC 20_newsgroups e wikinews estável entre N=50 e N=200 |
| told_br continua irrecuperável (~0.49) | Confirma que não vale 3 seeds |
| N=50 é suficiente (ou melhor que N=200) | Comparar ROC por N — custo-benefício |

---

## Passo 3 — Análise do Pilot

Antes de rodar o full, responder:

1. **Qual modelo usar no full?** 7B, 14B, ou ambos?
2. **Qual estratégia inclui no full?** Eliminar score_guided se diversity dominar
3. **told_br entra no full com 3 seeds?** Só se pilot mostrar ROC > 0.55 em alguma config
4. **Qual N usar?** Provavelmente só o melhor — não os dois

---

## Passo 4 — v3 Full (3 seeds)

Rodar só as configs validadas pelo pilot.

| Parâmetro | Valor |
|---|---|
| Seeds | 0, 1, 42 |
| Resto | Definido após análise do pilot |

**Resultado esperado:** tabela com média ± desvio por dataset/config. Esta é a tabela do paper.

**Custo estimado (worst case — tudo confirmado):**
4 datasets × 2 modelos × 2 estratégias × 2 N × 3 seeds = **96 runs (~5h Colab)**

---

## Estrutura esperada da tabela final (paper)

| Dataset | Tarefa | Lang | DeepSVDD (baseline) | Pipeline melhor (média ± std) | Delta |
|---|---|---|---|---|---|
| 20_newsgroups | TC | EN | ~0.96 | ? | ? |
| wikinews | TC | PT | ~0.79 | ? | ? |
| tweets_hs | HD | EN | ~0.55 | ? | ? |
| told_br | HD | PT | ~0.50 | ? | ? |

---

## Narrativa que os dados devem sustentar

> "O pipeline LLM zero-shot melhora detecção de anomalias quando o espaço de embedding tem separabilidade básica (DeepSVDD baseline > 0.55). Em tweets_hs (EN, hate explícito), o pipeline supera o baseline em +0.30 ROC sem nenhum rótulo humano. Em told_br (PT-BR, hate implícito), tanto o pipeline quanto o upperbound supervisionado (SetFit GT k=40, ROC ~0.69) confirmam que o problema é estrutural — a geometria do embedding não separa hate implícito PT-BR."

---

## Checklist

- [ ] Implementar `diversity` em `select_samples` (llm_runner.py)
- [ ] Testar `diversity` localmente com N=10 (sanity check)
- [ ] Rodar v3 pilot no Colab (seed=42)
- [ ] Analisar pilot — decidir configs do full
- [ ] Rodar v3 full (seeds 0, 1, 42)
- [ ] Montar tabela final com média ± std
- [ ] Comparar com DeepSVDD baseline e SetFit GT upperbound
