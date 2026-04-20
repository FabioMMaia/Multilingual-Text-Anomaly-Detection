# PAPER ROADMAP
## Zero-Cost Supervision for Multilingual Text Anomaly Detection via Large Language Models

> Este é o guia de escrita do paper. Cada seção tem: o que escrever, quais dados/figuras usar, e o status atual.  
> Trabalhamos seção por seção — marque ✅ quando terminar cada item.

---

## Status Geral

| Seção | Status | Prioridade |
|---|---|---|
| Abstract | ⏳ escrever por último | — |
| 1. Introduction | 🔲 rascunho pendente | Alta |
| 2. Related Work | 🔲 rascunho pendente | Média |
| 3. Methodology | 🔲 rascunho pendente | Alta |
| 4. Experiments & Results | 🔲 tabelas + texto | Alta |
| 5. Conclusion | 🔲 rascunhar depois de 4 | Baixa |
| Figuras | 🔲 gerar scripts | Alta |
| Referências (refs.bib) | ✅ skeleton criado | verificar |

---

## Bloco 1 — Introduction

### O que escrever (5 parágrafos):
1. **Contexto + problema:** AD em texto é importante; labels são caros → a bottleneck
2. **O gap:** trabalho anterior ([stil]) mostrou que 5% oracle labels = grandes ganhos. Mas esses labels requerem anotação humana.
3. **Nossa proposta:** substituir o humano por LLM no loop de anotação — zero custo de anotação.
4. **Preview dos resultados:** 22–67% do gap unsup→oracle fechado; 15–78× menos anomalias que oracle.
5. **Contribuições + RQs**

### Dados necessários:
- Tabela de resultados finais (v8 vs unsup vs oracle) — já computada
- Número de datasets/runs para mencionar

### Decisão: **começar por aqui?**

---

## Bloco 2 — Related Work (3 sub-seções)

### 2.1 Text Anomaly Detection
- Citar: pang2021deep, xu-etal-2023-comparative, ruff-etal-2019-self, **stil** (nosso paper anterior)
- Enfatizar: trabalhos existentes assumem labels humanos

### 2.2 LLMs como Anotadores
- Citar: wang2021want, he2023annollm, zhang2024llmaad
- O que sabemos: LLMs bons para classificação; menos explorado para AD
- Diferencial nosso: loop ativo + combinação com modelos de AD

### 2.3 Few-Shot Embedding Fine-Tuning (SetFit)
- Citar: tunstall2022efficient
- Por que relevante: nossos labels são poucos e ruidosos — SetFit é o candidato natural

---

## Bloco 3 — Methodology

### 3.1 Problem Formulation ✅ (já no main.tex)

### 3.2 Pipeline — precisa de Figura 1
```
[ Corpus ] → [ Sampler (random/diversity) ] → [ LLM (Qwen 7B/14B via Groq) ]
    ↓                                               ↓
 Embeddings                                   Weak Labels
    ↓                                               ↓
[ SetFit fine-tune? ] ──────────────────→ [ MLP / DeepSAD ]
                                                    ↓
                                           [ Anomaly Scores ]
```
**TODO:** criar figura como diagrama em LaTeX (tikz) ou como PNG em `figs/pipeline.png`

### 3.3 Sampling Strategies ✅ (já no main.tex)

### 3.4 SetFit Fine-Tuning
- Descrever: quando há ≥ K anomalias LLM → fine-tune contrastivo
- Quando skipped: 20_newsgroups nunca tem exemplos suficientes
- Parâmetro: K mínimo (verificar no código)

### 3.5 Modelos AD ✅ (já no main.tex)

### 3.6 Datasets ✅ (tabela rascunho no main.tex)
- **Verificar:** números exatos de train/test split para a tabela
- Completar: contamination=5%, seeds=0/1/42

---

## Bloco 4 — Experiments & Results (4 RQs)

### Tabelas a criar:

#### Tabela Principal — Results (RQ1)
| Dataset | Unsup best | v5 DS+SF | v6 DS | v7 MLP | **v8 MLP+SF** | Oracle |
|---|---|---|---|---|---|---|
| 20_newsgroups | 0.920 | 0.879 | 0.850 | 0.937 | **0.937** | 0.998 |
| hatebr | 0.561 | 0.620 | 0.595 | 0.672 | **0.731** | 0.873 |
| tweets_hs | 0.575 | 0.791 | 0.806 | 0.819 | **0.829** | 0.956 |
| wikinews | 0.776 | 0.719 | 0.748 | 0.827 | **0.778** | 0.943 |
| Global | 0.708 | 0.752 | 0.750 | 0.814 | **0.819** | 0.942 |

> **Script:** `scripts/_overview.py` já calcula isso — adaptar para LaTeX

#### Tabela 2×2 (RQ2 + RQ3)
| | sem SetFit | com SetFit |
|---|---|---|
| DeepSAD | 0.750 (v6) | 0.752 (v5) |
| MLP | 0.814 (v7) | **0.819 (v8)** |

#### Tabela SetFit Skipped (RQ3)
| Dataset | N | SetFit ran | AUC ran | AUC skipped |
|---|---|---|---|---|
| ... | ... | ... | ... | ... |

#### Tabela Label Efficiency (RQ4)
| Dataset | Oracle anom. | Pipeline N=200 | Ratio | AUC gap |
|---|---|---|---|---|
| ... | ... | ... | ... | ... |

### Figuras a criar:

#### Figura 1 — Pipeline diagram
- `figs/pipeline.png` ou TikZ
- **TODO**

#### Figura 2 — Main results bar chart
- 1 gráfico por dataset: barras para unsup / v7 / v8 / oracle
- ou heatmap 2×2 com delta de AUC
- **Script a criar:** `scripts/fig_results.py`
- **TODO**

#### Figura 3 — SetFit effect (N=50 vs N=200, ran vs skipped)
- Scatter ou barras por dataset
- **Script a criar:** `scripts/fig_setfit.py`
- **TODO**

---

## Bloco 5 — Conclusion

Escrever depois das seções de resultado. Pontos principais:
- Pipeline fecha 22–67% do gap com zero anotação humana
- MLP + SetFit é a melhor configuração
- SetFit é condicional (precisa de anomalias LLM suficientes)
- Label efficiency: 15–78× razão oracle/pipeline, gap só 5–17pp

---

## Ordem de execução sugerida

```
1. [ ] Gerar Figura 1 (pipeline diagram)
2. [ ] Escrever Seção 3 — Methodology (com figura)
3. [ ] Gerar tabelas LaTeX dos resultados
4. [ ] Escrever Seção 4 — Results (RQ1–RQ4)
5. [ ] Escrever Seção 1 — Introduction
6. [ ] Escrever Seção 2 — Related Work
7. [ ] Escrever Seção 5 — Conclusion
8. [ ] Gerar Figura 2 e Figura 3
9. [ ] Escrever Abstract
10. [ ] Revisar referências (refs.bib)
11. [ ] Compilar PDF e ajustar layout
```

---

## Notas técnicas

- **Venue alvo:** STIL 2026 ou similar SBC — verificar deadline
- **Limite de páginas:** SBC normalmente 8–10pp
- **Idioma:** Inglês
- **Template:** SBC (`sbc-template.sty`) — mesmo do STIL 2025
- **Compilar:** `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`
- **Dados experimentais:** `scripts/_overview.py`, `scripts/_v8_paired.py`, `scripts/_label_analysis.py`
- **Modelos LLM usados:** Qwen 2.5 7B e 14B via Groq API

---

## Checklist final antes de submeter

- [ ] Abstract ≤ 10 linhas
- [ ] Figuras em alta resolução (≥ 300 dpi)  
- [ ] Todas as tabelas com `\caption` e `\label`
- [ ] Referências completas no refs.bib
- [ ] Compilação sem erros/warnings críticos
- [ ] Limite de páginas respeitado
- [ ] Acknowledgments atualizado
