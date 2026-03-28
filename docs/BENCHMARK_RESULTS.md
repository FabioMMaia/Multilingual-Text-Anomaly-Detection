# 📄 Paper Takeaways — STIL 2025

> **"Learning with Few: A Comparative Study of Multilingual Text Anomaly Detection"**  
> Submetido ao STIL 2025 (Simpósio Brasileiro de## 5. Conclusão Principal

> *"Semi-supervis## 7. Posicionamento do Próximo Paper

O próximo trabalho deve responder diretamente ao future work item (i):

> **"LLMs como geradores de rótulos fracos para detecção de anomalias em texto multilíngue"**

Condições propostas baseadas nos achados:
1. **Baseline unsupervised** (sem rótulos)
2. **N rótulos humanos aleatórios** → DevNet/MLP/XGBOD (resultado já estabelecido neste paper)
3. **N rótulos gerados por LLM** → DevNet/MLP/XGBOD (novo experimento)

Datasets prioritários: **TOLD-Br** (mais difícil, sem saturação) e **Hate Speech Tweets EN** (segundo mais difícil na classe HS).

Modelos foco: **MLP e DevNet** — os dois que aparecem consistentemente no topo e têm comportamentos complementares (MLP mais estável, DevNet mais sensível ao embedding).

> ⚠️ **EtaDevNet não entra no próximo experimento.** O benchmark STIL mostrou que ele não oferece vantagem consistente sobre DevNet padrão e é muito sensível ao hiperparâmetro eta. A contribuição do próximo paper está na estratégia de anotação (LLM labels), não na arquitetura do modelo AD.

A hipótese central:
> *Rótulos LLM — enriquecidos com raciocínio explícito e sensibilidade cultural — são mais informativos do que rótulos aleatórios para datasets semanticamente complexos, levando a AUC superior com o mesmo número de exemplos anotados.*as DevNet, XGBOD, and DeepSAD consistently outperformed unsupervised baselines. Multilingual encoders demonstrated greater robustness when labeled data was scarce, while language-specific models proved highly effective when more labels were available."*

Pelos dados reais:
- **MLP lidera** o ranking geral (AUC 0.884) — à frente de DevNet (0.881) e XGBOD (0.874)
- Supervisão mínima é suficiente para a maioria dos datasets
- **TOLD-Br é o outlier** — precisa de mais supervisão, mas não satura
- A qualidade da representação é crítica — bons embeddings + modelo simples > modelo complexo + embeddings ruins
- **EtaDevNet (paper anterior) não superou** os modelos base do STIL — o melhor EtaDevNet empata com DevNet padrão, mas com muito mais sensibilidade a hiperparâmetroogia da Informação e da Linguagem Humana)

---

## 1. Objetivo e Contribuição

O paper apresenta um **benchmark abrangente de detecção de anomalias em texto** sob dois regimes de supervisão (unsupervised e semi-supervised), com foco em:

- Cenários de **escassez de rótulos** (few-shot)
- **Diversidade linguística**: Inglês e Português
- **Diversidade de encoders**: multilinguais vs. específicos do Português
- **Diversidade de modelos**: 8 unsupervised + 4 semi-supervised

É uma extensão direta de [Maia & Costa, 2024], que aplicou semi-supervised AD em dois datasets portugueses.

---

## 2. Setup Experimental

### Datasets (6 no total)

| Dataset | Task | Lang | Size | Normal | Anomalia |
|---------|------|------|------|--------|---------|
| WikiNews (pt) | TC | PT | 10.581 | "Política" | demais categorias |
| 20 Newsgroups | TC | EN | 18.548 | "comp.graphics" | demais categorias |
| Portuguese Tweets | SA | PT | 60.000 | Negativo | Positivo |
| TweetEval | SA | EN | 59.899 | Neutral/Positive | Negative |
| TOLD-Br | HS | PT | 21.000 | Non-hate | Hate |
| Hate Speech Tweets | HS | EN | 31.962 | Non-hate | Hate |

- **Contaminação fixada em 5%** (proporção de anomalias no treino)
- Classe mais frequente = normal; demais = anomalia

### Encoders (6 no total)

| ShortName | Modelo | Tipo | Dim |
|-----------|--------|------|-----|
| distiluse-v1 | distiluse-base-multilingual-cased-v1 | Multilingual | 512 |
| distiluse-v2 | distiluse-base-multilingual-cased-v2 | Multilingual | 512 |
| XLM-RoBERTa | xlm-roberta-large | Multilingual | 1024 |
| BERT-base-PT | bert-base-portuguese-cased | Português | 768 |
| BERT-large-PT | bert-large-portuguese-cased | Português | 1024 |
| Serafim | serafim-100m-portuguese-pt | Português | 768 |

### Modelos (12 no total)

**Unsupervised:** OCSVM, IForest, LOF, HBOS, KDE, AE, VAE, DeepSVDD  
**Semi-supervised:** DeepSAD, DevNet, XGBOD, MLP

### Infraestrutura
- Python 3 + NVIDIA Tesla T4 (16 GB) + CUDA 12.4
- Scikit-learn + PyOD + DeepOD + wrappers customizados
- Múltiplos seeds → distribuição de AUC reportada

---

## 3. Resultados Quantitativos (dados reais do CSV)

### Ganho de supervisão por dataset (unsupervised → semi-supervised, média de todos modelos)

| Dataset | Task | Lang | AUC Unsup | AUC Semi | Δ |
|---------|------|------|-----------|----------|---|
| Tweets HS | HS | EN | 0.492 | 0.934 | **+0.443** |
| PT Tweets | SA | PT | 0.568 | 0.942 | **+0.374** |
| TOLD-Br | HS | PT | 0.454 | 0.718 | +0.264 |
| TweetEval | SA | EN | 0.478 | 0.712 | +0.234 |
| WikiNews | TC | PT | 0.717 | 0.930 | +0.212 |
| 20 Newsgroups | TC | EN | 0.800 | 0.991 | +0.191 |

### Ranking dos modelos semi-supervised por dataset (test_auc médio, n_known > 0)

| Dataset | 1º | 2º | 3º | 4º |
|---------|----|----|----|----|
| 20 Newsgroups | MLP **0.995** | DevNet 0.992 | DeepSAD 0.991 | XGBOD 0.987 |
| PT Tweets | MLP **0.982** | DevNet 0.940 | DeepSAD 0.933 | XGBOD 0.912 |
| Tweets HS | MLP **0.956** | XGBOD 0.943 | DevNet 0.933 | DeepSAD 0.905 |
| WikiNews | DevNet **0.948** | MLP 0.943 | XGBOD 0.937 | DeepSAD 0.890 |
| TOLD-Br | XGBOD **0.745** | DevNet 0.740 | MLP 0.722 | DeepSAD 0.666 |
| TweetEval | DevNet **0.747** | XGBOD 0.743 | MLP 0.715 | DeepSAD 0.642 |

### EtaDevNet vs modelos base (ranking geral, semi-supervised)

| Modelo | AUC médio | Posição |
|--------|-----------|---------|
| **MLP** | **0.884** | 🥇 |
| **DevNet** | **0.881** | 🥈 |
| **XGBOD** | **0.874** | 🥉 |
| EtaDevNetMC_eta_0.6 | 0.873 | 4º |
| EtaDevNet_eta_0.6 | 0.862 | 5º |
| EtaDevNet_eta_0.7 | 0.847 | 6º |
| EtaDevNetMC_eta_0.7 | 0.842 | 7º |
| EtaDevNetMC_eta_0.8 | 0.835 | 8º |
| **DeepSAD** | **0.835** | 8º |
| ... | ... | ... |
| EtaDevNetMC_eta_0.9 | 0.672 | último |

> ⚠️ **Nota importante sobre EtaDevNet:** O EtaDevNet era o modelo proposto no paper anterior (Maia & Costa, 2024). **Ele não se mostrou superior** aos modelos base do STIL — o melhor EtaDevNet (eta_0.6) empata ou fica levemente abaixo do DevNet padrão e do MLP. Além disso, apresenta **alta sensibilidade ao hiperparâmetro eta** — valores extremos (eta=0.9) colapsam a performance (ex: TweetEval 0.525, WikiNews 0.791). Isso descredencia EtaDevNet como contribuição robusta e reforça o foco nos modelos base para o próximo paper.

---

## 4. Resultados por RQ

### RQ1 — Supervisão melhora a detecção?

**✅ Sim, de forma consistente e estatisticamente significativa (p < 0.001, Wilcoxon).**

- Ganhos variam de **+0.19 (20ng) a +0.44 (Tweets HS)**
- O efeito é **maior em tarefas semanticamente complexas** (HS, SA) do que em TC
- Em TC (20ng, WikiNews), os unsupervised já funcionam razoavelmente — sinal lexical forte
- Em HS/SA, modelos unsupervised frequentemente **falham em capturar ambiguidade, ironia e hostilidade culturalmente específica**
- Datasets em Português mostram **maior variância e AUC base menor** que os em Inglês

> 💡 *Implicação:* A simples disponibilidade de alguns rótulos muda o jogo para tarefas difíceis — o custo marginal da anotação é alto em termos de qualidade, não de quantidade.

---

### RQ2 — Multilingual vs. encoders específicos do Português?

**〰️ Sem diferença estatisticamente significativa.**

- Semi-supervised: p = 0.77 (Mann-Whitney U)
- Unsupervised: p = 0.39
- XLM-RoBERTa (multilingual) e BERT-large-PT (específico) foram os mais fortes individualmente
- **Serafim foi o mais fraco** entre os encoders portugueses
- A variância intra-encoder é alta — os resultados dependem mais do **dataset e do modelo AD** do que do encoder

> 💡 *Implicação:* Não há vantagem sistemática de encoders específicos do Português. Isso **justifica usar distiluse-v2 como encoder padrão** nos experimentos SetFit.

---

### RQ3 — Quais modelos são mais robustos?

**🏆 MLP, DevNet e XGBOD dominam no regime semi-supervised.**

| Modelo | AUC médio | Destaque |
|--------|-----------|---------|
| **MLP** | **0.884** | Modelo mais simples — consistentemente no topo |
| **DevNet** | **0.881** | Forte em TC e HS; mais instável em SA difícil |
| **XGBOD** | **0.874** | Estável — combina unsupervised scores com XGBoost |
| DeepSAD | 0.835 | Bom, mas atrás dos três líderes |
| AE / VAE | ~0.60 | Unsupervised — desempenho intermediário |
| DeepSVDD | 0.549 | Fraco mesmo com supervisão |
| OCSVM / LOF | ~0.57–0.59 | Limitados |

- O sucesso do **MLP** (modelo mais simples) reforça que **representação de qualidade > complexidade arquitetural**
- Em TC: modelos unsupervised são mais viáveis (WikiNews unsup já em 0.717)
- Em SA/HS: supervisão é necessária para performance competitiva

---

### RQ4 — Quanto de supervisão é necessário para superar o unsupervised?

**Depende fortemente da complexidade da tarefa.**

Análise com DevNet progressivo (10 rótulos → 5% do treino), 5 seeds, 3 datasets PT:

| Dataset | Task | Saturação | Variância |
|---------|------|-----------|---------|
| **PT Tweets** | SA | < 1% de labels | Baixa — padrão aprendido rapidamente |
| **WikiNews** | TC | ~2–3% | Moderada — satura mais cedo e modestamente |
| **TOLD-Br** | HS | **Sem saturação até 5%** | **Alta — curva ruidosa e gradual** |

- PT Tweets: near-optimal AUC com menos de 1% de rótulos
- TOLD-Br: **nenhuma saturação visível até 5%**, alta variância até o fim
- Mesmo rótulos **aleatórios** já geram ganhos significativos

> 💡 *Implicação central:* **TOLD-Br é o caso mais difícil** — semanticamente ambíguo, culturalmente específico, label-hungry. É o dataset que mais se beneficiaria de rótulos de **melhor qualidade** (não só mais rótulos aleatórios).

---

## 4. Conclusão Principal

> *"Semi-supervised models such as DevNet, XGBOD, and DeepSAD consistently outperformed unsupervised baselines. Multilingual encoders demonstrated greater robustness when labeled data was scarce, while language-specific models proved highly effective when more labels were available."*

- Supervisão mínima é suficiente para a maioria dos datasets
- TOLD-Br é o outlier — precisa de mais supervisão, mas não satura
- A qualidade da representação é crítica — bons embeddings + modelo simples > modelo complexo + embeddings ruins

---

## 5. Trabalhos Futuros (explícitos no paper)

> *"(i) leveraging large language models (LLMs) to generate weak labels, assist with difficult cases, and enable real-time adaptation to new data"*

Este é o **ponto de partida direto** para o próximo experimento: usar LLMs como anotadores fracos para substituir ou complementar rótulos humanos — com foco especial em TOLD-Br e Hate Speech EN, os datasets mais difíceis identificados neste benchmark.

As demais direções futuras:
- (ii) análise além do nível de sentença (entidades, frases, documentos)
- (iii) extensão a mais línguas e domínios low-resource
- (iv) explorar objetivos de representation learning diferentes (contrastivo, attention-based)

> 💡 *O experimento SetFit já cobriu parcialmente o item (iv) — e confirmou que contrastivo + rótulos aleatórios não é suficiente para TOLD-Br.*

---

## 6. Conexão com os Experimentos SetFit

| Achado do Paper | Confirmado pelo SetFit? |
|----------------|------------------------|
| TOLD-Br não satura com 5% de rótulos | ✅ — AUC ~0.5 com k=20, ~0.62 com k=40 |
| Alta variância em TOLD-Br | ✅ — resultados instáveis especialmente no DevNet |
| DevNet é o modelo mais sensível à qualidade do embedding | ✅ — colapsa sem SetFit em 20ng; instável no k=40 em alguns datasets |
| Encoders multilinguais são suficientes | ✅ — distiluse-v2 (multilingual) mantém resultados competitivos |
| MLP é surpreendentemente competitivo | ✅ — MLP com SetFit é o mais estável nos experimentos |

---

## 7. Posicionamento do Próximo Paper

O próximo trabalho deve responder diretamente ao future work item (i):

> **"LLMs como geradores de rótulos fracos para detecção de anomalias em texto multilíngue"**

Condições propostas baseadas nos achados:
1. **Baseline unsupervised** (sem rótulos)
2. **N rótulos humanos aleatórios** → DevNet/MLP (resultado já estabelecido neste paper)
3. **N rótulos gerados por LLM** → DevNet/MLP (novo experimento)

Datasets prioritários: **TOLD-Br** (mais difícil, sem saturação) e **Hate Speech Tweets EN** (segundo mais difícil na classe HS).

A hipótese central:
> *Rótulos LLM — enriquecidos com raciocínio explícito e sensibilidade cultural — são mais informativos do que rótulos aleatórios para datasets semanticamente complexos, levando a AUC superior com o mesmo número de exemplos anotados.*
