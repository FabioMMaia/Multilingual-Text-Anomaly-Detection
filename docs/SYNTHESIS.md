# 🔬 Síntese Crítica dos Experimentos — STIL 2025 + SetFit

> Este documento combina os achados do benchmark STIL 2025 com os experimentos SetFit (k=20, k=40)
> para extrair uma leitura crítica e integrada, identificando o que cada experimento prova, onde eles
> se contradizem, e o que ainda está em aberto.

---

## 1. Alinhamento dos Experimentos

Os dois experimentos **não são diretamente comparáveis** sem ajuste de contexto:

| Dimensão | STIL 2025 | SetFit (k=20/40) |
|----------|-----------|-----------------|
| Encoder | 6 encoders (distiluse-v1/v2, XLM-R, BERT-base-PT, BERT-large-PT, Serafim) | distiluse-v2 apenas |
| Fine-tuning | ❌ Nenhum — embeddings fixos | ✅ SetFit contrastivo com k amostras/classe |
| Modelos AD | DevNet, MLP, XGBOD, DeepSAD + 8 unsupervised | DevNet, MLP, DeepSAD |
| n_known (rótulos) | Todos os outliers do treino (ex: TOLD-Br = **470**) | 5, 10, 20, 50, 100, **200** |
| Seleção de rótulos | Aleatória | Aleatória |
| Datasets | 6 (inclui 20ng) | 5 (sem 20ng para n>20) |

A lacuna mais importante: o **STIL usa ~2–3x mais rótulos** do que o máximo testado no SetFit.

---

## 2. Comparação Direta — MLP, distiluse-v2, n máximo disponível

| Dataset | STIL MLP (470–1596 rótulos) | STIL best | SetFit Orig MLP (200) | SetFit FT MLP (200) | Δ FT−Orig |
|---------|----------------------------|-----------|----------------------|---------------------|-----------|
| PT Tweets | 0.982 | 0.994 | 0.857 | **1.000** | +0.143 |
| WikiNews | 0.943 | 0.965 | 0.931 | 0.939 | +0.007 |
| Tweets HS | 0.956 | 0.960 | 0.894 | 0.933 | +0.039 |
| TweetEval | 0.715 | 0.767 | 0.679 | 0.702 | +0.023 |
| **TOLD-Br** | **0.701** *(470 rótulos)* | **0.787** | **0.647** *(200 rótulos)* | **0.685** *(200 rótulos)* | **+0.038** |

### Leitura crítica

**PT Tweets:** SetFit FT com 200 rótulos **supera** STIL com 982 rótulos (1.000 vs 0.982). Fine-tuning contrastivo com poucos exemplos substitui completamente a necessidade de mais rótulos. ✅ SetFit vence claramente.

**WikiNews / Tweets HS:** STIL com muito mais rótulos tem vantagem pequena a moderada. SetFit FT recupera parte do gap mas não fecha completamente. Empate prático, STIL leva por margem.

**TweetEval:** Ambos fracos. Nenhuma das abordagens resolve esta tarefa — nem mais rótulos (STIL), nem fine-tuning (SetFit). Problema estrutural da tarefa.

**TOLD-Br — o caso mais revelador:**
- STIL com **470 rótulos aleatórios** → MLP 0.701
- SetFit FT com **200 rótulos aleatórios** → MLP 0.685
- Diferença: +0.016 a favor do STIL, mas com **mais que o dobro de rótulos**
- Fine-tuning contrastivo adiciona apenas +0.038 sobre o embedding original com 200 rótulos
- **Conclusão:** Nem mais rótulos nem fine-tuning contrastivo resolve TOLD-Br de forma satisfatória

---

## 3. O que cada experimento prova

### STIL 2025 prova:
1. ✅ Supervisão mínima melhora AD em texto — estatisticamente robusto (p < 0.001)
2. ✅ MLP, DevNet e XGBOD são os melhores modelos semi-supervised
3. ✅ Encoders multilinguais e PT-específicos são equivalentes (sem diferença significativa)
4. ✅ TOLD-Br não satura até 5% dos rótulos — alta variância, problema difícil
5. ❌ **Não testa** qualidade dos rótulos — usa sempre rótulos aleatórios

### SetFit prova:
1. ✅ Fine-tuning contrastivo com poucos exemplos pode ser mais eficiente que mais rótulos (PT Tweets)
2. ✅ Para tarefas simples/separáveis, SetFit com k=20 já atinge teto de performance
3. ✅ Aumentar k=20→k=40 ajuda em casos difíceis (TOLD-Br DevNet: 0.469→0.608)
4. ⚠️ DevNet é instável com SetFit em alguns datasets — colapso de performance
5. ❌ **Não resolve** TOLD-Br de forma satisfatória mesmo com k=40 (máx ~0.685)
6. ❌ **Não testa** qualidade dos rótulos — ainda usa seleção aleatória para o fine-tuning

---

## 4. A Tensão Central — O que ambos deixam em aberto

Ambos os experimentos partem da mesma premissa implícita:

> *"Os rótulos disponíveis são uma amostra aleatória dos dados reais"*

Nenhum dos dois testa o que acontece quando os **rótulos têm qualidade superior** — isto é, quando são selecionados ou gerados de forma mais informativa.

### O problema de TOLD-Br em números

| Abordagem | # Rótulos | MLP AUC | Eficiência (AUC/rótulo) |
|-----------|-----------|---------|-------------------------|
| STIL random | 470 | 0.701 | 0.00149 |
| SetFit FT random k=40 | 200 (fine-tuning) + 200 (n_known) | 0.685 | ~0.00171 |
| SetFit FT random k=20 | 200 (fine-tuning apenas) | 0.689 | ~0.00345 |
| **Teto observado** | **470** | **0.787** *(best model, XGBOD)* | — |

O teto observado de 0.787 com XGBOD e 470 rótulos ainda é **insatisfatório** para uma tarefa de moderação de conteúdo. A questão é: o problema é **quantidade** (precisamos de mais rótulos?) ou **qualidade** (os rótulos aleatórios não ensinam os casos difíceis?).

### Evidência que aponta para qualidade

1. **Sem saturação até 5%** (STIL RQ4): se fosse só quantidade, veríamos saturação
2. **Alta variância** mesmo com muitos rótulos: o modelo fica instável — sinal de que os rótulos não discriminam bem os casos difíceis
3. **SetFit k=20→k=40**: dobrar os exemplos de fine-tuning muda DevNet de -0.048 para +0.091 em n=5 — mas o impacto diminui com mais rótulos (n=200: +0.038). Fine-tuning com mais exemplos aleatórios tem retornos decrescentes
4. **DevNet instabilidade**: o modelo mais sensível ao alinhamento de embedding colapsa quando o espaço contrastivo não reflete a fronteira real — sinal de que os exemplos de treino SetFit não capturam o sinal discriminativo correto

---

## 5. Síntese — Diagrama de Progressão dos Experimentos

```
EXPERIMENTO 1: STIL 2025
  ↓ Pergunta: Supervisão ajuda?
  ↓ Resposta: Sim, muito. Mas rótulos são aleatórios.
  ↓ Gap: TOLD-Br não satura → qualidade dos rótulos é o gargalo?

EXPERIMENTO 2: SetFit (k=20, k=40)
  ↓ Pergunta: Fine-tuning contrastivo com poucos exemplos ajuda?
  ↓ Resposta: Sim, para tarefas separáveis. Para TOLD-Br, ajuda pouco.
  ↓ Gap: Os k exemplos de fine-tuning ainda são aleatórios.
  ↓ Novo sinal: k=40 ajuda TOLD-Br — quantidade importa ALGUM POUCO.
  ↓ Conclusão: Não é só quantidade. A seleção/qualidade dos exemplos é crítica.

EXPERIMENTO 3 (próximo): LLM como anotador
  ↓ Pergunta: Rótulos gerados por LLM (com raciocínio) superam rótulos aleatórios?
  ↓ Setup: Mesmo pipeline, mesmos modelos, mesma quantidade de rótulos.
  ↓ Variável: origem do rótulo (humano aleatório vs LLM raciocínio explícito)
  ↓ Hipótese: LLM captura nuance cultural/semântica que amostra aleatória não captura.
```

---

## 6. O Que o Próximo Experimento Precisa Controlar

Para que o experimento LLM seja conclusivo e publicável:

| Variável | Como controlar |
|----------|---------------|
| **Quantidade de rótulos** | Fixar N igual para humano-aleatório e LLM (ex: N=50, N=100, N=200) |
| **Encoder** | Fixar distiluse-v2 (ou testar XLM-R também) |
| **Modelos AD** | Fixar MLP + DevNet (os mais sensíveis à qualidade do input) |
| **Datasets** | Priorizar TOLD-Br e Tweets HS — os que não saturaram |
| **Baseline** | Incluir: (a) unsupervised, (b) N rótulos aleatórios, (c) N rótulos LLM |
| **LLM** | Usar AD-LLM Setting 2 (com descrição da anomalia) para TOLD-Br |
| **Repetições** | Mínimo 5 seeds por condição |

### Métricas de sucesso para publicação

> O experimento é positivo se: **AUC(LLM labels) > AUC(random labels)** com o mesmo N, em TOLD-Br e/ou Tweets HS, com efeito consistente em pelo menos 2 modelos.

> O experimento é negativo (mas ainda publicável) se: LLM não supera aleatório → conclusão é que "para AD semi-supervised, a qualidade do rótulo importa menos do que a quantidade" — resultado inesperado e igualmente relevante.

---

## 7. Problemas Estruturais do TOLD-Br

O TOLD-Br merece uma análise crítica separada porque concentra **dois problemas independentes** que se reforçam mutuamente — e ambos afetam diretamente a interpretação dos resultados.

### 7.1 — O critério de anomalia por frequência é problemático

A convenção adotada no paper é: **classe mais frequente = normal, todo o resto = anomalia**. Para a maioria dos datasets isso funciona bem porque a classe majoritária tem um sinal semântico coerente:

| Dataset | Normal (frequente) | Anomalia | Coerência da fronteira |
|---------|-------------------|----------|----------------------|
| 20 Newsgroups | comp.graphics | outros tópicos | ✅ Alta — diferenças lexicais claras |
| PT Tweets | Negativo | Positivo | ✅ Alta — sentimento bem definido |
| WikiNews | Política | outras editorias | ✅ Alta — fronteira temática nítida |
| **TOLD-Br** | **Non-hate** | **Hate (todos os tipos)** | ❌ **Baixa — múltiplos subtipos colapsados** |

No TOLD-Br, a categoria "anomalia" agrupa **subtipos heterogêneos de ódio**: misoginia, homofobia, xenofobia, racismo, capacitismo, etc. Cada subtipo ocupa uma região diferente do espaço de embedding — um tweet misógino e um tweet racista não são semelhantes entre si, apenas semelhantes ao "não-ofensivo" por contraste. Isso **polui o sinal** que os modelos semi-supervised tentam aprender: com 50 rótulos aleatórios, o modelo pode receber 30 exemplos de misoginia e 20 de racismo — e construir uma fronteira totalmente diferente do que se recebesse 50 exemplos de xenofobia.

É provável que isso explique diretamente a **alta variância entre seeds** observada no RQ4: cada seed sorteia uma composição diferente de subtipos, resultando em fronteiras de decisão inconsistentes.

### 7.2 — A qualidade intrínseca dos rótulos é baixa

O TOLD-Br foi construído via crowdsourcing (Twitter/Reddit) com anotação multi-label, e carrega problemas documentados:

- **Baixo acordo entre anotadores** para casos limítrofes — ironia, código velado, gírias PT-BR
- **Viés cultural e regional** — o que é considerado ofensivo varia por contexto, e o dataset mistura anotadores de diferentes origens
- **Fenômeno contínuo forçado em binário** — hate speech não é uma categoria discreta, e a binarização non-hate/hate descarta toda a gradação de intensidade

Na prática: parte dos exemplos rotulados como "hate" foram contestados entre anotadores humanos. O modelo recebe um rótulo com ruído intrínseco — não porque o pipeline de AD seja ruim, mas porque **o próprio ground truth é ruidoso**.

### 7.3 — Interação entre os dois problemas

```
Critério por frequência          Rótulos ruidosos
       ↓                                ↓
"Hate" = categoria heterogênea    ground truth contestável
       ↓                                ↓
         Sinal de supervisão fraco e inconsistente
                        ↓
         Alta variância entre seeds (observado)
         Ausência de saturação até 5% (observado)
         SetFit k=40 ajuda pouco (observado)
```

Nenhum desses problemas é resolvido por **mais rótulos aleatórios** — o que explicaria a ausência de saturação. A questão passa a ser: um LLM consegue superar esses dois problemas ao mesmo tempo?

- Para o **problema de heterogeneidade**: um LLM com prompt bem desenhado pode identificar o subtipo de ódio, fornecendo um sinal mais específico e coerente
- Para o **problema de ruído no rótulo**: um LLM que raciocina sobre contexto pode ser mais consistente que anotadores humanos em casos ambíguos — ou pode amplificar o viés

> ⚠️ **Implicação para o próximo experimento:** Se o LLM não superar rótulos aleatórios em TOLD-Br, a causa pode não ser a estratégia de anotação em si — mas a insolubilidade estrutural da tarefa sob o critério de anomalia por frequência. Isso é um resultado em si mesmo, e precisa ser reportado com essa interpretação.

---

## 8. Resumo Executivo das Lacunas

| # | Lacuna identificada | Experimento que a revelou | Como o próximo experimento aborda |
|---|---------------------|--------------------------|----------------------------------|
| L1 | Rótulos aleatórios não saturam TOLD-Br | STIL RQ4 | Substitui aleatório por LLM |
| L2 | Fine-tuning contrastivo com exemplos aleatórios tem retornos decrescentes | SetFit k=20→k=40 | LLM melhora qualidade dos exemplos do SetFit |
| L3 | DevNet instável no espaço SetFit — sinal de embedding mal alinhado | SetFit comparação k | LLM pode fornecer exemplos mais representativos da fronteira |
| L4 | Nenhum experimento testou qualidade de rótulo | Ambos | Variável principal do próximo paper |
| L5 | EtaDevNet não superou baseline simples (MLP) | STIL CSV | Descartado — foco em LLM como contrib. do próximo paper |
| **L6** | **Critério de anomalia por frequência colapsa subtipos heterogêneos** | **Análise crítica TOLD-Br** | **Considerar estratificação por subtipo no próximo paper** |
| **L7** | **Rótulos do TOLD-Br têm baixo acordo entre anotadores** | **Análise crítica TOLD-Br** | **LLM pode ser mais consistente — ou amplificar viés** |
