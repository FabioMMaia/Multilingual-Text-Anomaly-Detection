# Related Works

Materiais de referência para o paper *Zero-Cost Supervision for Multilingual Text Anomaly Detection via Large Language Models*.

---

## Estrutura

```
related_works/
├── ad_nlp/           → Papers sobre anomaly detection em texto (base do campo)
└── llm_ad/           → Papers sobre LLMs aplicados a anomaly/OOD detection
    ├── citations_from_original/   → Citados pelo paper STIL 2025 — relevância comprovada
    ├── scopus/                    → Resultado de busca Scopus (LLM + AD) — filtrar
    └── web_of_knowledge/          → Resultado de busca WoK (LLM + AD) — filtrar
```

---

## `ad_nlp/` — Text Anomaly Detection (base)

Estes papers formam o estado da arte em detecção de anomalias em texto. São os mais relevantes para o Related Work (§2.1).

| Paper | Relevância | Seção do paper |
|---|---|---|
| **Pang et al. (2021)** — *Deep Learning for Anomaly Detection: A Review* — ACM CSUR | Survey geral; taxonomia dos métodos (unsup / semi-sup / sup). Citação fundamental. | §2.1 + §1 |
| **Ruff et al. (2019)** — *Self-Attentive Multi-Context One-Class Classification* — ACL | Método CVDD: AD em texto com self-attention e embeddings estáticos. Baseline textual. | §2.1 |
| **Manolache et al. (2021)** — *DATE: Detecting Anomalies in Text via Self-Supervision* — NAACL | Self-supervised AD em texto via corrupção de tokens. Estado da arte em AD textual puro. | §2.1 |
| **Das et al. (2023)** — *Few-Shot Anomaly Detection in Text with Deviation Learning* | Estende DevNet para few-shot em texto. Diretamente relacionado — mesma família de modelos. | §2.1 + §3.5 |
| **Xu et al. (2023)** — *Comparative Analysis of AD Algorithms in Text* — RANLP | Benchmark de 22 algoritmos em 17 corpora. Contexto para comparação. | §2.1 |
| **Boutalbi et al. (2023)** — *ML for Text Anomaly Detection: A Systematic Review* | Survey sistemático específico de AD em texto. Boa cobertura do campo. | §2.1 |
| **Ait-Saada & Nadif (2023)** — *Unsupervised AD in Multi-Topic Short-Text Corpora* — EACL | AD não-supervisionado em corpora multi-tópico — caso de uso próximo ao nosso. | §2.1 |
| **Park (2022)** — *Comparative Study for Outlier Detection in High Dimensional Text* | Comparação de métodos clássicos em texto de alta dimensão. Baseline shallow. | §2.1 |
| **Novoa-Paradela et al. (2024)** — *Explained Anomaly Detection in Text Reviews* — EAAI | AD com explicabilidade em reviews. Menos central, mas útil para contextualizar. | Opcional |

---

## `llm_ad/citations_from_original/`

Papers que citam o STIL 2025 + foram selecionados manualmente como relevantes.

| Paper | Relevância |
|---|---|
| **Liu et al. (2024)** — *How Good Are LLMs at Out-of-Distribution Detection?* — LREC-COLING | LLMs para OOD/AD: avalia capacidade dos LLMs de detectar anomalias diretamente. **Muito relevante** — posiciona nossa abordagem (LLM como anotador, não como detector). |

---

## `llm_ad/scopus/` — Busca sistemática Scopus

Resultado de busca por (LLM + anomaly detection). Alguns resultados são ruído (domínio incorreto).

| Paper | Relevância | Usar? |
|---|---|---|
| **Yang et al. (2025)** — *AD-LLM: Benchmarking LLMs for Anomaly Detection* — ACL Findings | Benchmark de LLMs como detectores de anomalia. **Muito relevante** — posiciona diferenciação: nós usamos LLM como *anotador*, não como *detector*. | ✅ Sim |
| **Xu & Ding (2024)** — *LLMs for Anomaly and OOD Detection: A Survey* — arXiv 2409.01980 | Survey abrangente de LLMs para AD/OOD. Boa visão do estado da arte. **Citar no §2**. | ✅ Sim |
| **Fetaji & Samanta (2025)** — *Countering LLM-Generated Phishing via Semantic AD* | LLMs gerando ataques + AD para defesa. Domínio específico (cybersecurity). | ⚠️ Opcional |
| **Valadão et al.** — *Using LLMs for Audit in Notary Offices* | LLMs para detecção de fraude em documentos. Diferente do nosso foco. | ❌ Ruído |
| **Zhang** — *Abnormal Event Extraction via LLM Retrieval Enhancement* | Extração de eventos anômalos com RAG. Domínio diferente (event extraction). | ❌ Ruído |
| **Arambepola et al. (2025)** — *Systematic Review: Mobile App Review Analysis with LLMs* | Review analysis. Não é AD. | ❌ Ruído |
| **Anupriya et al.** — *LLM-Powered UPI Transaction Monitoring* | Fraud detection em transações financeiras. Não é texto. | ❌ Ruído |

---

## `llm_ad/web_of_knowledge/` — Busca sistemática Web of Knowledge

| Paper | Relevância | Usar? |
|---|---|---|
| **Li et al. (2022)** — *GPT-D: Inducing Dementia-related Linguistic Anomalies* — ACL | Usa LLM para *induzir* anomalias linguísticas (não detectar). Perspectiva inversa interessante. | ⚠️ Opcional |
| **de Giorgio et al.** — *Detecting LLMs in Exam Essays* | Detectar texto gerado por LLM. Diferente do foco — LLM como anomalia, não como ferramenta. | ❌ Ruído |
| **Song et al. (2024)** — *LUNA: Universal Analysis Framework for LLMs* — IEEE TSE | Framework de análise de LLMs (confiabilidade, segurança). Não é AD aplicado. | ❌ Ruído |
| **Fox et al.** — *Leverage LLMs for Enhanced Aviation Safety* | LLMs para segurança aviação. Domínio específico, metodologia diferente. | ❌ Ruído |
| **s41598-025-09138-0.pdf** | Não identificado completamente — ler manualmente. | ❓ Verificar |
| **Meticulous_Thought_Defender...** — *CoT for Detecting Prompt Injection* | Prompt injection detection com CoT. Diferente do foco. | ❌ Ruído |

---

## Resumo: papers para citar no paper

### Obrigatórios (§2 Related Work)
- Pang et al. 2021 (survey geral AD)
- Ruff et al. 2019 (CVDD — AD textual)
- Manolache et al. 2021 (DATE — AD textual)
- Das et al. 2023 (few-shot AD textual)
- Xu et al. 2023 (comparative benchmark AD)
- Yang et al. 2025 (AD-LLM — LLM como detector, diferente do nosso)
- Xu & Ding 2024 (survey LLMs para AD)
- Liu et al. 2024 (LLMs para OOD detection)

### Secundários (usar se houver espaço)
- Boutalbi et al. 2023 (systematic review AD texto)
- Ait-Saada & Nadif 2023 (unsupervised multi-topic text AD)
- Park 2022 (comparative high-dim text)
- Li et al. 2022 (GPT-D — perspectiva inversa)
