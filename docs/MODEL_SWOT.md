# LLM Annotator — SWOT por Modelo

> **Contexto de avaliação:** anotação de hate speech (PT-BR e EN) em pipeline de anomaly detection.
> Tarefa: classificar textos como hate/non-hate via prompt zero-shot/few-shot, threshold 0.45.
> Hardware alvo: Colab T4 (15 GB VRAM) para modelos locais; API para o resto.

---

## Visão Geral — Tabela Comparativa

| Modelo | Parâmetros | Custo | VRAM (Q4) | PT-BR | Implicit HS | JSON | Velocidade |
|---|---|---|---|---|---|---|---|
| Qwen 2.5 7B | 7B | Grátis (local) | ~4.7 GB | ★★★★ | ★★☆ | ★★★★ | ~15 tok/s (T4) |
| Qwen 2.5 14B | 14B | Grátis (local) | ~9.5 GB | ★★★★ | ★★★☆ | ★★★★ | ~8 tok/s (T4) |
| Llama 3.3 70B | 70B | Grátis (Groq, 1K/day) | N/A | ★★★☆ | ★★★★ | ★★★★ | ~200 tok/s |
| GPT-4o-mini | ~8B MoE | ~$0.03/run (N=200) | N/A | ★★★★ | ★★★★ | ★★★★★ | ~300 tok/s |
| **Gemini 2.5 Flash-Lite** | não divulgado¹ | **Grátis (1.5K/day)** | N/A | ★★★★ | ★★★☆ | ★★★★ | ~366 tok/s |

> ¹ Google não divulga contagem de parâmetros para modelos Gemini (proprietários). A família Flash-Lite é descrita como "distilada" — estimativas da comunidade: 8–30B. O indicador prático é o MMMLU score: **84.5%** em 57 idiomas, incluindo PT-BR.

---

## SWOT por Modelo

### 1. Qwen 2.5 7B Instruct Q4_K_M ← modelo atual

| | |
|---|---|
| **Strengths** | ✔ Zero custo, roda 100% offline no Colab T4  ✔ Melhor multilingual 7B no Open LLM Leaderboard (2024)  ✔ JSON instruction-following consistente  ✔ 4.7 GB VRAM — cabe na T4 com folga para DeepSAD |
| **Weaknesses** | ✘ Hate speech implícito/irônico em PT-BR não dispara threshold  ✘ Calibrado para segurança → conservador demais em HD  ✘ Recall ~0 em told_br nos testes originais  ✘ Sem treinamento específico em hate speech |
| **Opportunities** | → Prompt melhorado (v2) + threshold 0.45 pode resolver parcialmente  → Serve como baseline local para comparação com APIs  → Se funcionar, argumento de "zero custo, zero infraestrutura" é forte para a tese |
| **Threats** | → Resultados frágeis: único caso de sucesso (tweets_hs N=200) pode não replicar  → Com múltiplos seeds, variância pode ser alta |

**Como obter:** HuggingFace — [Qwen/Qwen2.5-7B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF)
```bash
# Arquivo recomendado (~4.7 GB):
qwen2.5-7b-instruct-q4_k_m.gguf
```

---

### 2. Qwen 2.5 14B Instruct Q4_K_M

| | |
|---|---|
| **Strengths** | ✔ Mesma família do 7B — prompt idêntico, sem mudança de código  ✔ Melhor raciocínio em textos ambíguos (comparado ao 7B)  ✔ Maior precision em wikinews/pt_tweets  ✔ 9.5 GB → ainda cabe na T4 |
| **Weaknesses** | ✘ **Mais conservador que o 7B para HD**: encontrou menos anomalias, não mais  ✘ ~2x mais lento que o 7B (8 tok/s vs 15 tok/s)  ✘ Não confirmou hipótese "maior = melhor recall" |
| **Opportunities** | → Vale re-testar com prompt v2 + threshold 0.45 — pode mudar comportamento  → Útil como comparação local "low cost" vs. APIs |
| **Threats** | → Gasta o dobro do tempo de runtime no Colab  → Se 7B funcionar com prompt v2, o 14B não acrescenta muito |

**Como obter:** HuggingFace — [Qwen/Qwen2.5-14B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF)
```bash
# Arquivo recomendado (~9.5 GB — verifica VRAM disponível):
qwen2.5-14b-instruct-q4_k_m.gguf
```

---

### 3. Llama 3.3 70B Versatile — via Groq

| | |
|---|---|
| **Strengths** | ✔ **70B parâmetros** → melhor compreensão de hate implícito entre os testados  ✔ ~200 tok/s via API — muito mais rápido que qualquer local  ✔ Suporte a PT (Meta multilingual training)  ✔ Backend já implementado no `LLMAnnotator` |
| **Weaknesses** | ✘ **Só 1.000 req/day no free tier** (não 14.400 — esse era do 8B)  ✘ TPM 12K → exige delay de 8s entre chamadas  ✘ N=200 × 8s = ~27min por run; grade completa em vários dias  ✘ Modelo não é open-weight nessa versão — não auditável localmente |
| **Opportunities** | → Maior qualidade provável em hate speech implícito vs. modelos menores  → Com cache de anotações, 1.000 calls/day são suficientes para anotar N=500 todo o corpus de uma vez  → Groq também oferece Llama 3.1 8B (14.400 RPD) para comparação de escala |
| **Threats** | → Rate limit diário (1K) esgota rápido se houver bugs na execução  → Groq pode mudar free tier a qualquer momento  → Se qualidade < Gemini 2.5 Flash-Lite, perdeu o único diferencial (tamanho) |

**Como obter:** [console.groq.com](https://console.groq.com) → Sign up → API Keys → gerar chave
```bash
# No .env do projeto:
GROQ_API_KEY=gsk_...

# No runner:
--backend groq --llm_model llama-3.3-70b-versatile
```
> O modelo padrão do backend `groq` já é `llama-3.3-70b-versatile` — não precisa passar `--llm_model`.

---

### 4. GPT-4o-mini — via OpenAI

| | |
|---|---|
| **Strengths** | ✔ Melhor instruction-following do mercado para JSON estruturado  ✔ Excelente compreensão de hate implícito, ironia, coded language em PT e EN  ✔ response_format JSON nativo — zero parse errors  ✔ $10 = ~300+ runs de N=200 → grade completa com múltiplos seeds |
| **Weaknesses** | ✘ Único modelo pago da lista  ✘ Pode ser excessivamente cauteloso com HD (RLHF de segurança agressivo)  ✘ Requer cartão de crédito cadastrado |
| **Opportunities** | → Serve como ceiling: se GPT-4o-mini falhar, nenhum outro modelo vai funcionar → conclusão forte  → Comparação "SLM local 7B vs. SOTA API" é argumento direto para a tese  → Backend já implementado |
| **Threats** | → Filtros de segurança podem recusar amostras de hate speech reais → parse_error artificial  → Custo cresce se decidir expandir para N=500 ou mais seeds |

**Como obter:** [platform.openai.com](https://platform.openai.com) → Billing → Add credit ($10 mínimo)
```bash
# No .env do projeto:
OPENAI_API_KEY=sk-...

# No runner:
--backend openai --llm_model gpt-4o-mini
```

---

### 5. Gemini 2.5 Flash-Lite — via Google AI Studio ⭐ alternativa ao Groq

| | |
|---|---|
| **Strengths** | ✔ **Grátis: ~1.500 req/day, 1M TPM** — limite diário superior ao Groq 70B  ✔ **Brasil suportado oficialmente** — e Colab usa região EUA, sem problema  ✔ Parâmetros não divulgados, mas **MMMLU 84.5%** em 57 idiomas (PT-BR incluso)  ✔ 366 tok/s — mais rápido que todos os locais  ✔ Delay necessário só 3s (30 RPM free tier) vs. 8s do Groq  ✔ Structured outputs (JSON) nativos  ✔ Contexto 1M tokens — sem risco de truncagem |
| **Weaknesses** | ✘ Parâmetros desconhecidos — não auditável como open-weight  ✘ Qualidade em hate speech implícito PT-BR inferior ao Llama 70B (menor escala)  ✘ **2.0 Flash-Lite está deprecated** (desliga 1 jun 2026) — usar apenas 2.5 |
| **Opportunities** | → **Volume total por dia**: 1.500 calls/day × N=200 = 7 runs completos/dia  → Com cache de anotações, 1 run de anotação (N=400) cobre toda a grade  → Se funcionar, tem o argumento "multilingual API grátis de alta escala" para a tese  → MMMLU 84.5% sugere boa cobertura de PT-BR coloquial e gírias |
| **Threats** | → Google pode mudar política de free tier  → Qualidade menor que 70B para hate speech implícito (hipótese não confirmada)  → Sem parâmetro oficial de escala para citar na tese |

**Escala e multilingual (o que se sabe):**
- Parâmetros: **não divulgado** (Google mantém proprietário para toda a família Gemini)
- MMMLU (57 idiomas, Q&A multilingual): **84.5%** — supera Llama 3.1 8B (83%), comparável ao Llama 3.3 70B (84.9%)
- PT-BR: incluso no treinamento multilingual do Google desde o início da família Gemini
- Velocidade medida: **366 tok/s** (benchmark oficial deepmind.google)

**Como obter:** [aistudio.google.com](https://aistudio.google.com) → Get API Key (grátis, sem cartão)
```bash
GEMINI_API_KEY=AIza...
--backend gemini
# modelo padrão já é gemini-2.5-flash-lite (atualizado no código)
```

---

## Recomendação para o Projeto

```
Rodada atual — Groq Llama 3.3 70B (em andamento)
  Modelo : Llama 3.3 70B (Groq, 1K req/day)
  Config : told_br + tweets_hs, N∈{50,100,150,200}, random + score_guided, seed=42
  Risco  : rate limit 1K/day — cache de anotações resolve (anotar uma vez, treinar N vezes)

Próxima rodada — SE Groq for lento/instável
  Modelo : Gemini 2.5 Flash-Lite (Google AI Studio, grátis, 1.5K req/day)
  Config : mesma grade
  Vantagem: 1M TPM → delay de 3s (vs 8s Groq); Brasil suportado; Colab sem VPN
  Custo  : zero

Rodada parallela — Ablation de escala (Groq 8B)
  Modelo : llama-3.1-8b-instant (Groq, 14.4K req/day!)
  Config : told_br, N=200, random, seed 42
  Objetivo: comparar 8B vs 70B → quantificar ganho de escala no pipeline
  Custo  : zero

Rodada final — SE APIs falharem ou para ceiling
  Modelo : GPT-4o-mini (OpenAI)
  Config : told_br + tweets_hs, N=200, 3 seeds
  Objetivo: ceiling — se GPT-4o-mini falhar, conclusão é argumento da tese
  Custo  : ~$2–4 dos $10 disponíveis
```

---

## Outros modelos no Groq (free tier) para ablation futura

| Modelo | Parâmetros | PT-BR | RPD (free) | Uso sugerido |
|---|---|---|---|---|
| `llama-3.1-8b-instant` | 8B | ★★★☆ | **14.400** | Anotação em volume barato; ablation de escala vs. 70B |
| `meta-llama/llama-4-scout-17b-16e-instruct` | 17B×16E MoE | ★★★☆ | 1.000 | Alternativa arquitetura MoE se Llama 70B falhar |
| `qwen/qwen3-32b` | 32B | ★★★★ | 1.000 | Boa cobertura PT-BR; 60 RPM (mais generoso) |

> **Atenção ao limite real:** só o `llama-3.1-8b-instant` tem 14.400 RPD. Os demais modelos Groq têm 1.000 RPD no free tier — o mesmo que o 70B.
