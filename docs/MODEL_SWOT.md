# LLM Annotator — SWOT por Modelo

> **Contexto de avaliação:** anotação de hate speech (PT-BR e EN) em pipeline de anomaly detection.
> Tarefa: classificar textos como hate/non-hate via prompt zero-shot/few-shot, threshold 0.45.
> Hardware alvo: Colab T4 (15 GB VRAM) para modelos locais; API para o resto.

---

## Visão Geral — Tabela Comparativa

| Modelo | Parâmetros | Custo | VRAM (Q4) | PT-BR | Implicit HS | JSON | Velocidade (T4) |
|---|---|---|---|---|---|---|---|
| Qwen 2.5 7B | 7B | Grátis (local) | ~4.7 GB | ★★★★ | ★★☆ | ★★★★ | ~15 tok/s |
| Qwen 2.5 14B | 14B | Grátis (local) | ~9.5 GB | ★★★★ | ★★★☆ | ★★★★ | ~8 tok/s |
| Llama 3.3 70B | 70B | Grátis (Groq API) | N/A | ★★★☆ | ★★★★ | ★★★★ | ~200 tok/s |
| GPT-4o-mini | ~8B MoE | ~$0.03/run (N=200) | N/A | ★★★★ | ★★★★ | ★★★★★ | ~300 tok/s |
| Gemini 2.0 Flash | ~? | Grátis* | N/A | ★★★☆ | ★★★☆ | ★★★★ | ~400 tok/s |

> *Gemini free tier geo-restrito no Brasil — pode não funcionar sem VPN.

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

### 3. Llama 3.3 70B Versatile — via Groq ⭐ recomendado como próximo passo

| | |
|---|---|
| **Strengths** | ✔ **Grátis: 14.400 req/day, sem cartão de crédito**  ✔ 70B parâmetros → muito melhor compreensão de hate implícito  ✔ ~200 tok/s via API — mais rápido que qualquer local  ✔ Suporte oficial a PT (Meta multilingual training)  ✔ Backend já implementado no `LLMAnnotator` |
| **Weaknesses** | ✘ Rate limit: 6.000 tokens/min → N=200 leva ~3–5 min (aceitável)  ✘ Requer conexão à internet no Colab  ✘ Modelo não é open-weight nessa versão — não auditável localmente  ✘ PT-BR mais fraco que Qwen em benchmarks específicos |
| **Opportunities** | → Se funcionar, você tem resultado "modelo grande gratuito vs. SLM local" — contribuição real  → 14.400 req/day = ~72 runs de N=200/dia → grade completa em 2–3 dias grátis  → Groq também oferece Gemma 2 9B e Mixtral 8x7B para ablation |
| **Threats** | → Rate limit pode causar esperas se rodar muitos runs em paralelo  → Groq pode mudar free tier a qualquer momento |

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

### 5. Gemini 2.0 Flash Lite — via Google AI Studio

| | |
|---|---|
| **Strengths** | ✔ Grátis no free tier (15 req/min, 1M tokens/day)  ✔ Muito rápido via API  ✔ Bom instruction-following para JSON |
| **Weaknesses** | ✘ **Free tier bloqueado no Brasil** — requer VPN ou conta com billing fora do BR  ✘ PT-BR menos robusto que Qwen/GPT em hate speech  ✘ Comportamento menos previsível em prompts longos |
| **Opportunities** | → Se você tiver VPN ou conta com endereço fora do BR, é uma opção válida gratuita |
| **Threats** | → Geo-restrição torna inviável sem workaround  → Google pode mudar política de free tier |

**Como obter:** [aistudio.google.com](https://aistudio.google.com) → Get API Key
```bash
GEMINI_API_KEY=AIza...
--backend gemini
```

---

## Recomendação para o Projeto

```
Rodada 1 — HOJE
  Modelo : Qwen 2.5 7B (local, llamacpp)
  Config : told_br, N=200, random + score_guided, seed=42
  Objetivo: validar se prompt v2 + threshold 0.45 resolve o setfit_skipped
  Custo  : zero

Rodada 2 — SE RODADA 1 MELHORAR MAS NÃO RESOLVER
  Modelo : Llama 3.3 70B (Groq, grátis)
  Config : told_br + tweets_hs, N=200, random, seeds 42/0/1
  Objetivo: confirmar se é problema de modelo ou de prompt
  Custo  : zero (criar conta em console.groq.com)

Rodada 3 — SE GROQ FUNCIONAR
  Modelo : Llama 3.3 70B (Groq)
  Config : grade completa — N ∈ {100, 200, 300}, 5 seeds
  Objetivo: resultados finais para a tese
  Custo  : zero

Rodada 4 — SOMENTE SE GROQ FALHAR
  Modelo : GPT-4o-mini (OpenAI)
  Config : told_br + tweets_hs, N=200, 3 seeds
  Objetivo: ceiling — se GPT-4o-mini falhar, conclusão é argumento da tese
  Custo  : ~$2–4 dos $10 disponíveis
```

---

## Outros modelos no Groq (free tier) para ablation futura

| Modelo | Parâmetros | PT-BR | Uso sugerido |
|---|---|---|---|
| `gemma2-9b-it` | 9B | ★★★☆ | Comparação SLM via API (sem custo de GPU) |
| `mixtral-8x7b-32768` | 47B MoE | ★★★☆ | Alternativa se Llama 3.3 atingir rate limit |
| `llama-3.1-8b-instant` | 8B | ★★★☆ | Baseline leve — comparar com Qwen 7B local |

> Todos os modelos Groq são free tier, sem cartão de crédito. Limite compartilhado de 14.400 req/day.
