# Notas de sessão — v6 fix (TEMPORÁRIO, pode deletar após rerun)

## Problema identificado

Os arquivos `_llm_labels.csv` do **v5** contêm runs de **dois modelos empilhados** (Qwen 7B e Qwen 14B) no mesmo arquivo:

- 300 linhas por (strategy, N, dataset) = **6 run_ids × 50 labels** (3 seeds × 7B + 3 seeds × 14B)
- v6 usava `seed == X` para filtrar → retornava **100 linhas** (50 do 7B + 50 do 14B) em vez de 50
- Resultado: v6 atual está **contaminado** com labels misturadas dos dois modelos

## Fix implementado

### `scripts/run_llm_active_loop.py`
- Novo argumento `--load_labels_model` (ex: `"qwen2.5-14b"`)
- Campo `llm_model` no CSV de métricas agora usa `args.load_labels_model` quando carregando de arquivo (estava hardcoded como `"loaded_from_v5"`)

### `src/pipeline/llm_runner.py`
- Novo parâmetro `load_labels_model: Optional[str] = None`
- Lógica de filtro: lê o CSV de métricas companion (mesmo dir, sem `_llm_labels`), filtra por `llm_model.str.contains(tag)`, obtém os `run_id`s correspondentes e filtra o labels CSV por esses `run_id`s — **antes** do filtro por seed
- Docstring atualizado com os 3 novos parâmetros

## Decisão de modelo para v6

Usar **Qwen 14B** (vence em 3/4 datasets pelo v5 README):
- `20_newsgroups`: ≈ empate
- `tweets_hs`: +0.028 (14B)
- `wikinews`: +0.123 (14B)
- `hatebr`: +0.037 (7B ganha aqui, mas minoria)

## O que fazer antes de rodar v6

- [ ] **Limpar** os resultados atuais de `data/llm_results/v6/` (estão inválidos)
- [ ] **Atualizar** o notebook/script do v6 adicionando `"--load_labels_model", "qwen2.5-14b"` ao comando
- [ ] Certificar que o `--load_labels_from` aponta para os arquivos corretos do v5 (`score_guided` e `diversity`, N=50)
- [ ] Rerun todos os 96 runs do v6 no Colab

## Estrutura esperada do comando v6 (Cell 4 do Colab)

```python
cmd = [
    "python", "scripts/run_llm_active_loop.py",
    "--project_path", PROJECT_PATH,
    "--dataset", dataset,
    "--strategy", strategy,
    "--n_llm_calls", str(n),
    "--seed", str(seed),
    "--load_labels_from", <caminho_v5_labels_csv>,
    "--load_labels_model", "qwen2.5-14b",   # <-- NOVO
    "--no_setfit",
    "--results_dir", "data/llm_results/v6",
    # ... demais args
]
```

## Observação sobre v7+

A mesma lógica de `--load_labels_model` serve para qualquer experimento futuro que reutilize labels de arquivos com múltiplos modelos. Basta passar o tag correto (chave do `LLAMACPP_MODEL_MAP`).
