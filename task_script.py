import pandas as pd

labels = pd.read_csv('data/llm_results/v5/random/N_50/tweets_hs_llm_labels.csv')
metrics = pd.read_csv('data/llm_results/v5/random/N_50/tweets_hs.csv')

print('=== _llm_labels.csv ===')
print('Total linhas:', len(labels))
print('run_ids únicos:', labels['run_id'].nunique())
print('linhas por run_id:')
print(labels.groupby('run_id').size())

print()
print('=== metrics.csv (run_id -> seed -> llm_model) ===')
print(metrics[['run_id','seed','llm_model']].to_string())
