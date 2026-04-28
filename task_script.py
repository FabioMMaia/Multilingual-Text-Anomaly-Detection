import pandas as pd, glob, os
df_meta = pd.read_csv('data/llm_results/consolidated_results.csv')
runs_7b  = df_meta[(df_meta['model_short']=='7b') & (df_meta['strategy']=='random') & (df_meta['n_llm_calls']==200) & (df_meta['role']=='deepsad_sf')]
runs_14b = df_meta[(df_meta['model_short']=='14b') & (df_meta['strategy']=='random') & (df_meta['n_llm_calls']==200) & (df_meta['role']=='deepsad_sf')]
print('7B run_ids:')
print(runs_7b[['run_id','dataset','seed','llm_precision','llm_recall','n_anomalies_found','roc_auc']].to_string())
print('\n14B run_ids:')
print(runs_14b[['run_id','dataset','seed','llm_precision','llm_recall','n_anomalies_found','roc_auc']].to_string())
