import pandas as pd, glob, re

files = glob.glob('data/llm_results/v5/**/*_llm_labels.csv', recursive=True)

rows = []
for f in files:
    df = pd.read_csv(f)
    f_norm = f.replace(chr(92), '/')
    dataset_match = re.search(r'N_\d+/(\w+)_llm_labels', f_norm)
    ds = dataset_match.group(1) if dataset_match else 'unknown'
    strategy = 'diversity' if 'diversity' in f_norm else 'random'
    N = int(re.search(r'N_(\d+)', f_norm).group(1))

    if 'llm_model' in df.columns:
        df['model_short'] = df['llm_model'].apply(lambda x: '7B' if '7b' in str(x).lower() else '14B')
        for (seed, model), g in df.groupby(['seed', 'model_short']):
            rows.append({
                'ds': ds, 'strategy': strategy, 'N': N,
                'seed': seed, 'model': model,
                'total_annotated': len(g),
                'llm_anomalies': int(g['llm_label'].sum()),
            })

rdf = pd.DataFrame(rows)
print('=== Anomalias LLM por run individual (mean ± std over seeds x strategies) ===')
agg = rdf.groupby(['ds', 'N', 'model'])['llm_anomalies'].agg(['mean', 'std']).round(1)
print(agg.to_string())

print()
print('=== Oracle: anomalias conhecidas (GT, 5% treino) ===')
bench = pd.read_csv('data/benchmark_results/benchmark_results.csv')
name_map = {
    'tweets_hate_speech_detection_distiluse-base-multilingual-cased-v2': 'tweets_hs',
    'HateBR_distiluse-base-multilingual-cased-v2': 'hatebr',
    '20_newsgroups_distiluse-base-multilingual-cased-v2': '20_newsgroups',
    'wikinews_distiluse-base-multilingual-cased-v2': 'wikinews',
}
bench['dataset'] = bench['dataset'].map(name_map)
print(bench[bench['model'] == 'MLP'][['dataset', 'n_known_outliers', 'train_size']].to_string())

print()
print('=== Oracle: anomalias conhecidas (GT, 5% treino) ===')
bench = pd.read_csv('data/benchmark_results/benchmark_results.csv')
name_map = {
    'tweets_hate_speech_detection_distiluse-base-multilingual-cased-v2': 'tweets_hs',
    'HateBR_distiluse-base-multilingual-cased-v2': 'hatebr',
    '20_newsgroups_distiluse-base-multilingual-cased-v2': '20_newsgroups',
    'wikinews_distiluse-base-multilingual-cased-v2': 'wikinews',
}
bench['dataset'] = bench['dataset'].map(name_map)
print(bench[bench['model'] == 'MLP'][['dataset', 'n_known_outliers', 'train_size']].to_string())
