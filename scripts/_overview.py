import pandas as pd, glob

def load_v(version):
    files = [f for f in glob.glob('data/llm_results/' + version + '/**/*.csv', recursive=True) if 'llm_labels' not in f]
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

df5 = load_v('v5')
df6 = load_v('v6')
df7 = load_v('v7')

# Normalize llm_model to short name
df5['model_short'] = df5['llm_model'].apply(lambda x: '7B' if '7b' in str(x).lower() else '14B')

bench = pd.read_csv('data/benchmark_results/benchmark_results.csv')
name_map = {
    'tweets_hate_speech_detection_distiluse-base-multilingual-cased-v2': 'tweets_hs',
    'HateBR_distiluse-base-multilingual-cased-v2': 'hatebr',
    '20_newsgroups_distiluse-base-multilingual-cased-v2': '20_newsgroups',
    'wikinews_distiluse-base-multilingual-cased-v2': 'wikinews',
}
bench['dataset'] = bench['dataset'].map(name_map)

unsup_models  = ['IForest','LOF','DeepSVDD','OCSVM','AutoEncoder','VAE','HBOS']
oracle_models = ['DeepSAD','DevNet','MLP','XGBOD']

unsup  = bench[bench['model'].isin(unsup_models)].groupby('dataset')['test_auc'].max()
oracle = bench[bench['model'].isin(oracle_models)].groupby('dataset')['test_auc'].max()

v5_7b  = df5[df5['model_short']=='7B'].groupby('dataset')['roc_auc'].mean()
v5_14b = df5[df5['model_short']=='14B'].groupby('dataset')['roc_auc'].mean()
v6_mean = df6.groupby('dataset')['roc_auc'].mean()
v7_mean = df7.groupby('dataset')['roc_auc'].mean()
v5_best = df5.groupby('dataset')['roc_auc'].max()
v6_best = df6.groupby('dataset')['roc_auc'].max()
v7_best = df7.groupby('dataset')['roc_auc'].max()

datasets = ['20_newsgroups','hatebr','tweets_hs','wikinews']

print("=== MEAN — v5 quebrado por modelo LLM ===")
print(f"{'Dataset':<20} {'Unsup':>7} {'v5 7B':>7} {'v5 14B':>8} {'v6 DS':>7} {'v7 MLP':>8} {'Oracle':>8} {'v7 vs unsup':>12} {'% gap':>7}")
for d in datasets:
    u = unsup[d]; o = oracle[d]
    gap = (v7_mean[d]-u)/(o-u)*100
    print(f"{d:<20} {u:>7.3f} {v5_7b[d]:>7.3f} {v5_14b.get(d, float('nan')):>8.3f} {v6_mean[d]:>7.3f} {v7_mean[d]:>8.3f} {o:>8.3f} {v7_mean[d]-u:>+12.3f} {gap:>6.1f}%")

print()
print("=== BEST CONFIG ===")
print(f"{'Dataset':<20} {'Unsup':>7} {'v5 best':>8} {'v6 best':>8} {'v7 best':>8} {'Oracle':>8}")
for d in datasets:
    print(f"{d:<20} {unsup[d]:>7.3f} {v5_best[d]:>8.3f} {v6_best[d]:>8.3f} {v7_best[d]:>8.3f} {oracle[d]:>8.3f}")
