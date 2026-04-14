import pandas as pd, glob

def load_v(version):
    files = [f for f in glob.glob('data/llm_results/' + version + '/**/*.csv', recursive=True) if 'llm_labels' not in f]
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

df5 = load_v('v5')
df6 = load_v('v6')
df7 = load_v('v7')
df8 = load_v('v8')

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
v8_mean = df8.groupby('dataset')['roc_auc'].mean()
v8_best = df8.groupby('dataset')['roc_auc'].max()
v5_ds_setfit    = df5.groupby('dataset')['roc_auc'].mean()   # DeepSAD + SetFit
v6_ds_nosetfit  = df6.groupby('dataset')['roc_auc'].mean()   # DeepSAD + no SetFit

datasets = ['20_newsgroups', 'hatebr', 'tweets_hs', 'wikinews']

print("=== v8 raw means ===")
print(v8_mean.round(4))

print()
print("=== v7 vs v8 (SetFit effect on MLP) ===")
print(f"{'Dataset':<20} {'Unsup':>7} {'v7 MLP':>8} {'v8 MLP+SF':>10} {'Delta':>7} {'Oracle':>8} {'% gap v8':>9}")
for d in datasets:
    u = unsup[d]; o = oracle[d]
    delta = v8_mean[d] - v7_mean[d]
    gap = (v8_mean[d]-u)/(o-u)*100
    print(f"{d:<20} {u:>7.3f} {v7_mean[d]:>8.3f} {v8_mean[d]:>10.3f} {delta:>+7.3f} {o:>8.3f} {gap:>8.1f}%")

print()
print("=== SetFit skipped analysis ===")
print("SetFit ran (False=ran, True=skipped):")
print(df8.groupby(['dataset','n_llm_calls'])['setfit_skipped'].value_counts().unstack().fillna(0).astype(int))
print()
print("AUC when SetFit ran vs skipped:")
print(df8.groupby(['dataset','setfit_skipped'])['roc_auc'].agg(['mean','count']).round(4))

print()
print("=== 2x2 TABLE (mean AUC) ===")
print(f"{'Dataset':<20} {'DeepSAD+SF':>12} {'DeepSAD':>9} {'MLP':>8} {'MLP+SF':>9}")
for d in datasets:
    print(f"{d:<20} {v5_ds_setfit[d]:>12.3f} {v6_ds_nosetfit[d]:>9.3f} {v7_mean[d]:>8.3f} {v8_mean[d]:>9.3f}")

print()
print("=== Full overview (all versions) ===")
print(f"{'Dataset':<20} {'Unsup':>7} {'v5 7B':>8} {'v5 14B':>8} {'v6':>7} {'v7':>7} {'v8':>7} {'Oracle':>8}")
for d in datasets:
    print(f"{d:<20} {unsup[d]:>7.3f} {v5_7b[d]:>8.3f} {v5_14b.get(d, float('nan')):>8.3f} {v6_mean[d]:>7.3f} {v7_mean[d]:>7.3f} {v8_mean[d]:>7.3f} {oracle[d]:>8.3f}")

print()
print("=== Global means ===")
def gmean(s): return s.loc[datasets].mean()
print(f"Unsup:    {gmean(unsup):.4f}")
print(f"v7 MLP:   {gmean(v7_mean):.4f}")
print(f"v8 MLP+SF:{gmean(v8_mean):.4f}")
print(f"Oracle:   {gmean(oracle):.4f}")
print(f"v8 - v7:  {gmean(v8_mean) - gmean(v7_mean):+.4f}")
