import pandas as pd, glob

def load_v(version):
    files = [f for f in glob.glob('data/llm_results/' + version + '/**/*.csv', recursive=True) if 'llm_labels' not in f]
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

df5 = load_v('v5')
df6 = load_v('v6')
df7 = load_v('v7')
df8 = load_v('v8')

key = ['dataset','strategy','n_llm_calls','seed']

v5_mean_key = df5.groupby(key)['roc_auc'].mean().reset_index()
v7_key      = df7[key + ['roc_auc']].copy()
v8_key      = df8[key + ['roc_auc', 'setfit_skipped']].copy()

# v8 vs v7 paired
m_v87 = v7_key.merge(v8_key, on=key, suffixes=('_v7','_v8'))
m_v87['delta'] = m_v87['roc_auc_v8'] - m_v87['roc_auc_v7']
print("=== Paired v8 - v7 (SetFit effect on MLP) ===")
print(m_v87.groupby('dataset')['delta'].agg(['mean','std']).round(4))
print("Global: {:+.4f} +/- {:.4f}".format(m_v87['delta'].mean(), m_v87['delta'].std()))

# v8 vs v5 paired
m_v85 = v5_mean_key.merge(v8_key, on=key, suffixes=('_v5','_v8'))
m_v85['delta'] = m_v85['roc_auc_v8'] - m_v85['roc_auc_v5']
print()
print("=== Paired v8 - v5 (MLP+SF vs DeepSAD+SF) ===")
print(m_v85.groupby('dataset')['delta'].agg(['mean','std']).round(4))
print("Global: {:+.4f} +/- {:.4f}".format(m_v85['delta'].mean(), m_v85['delta'].std()))

# SetFit skipped breakdown
print()
print("=== v8: AUC por setfit_skipped e dataset/N ===")
g = df8.groupby(['dataset','setfit_skipped','n_llm_calls'])['roc_auc'].agg(['mean','count']).round(4)
print(g.to_string())

# Global summary
datasets = ['20_newsgroups','hatebr','tweets_hs','wikinews']
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

v5_ds = df5.groupby('dataset')['roc_auc'].mean()
v6_ds = df6.groupby('dataset')['roc_auc'].mean()
v7_ds = df7.groupby('dataset')['roc_auc'].mean()
v8_ds = df8.groupby('dataset')['roc_auc'].mean()

print()
print("=== Full 2x2 + Gap% table ===")
print("{:<20} {:>8} {:>10} {:>9} {:>9} {:>9} {:>8}".format(
    'Dataset','Unsup','DS+SF(v5)','DS(v6)','MLP(v7)','MLP+SF(v8)','Oracle'))
for d in datasets:
    u = unsup[d]; o = oracle[d]
    gap_v7 = (v7_ds[d]-u)/(o-u)*100
    gap_v8 = (v8_ds[d]-u)/(o-u)*100
    print("{:<20} {:>8.3f} {:>10.3f} {:>9.3f} {:>9.3f} {:>9.3f} {:>8.3f}  (v7:{:.0f}%, v8:{:.0f}%)".format(
        d, u, v5_ds[d], v6_ds[d], v7_ds[d], v8_ds[d], o, gap_v7, gap_v8))
