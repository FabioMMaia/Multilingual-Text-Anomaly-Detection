import pandas as pd, glob, os, numpy as np

base = r'c:\Users\fabio\Desktop\Mestrado\Multilingual-Text-Anomaly-Detection\data\llm_results\v5'
files = [f for f in glob.glob(os.path.join(base, '**', '*.csv'), recursive=True) if 'llm_labels' not in f]
dfs = []
for f in files:
    rel = os.path.relpath(f, base)
    parts = rel.split(os.sep)
    df = pd.read_csv(f)
    df['strategy_folder'] = parts[0]
    df['n_folder'] = parts[1]
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
df['model_size'] = df['llm_model'].str.extract(r'qwen2\.5-(\d+b)')

def fmt(mean, std):
    return f"{mean:.3f} ±{std:.3f}"

# TABLE 1: mean±std ROC-AUC por dataset x strategy x N x model
grp = df.groupby(['dataset','strategy','n_llm_calls','model_size'])['roc_auc']
tbl = grp.agg(['mean','std']).reset_index()
tbl['val'] = tbl.apply(lambda r: fmt(r['mean'], r['std']), axis=1)
print('=== TABLE 1: ROC-AUC mean (std) ===')
pivot = tbl.pivot_table(index=['dataset','strategy','n_llm_calls'], columns='model_size', values='val', aggfunc='first')
print(pivot.to_string())
print()

# TABLE 2: mean AUC per dataset x model (collapsed over strategy+N)
print('=== TABLE 2: Mean AUC per dataset x model ===')
best = df.groupby(['dataset','model_size'])['roc_auc'].agg(['mean','std','max']).reset_index()
print(best.round(3).to_string(index=False))
print()

# Q1: 7B vs 14B
print('=== Q1: 7B vs 14B overall ===')
for m in ['7b','14b']:
    vals = df[df['model_size']==m]['roc_auc']
    print(f'  {m}: mean={vals.mean():.3f} std={vals.std():.3f} n={len(vals)}')
print()

# Q2: N=50 vs N=200
print('=== Q2: N=50 vs N=200 ===')
for n in [50, 200]:
    vals = df[df['n_llm_calls']==n]['roc_auc']
    print(f'  N={n}: mean={vals.mean():.3f} std={vals.std():.3f} n={len(vals)}')
print()

# Q3: random vs diversity
print('=== Q3: random vs diversity ===')
for s in ['random','diversity']:
    vals = df[df['strategy']==s]['roc_auc']
    print(f'  {s}: mean={vals.mean():.3f} std={vals.std():.3f} n={len(vals)}')
print()

# Q4: dataset difficulty
print('=== Q4: dataset difficulty ===')
for ds in ['20_newsgroups','tweets_hs','wikinews','hatebr']:
    vals = df[df['dataset']==ds]['roc_auc']
    print(f'  {ds:20s}: mean={vals.mean():.3f} std={vals.std():.3f}')
print()

# Q5: N effect per dataset
print('=== Q5: N effect per dataset ===')
n_effect = df.groupby(['dataset','n_llm_calls'])['roc_auc'].mean().unstack()
n_effect['delta_200_minus_50'] = n_effect[200] - n_effect[50]
print(n_effect.round(3).to_string())
print()

# Q6: strategy x N interaction
print('=== Q6: strategy x N interaction ===')
sn = df.groupby(['strategy','n_llm_calls'])['roc_auc'].mean().unstack()
print(sn.round(3).to_string())
print()

# Q7: best single config per dataset
print('=== Q7: best single config per dataset (max mean AUC, 3 seeds) ===')
top = tbl.sort_values('mean', ascending=False).groupby('dataset').first().reset_index()
for _, row in top.iterrows():
    print(f"  {row['dataset']:20s} {row['strategy']:10s} N={row['n_llm_calls']:3d} {row['model_size']}  AUC={row['mean']:.3f} ±{row['std']:.3f}")
print()

# Q8: timing summary
print('=== Q8: timing (elapsed_seconds) ===')
timing = df.groupby(['n_llm_calls','model_size'])['elapsed_seconds'].mean() / 60
print(timing.round(1).to_string())
