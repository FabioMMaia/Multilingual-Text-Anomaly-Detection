"""
Generate all figures for the paper:
    Fig 1: pipeline diagram         — figs/pipeline.png  (placeholder, use DALL-E prompt)
    Fig 2: main results bar chart   — figs/main_results.png
    Fig 3: SetFit effect chart      — figs/setfit_effect.png
    Fig 4: label efficiency chart   — figs/label_efficiency.png

Run from project root:
    python scripts/_make_paper_figs.py
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 200,
})

OUT_DIR = "puplication/ZeroCost-LLM-Supervision-Multilingual-Text-AD/figs"
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS   = ["20_newsgroups", "hatebr", "tweets_hs", "wikinews"]
DS_LABELS  = ["20 Newsgroups", "HateBR", "Hate Speech\nTweets", "WikiNews"]

# ─── load pipeline results ───────────────────────────────────────────────────

def load_v(version):
    files = [f for f in glob.glob(f"data/llm_results/{version}/**/*.csv", recursive=True)
             if "llm_labels" not in f]
    if not files:
        raise FileNotFoundError(f"No result CSVs found for version '{version}'")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

df5 = load_v("v5")
df6 = load_v("v6")
df7 = load_v("v7")
df8 = load_v("v8")

# ─── load benchmark (unsup / oracle) ────────────────────────────────────────

bench = pd.read_csv("data/benchmark_results/benchmark_results.csv")
name_map = {
    "tweets_hate_speech_detection_distiluse-base-multilingual-cased-v2": "tweets_hs",
    "HateBR_distiluse-base-multilingual-cased-v2": "hatebr",
    "20_newsgroups_distiluse-base-multilingual-cased-v2": "20_newsgroups",
    "wikinews_distiluse-base-multilingual-cased-v2": "wikinews",
}
bench["dataset"] = bench["dataset"].map(name_map)

UNSUP_MODELS  = ["IForest", "LOF", "DeepSVDD", "OCSVM", "AutoEncoder", "VAE", "HBOS"]
ORACLE_MODELS = ["DeepSAD", "DevNet", "MLP", "XGBOD"]
unsup  = bench[bench["model"].isin(UNSUP_MODELS)].groupby("dataset")["test_auc"].max()
oracle = bench[bench["model"].isin(ORACLE_MODELS)].groupby("dataset")["test_auc"].max()

v5_mean = df5.groupby("dataset")["roc_auc"].mean()
v6_mean = df6.groupby("dataset")["roc_auc"].mean()
v7_mean = df7.groupby("dataset")["roc_auc"].mean()
v8_mean = df8.groupby("dataset")["roc_auc"].mean()

# add global row
def with_global(s):
    s = s.reindex(DATASETS)
    return dict(s.to_dict(), **{"global": s.mean()})

DATA = {
    "Unsupervised": with_global(unsup),
    "DS+SF":        with_global(v5_mean),
    "DeepSAD":      with_global(v6_mean),
    "MLP":          with_global(v7_mean),
    "MLP+SF":       with_global(v8_mean),
    "Oracle":       with_global(oracle),
}

# ── print stats to terminal ────────────────────────────────────────────────────
print("=== PAPER STATS ===")
print(f"\n{'Dataset':<20} {'Unsup':>7} {'DS+SF':>7} {'DeepSAD':>9} {'MLP':>7} {'MLP+SF':>8} {'Oracle':>8}")
for d in DATASETS:
    print(f"{d:<20} {unsup[d]:>7.3f} {v5_mean[d]:>7.3f} {v6_mean[d]:>9.3f} "
          f"{v7_mean[d]:>7.3f} {v8_mean[d]:>8.3f} {oracle[d]:>8.3f}")
print(f"{'Global':<20} "
      f"{unsup.reindex(DATASETS).mean():>7.3f} "
      f"{v5_mean.reindex(DATASETS).mean():>7.3f} "
      f"{v6_mean.reindex(DATASETS).mean():>7.3f} "
      f"{v7_mean.reindex(DATASETS).mean():>7.3f} "
      f"{v8_mean.reindex(DATASETS).mean():>7.3f} "
      f"{oracle.reindex(DATASETS).mean():>8.3f}")

print("\n=== SetFit paired delta v8 - v7 ===")
for d in DATASETS:
    delta = v8_mean[d] - v7_mean[d]
    print(f"  {d:<22}: {delta:+.3f}")

print("\n=== % of unsup->oracle gap closed by MLP+SF ===")
for d in DATASETS:
    pct = (v8_mean[d] - unsup[d]) / (oracle[d] - unsup[d]) * 100
    print(f"  {d:<22}: {pct:.0f}%")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — Main results: grouped bar chart (per dataset + global)
# ─────────────────────────────────────────────────────────────────────────────

COLS   = ["Unsupervised", "DS+SF", "DeepSAD", "MLP", "MLP+SF", "Oracle"]
COLORS = ["#bdbdbd", "#9ecae1", "#6baed6", "#2171b5", "#08306b", "#e6550d"]
HATCHES = ["", "", "", "", "", "//"]

ALL_KEYS = DATASETS + ["global"]
ALL_XLABELS = DS_LABELS + ["Global\nMean"]

n_groups = len(ALL_KEYS)
n_bars   = len(COLS)
width    = 0.12
x        = np.arange(n_groups)

fig, ax = plt.subplots(figsize=(7.5, 3.4))

for i, (col, color, hatch) in enumerate(zip(COLS, COLORS, HATCHES)):
    vals = [DATA[col][k] for k in ALL_KEYS]
    offset = (i - n_bars / 2 + 0.5) * width
    ax.bar(x + offset, vals, width, label=col.replace("\n", " "),
           color=color, hatch=hatch, edgecolor="white", linewidth=0.4)

ax.set_xticks(x)
ax.set_xticklabels(ALL_XLABELS)
ax.set_ylabel("ROC-AUC")
ax.set_ylim(0.45, 1.05)
ax.set_title("Mean ROC-AUC by Configuration and Dataset", pad=6)
ax.axvline(n_groups - 1.5, color="gray", linewidth=0.8, linestyle="--")
ax.legend(loc="upper left", framealpha=0.85, ncol=3)
ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax.set_axisbelow(True)

# annotate best pipeline per dataset
for ki, k in enumerate(ALL_KEYS):
    best_v = max(DATA[col][k] for col in ["DS+SF", "DeepSAD", "MLP", "MLP+SF"])
    ax.plot(x[ki], best_v + 0.012, marker="*", color="#08306b", markersize=6, zorder=5)

fig.tight_layout()
out = os.path.join(OUT_DIR, "main_results.png")
fig.savefig(out, bbox_inches="tight")
print(f"\nSaved: {out}")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — SetFit effect: v7 vs v8, per dataset
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 4, figsize=(7.5, 2.6), sharey=False)

for ax, d, label in zip(axes, DATASETS, DS_LABELS):
    sub = df8[df8["dataset"] == d]
    v7_sub = df7[df7["dataset"] == d]["roc_auc"].values
    v8_ran = sub[sub["setfit_skipped"] == False]["roc_auc"].values
    v8_skp = sub[sub["setfit_skipped"] == True]["roc_auc"].values

    parts = []
    labs  = []
    if len(v7_sub):
        parts.append(v7_sub); labs.append("MLP (no SetFit)")
    if len(v8_ran):
        parts.append(v8_ran); labs.append("MLP+SF (ran)")
    if len(v8_skp):
        parts.append(v8_skp); labs.append("MLP+SF (skipped)")

    bp = ax.boxplot(parts, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.2),
                    whiskerprops=dict(linewidth=0.8),
                    capprops=dict(linewidth=0.8),
                    flierprops=dict(marker=".", markersize=3))

    palette = ["#6baed6", "#08306b", "#bdbdbd"]
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_xticks(range(1, len(parts) + 1))
    ax.set_xticklabels(labs, fontsize=6.5, rotation=20, ha="right")
    ax.set_title(label, fontsize=8.5)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    if ax == axes[0]:
        ax.set_ylabel("ROC-AUC")

fig.suptitle("SetFit Effect: MLP vs MLP+SF Distribution", y=1.01, fontsize=9.5)
fig.tight_layout()
out = os.path.join(OUT_DIR, "setfit_effect.png")
fig.savefig(out, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 — Label efficiency: bubble / horizontal bar
# ─────────────────────────────────────────────────────────────────────────────

label_data = {
    "20 Newsgroups": dict(oracle=39,  pipeline=1,  ratio=49, gap=0.061, v8=v8_mean["20_newsgroups"],  unsup=unsup["20_newsgroups"],  oracle_auc=oracle["20_newsgroups"]),
    "HateBR":        dict(oracle=140, pipeline=8,  ratio=18, gap=0.142, v8=v8_mean["hatebr"],         unsup=unsup["hatebr"],         oracle_auc=oracle["hatebr"]),
    "HS Tweets":     dict(oracle=1189,pipeline=15, ratio=78, gap=0.127, v8=v8_mean["tweets_hs"],      unsup=unsup["tweets_hs"],      oracle_auc=oracle["tweets_hs"]),
    "WikiNews":      dict(oracle=233, pipeline=15, ratio=15, gap=0.165, v8=v8_mean["wikinews"],       unsup=unsup["wikinews"],       oracle_auc=oracle["wikinews"]),
}

fig, ax = plt.subplots(figsize=(6.5, 3.0))

ds_names = list(label_data.keys())
y        = np.arange(len(ds_names))

for yi, (name, d) in enumerate(label_data.items()):
    u, v8_auc, orc = d["unsup"], d["v8"], d["oracle_auc"]
    # draw the full bar from unsup->oracle in light grey
    ax.barh(yi, orc - u, left=u, height=0.35, color="#eeeeee", edgecolor="#aaaaaa", linewidth=0.5)
    # draw the pipeline's portion in blue
    ax.barh(yi, v8_auc - u, left=u, height=0.35, color="#2171b5", alpha=0.85, edgecolor="none")
    # mark oracle
    ax.plot(orc, yi, marker="|", markersize=12, color="#e6550d", linewidth=2, zorder=5)
    # annotate ratio and gap-closed %
    pct = (v8_auc - u) / (orc - u) * 100
    ax.text(orc + 0.005, yi, f"×{d['ratio']}  {pct:.0f}%", va="center", fontsize=7.5, color="#333333")

ax.set_yticks(y)
ax.set_yticklabels(ds_names)
ax.set_xlabel("ROC-AUC")
ax.set_xlim(0.45, 1.18)
ax.set_title("Label Efficiency: MLP+SF vs Oracle\n"
             "(blue = AUC gain over unsupervised,  grey = remaining gap,\n"
             "orange | = oracle AUC,  \u00d7N = label disadvantage,  % = gap closed)", fontsize=8)
ax.xaxis.grid(True, linewidth=0.4, alpha=0.5)
ax.set_axisbelow(True)

blue_patch  = mpatches.Patch(color="#2171b5", alpha=0.85, label="MLP+SF gain")
grey_patch  = mpatches.Patch(facecolor="#eeeeee", edgecolor="#aaaaaa", label="remaining gap")
oracle_line = mpatches.Patch(color="#e6550d", label="oracle AUC")
ax.legend(handles=[blue_patch, grey_patch, oracle_line], loc="upper left", fontsize=7.5)

fig.tight_layout()
out = os.path.join(OUT_DIR, "label_efficiency.png")
fig.savefig(out, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()

print("\nAll figures saved to:", OUT_DIR)

