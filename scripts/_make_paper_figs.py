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

OUT_DIR = "puplication/STIL2026-LLM-Generated-Weak-Labels-for-Lightweight-Text-Anomaly-Detection/figs"
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

# Use N=200-only MLP+SF 14B for label efficiency (consistent with Tab 2)
df_all  = pd.read_csv("data/llm_results/consolidated_results.csv")
df_best = df_all[(df_all["n_llm_calls"] == 200) & (df_all["model_short"] == "14b") & (df_all["role"] == "mlp_sf")]
_v8_200 = df_best.groupby("dataset")["roc_auc"].mean()
_n_anom = df_best.groupby("dataset")["n_anomalies_found"].mean().round().astype(int)

label_data = {
    "20 Newsgroups": dict(oracle=39,  pipeline=int(_n_anom.get("20_newsgroups", 1)),  ratio=max(1, 39 //  max(1, int(_n_anom.get("20_newsgroups", 1)))),  v8=_v8_200.get("20_newsgroups", v8_mean["20_newsgroups"]),  unsup=unsup["20_newsgroups"],  oracle_auc=oracle["20_newsgroups"]),
    "HateBR":        dict(oracle=140, pipeline=int(_n_anom.get("hatebr",       8)),  ratio=max(1, 140 // max(1, int(_n_anom.get("hatebr",       8)))),  v8=_v8_200.get("hatebr",       v8_mean["hatebr"]),         unsup=unsup["hatebr"],         oracle_auc=oracle["hatebr"]),
    "HS Tweets":     dict(oracle=1189,pipeline=int(_n_anom.get("tweets_hs",   15)),  ratio=max(1, 1189 //max(1, int(_n_anom.get("tweets_hs",  15)))),  v8=_v8_200.get("tweets_hs",   v8_mean["tweets_hs"]),      unsup=unsup["tweets_hs"],      oracle_auc=oracle["tweets_hs"]),
    "WikiNews":      dict(oracle=233, pipeline=int(_n_anom.get("wikinews",    15)),  ratio=max(1, 233 //  max(1, int(_n_anom.get("wikinews",    15)))),  v8=_v8_200.get("wikinews",     v8_mean["wikinews"]),       unsup=unsup["wikinews"],       oracle_auc=oracle["wikinews"]),
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

# ─────────────────────────────────────────────────────────────────────────────
# FIG 5 — Ablation boxplot: 2×2 grid, N=200, grouped by role, hue=LLM size
# ─────────────────────────────────────────────────────────────────────────────

ROLE_ORDER  = ["deepsad", "deepsad_sf", "mlp", "mlp_sf"]
ROLE_XLABELS = ["DeepSAD", "DeepSAD\n+SF", "MLP", "MLP\n+SF"]
COLOR_7B  = "#4292c6"   # blue
COLOR_14B = "#f16913"   # orange
MODEL_COLORS = {"7b": COLOR_7B, "14b": COLOR_14B}

DS_ORDER   = ["20_newsgroups", "wikinews", "tweets_hs", "hatebr"]
DS_DISPLAY = ["20 Newsgroups", "WikiNews", "Hate Speech Tweets", "HateBR"]

df200 = df_all[df_all["n_llm_calls"] == 200].copy()

fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharey=False)
axes_flat = axes.flatten()

_box_w  = 0.30
_offset = 0.18   # half-distance between 7B/14B boxes

for ax, ds, ds_label in zip(axes_flat, DS_ORDER, DS_DISPLAY):
    sub = df200[df200["dataset"] == ds]
    u_val = float(unsup[ds])
    o_val = float(oracle[ds])

    all_data  = []
    positions = []
    colors    = []

    for ri, role in enumerate(ROLE_ORDER):
        x_ctr = ri + 1
        for mi, model in enumerate(["7b", "14b"]):
            vals = sub[(sub["role"] == role) & (sub["model_short"] == model)]["roc_auc"].values
            all_data.append(vals)
            positions.append(x_ctr + _offset * (mi * 2 - 1))
            colors.append(MODEL_COLORS[model])

    bp = ax.boxplot(
        all_data, positions=positions, widths=_box_w,
        patch_artist=True,
        medianprops=dict(color="#222222", linewidth=1.6),
        whiskerprops=dict(color="#555555", linewidth=0.9),
        capprops=dict(color="#555555", linewidth=0.9),
        flierprops=dict(marker="o", markersize=3.5, markerfacecolor="#888888",
                        markeredgecolor="none", alpha=0.7),
        boxprops=dict(linewidth=0.8),
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.82)

    # jittered individual points
    rng = np.random.default_rng(0)
    for vals, pos, c in zip(all_data, positions, colors):
        jitter = rng.uniform(-0.07, 0.07, len(vals))
        ax.scatter(pos + jitter, vals, s=10, color=c, alpha=0.55, zorder=3,
                   linewidths=0)

    # reference lines
    ax.axhline(u_val, color="#555555", linewidth=0.9, linestyle="--",
               alpha=0.7, zorder=1, label="Best unsup.")
    ax.axhline(o_val, color="#cb181d", linewidth=0.9, linestyle=":",
               alpha=0.8, zorder=1, label="GT upper bound")

    # vertical separators between classifier groups
    for xi in [1.5, 2.5, 3.5]:
        ax.axvline(xi, color="#dddddd", linewidth=0.7, zorder=0)

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(ROLE_XLABELS, fontsize=8)
    ax.set_title(ds_label, fontsize=9.5, fontweight="bold", pad=4)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    if ax in axes[:, 0]:
        ax.set_ylabel("ROC-AUC", fontsize=8.5)

# shared legend
_7b_patch  = mpatches.Patch(color=COLOR_7B,  alpha=0.85, label="Qwen 7B")
_14b_patch = mpatches.Patch(color=COLOR_14B, alpha=0.85, label="Qwen 14B")
_unsup_ln  = plt.Line2D([0], [0], color="#555555", linewidth=1.2,
                        linestyle="--", label="Best unsupervised")
_gt_ln     = plt.Line2D([0], [0], color="#cb181d", linewidth=1.2,
                        linestyle=":", label="GT upper bound")
fig.legend(
    handles=[_7b_patch, _14b_patch, _unsup_ln, _gt_ln],
    loc="lower center", ncol=4, fontsize=8,
    bbox_to_anchor=(0.5, -0.01), framealpha=0.92,
    edgecolor="#cccccc",
)

fig.suptitle(
    "Ablation: classifier $\\times$ embedding $\\times$ LLM size  ($N=200$, 6 runs per box)",
    fontsize=9.5, y=1.005,
)
fig.tight_layout(rect=[0, 0.06, 1, 1])
out = os.path.join(OUT_DIR, "ablation_boxplot.png")
fig.savefig(out, bbox_inches="tight", dpi=200)
print(f"Saved: {out}")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# FIG 6 — Sampling boxplot: 4 conditions, short labels, DeepSAD+SF 14B
# ─────────────────────────────────────────────────────────────────────────────
from scipy import stats as _stats

df_samp = df_all[
    (df_all["role"] == "deepsad_sf") & (df_all["model_short"] == "14b")
].copy()

# 4 conditions in logical order: Rnd-50 | Cov-50 || Rnd-200 | Cov-200
SAMP_N      = [50,       50,         200,      200      ]
SAMP_STRAT  = ["random", "diversity","random", "diversity"]
SAMP_LABELS = ["Rnd.\n50", "Cov.\n50", "Rnd.\n200", "Cov.\n200"]
SAMP_COLORS = ["#9ecae1", "#2171b5", "#9ecae1", "#2171b5"]   # light=Random, dark=Coverage
SAMP_HATCHES= ["",        "",        "///",     "///"]        # hatching marks N=200

fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharey=False)
axes_flat = axes.flatten()

for ax, ds, ds_label in zip(axes_flat, DS_ORDER, DS_DISPLAY):
    sub   = df_samp[df_samp["dataset"] == ds]
    u_val = float(unsup[ds])
    o_val = float(oracle[ds])

    all_data = []
    sf_info  = []
    for n, strat in zip(SAMP_N, SAMP_STRAT):
        v  = sub[(sub["n_llm_calls"] == n) & (sub["strategy"] == strat)]["roc_auc"].values
        sf = sub[(sub["n_llm_calls"] == n) & (sub["strategy"] == strat)]["setfit_skipped"].values
        all_data.append(v)
        sf_info.append(int((~sf).sum()) if len(sf) else 0)

    bp = ax.boxplot(
        all_data, positions=[1, 2, 3, 4], widths=0.55,
        patch_artist=True,
        medianprops=dict(color="white", linewidth=1.8),
        whiskerprops=dict(color="#555555", linewidth=0.9),
        capprops=dict(color="#555555", linewidth=0.9),
        flierprops=dict(marker="o", markersize=3.5, markerfacecolor="#888888",
                        markeredgecolor="none", alpha=0.7),
        boxprops=dict(linewidth=0.8),
    )
    for patch, c, h in zip(bp["boxes"], SAMP_COLORS, SAMP_HATCHES):
        patch.set_facecolor(c)
        patch.set_hatch(h)
        patch.set_alpha(0.85)

    # individual seed points
    rng = np.random.default_rng(1)
    for vals, pos, c in zip(all_data, [1, 2, 3, 4], SAMP_COLORS):
        jitter = rng.uniform(-0.09, 0.09, len(vals))
        ax.scatter(pos + jitter, vals, s=13, color=c, alpha=0.7,
                   edgecolors="white", linewidths=0.4, zorder=4)

    # SF activation label — placed neatly above each whisker top
    ax.autoscale(enable=True, axis="y")
    ax.figure.canvas.draw()          # force y-limits to settle
    y0, y1 = ax.get_ylim()
    pad = (y1 - y0) * 0.04
    for pos, sf_ran, vals in zip([1, 2, 3, 4], sf_info, all_data):
        if len(vals) == 0:
            continue
        y_top = max(vals) + pad
        ax.text(pos, y_top, f"SF {sf_ran}/3",
                ha="center", va="bottom", fontsize=6.2,
                color="#444444", style="italic")
    # expand y-limit to fit annotations
    ax.set_ylim(y0, ax.get_ylim()[1] + (y1 - y0) * 0.10)

    # group separator between N=50 and N=200 pairs
    ax.axvline(2.5, color="#aaaaaa", linewidth=1.0, linestyle="--",
               alpha=0.65, zorder=0)

    # reference lines
    ax.axhline(u_val, color="#555555", linewidth=0.9, linestyle="--",
               alpha=0.7, zorder=1)
    ax.axhline(o_val, color="#cb181d", linewidth=0.9, linestyle=":",
               alpha=0.8, zorder=1)

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(SAMP_LABELS, fontsize=8.5)
    ax.set_title(ds_label, fontsize=9.5, fontweight="bold", pad=4)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    if ax in axes[:, 0]:
        ax.set_ylabel("ROC-AUC", fontsize=8.5)

# shared legend
_rnd_p  = mpatches.Patch(color="#9ecae1", alpha=0.85, label="Random")
_cov_p  = mpatches.Patch(color="#2171b5", alpha=0.85, label="Coverage ($k$-means)")
_50_p   = mpatches.Patch(facecolor="#dddddd", edgecolor="#555555", label="$N=50$  (no hatch)")
_200_p  = mpatches.Patch(facecolor="#dddddd", edgecolor="#555555", hatch="///", label="$N=200$  (hatch)")
_unsup2 = plt.Line2D([0], [0], color="#555555", linewidth=1.2,
                     linestyle="--", label="Best unsupervised")
_gt2    = plt.Line2D([0], [0], color="#cb181d", linewidth=1.2,
                     linestyle=":", label="GT upper bound")
fig.legend(
    handles=[_rnd_p, _cov_p, _50_p, _200_p, _unsup2, _gt2],
    loc="lower center", ncol=6, fontsize=7.2,
    bbox_to_anchor=(0.5, -0.01), framealpha=0.92,
    edgecolor="#cccccc",
)

fig.suptitle(
    "Sampling strategy and budget  (DeepSAD+SF, Qwen 14B, 3 seeds per box)",
    fontsize=9.5, y=1.005,
)
fig.tight_layout(rect=[0, 0.055, 1, 1])
out = os.path.join(OUT_DIR, "sampling_boxplot.png")
fig.savefig(out, bbox_inches="tight", dpi=200)
print(f"Saved: {out}")
plt.close()

print("\n=== N=200-only Tab 2 values (deepsad_sf and mlp_sf) ===")
for role, rname in [("deepsad_sf", "DeepSAD"), ("mlp_sf", "MLP")]:
    for model in ["7b", "14b"]:
        sub = df200[(df200["role"]==role) & (df200["model_short"]==model)]
        parts = []
        for ds in ["20_newsgroups", "wikinews", "tweets_hs", "hatebr"]:
            m = sub[sub["dataset"]==ds]["roc_auc"].mean()
            s = sub[sub["dataset"]==ds]["roc_auc"].std()
            parts.append(f"${m:.3f}_{{\\pm.{int(round(s*1000)):03d}}}$")
        print(f"  {rname} ({model.upper()}): " + " & ".join(parts))
print()
print("=== Gap (N=200, best per column) ===")
best_per_ds = {}
for ds in ["20_newsgroups", "wikinews", "tweets_hs", "hatebr"]:
    best_auc = max(
        df200[(df200["role"]==r) & (df200["model_short"]==m)][df200["dataset"]==ds
             ]["roc_auc"].mean() if len(df200[(df200["role"]==r) &
             (df200["model_short"]==m) & (df200["dataset"]==ds)]) > 0 else 0
        for r in ["deepsad_sf", "mlp_sf"] for m in ["7b", "14b"]
    )
    u = float(unsup[ds]); o = float(oracle[ds])
    gap = (best_auc - u) / (o - u) * 100 if (o-u) > 0 else 0
    best_per_ds[ds] = (best_auc, gap)
    print(f"  {ds}: best={best_auc:.3f}, gap={gap:.0f}%")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 7 — Annotation quality: precision and recall by dataset × model
# ─────────────────────────────────────────────────────────────────────────────
df_ann = df_all[
    (df_all["strategy"] == "random") & (df_all["n_llm_calls"] == 200) &
    (df_all["role"] == "deepsad_sf")
].copy()

ann_rows = []
for ds in DS_ORDER:
    for model in ["7b", "14b"]:
        sub = df_ann[(df_ann["dataset"] == ds) & (df_ann["model_short"] == model)]
        ann_rows.append({
            "dataset": DS_DISPLAY[DS_ORDER.index(ds)],
            "model": "Qwen 7B" if model == "7b" else "Qwen 14B",
            "precision": sub["llm_precision"].mean(),
            "recall": sub["llm_recall"].mean(),
        })
df_ann_agg = pd.DataFrame(ann_rows)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
metrics = [("precision", "Precision"), ("recall", "Recall")]
colors = {"Qwen 7B": "#4292c6", "Qwen 14B": "#e6550d"}
x = np.arange(len(DS_ORDER))
w = 0.34

for ax, (metric, mlabel) in zip(axes, metrics):
    for i, (mname, c) in enumerate(colors.items()):
        vals = [df_ann_agg[(df_ann_agg["dataset"] == DS_DISPLAY[j]) &
                            (df_ann_agg["model"] == mname)][metric].values[0]
                for j in range(len(DS_ORDER))]
        offset = (i - 0.5) * w
        bars = ax.bar(x + offset, vals, w, color=c, alpha=0.85, label=mname,
                      edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)

    ax.axhline(0.5, color="#888888", linewidth=0.8, linestyle="--",
               alpha=0.6, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace(" ", "\n") for d in DS_DISPLAY], fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel(mlabel, fontsize=9)
    ax.set_title(mlabel, fontsize=9.5, fontweight="bold")
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    if ax == axes[0]:
        ax.legend(fontsize=7.5, framealpha=0.9, edgecolor="#cccccc")

fig.suptitle(
    "LLM annotation quality by dataset  (random, $N=200$, mean over 3 seeds)",
    fontsize=9.5, y=1.02,
)
fig.tight_layout()
out = os.path.join(OUT_DIR, "annotation_quality.png")
fig.savefig(out, bbox_inches="tight", dpi=200)
print(f"Saved: {out}")
plt.close()

