"""
_consolidate_results.py
-----------------------
Consolidates LLM-supervised experiment results from v5–v8 into final paper tables.

Version-to-role mapping (all 4 datasets in each version):
  v5  →  DeepSAD + SetFit   (LLM annotation + AD)
  v6  →  DeepSAD, no SetFit (reads v5 labels)
  v7  →  MLP, no SetFit     (reads v5 labels)
  v8  →  MLP + SetFit       (reads v5 labels)

Note: v9 was a staging area for the 20_newsgroups hotfix. After migrating v9 → v5
(see data/llm_results/v9/README.md), this script just reads v5–v8 normally.

Usage (local):
    python scripts/_consolidate_results.py --project_path .

Usage (Colab):
    !python scripts/_consolidate_results.py \\
        --project_path "/content/drive/MyDrive/Projeto ML/2026/Master/Multilingual-Text-Anomaly-Detection"
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd


# ─── Dataset display order ───────────────────────────────────────────────────
DATASETS = ["20_newsgroups", "wikinews", "tweets_hs", "hatebr"]
DATASET_LABELS = {
    "20_newsgroups": "20 News",
    "wikinews":      "WikiNews",
    "tweets_hs":     "HS Tweets",
    "hatebr":        "HateBR",
}

# ─── Version-to-role map ─────────────────────────────────────────────
# After migrating v9 → v5, all 4 datasets live in v5/v6/v7/v8.
# See data/llm_results/v9/README.md for migration instructions.
VERSION_MAP = {
    "deepsad_sf": "v5",   # DeepSAD + SetFit  (LLM annotation + AD)
    "deepsad":    "v6",   # DeepSAD, no SetFit
    "mlp":        "v7",   # MLP, no SetFit
    "mlp_sf":     "v8",   # MLP + SetFit (proposed)
}


def parse_args():
    parser = argparse.ArgumentParser(description="Consolidate experiment results for paper tables.")
    parser.add_argument("--project_path", type=str, default=".", help="Root path of the project.")
    parser.add_argument("--results_base", type=str, default="data/llm_results",
                        help="Base dir for llm_results (relative to project_path).")
    parser.add_argument("--benchmark_csv", type=str,
                        default="data/benchmark_results/benchmark_results.csv",
                        help="Path to benchmark_results.csv (relative to project_path).")
    return parser.parse_args()


def load_version(version_dir: str) -> pd.DataFrame:
    """Load all metrics CSVs from a version directory (excluding *_llm_labels.csv)."""
    pattern = os.path.join(version_dir, "**", "*.csv")
    files = [f for f in glob.glob(pattern, recursive=True) if "llm_labels" not in f]
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, engine="python", on_bad_lines="skip")
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: could not read {f}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def tag_model(df: pd.DataFrame) -> pd.DataFrame:
    """Add model_short column ('7b' or '14b') from llm_model string."""
    def _tag(s):
        s = str(s).lower()
        if "14b" in s:
            return "14b"
        elif "7b" in s:
            return "7b"
        return "unknown"
    df = df.copy()
    df["model_short"] = df["llm_model"].apply(_tag)
    return df


def load_role(role: str, results_base: str) -> pd.DataFrame:
    """Load all data for a given role from its version directory."""
    version = VERSION_MAP[role]
    version_dir = os.path.join(results_base, version)
    df = load_version(version_dir)
    if df.empty:
        return pd.DataFrame()
    df = tag_model(df)
    df["role"] = role
    return df


def load_benchmark(benchmark_csv: str) -> tuple[pd.Series, pd.Series]:
    """Return (unsup_best, gt_best) Series indexed by dataset name."""
    bench = pd.read_csv(benchmark_csv)
    name_map = {
        "tweets_hate_speech_detection_distiluse-base-multilingual-cased-v2": "tweets_hs",
        "HateBR_distiluse-base-multilingual-cased-v2":                       "hatebr",
        "20_newsgroups_distiluse-base-multilingual-cased-v2":                "20_newsgroups",
        "wikinews_distiluse-base-multilingual-cased-v2":                     "wikinews",
    }
    bench["dataset"] = bench["dataset"].map(name_map)
    bench = bench.dropna(subset=["dataset"])

    unsup_models  = ["IForest", "LOF", "DeepSVDD", "OCSVM", "AutoEncoder", "VAE", "HBOS"]
    oracle_models = ["DeepSAD", "DevNet", "MLP", "XGBOD"]

    unsup_best = bench[bench["model"].isin(unsup_models)].groupby("dataset")["test_auc"].max()
    gt_best    = bench[bench["model"].isin(oracle_models)].groupby("dataset")["test_auc"].max()
    return unsup_best, gt_best


def compute_stats(df: pd.DataFrame, model_short: str) -> pd.Series:
    """Mean ± std ROC-AUC over all runs for a given model_short, indexed by dataset."""
    sub = df[df["model_short"] == model_short]
    if sub.empty:
        return pd.Series(dtype=float)
    return sub.groupby("dataset")["roc_auc"].agg(["mean", "std"])


# ─── Table printers ──────────────────────────────────────────────────────────

def print_main_results(all_roles: dict[str, pd.DataFrame], unsup_best: pd.Series, gt_best: pd.Series):
    """Print tab:main-results equivalent (Table 2 in the paper)."""
    print("\n" + "="*80)
    print("TABLE: Main Results — ROC-AUC (mean ± std, 12 runs per cell)")
    print("Rows: DeepSAD 7B/14B, MLP 7B/14B | Cols: 4 datasets")
    print("="*80)

    col_w = 20
    header = f"{'Method':<30}" + "".join(f"{DATASET_LABELS[d]:>{col_w}}" for d in DATASETS)
    print(header)
    print("-" * len(header))

    rows = [
        ("DeepSAD (Qwen 7B)",   "deepsad_sf", "7b"),
        ("DeepSAD (Qwen 14B)",  "deepsad_sf", "14b"),
        ("MLP (Qwen 7B)",       "mlp_sf",     "7b"),
        ("MLP (Qwen 14B) [†]",  "mlp_sf",     "14b"),
    ]

    best_per_col = {d: 0.0 for d in DATASETS}
    row_data = []
    for label, role, model in rows:
        df = all_roles.get(role, pd.DataFrame())
        vals = {}
        for d in DATASETS:
            sub = df[(df["dataset"] == d) & (df["model_short"] == model)]
            if sub.empty:
                vals[d] = ("---", 0.0)
            else:
                m, s = sub["roc_auc"].mean(), sub["roc_auc"].std()
                vals[d] = (f"{m:.3f}±{s:.3f}", m)
                if m > best_per_col[d]:
                    best_per_col[d] = m
        row_data.append((label, vals))

    for label, vals in row_data:
        line = f"{label:<30}"
        for d in DATASETS:
            cell_str, cell_val = vals[d]
            marker = "*" if abs(cell_val - best_per_col[d]) < 1e-6 and cell_val > 0 else " "
            line += f"{marker+cell_str:>{col_w}}"
        print(line)

    print("-" * len(header))
    print(f"{'Best unsup':<30}" + "".join(f"{unsup_best.get(d, float('nan')):>{col_w}.3f}" for d in DATASETS))
    print(f"{'Best (GT)':<30}"  + "".join(f"{gt_best.get(d, float('nan')):>{col_w}.3f}"  for d in DATASETS))

    print("\nGap recovery (best LLM-sup row vs [unsup, GT] interval):")
    for d in DATASETS:
        u = unsup_best.get(d, np.nan)
        g = gt_best.get(d, np.nan)
        b = best_per_col[d]
        gap = (b - u) / (g - u) * 100 if (g - u) > 0 else float("nan")
        print(f"  {DATASET_LABELS[d]:<12}: best={b:.3f}  unsup={u:.3f}  GT={g:.3f}  gap={gap:.0f}%")


def print_2x2_ablation(all_roles: dict[str, pd.DataFrame]):
    """Print 2×2 ablation table (DeepSAD/MLP × no-SF/SF), merged over models."""
    print("\n" + "="*80)
    print("TABLE: 2×2 Ablation — Mean ROC-AUC (all models combined)")
    print("Rows: DeepSAD / MLP | Cols: Original embed / SetFit fine-tuned")
    print("="*80)

    col_w = 16
    header = f"{'Dataset':<18}" + "".join(f"{h:>{col_w}}" for h in ["DeepSAD", "DeepSAD+SF", "MLP", "MLP+SF"])
    print(header)
    print("-" * len(header))

    for d in DATASETS:
        def _mean(role):
            df = all_roles.get(role, pd.DataFrame())
            sub = df[df["dataset"] == d]
            return sub["roc_auc"].mean() if not sub.empty else float("nan")

        vals = [_mean("deepsad"), _mean("deepsad_sf"), _mean("mlp"), _mean("mlp_sf")]
        line = f"{DATASET_LABELS[d]:<18}" + "".join(f"{v:>{col_w}.3f}" for v in vals)
        print(line)


def print_setfit_effect(all_roles: dict[str, pd.DataFrame]):
    """Print tab:setfit-effect: conditional AUC gain of SetFit on MLP, at N=200."""
    print("\n" + "="*80)
    print("TABLE: SetFit Effect on MLP — N=200, runs where SetFit executed")
    print("Columns: Ran/6 (out of 3 seeds × 2 strategies), Δ AUC (MLP+SF − MLP)")
    print("="*80)

    col_w = 10
    header = (f"{'Dataset':<18}"
              f"{'7B Ran/6':>{col_w}}{'7B ΔAUC':>{col_w}}"
              f"{'14B Ran/6':>{col_w}}{'14B ΔAUC':>{col_w}}")
    print(header)
    print("-" * len(header))

    for d in DATASETS:
        row = f"{DATASET_LABELS[d]:<18}"
        for model in ["7b", "14b"]:
            df_mlp    = all_roles.get("mlp",    pd.DataFrame())
            df_mlp_sf = all_roles.get("mlp_sf", pd.DataFrame())

            sub_base = df_mlp[   (df_mlp["dataset"]    == d) & (df_mlp["model_short"]    == model) & (df_mlp["n_llm_calls"]    == 200)]
            sub_sf   = df_mlp_sf[(df_mlp_sf["dataset"] == d) & (df_mlp_sf["model_short"] == model) & (df_mlp_sf["n_llm_calls"] == 200)]

            # runs where SetFit actually executed
            sub_sf_ran = sub_sf[sub_sf["setfit_skipped"] == False] if "setfit_skipped" in sub_sf.columns else sub_sf

            ran = len(sub_sf_ran)

            if ran == 0 or sub_base.empty:
                row += f"{'0':>{col_w}}{'---':>{col_w}}"
            else:
                # paired delta: match by seed + strategy
                key = ["seed", "strategy"]
                if all(c in sub_base.columns for c in key) and all(c in sub_sf_ran.columns for c in key):
                    merged = sub_sf_ran[key + ["roc_auc"]].merge(
                        sub_base[key + ["roc_auc"]], on=key, suffixes=("_sf", "_base"))
                    delta = (merged["roc_auc_sf"] - merged["roc_auc_base"]).mean()
                else:
                    delta = sub_sf_ran["roc_auc"].mean() - sub_base["roc_auc"].mean()

                sign = "+" if delta >= 0 else ""
                row += f"{ran:>{col_w}}{sign+f'{delta:.3f}':>{col_w}}"

        print(row)


def print_coverage_check(all_roles: dict[str, pd.DataFrame]):
    """Print run counts per role/dataset/model to verify completeness before consolidating."""
    print("\n" + "="*80)
    print("COVERAGE CHECK — run counts (expect 12 per cell: 3 seeds × 2 strats × 2 N)")
    print("="*80)
    print(f"{'Role':<14}{'Dataset':<18}{'Model':<8}{'Count':>8}  {'Source version'}")
    print("-" * 70)

    for role, version in VERSION_MAP.items():
        df = all_roles.get(role, pd.DataFrame())
        if df.empty:
            print(f"  {role:<12}  (no data)")
            continue
        for d in DATASETS:
            for model in ["7b", "14b"]:
                sub = df[(df["dataset"] == d) & (df["model_short"] == model)]
                count = len(sub)
                flag = " ⚠️  INCOMPLETE" if count < 12 else ""
                print(f"  {role:<12}  {DATASET_LABELS[d]:<16}  {model:<6}  {count:>5}  ({version}){flag}")


def main():
    args = parse_args()
    project_path  = os.path.abspath(args.project_path)
    results_base  = os.path.join(project_path, args.results_base)
    benchmark_csv = os.path.join(project_path, args.benchmark_csv)

    sys.path.insert(0, os.path.join(project_path, "src"))

    print(f"Project path : {project_path}")
    print(f"Results base : {results_base}")
    print(f"Benchmark CSV: {benchmark_csv}")

    # Load all roles
    print("\nLoading data…")
    all_roles = {}
    for role in VERSION_MAP:
        df = load_role(role, results_base)
        all_roles[role] = df
        n = len(df) if not df.empty else 0
        print(f"  {role:<14}: {n} rows")

    # Load benchmark baselines
    unsup_best, gt_best = load_benchmark(benchmark_csv)

    # Print all tables
    print_coverage_check(all_roles)
    print_main_results(all_roles, unsup_best, gt_best)
    print_2x2_ablation(all_roles)
    print_setfit_effect(all_roles)

    # Save consolidated CSV
    all_dfs = []
    for role, df in all_roles.items():
        if not df.empty:
            all_dfs.append(df)
    if all_dfs:
        consolidated = pd.concat(all_dfs, ignore_index=True)
        out_path = os.path.join(project_path, "data", "llm_results", "consolidated_results.csv")
        consolidated.to_csv(out_path, index=False)
        print(f"\nSaved consolidated CSV → {out_path}")


if __name__ == "__main__":
    main()
