
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plot_hist(scores, y_true):
    # Create a DataFrame for seaborn
    df = pd.DataFrame({
        "score": scores,
        "label": ["Normal" if y == 0 else "Anomaly" for y in y_true]
    })

    # Plot
    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=df,
        x="score",
        hue="label",
        bins=30,
        kde=True,
        stat="density",
        common_norm=False,
        palette={"Normal": "tab:blue", "Anomaly": "tab:red"},
        hue_order=["Normal", "Anomaly"]
    )
    plt.title("Score Distribution by Class")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Density")
    plt.legend(title="Label")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compare_auc_curves(path_csv, models, dataset, metric='test_auc', sort_values=False):
    plt.figure(figsize=(12, 6))

    for model_name in models:
        # Load data
        df = pd.read_csv(os.path.join(path_csv, f"{model_name}_{dataset}.csv"))
        df['model'] = model_name  # tag model name for legend

        # Optionally sort
        if sort_values:
            df = df.sort_values(by='n_known_outliers')

        sns.lineplot(
            data=df,
            x='n_known_outliers',
            y=metric,
            label=f"{model_name.replace('_', ' ')}"
        )

    plt.title(f'{metric.replace("_", " ").upper()} vs. Number of Known Outliers')
    plt.xlabel('Number of Known Outliers')
    plt.ylabel(metric.replace("_", " ").title())
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()

def compare_auc_curves_percentage(path_csv, models, dataset, metric='test_auc', sort_values=False):
    plt.figure(figsize=(12, 6))

    for model_name in models:
        df = pd.read_csv(os.path.join(path_csv, f"{model_name}_{dataset}.csv"))
        df['model'] = model_name

        # Convert to percentage
        df['labeled_percentage'] = 100 * df['n_known_outliers'] / df['train_size']

        if sort_values:
            df = df.sort_values(by='labeled_percentage')

        sns.lineplot(
            data=df,
            x='labeled_percentage',
            y=metric,
            label=f"{model_name.replace('_', ' ').title()}",
            marker='o'
        )

    plt.title(f'{metric.replace("_", " ").upper()} vs. % of Labeled Outliers')
    plt.xlabel('% of Labeled Outliers')
    plt.ylabel(metric.replace("_", " ").title())
    plt.grid(True)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()


# ---------------------------------------------------------------------------
# SetFit / embedding comparison plots
# ---------------------------------------------------------------------------

def plot_tsne_comparison(
    embeddings_a,
    embeddings_b,
    labels,
    title_a,
    title_b,
    dataset_short,
    encoder_short,
    save_dir,
    stage="comparison",
    max_points=5000,
    pca_dims=50,
    random_state=42,
    save=True,
):
    """
    Side-by-side t-SNE scatter of two embedding sets (e.g. before/after SetFit).

    Args:
        embeddings_a: First embedding matrix (N x D).
        embeddings_b: Second embedding matrix (N x D).
        labels: Binary labels (0 = normal, 1 = anomaly).
        title_a / title_b: Subplot titles.
        dataset_short: Short dataset name used in filename.
        encoder_short: Short encoder name used in filename.
        save_dir: Directory to save the PNG.
        stage: Stage label used in filename (e.g. "pre_vs_post_setfit").
        max_points: Subsample cap (shared for both embeddings).
        pca_dims: Reduce to this many PCA dims before t-SNE.
        random_state: Reproducibility seed.
        save: If True, save the figure to disk.
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    labels = np.asarray(labels)
    n = embeddings_a.shape[0]
    if n < 2:
        return

    if n > max_points:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=max_points, replace=False)
    else:
        idx = np.arange(n)

    X1 = embeddings_a[idx].copy()
    X2 = embeddings_b[idx].copy()
    y = labels[idx].copy()
    n_used = len(idx)
    print(f"Running t-SNE on {n_used} samples")

    if X1.shape[1] > pca_dims:
        pca = PCA(n_components=pca_dims, random_state=random_state)
        X1 = pca.fit_transform(X1)
        X2 = pca.transform(X2)

    def _run_tsne(X):
        return TSNE(
            n_components=2,
            perplexity=min(30, len(X) - 1),
            method="barnes_hut",
            init="pca",
            learning_rate="auto",
            max_iter=1000,
            random_state=random_state,
        ).fit_transform(X)

    X1_2d = _run_tsne(X1)
    X2_2d = _run_tsne(X2)

    df1 = pd.DataFrame({"x": X1_2d[:, 0], "y": X1_2d[:, 1], "label": y})
    df2 = pd.DataFrame({"x": X2_2d[:, 0], "y": X2_2d[:, 1], "label": y})

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    sns.scatterplot(data=df1, x="x", y="y", hue="label", alpha=0.7, ax=axes[0])
    axes[0].set_title(title_a)
    sns.scatterplot(data=df2, x="x", y="y", hue="label", alpha=0.7, ax=axes[1], legend=False)
    axes[1].set_title(title_b)
    fig.suptitle(f"t-SNE Embedding Comparison — {dataset_short}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save:
        os.makedirs(save_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"tsne_compare_{dataset_short}_{encoder_short}_{stage}_n{n_used}_{ts}.png"
        fpath = os.path.join(save_dir, fname)
        plt.savefig(fpath, dpi=200, bbox_inches="tight")
        print(f"Saved: {fpath}")

    plt.show()


def plot_semi_supervised_benchmark(
    results_df,
    metric="roc_auc",
    baseline_value=None,
    baseline_label=None,
    title=None,
    figsize=(10, 6),
    save_path=None,
):
    """
    Line plot of benchmark results from run_semi_supervised_benchmark.

    Args:
        results_df: DataFrame returned by run_semi_supervised_benchmark.
        metric: Column to plot on the y-axis.
        baseline_value: Optional horizontal reference line (e.g. unsupervised AUC).
        baseline_label: Legend label for the baseline line.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional file path to save the figure.
    """
    plt.figure(figsize=figsize)
    sns.lineplot(
        data=results_df,
        x="n_known_anomalies",
        y=metric,
        hue="method",
        marker="o",
        linewidth=2.5,
    )

    if baseline_value is not None:
        label = baseline_label or f"Baseline ({metric.upper()}={baseline_value:.3f})"
        plt.axhline(y=baseline_value, linestyle="--", linewidth=2, label=label)

    xs = sorted(results_df["n_known_anomalies"].unique())
    plt.xscale("log")
    plt.xticks(xs, xs)
    plt.xlabel("Number of Known Anomalies (log scale)")
    plt.ylabel(metric.replace("_", " ").upper())
    plt.title(title or "Semi-Supervised Anomaly Detection Benchmark")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    plt.show()