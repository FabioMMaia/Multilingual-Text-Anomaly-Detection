
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

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