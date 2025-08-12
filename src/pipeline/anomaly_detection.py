from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, pairwise_distances
import pandas as pd
import os
import uuid

def label_normal_vs_anomaly(labels_df, verbose=True, as_df=False):
    """
    Converts a multiclass label DataFrame into binary anomaly labels.
    
    The most frequent class is considered 'normal' (label 0),
    and all other classes are labeled as anomalies (label 1).
    
    Parameters:
    - labels_df: pandas.DataFrame with a column 'label'.
    - verbose: bool, if True, prints info about label mapping.
    - as_df: bool, if True, returns a DataFrame with column 'label'.
    
    Returns:
    - binary_labels: numpy.ndarray or pandas.DataFrame (based on as_df)
    """
    import numpy as np
    import pandas as pd

    labels = labels_df['label'].values

    # Count label frequencies
    unique, counts = np.unique(labels, return_counts=True)
    label_freq = dict(zip(unique, counts))

    # Find the most frequent label (normal class)
    normal_label = max(label_freq, key=label_freq.get)

    if verbose:
        print(f"✅ Selected normal label (inlier → 0): {normal_label}")
        anomaly_labels = [lbl for lbl in unique if lbl != normal_label]
        print(f"⚠️  Anomaly labels (outlier → 1): {anomaly_labels}")

    # Convert to binary: 0 = inlier, 1 = outlier
    binary_labels = np.array([0 if label == normal_label else 1 for label in labels])
    
    if as_df:
        return pd.DataFrame({'label': binary_labels})
    else:
        return binary_labels

def adjust_contamination(texts, labels, embeddings, perc_anomalous=0.05, random_state=42):
    """
    Downsample anomalies to achieve a specified contamination rate.

    Parameters:
    - texts: list of str (original texts)
    - labels: array-like (binary labels: 0 = normal, 1 = anomaly)
    - embeddings: np.ndarray (embedding matrix, shape = [n_samples, dim])
    - perc_anomalous: float (desired proportion of anomalies, e.g., 0.05 = 5%)
    - random_state: int (for reproducibility)

    Returns:
    - filtered_texts: list of str
    - filtered_labels: np.ndarray
    - filtered_embeddings: np.ndarray
    """

    df = pd.DataFrame({
        "text": texts,
        "label": labels,
        "embedding": list(embeddings)
    })

    normal_df = df[df["label"] == 0]
    anomalous_df = df[df["label"] == 1]

    n_anomalies = int(len(normal_df) * perc_anomalous)
    if n_anomalies > len(anomalous_df):
        raise ValueError(f"Not enough anomalies to achieve {perc_anomalous*100:.1f}% contamination.")

    anomalous_sample = anomalous_df.sample(n=n_anomalies, random_state=random_state)

    df_balanced = pd.concat([normal_df, anomalous_sample])
    df_balanced = df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)

    filtered_texts = df_balanced["text"].tolist()
    filtered_labels = df_balanced["label"].values
    filtered_embeddings = np.stack(df_balanced["embedding"].values)

    return filtered_texts, filtered_labels, filtered_embeddings

def pretty_print_data_info(texts, labels, embeddings):
    """
    Print information about the dataset.
    Parameters:
    - texts: list of str (original texts)
    - labels: array-like (binary labels: 0 = normal, 1 = anomaly)
    - embeddings: np.ndarray (embedding matrix, shape = [n_samples, dim])
    """

    print('Shape of embeddings:', embeddings.shape)
    print('Shape of labels:', labels.shape)
    print('Number of anomalies:', np.sum(labels))
    print('Number of normal:', len(labels) - np.sum(labels))
    print('Percentage of anomalies:', np.sum(labels) / len(labels))
    print('Shape of texts:', len(texts))


def split_data(embeddings, labels, test_size=0.2, random_state=42):
    """
    Split the data into train and test sets in a stratified way to preserve class balance.

    Parameters:
    - embeddings (np.ndarray): Feature vectors (e.g., sentence embeddings), shape (n_samples, n_features)
    - labels (np.ndarray): Binary labels (0 = inlier, 1 = outlier)
    - test_size (float): Proportion of the dataset to include in the test split
    - random_state (int): Random seed for reproducibility

    Returns:
    - x_train, x_test, y_train, y_test: Stratified train/test splits
    """

    x_train, x_test, y_train, y_test = train_test_split(
        embeddings,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )

    # Counts
    n_train_outliers = np.sum(y_train == 1)
    n_test_outliers = np.sum(y_test == 1)

    # Print shapes and anomaly counts
    print(f"Train set: {x_train.shape} (with {n_train_outliers} outliers and {len(x_train) - n_train_outliers} inliers)")
    print(f"Test  set: {x_test.shape} (with {n_test_outliers} outliers and {len(x_test) - n_test_outliers} inliers)")
    return x_train, x_test, y_train, y_test


def label_partial_outliers(y_train, n_known=None, contamination=None, random_state=42):
    """
    Create a partially labeled version of the anomaly labels.

    Parameters
    ----------
    y_train : array-like
        Full binary ground-truth labels (1 = anomaly, 0 = normal).

    n_known : int, optional
        Absolute number of labeled anomalies to include in the semi-supervised label array.
        If provided, takes precedence over `contamination`.

    contamination : float, optional
        Fraction of the full dataset to label as anomalies (e.g., 0.01 for 1% of the data).
        Only used if `n_known` is None.

    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    y_train_semi : np.ndarray
        Array of same shape as `y_train`, where:
        - 1 indicates a labeled anomaly,
        - 0 indicates an unlabeled sample.
    """
    y_train = np.asarray(y_train)
    y_train_semi = np.zeros_like(y_train)
    out_idx = np.where(y_train == 1)[0]
    rng = np.random.default_rng(seed=random_state)

    if n_known is not None:
        n_select = min(n_known, len(out_idx))
    elif contamination is not None:
        n_select = min(int(contamination * len(y_train)), len(out_idx))
    else:
        raise ValueError("You must specify either `n_known` or `contamination`.")

    if n_select == 0:
        return y_train_semi

    known_out_idx = rng.choice(out_idx, n_select, replace=False)
    y_train_semi[known_out_idx] = 1

    return y_train_semi

