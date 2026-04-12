"""
metrics.py
----------
Diagnostic metrics for anomaly detection experiments.
These operate on pre-computed embeddings + ground-truth labels and are
intended for post-hoc analysis, NOT as training signals.
"""

import numpy as np


def separation_ratio(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Embedding separability diagnostic for anomaly detection.

    Computes the ratio of (mean cosine distance from anomalies to normals)
    over (mean cosine distance among anomalies).  A high value means
    anomalies are far from normals relative to how spread the anomalies
    are among themselves — i.e., good structural separability.

        separation_ratio = mean_dist(A → N) / mean_dist(A → A)

    Values > 1  →  anomalies are farther from normals than from each other
                   (hard to detect — they blend with normal space)
    Values < 1  →  anomalies cluster away from normals (good separability)

    Parameters
    ----------
    embeddings : np.ndarray, shape (n_samples, d)
        Pre-computed embedding matrix (e.g. distiluse or SetFit).
    labels : np.ndarray, shape (n_samples,)
        Binary labels: 0 = normal, 1 = anomaly.

    Returns
    -------
    float
        Separation ratio scalar.  Returns np.nan if fewer than 2 anomalies
        or fewer than 1 normal sample exist.
    """
    embeddings = np.array(embeddings, dtype=np.float32)
    labels = np.array(labels)

    A_idx = np.where(labels == 1)[0]
    N_idx = np.where(labels == 0)[0]

    if len(A_idx) < 2 or len(N_idx) < 1:
        return np.nan

    # L2-normalise for cosine distance via dot product
    def _normalise(X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # avoid division by zero
        return X / norms

    A = _normalise(embeddings[A_idx])  # (|A|, d)

    N = _normalise(embeddings[N_idx])  # (|N|, d)

    # cosine distance = 1 - cosine_similarity
    # A → N  (|A| × |N_sample|)
    sim_AN = A @ N.T                          # (|A|, |N_sample|)
    dist_AN = 1.0 - sim_AN
    mean_dist_AN = float(dist_AN.mean())

    # A → A  (|A| × |A|), excluding diagonal
    sim_AA = A @ A.T                          # (|A|, |A|)
    np.fill_diagonal(sim_AA, np.nan)          # mask self-similarity
    dist_AA = 1.0 - sim_AA
    mean_dist_AA = float(np.nanmean(dist_AA))

    if mean_dist_AA == 0:
        return np.nan

    return mean_dist_AN / mean_dist_AA
