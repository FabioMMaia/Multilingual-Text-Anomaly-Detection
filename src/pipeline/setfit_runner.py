"""
SetFit runner: data splitting, model training/evaluation, and semi-supervised
benchmark loop for anomaly detection experiments.
"""

import inspect
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(preds, labels):
    """Return accuracy, f1, precision and recall for binary classification."""
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary"),
        "precision": precision_score(labels, preds, average="binary"),
        "recall": recall_score(labels, preds, average="binary"),
    }


# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------

def prepare_data_splits_for_anomaly_detection(
    texts: list,
    embeddings: np.ndarray,
    labels: np.ndarray,
    initial_test_size: float = 0.2,
    k_per_class: int = 20,
    setfit_eval_size: float = 0.3,
    random_state: int = 42,
    verbose: bool = True,
) -> tuple:
    """
    Split data for the SetFit + downstream anomaly detection pipeline.

    1. Stratified train/test split.
    2. From the training pool, extract k_per_class few-shot samples per class
       for SetFit fine-tuning (split further into SetFit train / eval).
    3. The remaining training samples form the downstream training pool.

    Args:
        texts: Raw text samples.
        embeddings: Pre-computed sentence embeddings (N x D).
        labels: Binary labels — 0 = normal, 1 = anomaly.
        initial_test_size: Fraction held out as the final test set.
        k_per_class: Few-shot budget per class for SetFit.
        setfit_eval_size: Fraction of few-shot samples used for SetFit eval.
        random_state: Reproducibility seed.
        verbose: Print split statistics.

    Returns:
        downstream_train: dict with keys 'texts', 'embeddings', 'labels'.
            Full training pool (few-shot samples included for the anomaly models).
        test_set: dict with keys 'texts', 'embeddings', 'labels'.
        setfit_data: dict with keys 'train_dataset', 'eval_dataset'
            (Hugging Face Dataset objects for SetFit training).
        remaining_downstream: dict with keys 'texts', 'embeddings', 'labels'.
            Training pool *excluding* the few-shot samples.
    """
    texts_arr = np.array(texts, dtype=object)
    labels_arr = np.array(labels, dtype=int)

    (
        texts_train, texts_test,
        embs_train, embs_test,
        labels_train, labels_test,
    ) = train_test_split(
        texts_arr, embeddings, labels_arr,
        test_size=initial_test_size,
        stratify=labels_arr,
        random_state=random_state,
    )

    if verbose:
        print("Initial split:")
        print(f"  Train pool: {embs_train.shape}  |  Test set: {embs_test.shape}")

    rng = np.random.default_rng(random_state)
    idx_by_class = defaultdict(list)
    for i, y in enumerate(labels_train):
        idx_by_class[int(y)].append(i)

    few_idx = []
    for y, idxs in idx_by_class.items():
        take = min(k_per_class, len(idxs))
        few_idx.extend(rng.choice(idxs, size=take, replace=False).tolist())

    few_texts = texts_train[few_idx].tolist()
    few_labels = labels_train[few_idx].tolist()

    if verbose:
        print(
            f"\nFew-shot samples: {len(few_texts)} "
            f"({few_labels.count(1)} anomalies, {few_labels.count(0)} normal)"
        )

    sf_train_texts, sf_eval_texts, sf_train_labels, sf_eval_labels = train_test_split(
        few_texts, few_labels,
        test_size=setfit_eval_size,
        stratify=few_labels,
        random_state=random_state,
    )

    if verbose:
        print(
            f"  SetFit train: {len(sf_train_texts)} samples  |  "
            f"SetFit eval: {len(sf_eval_texts)} samples"
        )

    setfit_data = {
        "train_dataset": Dataset.from_dict({"text": sf_train_texts, "label": sf_train_labels}),
        "eval_dataset": Dataset.from_dict({"text": sf_eval_texts, "label": sf_eval_labels}),
    }

    remaining_idx = np.setdiff1d(np.arange(len(texts_train)), few_idx)

    downstream_train = {
        "texts": texts_train.tolist(),
        "embeddings": embs_train,
        "labels": labels_train.tolist(),
    }
    test_set = {
        "texts": texts_test.tolist(),
        "embeddings": embs_test,
        "labels": labels_test.tolist(),
    }
    remaining_downstream = {
        "texts": texts_train[remaining_idx].tolist(),
        "embeddings": embs_train[remaining_idx],
        "labels": labels_train[remaining_idx].tolist(),
    }

    if verbose:
        print(f"\nRemaining downstream pool: {len(remaining_downstream['texts'])} samples")

    return downstream_train, test_set, setfit_data, remaining_downstream


# ---------------------------------------------------------------------------
# SetFit training and evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate_setfit_model(
    model,
    setfit_data: dict,
    test_set: dict,
    loss_class=None,
    num_iterations: int = 10,
    num_epochs: int = 1,
    column_mapping: dict = None,
    batch_size: int = 4,
    seed: int = 42,
    verbose: bool = True,
) -> tuple:
    """
    Train a SetFit model and evaluate it on both the few-shot train split and
    the held-out test set.

    Args:
        model: A SetFitModel instance (from SetFitModel.from_pretrained).
        setfit_data: dict with 'train_dataset' and 'eval_dataset' (HF Datasets).
        test_set: dict with 'texts' and 'labels'.
        loss_class: Sentence-transformers loss class (default: CosineSimilarityLoss).
        num_iterations: SetFitTrainer num_iterations.
        num_epochs: SetFitTrainer num_epochs.
        column_mapping: Mapping of column names (default: {"text": "text", "label": "label"}).
        batch_size: SetFitTrainer batch_size.
        seed: Random seed for the trainer.
        verbose: Print training/evaluation progress.

    Returns:
        metrics: dict with keys 'train' and 'test', each containing accuracy,
            f1, precision, recall, roc_auc, pr_auc.
        trainer: Trained SetFitTrainer object (gives access to fine-tuned model).
    """
    from sentence_transformers.losses import CosineSimilarityLoss
    from setfit import SetFitTrainer

    if loss_class is None:
        loss_class = CosineSimilarityLoss
    if column_mapping is None:
        column_mapping = {"text": "text", "label": "label"}

    text_col = column_mapping["text"]
    label_col = column_mapping["label"]

    if verbose:
        print("\nStarting SetFit training...")

    trainer = SetFitTrainer(
        model=model,
        train_dataset=setfit_data["train_dataset"],
        eval_dataset=setfit_data["eval_dataset"],
        loss_class=loss_class,
        num_iterations=num_iterations,
        num_epochs=num_epochs,
        column_mapping=column_mapping,
        metric=compute_metrics,
        batch_size=batch_size,
        seed=seed,
    )
    trainer.train()

    # --- train-set metrics ---
    train_texts = setfit_data["train_dataset"][text_col]
    train_labels = setfit_data["train_dataset"][label_col]

    y_pred_train = trainer.model.predict(train_texts)
    y_proba_train = trainer.model.predict_proba(train_texts)[:, 1]

    train_metrics = compute_metrics(y_pred_train, train_labels)
    train_metrics["roc_auc"] = roc_auc_score(train_labels, y_proba_train)
    train_metrics["pr_auc"] = average_precision_score(train_labels, y_proba_train)

    # --- test-set metrics ---
    y_pred_test = trainer.model.predict(test_set["texts"])
    y_proba_test = trainer.model.predict_proba(test_set["texts"])[:, 1]

    test_metrics = compute_metrics(y_pred_test, test_set["labels"])
    test_metrics["roc_auc"] = roc_auc_score(test_set["labels"], y_proba_test)
    test_metrics["pr_auc"] = average_precision_score(test_set["labels"], y_proba_test)

    if verbose:
        print("\n=== SetFit — Train Set ===")
        print(train_metrics)
        print("\n=== SetFit — Test Set ===")
        print(test_metrics)

    return {"train": train_metrics, "test": test_metrics}, trainer


# ---------------------------------------------------------------------------
# SetFit embedding extraction
# ---------------------------------------------------------------------------

def encode_with_setfit(trainer, downstream_train: dict, test_set: dict) -> None:
    """
    Encode downstream_train and test_set texts with the fine-tuned SetFit model
    body and store the results in-place under the key 'embeddings_sf'.

    Args:
        trainer: A trained SetFitTrainer (from train_and_evaluate_setfit_model).
        downstream_train: dict with at least a 'texts' key. Modified in-place.
        test_set: dict with at least a 'texts' key. Modified in-place.
    """
    body = trainer.model.model_body
    downstream_train["embeddings_sf"] = body.encode(downstream_train["texts"])
    test_set["embeddings_sf"] = body.encode(test_set["texts"])


# ---------------------------------------------------------------------------
# Semi-supervised benchmark
# ---------------------------------------------------------------------------

def build_n_known_list(total_available: int, base: list = None) -> list:
    """
    Return the canonical n_known budget values that fit within the dataset.

    Uses a fixed set of absolute values so results are comparable across runs
    and datasets. Values exceeding total_available are silently dropped.

    Args:
        total_available: Number of anomalies available in the training pool.
        base: Candidate values (default: [5, 10, 20, 50, 100, 200, 500]).

    Returns:
        Sorted list of valid n_known values.
    """
    if base is None:
        base = [5, 10, 20, 50, 100, 200, 500]

    valid = [n for n in base if n <= total_available]
    if not valid:
        print(
            f"[WARNING] Dataset supports fewer than {min(base)} anomalies "
            f"(total_available={total_available}). No valid n_known values."
        )
    return valid


def run_semi_supervised_benchmark(
    model_cls,
    model_kwargs: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_known_list: list,
    method_name: str,
    seed: int = 42,
    device: str = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run a semi-supervised anomaly detection benchmark over varying n_known budgets.

    For each value n in n_known_list, exactly n anomaly indices are randomly
    sampled from the training pool and provided as supervision signal.  The
    remaining training labels are treated as fully unlabeled (0).

    Args:
        model_cls: Anomaly detector class (e.g. DevNet, DeepSAD, MLPTF).
            Must implement fit(X, y) and decision_function(X).
        model_kwargs: Keyword arguments passed to model_cls() (excluding
            device, random_state, input_dim — injected automatically when
            the constructor accepts them).
        X_train: Training embeddings (N x D).
        y_train: True binary training labels (used only to locate anomaly indices).
        X_test: Test embeddings.
        y_test: True binary test labels.
        n_known_list: List of known-anomaly budgets to evaluate.
        method_name: Label for this run (appears in the 'method' column).
        seed: Random seed for anomaly sampling.
        device: 'cuda' or 'cpu'. Auto-detected when None.
        verbose: Print per-n timing.

    Returns:
        DataFrame with columns: method, model, n_known_anomalies, roc_auc,
        pr_auc, train_time_sec, inference_time_sec, total_time_sec.
    """
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    all_anomaly_idxs = np.where(y_train == 1)[0]
    total_available = len(all_anomaly_idxs)

    if verbose:
        print(f"\n[{method_name}] Available anomalies: {total_available}")
        print(f"[{method_name}] n_known_list = {n_known_list}")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    init_params = inspect.signature(model_cls.__init__).parameters
    rng = np.random.default_rng(seed)
    results = []

    for n in n_known_list:
        if n > total_available:
            continue

        sampled_idxs = rng.choice(all_anomaly_idxs, size=n, replace=False)
        y_semi = np.zeros(len(y_train), dtype=int)
        y_semi[sampled_idxs] = 1

        kwargs = dict(model_kwargs)
        if "device" in init_params:
            kwargs["device"] = device
        if "random_state" in init_params:
            kwargs["random_state"] = seed
        if "input_dim" in init_params:
            kwargs["input_dim"] = X_train.shape[1]

        clf = model_cls(**kwargs)

        t0 = time.perf_counter()
        clf.fit(X_train, y_semi)
        train_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        scores = clf.decision_function(X_test)
        inference_time = time.perf_counter() - t1

        results.append({
            "method": method_name,
            "model": model_cls.__name__,
            "n_known_anomalies": n,
            "roc_auc": roc_auc_score(y_test, scores),
            "pr_auc": average_precision_score(y_test, scores),
            "train_time_sec": train_time,
            "inference_time_sec": inference_time,
            "total_time_sec": train_time + inference_time,
        })

        if verbose:
            print(
                f"  n={n:4d} | roc_auc={results[-1]['roc_auc']:.4f} | "
                f"train={train_time:.2f}s | infer={inference_time:.3f}s"
            )

    return pd.DataFrame(results)
