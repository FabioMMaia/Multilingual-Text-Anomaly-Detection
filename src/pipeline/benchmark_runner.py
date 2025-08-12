import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, pairwise_distances
from typing import Callable
import json
import uuid
from .anomaly_detection import label_partial_outliers

def evaluate_model(model, x_train, y_train_true, y_train_semi, x_test, y_test_true, uncertainty_model=False, use_labels=True):
    if use_labels:
        model.fit(x_train, y=y_train_semi)
    else:
        model.fit(x_train)

    scores = {}

    if uncertainty_model:
        train_scores, train_uncertainty = model.decision_function(x_train)
        test_scores, test_uncertainty = model.decision_function(x_test)
        scores.update({
            "train_scores": train_scores,
            "train_uncertainty": train_uncertainty,
            "test_scores": test_scores,
            "test_uncertainty": test_uncertainty
        })
    else:
        train_scores = model.decision_function(x_train)
        test_scores = model.decision_function(x_test)
        scores.update({
            "train_scores": train_scores,
            "test_scores": test_scores
        })

    metrics = {
        "train_size": len(x_train),
        "test_size": len(x_test),
        "train_auc_semi": roc_auc_score(y_train_semi, train_scores),
        "train_auc": roc_auc_score(y_train_true, train_scores),
        "test_auc": roc_auc_score(y_test_true, test_scores),
        "train_ap_semi": average_precision_score(y_train_semi, train_scores),
        "train_ap": average_precision_score(y_train_true, train_scores),
        "test_ap": average_precision_score(y_test_true, test_scores),
    }

    return metrics, scores

def save_results(metrics, model_name, dataset_name, n_known,  results_dir="experiments_results"):
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"{model_name}_{dataset_name}.csv")

    row = {
        "model": model_name,
        "dataset": dataset_name,
        "n_known_outliers": n_known,
        "train_size":metrics["train_size"],
        "test_size": metrics["test_size"],
        "train_auc_semi": metrics["train_auc_semi"],
        "train_auc": metrics["train_auc"],
        "test_auc": metrics["test_auc"],
        "train_ap_semi": metrics["train_ap_semi"],
        "train_ap": metrics["train_ap"],
        "test_ap": metrics["test_ap"],
    }

    df = pd.DataFrame([row])

    if os.path.exists(results_path):
        df.to_csv(results_path, mode="a", header=False, index=False)
    else:
        df.to_csv(results_path, index=False)


def append_labeled_indices_compact(run_id, contamination, round_id, indices, results_dir):
    """
    Save known labeled indices as a compact CSV: one row per labeling scenario.
    """
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "labelled_anomalies.csv")

    row = {
        "run_id": str(run_id),
        "contamination": contamination,
        "round": round_id,
        "indices": json.dumps(indices)  # safely store list as a string
    }

    df = pd.DataFrame([row])
    if os.path.exists(path):
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def save_benchmark_result(metrics, model_name, dataset_name, contamination, round_id,
                          known_indices, run_id, results_dir="experiments_results"):
    """
    Save the metrics from the model run into a single benchmark_results.csv.
    """
    row = {
        "run_id": str(run_id),
        "model": model_name,
        "dataset": dataset_name,
        "contamination": contamination,
        "round": round_id,
        "n_known_outliers": len(known_indices),
        "train_size": metrics.get("train_size"),
        "test_size": metrics.get("test_size"),
        "train_auc_semi": metrics.get("train_auc_semi"),
        "train_auc": metrics.get("train_auc"),
        "test_auc": metrics.get("test_auc"),
        "train_ap_semi": metrics.get("train_ap_semi"),
        "train_ap": metrics.get("train_ap"),
        "test_ap": metrics.get("test_ap"),
    }

    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "benchmark_results.csv")
    df = pd.DataFrame([row])
    if os.path.exists(path):
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)

def benchmark_unsupervised_models(
    x_train, y_train, x_test, y_test,
    model_constructor: dict[str, Callable],
    dataset_name="dataset",
    results_dir="experiments_results",
    random_state=42,
):
    # Single run only
    run_ids = []
    run_id = uuid.uuid4()

    for model_name, model_fn in model_constructor.items():
        print(f"→ Running {model_name} [unsupervised]")
        model = model_fn()

        try:
            metrics, scores = evaluate_model(
                model,
                x_train,
                y_train,
                np.zeros_like(y_train),  # no labels
                x_test,
                y_test,
                use_labels=False,
                uncertainty_model=False
            )

            save_benchmark_result(
                metrics=metrics,
                model_name=model_name,
                dataset_name=dataset_name,
                contamination=None,
                round_id=0,
                known_indices=[],
                run_id=run_id,
                results_dir=results_dir
            )

            run_ids.append(str(run_id))

        except Exception as e:
            print(f"[ERROR] {model_name} failed: {e}")
            continue

    return run_ids

def benchmark_semisupervised_models(
    x_train, y_train, x_test, y_test,
    model_constructor: dict[str, tuple[Callable, bool]],  # (model_fn, uncertainty_model)
    dataset_name="dataset",
    contamination_levels = [0.01, 0.02, 0.03, 0.04, 0.05],
    n_rounds=5,
    results_dir="experiments_results",
    random_state=42,
):
    run_ids = []

    for contamination in contamination_levels:
        for round_id in range(n_rounds):
            seed = random_state + round_id
            y_train_semi = label_partial_outliers(y_train=y_train, contamination=contamination, random_state=seed)
            known_indices = np.where(y_train_semi == 1)[0]
            run_id = uuid.uuid4()

            append_labeled_indices_compact(
                run_id=run_id,
                contamination=contamination,
                round_id=round_id,
                indices=known_indices.tolist(),
                results_dir=results_dir
            )

            for model_name, (model_fn, uncertainty) in model_constructor.items():
                print(f"→ Running {model_name} (uncertainty_model={uncertainty})")
                model = model_fn()

                try:
                    metrics, scores = evaluate_model(
                        model,
                        x_train,
                        y_train,
                        y_train_semi,
                        x_test,
                        y_test,
                        use_labels=True,
                        uncertainty_model=uncertainty
                    )

                    save_benchmark_result(
                        metrics=metrics,
                        model_name=model_name,
                        dataset_name=dataset_name,
                        contamination=contamination,
                        round_id=round_id,
                        known_indices=known_indices.tolist(),
                        run_id=run_id,
                        results_dir=results_dir
                    )

                    run_ids.append(str(run_id))

                except Exception as e:
                    print(f"[ERROR] {model_name} failed: {e}")
                    continue

    return run_ids

