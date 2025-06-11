import gc

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate


def sparse_to_dense(sparse):
    num_classes = len(sparse)
    num_samples = sparse[0].shape[0]
    dense = np.zeros(shape=(num_samples, num_classes), dtype=np.float32)
    for idx, preds in enumerate(sparse):
        dense[:, idx] = preds[:, 1]
    return dense


def tabulate_metrics(metrics):
    headers = list(metrics.keys())
    data = [list(metrics.values())]
    print(tabulate(data, headers=headers, tablefmt="grid"))


def eval_knn(
    x_train, y_train, x_test, y_test, k=5, scale=False, return_preds=False, multilabel=False
):
    if scale:
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

    model = KNeighborsClassifier(n_neighbors=k, algorithm="brute", n_jobs=4)

    model.fit(X=x_train, y=y_train)
    y_pred = model.predict(x_test)
    metrics = {
        "overall_accuracy": accuracy_score(y_test, y_pred),
        "overall_precision": precision_score(y_test, y_pred, average="micro"),
        "average_precision": precision_score(y_test, y_pred, average="macro"),
        "overall_recall": recall_score(y_test, y_pred, average="micro"),
        "average_recall": recall_score(y_test, y_pred, average="macro"),
        "overall_f1": f1_score(y_test, y_pred, average="micro"),
        "average_f1": f1_score(y_test, y_pred, average="macro"),
    }

    if multilabel:
        score = model.predict_proba(x_test)
        score = sparse_to_dense(score)

        metrics.update(
            {
                "overall_map": average_precision_score(y_test, score, average="micro"),
                "average_map": average_precision_score(y_test, score, average="macro"),
            }
        )

    tabulate_metrics(metrics)

    del model, x_train, x_test
    torch.cuda.empty_cache()
    gc.collect()

    if return_preds:
        return metrics, y_pred
    else:
        return metrics


def eval_linear_probe(
    x_train,
    y_train,
    x_test,
    y_test,
    scale=False,
    max_iter=1000,
    seed=0,
    return_preds=False,
    multilabel=False,
    verbose=False,
):
    if scale:
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

    model = LogisticRegression(
        max_iter=max_iter, solver="sag", random_state=seed, verbose=1 if verbose else 0
    )

    if multilabel:
        model = OneVsRestClassifier(model)

    model.fit(X=x_train, y=y_train)
    y_pred = model.predict(x_test)
    metrics = {
        "overall_accuracy": accuracy_score(y_test, y_pred),
        "overall_precision": precision_score(y_test, y_pred, average="micro"),
        "average_precision": precision_score(y_test, y_pred, average="macro"),
        "overall_recall": recall_score(y_test, y_pred, average="micro"),
        "average_recall": recall_score(y_test, y_pred, average="macro"),
        "overall_f1": f1_score(y_test, y_pred, average="micro"),
        "average_f1": f1_score(y_test, y_pred, average="macro"),
    }

    if multilabel:
        score = model.predict_proba(x_test)
        metrics.update(
            {
                "overall_map": average_precision_score(y_test, score, average="micro"),
                "average_map": average_precision_score(y_test, score, average="macro"),
            }
        )

    tabulate_metrics(metrics)

    del model, x_train, x_test
    torch.cuda.empty_cache()
    gc.collect()

    if return_preds:
        return metrics, y_pred
    else:
        return metrics
