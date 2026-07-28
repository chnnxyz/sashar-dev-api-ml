"""Classification task: LogisticRegression / SVM / LightGBM → MLRunResult.

Scatter coordinates are the 2D PCA projection of the standardized features
(matching the frontend's PCA-projected class scatter)."""
from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from app.data.loader import feature_columns, load_dataframe, seeded_split
from app.data.pca import project_2d
from app.ml import metrics
from app.schemas import MLRunResult, RunMLRequest, ScatterPoint

DEFAULT_DATASET = "iris"


def _build_classifier(model: str, hp: dict):
    if model == "logistic_regression":
        return LogisticRegression(
            C=float(hp.get("C", 1.0)), max_iter=int(hp.get("max_iter", 100)),
            tol=float(hp.get("tol", 1e-4)),
        )
    if model == "svm":
        return SVC(
            C=float(hp.get("C", 1.0)), gamma=float(hp.get("gamma", 0.1)),
            degree=int(hp.get("degree", 3)), coef0=float(hp.get("coef0", 0.0)),
            probability=True, random_state=42,
        )
    if model == "lightgbm":
        return LGBMClassifier(
            n_estimators=int(hp.get("n_estimators", 100)),
            learning_rate=float(hp.get("learning_rate", 0.1)),
            num_leaves=int(hp.get("num_leaves", 31)),
            max_depth=int(hp.get("max_depth", -1)), verbose=-1,
        )
    raise ValueError(f"unknown classification model: {model}")


def _auc(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    try:
        if n_classes == 2:
            return float(roc_auc_score(y_true, proba[:, 1]))
        return float(roc_auc_score(y_true, proba, multi_class="ovr"))
    except Exception:
        return 0.0


def run_classification(req: RunMLRequest) -> MLRunResult:
    dataset = req.dataset or DEFAULT_DATASET
    df = load_dataframe(dataset)
    features = feature_columns(df)

    x = df[features].to_numpy(dtype=float)
    labels = df["target"].astype(str).to_numpy()
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    n_classes = len(encoder.classes_)
    n = len(df)
    train_mask = seeded_split(n, req.seed)

    clf = _build_classifier(req.model, req.hyperparameters)
    clf.fit(x[train_mask], y[train_mask])
    y_pred = clf.predict(x)
    proba = clf.predict_proba(x)

    coords, _ = project_2d(x)

    scatter = [
        ScatterPoint(x=float(coords[i, 0]), y=float(coords[i, 1]),
                     label=labels[i], is_train=bool(train_mask[i]))
        for i in range(n)
    ]
    test_predictions = []
    for i in range(n):
        if train_mask[i]:
            continue
        pred_label = encoder.classes_[y_pred[i]]
        test_predictions.append(
            ScatterPoint(
                x=float(coords[i, 0]), y=float(coords[i, 1]), label=labels[i],
                is_train=False, correct=bool(y_pred[i] == y[i]),
                predicted_label=str(pred_label),
                predicted_prob=float(proba[i].max()),
            )
        )

    test_idx = ~train_mask
    acc = accuracy_score(y[test_idx], y_pred[test_idx])
    f1 = f1_score(y[test_idx], y_pred[test_idx], average="macro")
    auc = _auc(y[test_idx], proba[test_idx], n_classes)

    # CV card uses misclassification rate as the "error" proxy.
    train_err = 1.0 - accuracy_score(y[train_mask], y_pred[train_mask])
    test_err = 1.0 - acc
    folds = []
    for tr, te in KFold(n_splits=5, shuffle=True, random_state=42).split(x):
        est = clone(clf)
        est.fit(x[tr], y[tr])
        folds.append(1.0 - accuracy_score(y[te], est.predict(x[te])))

    return MLRunResult(
        scatter=scatter,
        test_predictions=test_predictions,
        metrics={
            "accuracy": round(float(acc), 4),
            "f1": round(float(f1), 4),
            "auc_roc": round(float(auc), 4),
        },
        model=req.model,
        cv=metrics.make_cv_metrics(
            train_err, test_err, int(train_mask.sum()), int(test_idx.sum()), folds
        ),
    )
