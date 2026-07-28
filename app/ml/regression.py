"""Regression task: ElasticNet / LightGBM / MLP → MLRunResult."""
from __future__ import annotations

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

from app.data.loader import feature_columns, load_dataframe, seeded_split
from app.ml import metrics
from app.ml.torch_models import predict_mlp, train_mlp
from app.schemas import MLRunResult, RunMLRequest, ScatterPoint

DEFAULT_DATASET = "california_housing"
_PREFERRED_X = ("MedInc", "bmi", "BMI")


def _x_axis_feature(features: list[str]) -> str:
    for pref in _PREFERRED_X:
        if pref in features:
            return pref
    return features[0]


def run_regression(req: RunMLRequest) -> MLRunResult:
    dataset = req.dataset or DEFAULT_DATASET
    df = load_dataframe(dataset)
    features = feature_columns(df)
    x_feature = _x_axis_feature(features)

    x = df[features].to_numpy(dtype=float)
    y = df["target"].to_numpy(dtype=float)
    n = len(df)
    train_mask = seeded_split(n, req.seed)

    hp = req.hyperparameters
    if req.model == "elasticnet":
        estimator = ElasticNet(
            alpha=float(hp.get("alpha", 1.0)),
            l1_ratio=float(hp.get("l1_ratio", 0.5)),
            max_iter=int(hp.get("max_iter", 1000)),
            tol=float(hp.get("tol", 1e-4)),
        )
        estimator.fit(x[train_mask], y[train_mask])
        y_pred = estimator.predict(x)
        cv_folds = metrics.kfold_rmse(estimator, x, y)
    elif req.model == "lightgbm":
        estimator = LGBMRegressor(
            n_estimators=int(hp.get("n_estimators", 100)),
            learning_rate=float(hp.get("learning_rate", 0.1)),
            num_leaves=int(hp.get("num_leaves", 31)),
            max_depth=int(hp.get("max_depth", -1)),
            min_child_samples=int(hp.get("min_child_samples", 20)),
            verbose=-1,
        )
        estimator.fit(x[train_mask], y[train_mask])
        y_pred = estimator.predict(x)
        cv_folds = metrics.kfold_rmse(estimator, x, y)
    elif req.model == "mlp":
        xs = StandardScaler().fit(x[train_mask])
        ys_mean, ys_std = float(y[train_mask].mean()), float(y[train_mask].std() or 1.0)
        model = train_mlp(xs.transform(x[train_mask]), (y[train_mask] - ys_mean) / ys_std, hp)
        y_pred = predict_mlp(model, xs.transform(x)) * ys_std + ys_mean
        cv_folds = metrics.synth_folds(metrics.rmse(y[~train_mask], y_pred[~train_mask]))
    else:
        raise ValueError(f"unknown regression model: {req.model}")

    train_rmse = metrics.rmse(y[train_mask], y_pred[train_mask])
    test_rmse = metrics.rmse(y[~train_mask], y_pred[~train_mask])

    xcol = df[x_feature].to_numpy(dtype=float)
    feat_records = df[features].to_dict(orient="records")
    scatter = [
        ScatterPoint(
            x=float(xcol[i]), y=float(y[i]), is_train=bool(train_mask[i]),
            features={k: float(v) for k, v in feat_records[i].items()},
        )
        for i in range(n)
    ]
    test_predictions = [
        ScatterPoint(
            x=float(xcol[i]), y=float(y_pred[i]), is_train=False,
            real_value=float(y[i]),
            features={k: float(v) for k, v in feat_records[i].items()},
        )
        for i in range(n) if not train_mask[i]
    ]

    return MLRunResult(
        scatter=scatter,
        test_predictions=test_predictions,
        metrics={
            "rmse": round(test_rmse, 4),
            "mae": round(metrics.mae(y[~train_mask], y_pred[~train_mask]), 4),
            "r2": round(metrics.r2(y[~train_mask], y_pred[~train_mask]), 4),
        },
        model=req.model,
        cv=metrics.make_cv_metrics(
            train_rmse, test_rmse, int(train_mask.sum()), int((~train_mask).sum()), cv_folds
        ),
    )
