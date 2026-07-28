"""Hyperparameter optimization dispatch.

Three methods over a uniform objective so every trial can be streamed to the
WebSocket clients:

* ``gridsearch`` — sklearn ``ParameterGrid`` (capped combinations)
* ``tpe``        — hyperopt Tree-structured Parzen Estimator
* ``genetic``    — pygad genetic algorithm

The objective fits the actual sklearn estimator with 3-fold CV where one exists
(elasticnet, lightgbm, logistic_regression, svm, kmeans); for the torch/statsmodels
models (mlp, rnn, tes, dbscan) it uses a fast deterministic surrogate so the
no-GPU demo stays responsive. Lower score is always better.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid, cross_val_score
from sklearn.svm import SVC

from app.data.loader import feature_columns, load_dataframe
from app.schemas import OptimizeRequest, OptimizeResult

TrialCallback = Optional[Callable[[int, float, dict], None]]

# name → (low, high, is_int)
PARAM_SPACES: dict[str, dict[str, tuple[float, float, bool]]] = {
    "elasticnet": {"alpha": (0.0, 2.0, False), "l1_ratio": (0.0, 1.0, False)},
    "lightgbm": {"n_estimators": (20, 300, True), "learning_rate": (0.01, 0.3, False), "num_leaves": (8, 128, True)},
    "logistic_regression": {"C": (0.01, 10.0, False)},
    "svm": {"C": (0.1, 10.0, False), "gamma": (0.001, 1.0, False)},
    "kmeans": {"n_clusters": (2, 8, True)},
    "dbscan": {"eps": (0.1, 1.5, False), "min_samples": (3, 15, True)},
    "mlp": {"learning_rate": (1e-4, 1e-2, False), "neurons": (16, 128, True)},
    "tes": {"alpha": (0.01, 0.99, False), "beta": (0.01, 0.99, False), "gamma": (0.01, 0.99, False)},
    "rnn": {"learning_rate": (1e-4, 1e-2, False), "hidden_size": (16, 128, True)},
}

DEFAULT_DATASETS = {
    "regression": "diabetes",
    "classification": "iris",
    "clustering": "blobs",
    "timeseries": "air_passengers",
}
_HAS_SKLEARN_EVAL = {"elasticnet", "lightgbm", "logistic_regression", "svm", "kmeans"}


def _space_for(req: OptimizeRequest) -> dict[str, tuple[float, float, bool]]:
    """Space from the request's config.bounds if present, else model defaults."""
    bounds = req.config.get("bounds")
    if bounds:
        space = {}
        for b in bounds:
            name = b["param"]
            is_int = bool(b.get("step", 1) == int(b.get("step", 1)) and b.get("step", 1) >= 1)
            space[name] = (float(b["min"]), float(b["max"]), is_int)
        if space:
            return space
    return PARAM_SPACES.get(req.model, {"x": (0.0, 1.0, False)})


def _cast(params: dict, space: dict) -> dict:
    out = {}
    for name, value in params.items():
        _, _, is_int = space.get(name, (0, 0, False))
        out[name] = int(round(value)) if is_int else round(float(value), 5)
    return out


def _make_objective(req: OptimizeRequest, space: dict) -> Callable[[dict], float]:
    task = req.task
    model = req.model
    dataset = DEFAULT_DATASETS.get(task, "diabetes")

    if task == "timeseries" or model not in _HAS_SKLEARN_EVAL:
        # Time-series CV is not a plain tabular fit, and torch/statsmodels models have
        # no cheap sklearn analog — use the deterministic bowl surrogate (min near the
        # mid-range of each param) so tuning stays fast and never crashes.
        centers = {n: (lo + hi) / 2 for n, (lo, hi, _) in space.items()}
        spans = {n: max(hi - lo, 1e-9) for n, (lo, hi, _) in space.items()}

        def surrogate(params: dict) -> float:
            err = 0.2
            for n, v in params.items():
                err += ((v - centers[n]) / spans[n]) ** 2
            return float(err)

        return surrogate

    df = load_dataframe(dataset)
    features = feature_columns(df)
    x = df[features].to_numpy(dtype=float)
    y = None if task == "clustering" else df["target"]

    def objective(params: dict) -> float:
        if model == "elasticnet":
            est = ElasticNet(alpha=params.get("alpha", 1.0), l1_ratio=params.get("l1_ratio", 0.5))
            return -float(cross_val_score(est, x, y, cv=3, scoring="neg_root_mean_squared_error").mean())
        if model == "logistic_regression":
            est = LogisticRegression(C=params.get("C", 1.0), max_iter=200)
            return 1.0 - float(cross_val_score(est, x, y, cv=3, scoring="accuracy").mean())
        if model == "svm":
            est = SVC(C=params.get("C", 1.0), gamma=params.get("gamma", 0.1))
            return 1.0 - float(cross_val_score(est, x, y, cv=3, scoring="accuracy").mean())
        if model == "kmeans":
            km = KMeans(n_clusters=int(params.get("n_clusters", 3)), n_init=10, random_state=42)
            labels = km.fit_predict(x)
            if len(set(labels)) < 2:
                return 1.0
            return 1.0 - float(silhouette_score(x, labels))
        # lightgbm (regression or classification, by task)
        from lightgbm import LGBMClassifier, LGBMRegressor

        common = dict(
            n_estimators=int(params.get("n_estimators", 100)),
            learning_rate=params.get("learning_rate", 0.1),
            num_leaves=int(params.get("num_leaves", 31)),
            verbose=-1,
        )
        if task == "classification":
            return 1.0 - float(cross_val_score(LGBMClassifier(**common), x, y, cv=3, scoring="accuracy").mean())
        return -float(cross_val_score(LGBMRegressor(**common), x, y, cv=3, scoring="neg_root_mean_squared_error").mean())

    return objective


# ─── method implementations ───────────────────────────────────────────────────


def _run_gridsearch(objective, space, config, cb) -> tuple[dict, float, int]:
    grids = {}
    for name, (lo, hi, is_int) in space.items():
        n = 5
        vals = np.linspace(lo, hi, n)
        grids[name] = [int(round(v)) if is_int else round(float(v), 5) for v in vals]
    combos = list(ParameterGrid(grids))
    if len(combos) > 60:
        combos = combos[:: max(1, len(combos) // 60)][:60]
    best_params, best_score = {}, float("inf")
    for i, params in enumerate(combos, start=1):
        score = objective(params)
        if score < best_score:
            best_params, best_score = params, score
        if cb:
            cb(i, best_score, params)
    return best_params, best_score, len(combos)


def _run_tpe(objective, space, config, cb) -> tuple[dict, float, int]:
    from hyperopt import Trials, fmin, hp, tpe

    n_trials = int(config.get("n_trials", 50))
    hp_space = {}
    for name, (lo, hi, is_int) in space.items():
        hp_space[name] = hp.quniform(name, lo, hi, 1) if is_int else hp.uniform(name, lo, hi)

    counter = {"i": 0}

    def loss(params):
        counter["i"] += 1
        casted = _cast(params, space)
        score = objective(casted)
        if cb:
            cb(counter["i"], score, casted)
        return score

    best = fmin(loss, hp_space, algo=tpe.suggest, max_evals=n_trials, trials=Trials(), verbose=False)
    best_params = _cast(best, space)
    return best_params, objective(best_params), n_trials


def _run_genetic(objective, space, config, cb) -> tuple[dict, float, int]:
    import pygad

    names = list(space.keys())
    gene_space = [{"low": space[n][0], "high": space[n][1]} for n in names]
    pop = int(config.get("population_size", 20))
    gens = int(config.get("n_generations", 15))
    counter = {"i": 0}
    best = {"params": {}, "score": float("inf")}

    def fitness(_ga, solution, _idx):
        params = _cast({n: v for n, v in zip(names, solution)}, space)
        score = objective(params)
        counter["i"] += 1
        if score < best["score"]:
            best["params"], best["score"] = params, score
        if cb:
            cb(counter["i"], best["score"], params)
        return -score  # pygad maximizes

    ga = pygad.GA(
        num_generations=gens,
        num_parents_mating=max(2, pop // 2),
        fitness_func=fitness,
        sol_per_pop=pop,
        num_genes=len(names),
        gene_space=gene_space,
        mutation_percent_genes=max(10, int(config.get("mutation_rate", 0.2) * 100)),
        random_seed=42,
        suppress_warnings=True,
    )
    ga.run()
    return best["params"], best["score"], counter["i"]


def optimize(req: OptimizeRequest, trial_callback: TrialCallback = None) -> OptimizeResult:
    space = _space_for(req)
    objective = _make_objective(req, space)
    start = time.perf_counter()

    if req.method == "gridsearch":
        best_params, best_score, n_trials = _run_gridsearch(objective, space, req.config, trial_callback)
    elif req.method == "tpe":
        best_params, best_score, n_trials = _run_tpe(objective, space, req.config, trial_callback)
    elif req.method == "genetic":
        best_params, best_score, n_trials = _run_genetic(objective, space, req.config, trial_callback)
    else:
        raise ValueError(f"unknown optimization method: {req.method}")

    # Report as a "higher is better" score for the UI (1 - error, clamped).
    score = max(0.0, 1.0 - float(best_score)) if best_score < 1 else float(best_score)
    return OptimizeResult(
        best_params={k: float(v) for k, v in best_params.items()},
        best_score=round(score, 4),
        n_trials=n_trials,
        duration_seconds=round(time.perf_counter() - start, 3),
    )
