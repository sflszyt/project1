"""Shared modeling helpers for notebooks 01-04 (imported through sys.path like ``common``; there is no package).

Pure pandas / numpy / scikit-learn / LightGBM / scipy: nothing here touches Spark, MLflow or the file system.

- Features: :func:`feature_columns_from_dictionary`, :func:`fit_categorical_levels`, :func:`prepare_matrix`. Training
  and scoring both go through :func:`prepare_matrix`, so numeric columns are always ``float32`` and string columns
  always ``pd.Categorical`` with the levels fitted on Train (unknown values become NaN).
- Models: :func:`fit_lgbm` (early stopping) and :func:`refit_lgbm` (fixed rounds) for the LightGBM champion and the
  Layer 2 heads; :func:`fit_logistic` for the logistic baseline.
- Evaluation: :func:`binary_metrics`, :func:`reliability_table`, :func:`cluster_bootstrap`.
- Evidence and calibration: :func:`count_events`, :func:`select_calibrator` and the picklable calibrators
  :class:`PlattCalibrator`, :class:`InterceptCalibrator`, :class:`IdentityCalibrator` (their pickles reference this
  module, so ``import modeling`` must work wherever they are loaded).
- Layer 2 (notebook 03 evaluates exactly what notebook 04 serves): :func:`layer2_probabilities` (fixed / head /
  share / rule per L2, restricted to the L2's valid values and the row's constraints) and :func:`layer2_guidance`
  (top value, margin, status incl. abstain).
- Reasons: :func:`feature_contributions` (LightGBM ``pred_contrib``), :func:`group_contributions` (sums per reason
  group) and :func:`explain_rows` (top explainable, non-sensitive positive groups per row).
"""

from __future__ import annotations

import math
import re

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# Label-like column prefixes that must never become features (docs/MVP_SPEC.md section 7).
FORBIDDEN_FEATURE_PREFIXES = ("y_", "elig_", "p_true_")
# Scores are clipped to [SCORE_CLIP, 1 - SCORE_CLIP] before logit (Platt, intercept, calibration slope).
SCORE_CLIP = 1e-6
# Log-loss clips probabilities to [LOGLOSS_CLIP, 1 - LOGLOSS_CLIP].
LOGLOSS_CLIP = 1e-15
# Logistic baseline: a numeric column is heavy tailed when its max exceeds this and it is never negative.
HEAVY_TAIL_MAX = 1000.0
# One-hot levels seen fewer times than this in the fit sample are merged into one "infrequent" column.
ONE_HOT_MIN_FREQUENCY = 50
EVIDENCE_TIERS = ("own", "limited", "insufficient")
_TIER_METHOD = {"own": "platt", "limited": "intercept", "insufficient": "none"}


# ---------------------------------------------------------------------------------------------------------------
# Parameters and time blocks
# ---------------------------------------------------------------------------------------------------------------

def resolve_l2_list(products_conf: dict, value: str | None) -> list[str]:
    """The L2s of the ``l2_list`` parameter: ``pilot`` (default; ``pilot: true`` in products.yaml, catalog order),
    ``all`` (catalog order) or a comma-separated list (given order, duplicates removed)."""
    catalog = products_conf["products"]
    value = (value or "pilot").strip()
    if value == "pilot":
        l2s = [l2 for l2, spec in catalog.items() if spec.get("pilot")]
    elif value == "all":
        l2s = list(catalog)
    else:
        l2s = list(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
        unknown = [l2 for l2 in l2s if l2 not in catalog]
        if unknown:
            raise ValueError(f"l2_list has L2s that are not in products.yaml: {unknown}")
    if not l2s:
        raise ValueError(f"l2_list {value!r} selects no L2")
    return l2s


def inner_split(train_snapshots, n_valid: int, n_gap: int) -> tuple[list, list, list]:
    """Split the sorted Train snapshots into ``(fit, gap, valid)``: the last ``n_valid`` are the inner validation
    block, the ``n_gap`` before them the inner gap, the rest the fit block (notebook 01 early stopping)."""
    dates = sorted(set(train_snapshots))
    n_valid, n_gap = int(n_valid), int(n_gap)
    if n_valid < 1 or n_gap < 0:
        raise ValueError(f"inner split needs n_valid >= 1 and n_gap >= 0, got {n_valid}, {n_gap}")
    if len(dates) <= n_valid + n_gap:
        raise ValueError(f"{len(dates)} Train snapshots leave no fit block for {n_valid} valid + {n_gap} gap")
    n_fit = len(dates) - n_valid - n_gap
    return dates[:n_fit], dates[n_fit:n_fit + n_gap], dates[n_fit + n_gap:]


# ---------------------------------------------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------------------------------------------

def feature_columns_from_dictionary(dictionary_pdf: pd.DataFrame, exclusions) -> list[str]:
    """Columns with ``role == 'feature'`` in the order of ``dictionary_pdf`` (``column_name``, ``role``), minus
    ``exclusions``. Raises when a label-like column (``y_*``, ``elig_*``, ``p_true_*``) would become a feature."""
    excluded = set(exclusions or [])
    features = [name for name, role in zip(dictionary_pdf["column_name"], dictionary_pdf["role"])
                if role == "feature" and name not in excluded]
    leaked = [name for name in features if name.lower().startswith(FORBIDDEN_FEATURE_PREFIXES)]
    if leaked:
        raise ValueError(f"label-like columns have role 'feature' in the data dictionary: {leaked}")
    if len(set(features)) != len(features):
        raise ValueError("the data dictionary repeats feature columns")
    if not features:
        raise ValueError("the data dictionary has no feature columns")
    return features


def fit_categorical_levels(df: pd.DataFrame, string_columns) -> dict[str, list[str]]:
    """Sorted distinct non-null values of each string column (the category levels used everywhere afterwards)."""
    return {column: sorted(str(v) for v in df[column].dropna().unique()) for column in string_columns}


def prepare_matrix(df: pd.DataFrame, feature_columns, categorical_levels: dict[str, list[str]]) -> pd.DataFrame:
    """The model matrix: exactly ``feature_columns`` in that order, numeric columns as ``float32``, columns in
    ``categorical_levels`` as ``pd.Categorical`` with those levels (values outside them become NaN). Keeps the index
    of ``df``. Used for training and scoring alike so that the encoding cannot drift."""
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise KeyError(f"feature columns missing from the input: {missing}")
    columns = {}
    for column in feature_columns:
        values = df[column]
        if column in categorical_levels:
            values = values.astype(object).where(values.notna(), None)
            columns[column] = pd.Categorical(values, categories=categorical_levels[column])
        else:
            columns[column] = pd.to_numeric(values, errors="raise").to_numpy(dtype=np.float32, na_value=np.nan)
    return pd.DataFrame(columns, index=df.index)


def detect_heavy_tailed(X: pd.DataFrame, numeric_columns, max_threshold: float = HEAVY_TAIL_MAX) -> list[str]:
    """Numeric columns with ``max > max_threshold`` and no negative value (the logistic baseline log1p's them)."""
    heavy = []
    for column in numeric_columns:
        values = X[column].to_numpy(dtype=np.float64)
        if np.isnan(values).all():
            continue
        if np.nanmax(values) > max_threshold and np.nanmin(values) >= 0:
            heavy.append(column)
    return heavy


def train_l1_pair(frame, y, fit_rows, valid_rows, refit_rows, feature_columns, string_columns, params):
    """Fit the tuning model/transform on fit rows, then a separate fixed-iteration champion on refit rows.

    Callers pass only eligible Train rows. Validation chooses iterations; its metric is a tuning diagnostic,
    not an unbiased performance estimate. Calibration and final test never enter this function.
    """
    numeric = [c for c in feature_columns if c not in string_columns]

    def spec_and_matrix(rows, role):
        levels = fit_categorical_levels(frame.iloc[rows], string_columns)
        matrix = prepare_matrix(frame, feature_columns, levels)
        spec = {"feature_columns": list(feature_columns), "categorical_levels": levels,
                "heavy_tailed_columns": detect_heavy_tailed(matrix.iloc[rows], numeric), "model_role": role}
        return spec, matrix

    tuning_spec, tuning_matrix = spec_and_matrix(fit_rows, "tuning")
    early, iteration, _ = fit_lgbm(tuning_matrix.iloc[fit_rows], np.asarray(y)[fit_rows],
                                   tuning_matrix.iloc[valid_rows], np.asarray(y)[valid_rows], params, string_columns)
    champion_spec, champion_matrix = spec_and_matrix(refit_rows, "champion_refit")
    champion = refit_lgbm(champion_matrix.iloc[refit_rows], np.asarray(y)[refit_rows], params, iteration, string_columns)
    return early, champion, tuning_spec, champion_spec, iteration


def layer2_feature_spec(data, train_rows, l1_spec):
    """Pooled Layer 2 needs all its training categories, not one product's eligible L1 categories."""
    return {"feature_columns": list(l1_spec["feature_columns"]),
            "categorical_levels": fit_categorical_levels(data.loc[train_rows], l1_spec["categorical_levels"]),
            "fitted_on": "layer2_train_purchases"}


# ---------------------------------------------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------------------------------------------

def _lgbm_train_params(params: dict, overrides: dict | None) -> tuple[dict, int, int | None]:
    """Split YAML-style LightGBM params into (lightgbm.train params, max_rounds, early_stopping_rounds)."""
    merged = {**params, **(overrides or {})}
    if "max_rounds" not in merged:
        raise KeyError("LightGBM params need max_rounds")
    num_boost_round = int(merged.pop("max_rounds"))
    early_stopping_rounds = merged.pop("early_stopping_rounds", None)
    return merged, num_boost_round, None if early_stopping_rounds is None else int(early_stopping_rounds)


def fit_lgbm(X_fit: pd.DataFrame, y_fit, X_valid: pd.DataFrame, y_valid, params: dict, categorical_columns,
             overrides: dict | None = None):
    """``lightgbm.train`` on the fit block with early stopping on the valid block.

    ``params`` come from YAML: ``max_rounds`` becomes ``num_boost_round``, ``early_stopping_rounds`` drives the
    early-stopping callback (first metric only), everything else goes to LightGBM unchanged. ``overrides`` replace
    keys of ``params`` (e.g. ``objective`` / ``metric`` / ``num_class`` for a multiclass head). Returns
    ``(booster, best_iteration, evals_result)``; ``evals_result["valid"][<metric>]`` is the validation curve.
    """
    train_params, num_boost_round, early_stopping_rounds = _lgbm_train_params(params, overrides)
    if early_stopping_rounds is None:
        raise KeyError("fit_lgbm needs early_stopping_rounds in params")
    fit_set = lgb.Dataset(X_fit, label=np.asarray(y_fit), categorical_feature=list(categorical_columns))
    valid_set = lgb.Dataset(X_valid, label=np.asarray(y_valid), reference=fit_set)
    evals_result: dict = {}
    booster = lgb.train(
        train_params, fit_set, num_boost_round=num_boost_round, valid_sets=[valid_set], valid_names=["valid"],
        callbacks=[lgb.early_stopping(early_stopping_rounds, first_metric_only=True, verbose=False),
                   lgb.record_evaluation(evals_result)],
    )
    best_iteration = int(booster.best_iteration) if booster.best_iteration > 0 else booster.current_iteration()
    return booster, best_iteration, evals_result


def refit_lgbm(X: pd.DataFrame, y, params: dict, num_boost_round: int, categorical_columns,
               overrides: dict | None = None):
    """``lightgbm.train`` for exactly ``num_boost_round`` rounds without validation (refit on all of Train)."""
    train_params, _, _ = _lgbm_train_params(params, overrides)
    if int(num_boost_round) < 1:
        raise ValueError(f"num_boost_round must be >= 1, got {num_boost_round}")
    data = lgb.Dataset(X, label=np.asarray(y), categorical_feature=list(categorical_columns))
    return lgb.train(train_params, data, num_boost_round=int(num_boost_round))


# ---------------------------------------------------------------------------------------------------------------
# Logistic baseline
# ---------------------------------------------------------------------------------------------------------------

def signed_log1p(x):
    """``sign(x) * log1p(|x|)``: log1p for non-negative values, defined for negatives, NaN stays NaN."""
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.log1p(np.abs(x))


def _numeric_pipeline(log_transform: bool) -> Pipeline:
    # add_indicator: SimpleImputer appends MissingIndicator(features="missing-only") columns, i.e. one indicator for
    # each column with missing values in the fit sample (new missing values at scoring time do not raise).
    steps = [("log1p", FunctionTransformer(signed_log1p, feature_names_out="one-to-one"))] if log_transform else []
    steps += [("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]
    return Pipeline(steps)


def fit_logistic(X_fit: pd.DataFrame, y_fit, numeric_columns, categorical_columns, heavy_tailed_columns,
                 conf: dict) -> Pipeline:
    """Logistic baseline as one sklearn Pipeline (input: the :func:`prepare_matrix` matrix).

    Numeric columns: median imputation + missing indicators (columns with missing values only) + standardisation;
    ``heavy_tailed_columns`` are :func:`signed_log1p` transformed first. Categorical columns: one-hot
    (``handle_unknown='ignore'``, ``min_frequency=50``; missing is its own level). Then
    ``LogisticRegression(C, max_iter, solver)`` from ``conf`` (training.yaml ``logistic``).
    """
    heavy = set(heavy_tailed_columns or [])
    unknown = sorted(heavy - set(numeric_columns))
    if unknown:
        raise ValueError(f"heavy-tailed columns that are not numeric columns: {unknown}")
    plain = [c for c in numeric_columns if c not in heavy]
    logged = [c for c in numeric_columns if c in heavy]
    transformers = []
    if plain:
        transformers.append(("numeric", _numeric_pipeline(log_transform=False), plain))
    if logged:
        transformers.append(("heavy_tailed", _numeric_pipeline(log_transform=True), logged))
    if categorical_columns:
        transformers.append(("categorical",
                             OneHotEncoder(handle_unknown="ignore", min_frequency=ONE_HOT_MIN_FREQUENCY),
                             list(categorical_columns)))
    pipeline = Pipeline([
        ("features", ColumnTransformer(transformers, remainder="drop")),
        ("model", LogisticRegression(C=float(conf["C"]), max_iter=int(conf["max_iter"]),
                                     solver=str(conf.get("solver", "lbfgs")))),
    ])
    return pipeline.fit(X_fit, np.asarray(y_fit))


# ---------------------------------------------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------------------------------------------

def clipped_logit(s, eps: float = SCORE_CLIP) -> np.ndarray:
    """``logit(clip(s, eps, 1 - eps))``."""
    s = np.clip(np.asarray(s, dtype=np.float64), eps, 1 - eps)
    return np.log(s) - np.log1p(-s)


def fraction_label(fraction: float) -> str:
    """Metric-name suffix of a top fraction: 0.01 -> ``1pct``, 0.1 -> ``10pct``."""
    return f"{float(fraction) * 100:g}pct"


def _both_classes(y: np.ndarray) -> bool:
    n_pos = int(y.sum())
    return 0 < n_pos < len(y)


def calibration_slope(y, p) -> tuple[float, float]:
    """``(slope, intercept)`` of the unpenalised logistic regression ``y ~ logit(clip(p))``; NaN for one class."""
    y = np.asarray(y, dtype=np.int64)
    if not _both_classes(y):
        return math.nan, math.nan
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000, tol=1e-10)
    model.fit(clipped_logit(p).reshape(-1, 1), y)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def binary_metrics(y, p, top_fractions=(0.01, 0.05, 0.10)) -> dict:
    """Discrimination and calibration of probabilities ``p`` for 0/1 labels ``y``.

    Keys: ``n_rows, n_pos, obs_rate, mean_pred, oe`` (obs_rate / mean_pred), ``auroc, pr_auc`` (average precision),
    ``logloss, brier, slope, intercept`` (:func:`calibration_slope`) and per top fraction f ``lift_<f>`` (positive
    rate among the top ceil(f * n) rows by descending p, ties in row order, over the overall rate) and
    ``recall_<f>`` (their share of all positives); suffixes from :func:`fraction_label`. Rank metrics and the slope
    are NaN when ``y`` has a single class.
    """
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError(f"y and p must be 1-d arrays of the same length, got {y.shape} and {p.shape}")
    if len(y) == 0:
        raise ValueError("binary_metrics needs at least one row")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y must be 0/1")
    if np.isnan(p).any():
        raise ValueError("p has NaN values")
    n, n_pos = len(y), int(y.sum())
    obs_rate, mean_pred = n_pos / n, float(p.mean())
    both = _both_classes(y)
    clipped = np.clip(p, LOGLOSS_CLIP, 1 - LOGLOSS_CLIP)
    slope, intercept = calibration_slope(y, p)
    metrics = {
        "n_rows": n,
        "n_pos": n_pos,
        "obs_rate": obs_rate,
        "mean_pred": mean_pred,
        "oe": obs_rate / mean_pred if mean_pred > 0 else math.nan,
        "auroc": float(roc_auc_score(y, p)) if both else math.nan,
        "pr_auc": float(average_precision_score(y, p)) if both else math.nan,
        "logloss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))),
        "brier": float(np.mean((p - y) ** 2)),
        "slope": slope,
        "intercept": intercept,
    }
    order = np.argsort(-p, kind="stable")
    for fraction in top_fractions:
        k = max(1, math.ceil(float(fraction) * n))
        top_pos = int(y[order[:k]].sum())
        label = fraction_label(fraction)
        metrics[f"lift_{label}"] = (top_pos / k) / obs_rate if n_pos > 0 else math.nan
        metrics[f"recall_{label}"] = top_pos / n_pos if n_pos > 0 else math.nan
    return metrics


def reliability_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Equal-count bins of ascending ``p`` (bin 1 = lowest scores; ties split by row order):
    ``bin, mean_pred, obs_rate, n, n_pos``."""
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    n = len(p)
    if n == 0:
        raise ValueError("reliability_table needs at least one row")
    bins = np.empty(n, dtype=np.int64)
    bins[np.argsort(p, kind="stable")] = np.arange(n) * int(n_bins) // n + 1
    table = (pd.DataFrame({"bin": bins, "p": p, "y": y})
             .groupby("bin")
             .agg(mean_pred=("p", "mean"), obs_rate=("y", "mean"), n=("y", "size"), n_pos=("y", "sum"))
             .reset_index())
    return table


def weighted_auroc(y, p, w) -> float:
    """AUROC with sample weights (a :func:`cluster_bootstrap` statistic); NaN for a single class."""
    y = np.asarray(y)
    if not _both_classes(y):
        return math.nan
    return float(roc_auc_score(y, p, sample_weight=w))


def weighted_oe(y, p, w) -> float:
    """Observed / expected with sample weights (a :func:`cluster_bootstrap` statistic)."""
    w = np.asarray(w, dtype=np.float64)
    expected = float(np.sum(w * np.asarray(p, dtype=np.float64)))
    return float(np.sum(w * np.asarray(y, dtype=np.float64))) / expected if expected > 0 else math.nan


def cluster_bootstrap(y, p, clusters, stat_fn, n_boot: int = 200, seed: int = 0,
                      ci: float = 0.95) -> tuple[float, float]:
    """Percentile CI of ``stat_fn(y, p, w)`` over a cluster bootstrap.

    Each replicate draws as many clusters as there are, with replacement; a row's weight ``w`` is the number of
    times its cluster was drawn (rows with weight 0 are left out), so no data is copied or concatenated.
    Non-finite replicate values are dropped; returns ``(nan, nan)`` when none is finite.
    """
    y = np.asarray(y)
    p = np.asarray(p, dtype=np.float64)
    codes, uniques = pd.factorize(np.asarray(clusters))
    if (codes < 0).any():
        raise ValueError("clusters has missing values")
    n_clusters = len(uniques)
    rng = np.random.default_rng(seed)
    values = np.empty(int(n_boot), dtype=np.float64)
    for b in range(int(n_boot)):
        draws = np.bincount(rng.integers(0, n_clusters, size=n_clusters), minlength=n_clusters)
        weights = draws[codes]
        keep = weights > 0
        values[b] = stat_fn(y[keep], p[keep], weights[keep].astype(np.float64))
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan
    alpha = (1.0 - float(ci)) / 2.0
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))


# ---------------------------------------------------------------------------------------------------------------
# Evidence and calibration
# ---------------------------------------------------------------------------------------------------------------

def count_events(purchases_pdf: pd.DataFrame, l2: str, first_snapshot, last_snapshot, horizon_months: int) -> int:
    """Distinct ``purchase_id`` of ``l2`` with ``elig_at_feature_snapshot == 1`` and ``application_date`` in
    ``(first_snapshot, last_snapshot + DateOffset(months=horizon_months)]`` (docs/MVP_SPEC.md section 5)."""
    start = pd.Timestamp(first_snapshot)
    end = pd.Timestamp(last_snapshot) + pd.DateOffset(months=int(horizon_months))
    application = pd.to_datetime(purchases_pdf["application_date"])
    selected = ((purchases_pdf["l2"] == l2) & (purchases_pdf["elig_at_feature_snapshot"] == 1)
                & (application > start) & (application <= end))
    return int(purchases_pdf.loc[selected, "purchase_id"].nunique())


def evidence_tier(event_count: int, tiers_conf: dict) -> str:
    """``own`` (>= own_model_min_events), ``limited`` (>= limited_min_events) or ``insufficient``."""
    if event_count >= int(tiers_conf["own_model_min_events"]):
        return "own"
    if event_count >= int(tiers_conf["limited_min_events"]):
        return "limited"
    return "insufficient"


def select_calibrator(event_count: int, tiers_conf: dict, calibration_conf: dict) -> tuple[str, str]:
    """``(evidence_tier, method)``: own -> ``platt``, limited -> ``intercept``, insufficient -> ``none``; every tier
    gets ``disabled`` when ``calibration.enabled`` is false."""
    tier = evidence_tier(event_count, tiers_conf)
    if not calibration_conf.get("enabled", True):
        return tier, "disabled"
    return tier, _TIER_METHOD[tier]


class _Calibrator:
    """Common interface: ``fit(s, y) -> self``, ``predict(s) -> probabilities``, ``method``, ``params()``."""

    method = "base"

    def __init__(self):
        self.a: float | None = None
        self.b: float | None = None
        self.n_fit = 0
        self.n_pos_fit = 0

    def _record_fit(self, s, y) -> tuple[np.ndarray, np.ndarray]:
        s = np.asarray(s, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        if s.shape != y.shape or s.ndim != 1:
            raise ValueError(f"s and y must be 1-d arrays of the same length, got {s.shape} and {y.shape}")
        self.n_fit, self.n_pos_fit = len(y), int(y.sum())
        return s, y

    def params(self) -> dict:
        return {"method": self.method, "a": self.a, "b": self.b, "n_fit": self.n_fit, "n_pos_fit": self.n_pos_fit}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(a={self.a}, b={self.b})"


class PlattCalibrator(_Calibrator):
    """``logit(p) = a + b * logit(clip(s, 1e-6, 1 - 1e-6))`` fitted by ``LogisticRegression(C)`` (C=1e6 is
    effectively unpenalised). The caller checks ``b > 0`` (a non-positive slope would reverse the ranking)."""

    method = "platt"

    def __init__(self, C: float = 1e6):
        super().__init__()
        self.C = float(C)

    def fit(self, s, y) -> "PlattCalibrator":
        s, y = self._record_fit(s, y)
        if not _both_classes(y):
            raise ValueError("Platt calibration needs both classes")
        model = LogisticRegression(C=self.C, solver="lbfgs", max_iter=1000, tol=1e-10)
        model.fit(clipped_logit(s).reshape(-1, 1), y)
        self.a, self.b = float(model.intercept_[0]), float(model.coef_[0, 0])
        return self

    def predict(self, s) -> np.ndarray:
        return expit(self.a + self.b * clipped_logit(s))


class InterceptCalibrator(_Calibrator):
    """``logit(p) = a + logit(clip(s))`` (``b`` fixed at 1); ``a`` is the one-dimensional log-loss MLE."""

    method = "intercept"
    _BOUNDS = (-20.0, 20.0)

    def fit(self, s, y) -> "InterceptCalibrator":
        s, y = self._record_fit(s, y)
        if not _both_classes(y):
            raise ValueError("intercept calibration needs both classes")
        x = clipped_logit(s)

        def log_loss(a: float) -> float:  # mean of log(1 + e^z) - y * z, z = a + x
            z = a + x
            return float(np.mean(np.logaddexp(0.0, z) - y * z))

        result = minimize_scalar(log_loss, bounds=self._BOUNDS, method="bounded", options={"xatol": 1e-10})
        if not result.success or min(abs(result.x - bound) for bound in self._BOUNDS) < 1e-6:
            raise RuntimeError(f"intercept calibration did not converge inside {self._BOUNDS}: {result}")
        self.a, self.b = float(result.x), 1.0
        return self

    def predict(self, s) -> np.ndarray:
        return expit(self.a + clipped_logit(s))


class IdentityCalibrator(_Calibrator):
    """No calibration (``not_calibrated`` / ``disabled``): ``predict`` returns the scores unchanged."""

    method = "identity"

    def fit(self, s, y) -> "IdentityCalibrator":
        self._record_fit(s, y)
        return self

    def predict(self, s) -> np.ndarray:
        return np.asarray(s, dtype=np.float64).copy()


# ---------------------------------------------------------------------------------------------------------------
# Layer 2 guidance (docs/MVP_SPEC.md section 9)
# ---------------------------------------------------------------------------------------------------------------

# <attribute>_status of a served value by method_used; layer2_guidance adds "abstain" (margin below the attribute's
# margin) and "not_applicable" (the L2 or the row's constraints leave no valid value).
LAYER2_STATUS = {"fixed": "fixed", "head": "model", "share": "share", "rule": "rule"}
# Methods whose guidance abstains below the margin (a fixed value or a rule is never uncertain).
LAYER2_ABSTAINING = ["head", "share"]
# layer2.yaml constraint on_missing: what a constraint does when a column it reads is missing for the row.
CONSTRAINT_ON_MISSING = ("allow", "block")
_QUOTED_TEXT = re.compile(r"'[^']*'|\"[^\"]*\"")
_NAME_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def class_probabilities(booster, X: pd.DataFrame, n_classes: int) -> np.ndarray:
    """``(n_rows, n_classes)`` probabilities of a Layer 2 head, classes in the order of the attribute's values: a
    binary head predicts class 1 (the second value), a multiclass head every class."""
    p = np.asarray(booster.predict(X), dtype=np.float64)
    if n_classes == 2 and p.ndim == 1:
        p = np.column_stack([1.0 - p, p])
    if p.shape != (len(X), n_classes):
        raise ValueError(f"the head predicts shape {p.shape}, expected ({len(X)}, {n_classes})")
    return p


def rule_value_index(X: pd.DataFrame, rule: dict) -> np.ndarray:
    """Value index from a ``{feature, higher_end_values}`` rule (layer2.yaml mass_higher, values [Mass, HigherEnd]):
    1 (the second value) where ``X[feature]`` is one of ``higher_end_values``, else 0 (also when it is missing)."""
    column = X[rule["feature"]].astype(object)
    return column.isin(list(rule["higher_end_values"])).to_numpy().astype(np.int64)


def constraint_columns(expression: str, columns) -> list[str]:
    """The ``columns`` that an expression refers to (names inside quotes are ignored), in ``columns`` order."""
    names = set(_NAME_TOKEN.findall(_QUOTED_TEXT.sub(" ", expression)))
    return [column for column in columns if column in names]


def normalize_constraints(constraints) -> list[dict]:
    """layer2.yaml ``constraints`` as ``[{value, require, on_missing}]``: ``on_missing`` defaults to ``allow`` and
    must be ``allow`` or ``block``; other keys are refused. layer2_spec.json stores this form, and notebook 04
    compares it with the normalized layer2.yaml."""
    normalized = []
    for constraint in constraints or []:
        unknown = sorted(set(constraint) - {"value", "require", "on_missing"})
        if unknown or "value" not in constraint or not constraint.get("require"):
            raise ValueError(f"constraint {constraint} needs value and require (optional on_missing) only")
        on_missing = str(constraint.get("on_missing", "allow"))
        if on_missing not in CONSTRAINT_ON_MISSING:
            raise ValueError(f"constraint {constraint}: on_missing must be one of {CONSTRAINT_ON_MISSING}")
        normalized.append({"value": constraint["value"], "require": str(constraint["require"]),
                           "on_missing": on_missing})
    return normalized


def constraint_mask(X: pd.DataFrame, values, constraints) -> np.ndarray:
    """``(n_rows, len(values))`` booleans: False where a constraint rules the value out for the row.

    ``constraints`` is the layer2.yaml list ``[{value, require, on_missing}]``; ``require`` is a ``DataFrame.eval``
    expression over the columns of ``X`` (e.g. ``"CURRENT_AGE < 50"``); several constraints of one value must all
    hold. ``on_missing`` decides rows with a missing value in a column the expression refers to: ``allow`` (default)
    keeps the value (an unknown input does not rule it out), ``block`` rules it out.
    """
    values = list(values)
    allowed = np.ones((len(X), len(values)), dtype=bool)
    for constraint in normalize_constraints(constraints):
        value, expression = constraint["value"], constraint["require"]
        if value not in values:
            raise ValueError(f"constraint for {value!r}, which is not one of {values}")
        referenced = constraint_columns(expression, X.columns)
        if not referenced:
            raise ValueError(f"constraint {expression!r} refers to no input column")
        if len(X) == 0:
            continue
        result = X.eval(expression, engine="python")
        if not isinstance(result, pd.Series) or len(result) != len(X):
            raise ValueError(f"constraint {expression!r} does not give one boolean per row")
        satisfied = result.astype(object).where(result.notna(), False).to_numpy().astype(bool)
        unknown = X[referenced].isna().any(axis=1).to_numpy()
        if constraint["on_missing"] == "allow":
            allowed[:, values.index(value)] &= satisfied | unknown
        else:
            allowed[:, values.index(value)] &= satisfied & ~unknown
    return allowed


def restrict_probabilities(probs, allowed) -> np.ndarray:
    """``probs`` with the values outside ``allowed`` set to 0 and every row renormalised; rows without positive
    allowed probability (or with NaN) become NaN."""
    p = np.where(allowed, np.asarray(probs, dtype=np.float64), 0.0)
    total = p.sum(axis=1)
    usable = np.isfinite(total) & (total > 0)
    out = np.full(p.shape, np.nan)
    out[usable] = p[usable] / total[usable, None]
    return out


def layer2_probabilities(attribute_spec: dict, l2s, X: pd.DataFrame, X_head: pd.DataFrame | None = None,
                         booster=None) -> tuple[np.ndarray, np.ndarray]:
    """Served probabilities of one Layer 2 attribute: ``(probs, methods)``.

    ``attribute_spec`` is one entry of layer2_spec.json ``attributes``: ``values``, ``rule``, ``constraints`` and
    ``per_l2`` ``{l2: {method_used, valid_values, fixed_value, share}}``. Row i belongs to L2 ``l2s[i]``; ``X``
    (Layer 1 features) feeds the rule and the constraints, ``X_head`` (Layer 1 features + ``l2``, same rows) and
    ``booster`` the head rows. By method_used: ``fixed`` -> 1 for the fixed value, ``share`` -> the train shares,
    ``head`` -> :func:`class_probabilities`, ``rule`` -> 1 for :func:`rule_value_index`; then restricted to the L2's
    valid values and the row's constraints (:func:`constraint_mask`) and renormalised. ``probs`` is ``(n_rows,
    n_values)`` in ``values`` order (NaN rows: no value applies); ``methods`` is the method_used of each row
    (``not_applicable`` for an L2 missing from ``per_l2``).
    """
    values = list(attribute_spec["values"])
    l2s = np.asarray(l2s, dtype=object)
    if len(X) != len(l2s) or (X_head is not None and len(X_head) != len(l2s)):
        raise ValueError("l2s, X and X_head must describe the same rows")
    per_l2 = attribute_spec["per_l2"]
    methods = np.array([per_l2[l2]["method_used"] if l2 in per_l2 else "not_applicable" for l2 in l2s],
                       dtype=object)
    probs = np.full((len(l2s), len(values)), np.nan)
    valid = np.zeros(probs.shape, dtype=bool)
    for l2 in pd.unique(l2s):
        if l2 not in per_l2:
            continue
        rows, entry = l2s == l2, per_l2[l2]
        valid[rows] = [value in entry["valid_values"] for value in values]
        if entry["method_used"] == "fixed":
            probs[rows] = [1.0 if value == entry["fixed_value"] else 0.0 for value in values]
        elif entry["method_used"] == "share":
            probs[rows] = [float(entry["share"].get(value, 0.0)) for value in values]
    head = np.flatnonzero(methods == "head")
    if len(head):
        if booster is None or X_head is None:
            raise ValueError("head rows need X_head and the head booster")
        probs[head] = class_probabilities(booster, X_head.iloc[head], len(values))
    rule = np.flatnonzero(methods == "rule")
    if len(rule):
        probs[rule] = np.eye(len(values))[rule_value_index(X.iloc[rule], attribute_spec["rule"])]
    allowed = valid & constraint_mask(X, values, attribute_spec.get("constraints"))
    return restrict_probabilities(probs, allowed), methods


def layer2_guidance(probs, methods, margin: float | None = None):
    """Per row ``(value_index, prob, margin, status)`` of :func:`layer2_probabilities` output.

    ``value_index``: the most probable value (first in value order on ties; -1 for NaN rows); ``prob``: its
    probability; ``margin``: its lead over the runner-up (a lone remaining value leads by its probability).
    ``status``: :data:`LAYER2_STATUS` of the row's method, ``abstain`` for head / share rows with a lead below
    ``margin`` (the caller drops the guidance and keeps ``prob``), ``not_applicable`` for NaN rows.
    """
    p = np.asarray(probs, dtype=np.float64)
    n, k = p.shape
    index = np.full(n, -1, dtype=np.int64)
    top = np.full(n, np.nan)
    lead = np.full(n, np.nan)
    usable = ~np.isnan(p).any(axis=1) if k else np.zeros(n, dtype=bool)
    if usable.any():
        q = p[usable]
        order = np.argsort(-q, axis=1, kind="stable")
        rows = np.arange(len(q))
        index[usable] = order[:, 0]
        top[usable] = q[rows, order[:, 0]]
        lead[usable] = top[usable] - (q[rows, order[:, 1]] if k > 1 else 0.0)
    methods = np.asarray(methods, dtype=object)
    status = np.array([LAYER2_STATUS.get(m, "not_applicable") for m in methods], dtype=object)
    status[~usable] = "not_applicable"
    if margin is not None:
        status[usable & np.isin(methods, LAYER2_ABSTAINING) & (lead < float(margin))] = "abstain"
    return index, top, lead, status


# ---------------------------------------------------------------------------------------------------------------
# Reasons (docs/MVP_SPEC.md sections 9 and 10)
# ---------------------------------------------------------------------------------------------------------------

def feature_contributions(booster, X: pd.DataFrame) -> np.ndarray:
    """``(n_rows, n_features)`` LightGBM ``pred_contrib`` (TreeSHAP, raw log-odds scale) of a binary model without
    the trailing bias column; columns in ``booster.feature_name()`` order (which ``X`` must have)."""
    if list(X.columns) != booster.feature_name():
        raise ValueError("the columns of X differ from the booster's features")
    if len(X) == 0:
        return np.empty((0, booster.num_feature()))
    contributions = np.asarray(booster.predict(X, pred_contrib=True), dtype=np.float64)
    if contributions.shape != (len(X), booster.num_feature() + 1):
        raise ValueError(f"pred_contrib has shape {contributions.shape}: explanations need a binary model")
    return contributions[:, :-1]


def group_contributions(contributions, feature_names, feature_to_group: dict, group_order=None):
    """Feature contributions summed per reason group: ``(group_ids, totals, top_feature, top_contribution)``.

    ``group_ids``: the groups of ``feature_names`` in ``group_order`` order (default: first appearance);
    ``totals``: ``(n_rows, n_groups)`` sums; ``top_feature``: index into ``feature_names`` of each group's top
    feature, the largest contribution in the direction of the group total (the largest for a total >= 0, the most
    negative for a total < 0); ``top_contribution``: its contribution. Raises when a feature has no group.
    """
    feature_names = list(feature_names)
    contributions = np.asarray(contributions, dtype=np.float64)
    if contributions.ndim != 2 or contributions.shape[1] != len(feature_names):
        raise ValueError(f"contributions of shape {contributions.shape} for {len(feature_names)} features")
    ungrouped = [f for f in feature_names if f not in feature_to_group]
    if ungrouped:
        raise KeyError(f"features without a reason group: {ungrouped}")
    present = list(dict.fromkeys(feature_to_group[f] for f in feature_names))
    if group_order is None:
        group_ids = present
    else:
        unknown = [g for g in present if g not in group_order]
        if unknown:
            raise KeyError(f"reason groups missing from the group configuration: {unknown}")
        group_ids = [g for g in group_order if g in present]
    n = len(contributions)
    totals = np.empty((n, len(group_ids)))
    top_feature = np.empty((n, len(group_ids)), dtype=np.int64)
    top_contribution = np.empty((n, len(group_ids)))
    rows = np.arange(n)
    for j, group in enumerate(group_ids):
        members = np.array([i for i, f in enumerate(feature_names) if feature_to_group[f] == group])
        block = contributions[:, members]
        totals[:, j] = block.sum(axis=1)
        direction = np.where(totals[:, j] < 0, -1.0, 1.0)
        best = np.argmax(block * direction[:, None], axis=1)
        top_feature[:, j] = members[best]
        top_contribution[:, j] = block[rows, best]
    return group_ids, totals, top_feature, top_contribution


def feature_values_as_text(X: pd.DataFrame, rows, columns) -> np.ndarray:
    """``X.iloc[rows[i], columns[i]]`` as text (object array): category labels unchanged, numbers as the shortest
    text of their float32 value (``"35.0"``), None when missing."""
    rows = np.asarray(rows, dtype=np.int64)
    columns = np.asarray(columns, dtype=np.int64)
    out = np.full(len(rows), None, dtype=object)
    for column in np.unique(columns):
        selected = np.flatnonzero(columns == column)
        series = X.iloc[:, int(column)]
        if isinstance(series.dtype, pd.CategoricalDtype):
            codes = series.cat.codes.to_numpy()[rows[selected]]
            labels = np.asarray(series.cat.categories, dtype=object)
            valid = codes >= 0
            out[selected[valid]] = labels[codes[valid]]
        else:
            numbers = series.to_numpy(dtype=np.float32)[rows[selected]]
            text = numbers.astype(str).astype(object)
            text[np.isnan(numbers)] = None
            out[selected] = text
    return out


def explain_rows(booster, X: pd.DataFrame, feature_to_group: dict, groups_conf: dict, top_n: int,
                 contributions: np.ndarray | None = None) -> list[list[dict]]:
    """Reasons per row of a binary LightGBM model: up to ``top_n`` dicts ``{group, contribution, top_feature,
    top_feature_value, top_feature_contribution}`` per row.

    ``booster.predict(X, pred_contrib=True)`` without the bias column (or ``contributions`` precomputed by
    :func:`feature_contributions` for the same rows) is summed per reason group (``feature_to_group``: feature ->
    group id from conf/feature_groups.csv); groups with ``explainable: false`` or ``sensitive: true`` in
    ``groups_conf`` (reason_groups.yaml ``groups``; a missing flag counts as not explainable / sensitive) are
    dropped and the remaining groups with a positive sum are returned by descending contribution (ties in
    ``groups_conf`` order). ``group`` is the group's ``name``; ``top_feature`` / ``top_feature_contribution`` the
    group's largest contributing feature, ``top_feature_value`` its value as text (:func:`feature_values_as_text`).
    Rows without a positive explainable group get an empty list.
    """
    top_n = int(top_n)
    if top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}")
    if contributions is None:
        contributions = feature_contributions(booster, X)
    elif np.shape(contributions) != X.shape:
        raise ValueError(f"contributions of shape {np.shape(contributions)} for X of shape {X.shape}")
    feature_names = list(X.columns)
    group_ids, totals, top_feature, top_contribution = group_contributions(contributions, feature_names,
                                                                          feature_to_group, list(groups_conf))
    shown = np.array([bool(groups_conf[g].get("explainable", False)) and not groups_conf[g].get("sensitive", True)
                      for g in group_ids], dtype=bool)
    candidates = np.where(shown[None, :] & (totals > 0), totals, -np.inf)
    k = min(top_n, len(group_ids))
    order = np.argsort(-candidates, axis=1, kind="stable")[:, :k]
    chosen = np.take_along_axis(candidates, order, axis=1)
    n = len(X)
    text = feature_values_as_text(X, np.repeat(np.arange(n), k),
                                  np.take_along_axis(top_feature, order, axis=1).ravel()).reshape(n, k)
    names = [str(groups_conf[g].get("name", g)) for g in group_ids]
    reasons = []
    for i in range(n):
        row = []
        for j in range(k):
            if not np.isfinite(chosen[i, j]):
                break
            g = order[i, j]
            row.append({"group": names[g], "contribution": float(totals[i, g]),
                        "top_feature": feature_names[top_feature[i, g]], "top_feature_value": text[i, j],
                        "top_feature_contribution": float(top_contribution[i, g])})
        reasons.append(row)
    return reasons
