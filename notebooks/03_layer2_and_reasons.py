# Databricks notebook source
# %%
# Notebook 03: Layer 2 configuration guidance and the reason-code smoke test (docs/MVP_SPEC.md section 9).
#
# Training data: raw.l2_purchase_config (elig_at_feature_snapshot = 1) joined to the ml.modeling_table features on
# (client_id, feature_snapshot_date = snapshot_date); the Layer 1 feature list and category levels come from the
# release's feature_columns.json, so Layer 2 encodes features exactly like Layer 1. layer2.yaml split by
# application_date: <= train_application_end -> train (fit), test_application_start..test_application_end -> test
# (report only).
# Per attribute of layer2.yaml and per L2 (valid values from l2_config_validity.csv; an attribute that is not a
# column of that file allows all its values for every L2, e.g. buying_for):
#   fixed  the L2 has one valid value
#   head   pooled LightGBM over the purchases of all L2s with >= 2 valid values (features + l2 as categorical;
#          binary, or multiclass in the order of `values`), early-stopped on the last split.inner_valid_months
#          application months of train and refit on all of train; probabilities restricted to the L2's valid values
#          (and the constraints of the row; NULL inputs per constraint on_missing) and renormalised
#   share  train share of each valid value (Laplace +1): method share, or a head L2 with a valid value that has
#          fewer than min_rows_per_value train purchases
#   rule   mass_higher from wealth_seg3 (fixed for an L2 with one valid value)
# Evaluation on the test purchases, by attribute and L2: log-loss / accuracy of the method used next to the share
# baseline, accuracy of rules, abstain rate at the attribute's margin (modeling.layer2_probabilities /
# layer2_guidance: the same code notebook 04 serves with).
# Outputs: MLflow experiment l2_training (one run per head + a "layer2" run with layer2_spec.json; layer2_version =
# its run_id), file bundle <artifacts_dir>/<release_id>/layer2/{<attribute>_model.txt, layer2_spec.json} (staged and
# swapped in by common.publish_directory) + manifest.json entries layer2_version / layer2_spec_file, then
# ml.l2_evaluation and ml.layer2_models (replaceWhere release_id; ml.layer2_models is what notebook 04 looks the
# release up in).
# Reasons: modeling.explain_rows on 100 test rows with a Layer 1 model of the release (printed, not written).
#
# Parameters: env (default local), release_id (required on Databricks, where the job passes mvp-{{job.run_id}} to
# every task; locally the default is the release with the latest selected_at in ml.model_selection). Models and
# feature lists load from MLflow or the file bundle per env model_source.
# Run locally: python notebooks/03_layer2_and_reasons.py --env local --release_id mvp-acceptance
import json
import os
import re
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


def _notebook_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:  # Databricks / interactive window: the cwd is the notebook directory or the repository root
        for candidate in (Path.cwd(), Path.cwd() / "notebooks"):
            if (candidate / "common.py").is_file():
                return candidate.resolve()
        raise RuntimeError(f"notebooks/common.py not found from {Path.cwd()}")


if str(_notebook_dir()) not in sys.path:
    sys.path.insert(0, str(_notebook_dir()))
import common  # noqa: E402
import l1_contract
import modeling  # noqa: E402

env = common.get_param("env", "local")
release_id = common.get_param("release_id", None)
started = time.time()
print(f"env={env} release_id={release_id or '(latest in ml.model_selection)'}")

# COMMAND ----------

# %%
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

env_conf = common.load_env(env)
products = common.load_conf("products")
training_conf = common.load_conf("training")
layer2_conf = common.load_conf("layer2")
reason_conf = common.load_conf("reason_groups")
CLIENT_ID = products["keys"]["client_id"]
SNAPSHOT_DATE = products["keys"]["snapshot_date"]
ALL_L2S = list(products["products"])
ATTRIBUTES = layer2_conf["attributes"]
LGBM_PARAMS = dict(layer2_conf["lightgbm"])
TOP_N = int(training_conf["explain"]["top_n_reasons"])
MODEL_SOURCE = str(env_conf.get("model_source", "mlflow"))
METHODS = ("head", "rule", "share")
L2_COLUMN = "l2"            # categorical feature of the heads; levels = the L2s of products.yaml
SMOKE_ROWS, SMOKE_EXAMPLES = 100, 3
PURCHASE_KEYS = ["purchase_id", "client_id", "l2", "application_date", "feature_snapshot_date"]


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def as_date(value):
    return pd.Timestamp(value).date()


TRAIN_END = as_date(layer2_conf["split"]["train_application_end"])
TEST_START = as_date(layer2_conf["split"]["test_application_start"])
TEST_END = as_date(layer2_conf["split"]["test_application_end"])
INNER_VALID_MONTHS = int(layer2_conf["split"]["inner_valid_months"])  # heads early-stop on these last train months
check(TRAIN_END < TEST_START <= TEST_END, f"layer2.yaml split is not ordered: {layer2_conf['split']}")
check(INNER_VALID_MONTHS >= 1, "layer2.yaml split.inner_valid_months must be >= 1")
check(MODEL_SOURCE in ("mlflow", "files"), f"model_source must be mlflow or files, got {MODEL_SOURCE!r}")
check(release_id is not None or not common.is_databricks(),
      "release_id is required on Databricks (the job passes mvp-{{job.run_id}}); the latest-release default is local")
check(release_id is None or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", release_id) is not None,
      f"release_id {release_id!r} must be letters, digits, '_', '.', '-'")

for attribute, conf in ATTRIBUTES.items():
    values = list(conf.get("values") or [])
    check(conf.get("method") in METHODS, f"{attribute}: method must be one of {METHODS}, got {conf.get('method')!r}")
    check(len(values) >= 2 and len(set(values)) == len(values), f"{attribute}: needs >= 2 distinct values")
    if conf["method"] == "head":
        check(int(conf.get("min_rows_per_value", 0)) > 0, f"{attribute}: head needs min_rows_per_value")
    if conf["method"] in ("head", "share"):
        check(conf.get("margin") is not None and 0 <= float(conf["margin"]) < 1, f"{attribute}: margin in [0, 1)")
    if conf["method"] == "rule":
        rule = conf.get("rule") or {}
        check(len(values) == 2 and rule.get("feature") and rule.get("higher_end_values"),
              f"{attribute}: a rule needs two values and rule.feature / rule.higher_end_values")
    for constraint in modeling.normalize_constraints(conf.get("constraints")):  # raises on a malformed constraint
        check(constraint["value"] in values, f"{attribute}: constraint {constraint} is not for a value of {values}")

# Valid values per attribute and L2: the l2_config_validity.csv combinations (in the order of `values`).
validity_path = Path(layer2_conf["validity_file"])
validity = pd.read_csv(validity_path if validity_path.is_absolute() else common.repo_root() / validity_path,
                       dtype=str)
check("l2" in validity.columns, f"{validity_path} has no l2 column")
unknown = sorted(set(validity["l2"]) - set(ALL_L2S))
check(not unknown, f"{validity_path} has L2s that are not in products.yaml: {unknown}")
VALID = {}
for attribute, conf in ATTRIBUTES.items():
    values = list(conf["values"])
    if attribute not in validity.columns:
        VALID[attribute] = {l2: list(values) for l2 in ALL_L2S}
        continue
    outside = sorted(set(validity[attribute].dropna()) - set(values))
    check(not outside, f"{validity_path}: {attribute} values {outside} are not in layer2.yaml {values}")
    found = validity.groupby("l2")[attribute].agg(set).to_dict()
    VALID[attribute] = {l2: [v for v in values if v in found.get(l2, set())] for l2 in ALL_L2S}
    missing = [l2 for l2 in ALL_L2S if not VALID[attribute][l2]]
    if missing:
        print(f"WARNING {attribute}: no valid value in {validity_path} for {missing} (status not_applicable)")

feature_groups = pd.read_csv(common.repo_root() / "conf" / "feature_groups.csv", dtype=str)
FEATURE_TO_GROUP = dict(zip(feature_groups["feature_name"], feature_groups["reason_group"]))
GROUPS = reason_conf["groups"]
print(f"Attributes: { {a: c['method'] for a, c in ATTRIBUTES.items()} }; train applications <= {TRAIN_END}, test "
      f"{TEST_START}..{TEST_END}; model_source {MODEL_SOURCE}")

# COMMAND ----------

# %%
import mlflow  # noqa: E402
import mlflow.lightgbm  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

spark = common.get_spark(env_conf)
experiment_id = common.mlflow_setup(env_conf, "l2_training")

selection = common.to_pandas(common.read_table(spark, "ml.model_selection", env_conf),
                             ["release_id", "l2", "model_run_id", "selected_at", "training_config_hash", "provenance_json"])
check(len(selection) > 0, "ml.model_selection is empty (run notebook 02)")
if release_id is None:
    release_id = selection.sort_values(["selected_at", "release_id"])["release_id"].iloc[-1]
    print(f"release_id defaults to the latest release in ml.model_selection: {release_id}")
RELEASE = selection[selection["release_id"] == release_id].set_index("l2")
check(len(RELEASE) > 0, f"release {release_id!r} is not in ml.model_selection")
check(RELEASE.index.is_unique, f"release {release_id} has several ml.model_selection rows for one L2")
RELEASE_L2S = [l2 for l2 in ALL_L2S if l2 in RELEASE.index]
check(len(RELEASE_L2S) == len(RELEASE), f"release {release_id} has L2s that are not in products.yaml")
artifacts_dir = Path(env_conf["paths"]["artifacts_dir"])
RELEASE_DIR = (artifacts_dir if artifacts_dir.is_absolute() else common.repo_root() / artifacts_dir) / release_id
MANIFEST_PATH = RELEASE_DIR / "manifest.json"
check(MANIFEST_PATH.is_file(), f"{MANIFEST_PATH} does not exist: notebook 02 publishes the release bundle")
with open(MANIFEST_PATH, encoding="utf-8") as f:
    check(json.load(f).get("release_id") == release_id, f"{MANIFEST_PATH} belongs to another release")
work_dir = Path(tempfile.mkdtemp(prefix="prodrec-03-"))


def release_artifact(l2: str, name: str) -> Path:
    """``name`` of a release L2: downloaded from its MLflow model run (model_source mlflow) or from the bundle."""
    if MODEL_SOURCE == "mlflow":
        target = work_dir / l2
        target.mkdir(exist_ok=True)
        return Path(mlflow.artifacts.download_artifacts(run_id=RELEASE.loc[l2, "model_run_id"], artifact_path=name,
                                                        dst_path=str(target)))
    path = RELEASE_DIR / l2 / name
    check(path.is_file(), f"{path} does not exist")
    return path


# One Layer 1 feature specification for the whole release (notebook 01 trains every L2 on the same columns).
feature_specs = {}
for l2 in RELEASE_L2S:
    with open(release_artifact(l2, "feature_columns.json"), encoding="utf-8") as f:
        feature_specs[l2] = json.load(f)
    provenance = json.loads(RELEASE.loc[l2, "provenance_json"])
    l1_contract.validate_provenance(provenance, training_conf, RELEASE.loc[l2, "model_run_id"])
    l1_contract.validate_feature_spec(feature_specs[l2], provenance, "champion_refit")
SPEC_L2 = RELEASE_L2S[0]
FEATURE_SPEC = feature_specs[SPEC_L2]
for l2, spec in feature_specs.items():
    check(spec["feature_columns"] == FEATURE_SPEC["feature_columns"],
          f"feature_columns.json of {l2} differs from {SPEC_L2}: the release mixes feature columns")
# Eligible L1 populations can differ; shared column order comes from L1, category levels from pooled L2 Train.
FEATURES = list(FEATURE_SPEC["feature_columns"])
LEVELS = dict(FEATURE_SPEC["categorical_levels"])
HEAD_FEATURES = FEATURES + [L2_COLUMN]
HEAD_LEVELS = {**LEVELS, L2_COLUMN: ALL_L2S}
HEAD_CATEGORICAL = list(LEVELS) + [L2_COLUMN]
check(L2_COLUMN not in FEATURES, f"a Layer 1 feature is called {L2_COLUMN!r}")
for attribute, conf in ATTRIBUTES.items():
    if conf["method"] == "rule":
        check(conf["rule"]["feature"] in FEATURES, f"{attribute}: rule feature {conf['rule']['feature']} is no feature")
    for constraint in conf.get("constraints") or []:
        check(modeling.constraint_columns(str(constraint["require"]), FEATURES),
              f"{attribute}: constraint {constraint['require']!r} refers to no feature")
print(f"Release {release_id}: L2s {RELEASE_L2S}; {len(FEATURES)} features from {SPEC_L2}/feature_columns.json "
      f"({MODEL_SOURCE})")

# COMMAND ----------

# %%
# Purchases (eligible at the feature snapshot, train or test window) with the features of their snapshot.
t0 = time.time()
purchase_columns = PURCHASE_KEYS + list(ATTRIBUTES)
purchases_sdf = common.read_table(spark, "raw.l2_purchase_config", env_conf)
missing = [c for c in purchase_columns + ["elig_at_feature_snapshot"] if c not in purchases_sdf.columns]
check(not missing, f"raw.l2_purchase_config lacks columns {missing}")
collisions = sorted(set(FEATURES) & set(purchase_columns))
check(not collisions, f"feature columns collide with purchase columns: {collisions}")
application = F.col("application_date")
purchases_sdf = (purchases_sdf.filter(F.col("elig_at_feature_snapshot") == 1)
                 .filter((application <= F.lit(TRAIN_END)) | application.between(F.lit(TEST_START), F.lit(TEST_END)))
                 .select(*purchase_columns))
features_sdf = common.read_table(spark, "ml.modeling_table", env_conf).select(
    F.col(CLIENT_ID).alias("_feature_client"), F.col(SNAPSHOT_DATE).alias("_feature_snapshot"), *FEATURES)
joined = purchases_sdf.join(features_sdf, (F.col("client_id") == F.col("_feature_client"))
                            & (F.col("feature_snapshot_date") == F.col("_feature_snapshot")), "inner")
n_purchases = purchases_sdf.count()
data = (common.to_pandas(joined, purchase_columns + FEATURES)
        .sort_values("purchase_id", kind="mergesort").reset_index(drop=True))  # deterministic row order
check(len(data) > 0, "no purchase matches a ml.modeling_table snapshot")
check(data["purchase_id"].is_unique, "raw.l2_purchase_config repeats purchase_id")
unknown = sorted(set(data["l2"]) - set(ALL_L2S))
check(not unknown, f"raw.l2_purchase_config has L2s that are not in products.yaml: {unknown}")
data["application_date"] = pd.to_datetime(data["application_date"])
data["block"] = np.where(data["application_date"].dt.date <= TRAIN_END, "train", "test")
IS_TRAIN = (data["block"] == "train").to_numpy()
IS_TEST = ~IS_TRAIN
load_seconds = round(time.time() - t0, 1)
print(f"Loaded {len(data):,} purchases with features ({n_purchases - len(data):,} of {n_purchases:,} eligible "
      f"purchases have no ml.modeling_table snapshot): train {IS_TRAIN.sum():,}, test {IS_TEST.sum():,} in "
      f"{load_seconds}s")

t0 = time.time()
LAYER2_FEATURE_SPEC = modeling.layer2_feature_spec(data, IS_TRAIN, FEATURE_SPEC)
LEVELS = LAYER2_FEATURE_SPEC["categorical_levels"]
HEAD_LEVELS = {**LEVELS, L2_COLUMN: ALL_L2S}
X = modeling.prepare_matrix(data, FEATURES, LEVELS)                  # Layer 2 encoding for rules / constraints
X_head = modeling.prepare_matrix(data, HEAD_FEATURES, HEAD_LEVELS)   # + l2 for the heads
L2_OF_ROW = data["l2"].to_numpy(dtype=object)
print(f"Prepared matrices in {time.time() - t0:.1f}s")

# COMMAND ----------

# %%
def l2_plan(attribute: str, value_ok: np.ndarray) -> dict:
    """per_l2 entries of one attribute: method_used, valid_values, fixed_value, share (head / share L2s), train_rows."""
    conf = ATTRIBUTES[attribute]
    train = data.loc[IS_TRAIN & value_ok]
    counts = train.groupby(["l2", attribute]).size()
    plan = {}
    for l2 in ALL_L2S:
        valid = VALID[attribute][l2]
        rows = {v: int(counts.get((l2, v), 0)) for v in valid}
        if not valid:
            method_used = "not_applicable"
        elif len(valid) == 1:
            method_used = "fixed"
        elif conf["method"] in ("rule", "share"):
            method_used = conf["method"]
        else:
            method_used = "head" if min(rows.values()) >= int(conf["min_rows_per_value"]) else "share"
        share = None
        if method_used in ("head", "share"):
            total = sum(rows.values()) + len(valid)
            share = {v: (rows[v] + 1) / total for v in valid}
        plan[l2] = {"method_used": method_used, "valid_values": valid,
                    "fixed_value": valid[0] if method_used == "fixed" else None, "share": share, "train_rows": rows}
    return plan


def train_head(attribute: str, value_ok: np.ndarray) -> tuple:
    """Pooled head over the purchases of the L2s with >= 2 valid values: ``(booster, objective overrides, info)``."""
    values = list(ATTRIBUTES[attribute]["values"])
    multi_l2s = [l2 for l2 in ALL_L2S if len(VALID[attribute][l2]) >= 2]
    rows = IS_TRAIN & value_ok & data["l2"].isin(multi_l2s).to_numpy()
    y = data[attribute].map({v: i for i, v in enumerate(values)}).to_numpy()
    months = data["application_date"].dt.to_period("M")
    valid_months = sorted(months[rows].unique())[-INNER_VALID_MONTHS:]
    in_valid = rows & months.isin(valid_months).to_numpy()
    in_fit = rows & ~in_valid
    check(in_fit.any() and in_valid.any(), f"{attribute}: the head has no fit or no inner validation rows")
    check(len(np.unique(y[in_fit])) >= 2, f"{attribute}: the head's fit rows have a single class")
    if len(values) == 2:
        overrides = {"objective": "binary", "metric": "binary_logloss"}
    else:
        overrides = {"objective": "multiclass", "metric": "multi_logloss", "num_class": len(values)}
    early, best_iteration, evals = modeling.fit_lgbm(X_head.loc[in_fit], y[in_fit].astype(np.int64),
                                                     X_head.loc[in_valid], y[in_valid].astype(np.int64),
                                                     LGBM_PARAMS, HEAD_CATEGORICAL, overrides)
    booster = modeling.refit_lgbm(X_head.loc[rows], y[rows].astype(np.int64), LGBM_PARAMS, best_iteration,
                                  HEAD_CATEGORICAL, overrides)
    check(booster.feature_name() == HEAD_FEATURES, f"{attribute}: LightGBM renamed the head's feature columns")
    info = {
        "objective": overrides["objective"], "best_iteration": best_iteration, "train_l2s": multi_l2s,
        "train_rows": int(rows.sum()), "fit_rows": int(in_fit.sum()), "inner_valid_rows": int(in_valid.sum()),
        "inner_valid_months": ",".join(str(m) for m in valid_months),
        "inner_valid_logloss": float(evals["valid"][overrides["metric"]][best_iteration - 1]),
    }
    return booster, overrides, info


def guidance_metrics(probs: np.ndarray, y: np.ndarray, methods: np.ndarray, margin, with_logloss: bool) -> dict:
    """n_rows, n_not_applicable, accuracy (+ logloss, abstain_rate, guided_accuracy) of served probabilities."""
    index, _, _, status = modeling.layer2_guidance(probs, methods, margin)
    usable = index >= 0
    metrics = {"n_rows": float(len(y)), "n_not_applicable": float((~usable).sum())}
    if not usable.any():
        return metrics
    hit = index[usable] == y[usable]
    metrics["accuracy"] = float(hit.mean())
    if with_logloss:
        p_true = probs[usable][np.arange(int(usable.sum())), y[usable]]
        metrics["logloss"] = float(-np.mean(np.log(np.clip(p_true, modeling.LOGLOSS_CLIP, 1.0)))) + 0.0  # no -0.0
        abstain = status[usable] == "abstain"
        metrics["abstain_rate"] = float(abstain.mean())
        metrics["guided_accuracy"] = float(hit[~abstain].mean()) if (~abstain).any() else float("nan")
    return metrics


def evaluate_attribute(attribute: str, spec: dict, value_ok: np.ndarray, booster) -> list[dict]:
    """ml.l2_evaluation rows of one attribute on the test purchases: per L2, per method_used ('all' L2s) and overall
    ('all', 'all'); head / share attributes also carry the share baseline (share_* metrics) of the same rows."""
    values = list(spec["values"])
    rows = np.flatnonzero(IS_TEST & value_ok)
    l2s = L2_OF_ROW[rows]
    y = data[attribute].iloc[rows].map({v: i for i, v in enumerate(values)}).to_numpy().astype(np.int64)
    X_rows, X_head_rows = X.iloc[rows], X_head.iloc[rows]
    probs, methods = modeling.layer2_probabilities(spec, l2s, X_rows, X_head_rows, booster)
    with_baseline = spec["method"] != "rule"
    if with_baseline:  # the share baseline: every head L2 served by its train shares instead
        baseline_spec = {**spec, "per_l2": {l2: ({**entry, "method_used": "share"} if entry["method_used"] == "head"
                                                 else entry) for l2, entry in spec["per_l2"].items()}}
        share_probs, share_methods = modeling.layer2_probabilities(baseline_spec, l2s, X_rows)
    subsets = [(l2, methods[l2s == l2][0], l2s == l2) for l2 in ALL_L2S if (l2s == l2).any()]
    subsets += [("all", method, methods == method) for method in pd.unique(methods)]
    subsets += [("all", "all", np.ones(len(rows), dtype=bool))]
    out = []
    for l2, method_used, mask in subsets:
        metrics = guidance_metrics(probs[mask], y[mask], methods[mask], spec["margin"], with_baseline)
        if with_baseline:
            baseline = guidance_metrics(share_probs[mask], y[mask], share_methods[mask], spec["margin"], True)
            metrics.update({f"share_{name}": value for name, value in baseline.items()
                            if name not in ("n_rows", "n_not_applicable")})
        out += [{"attribute": attribute, "l2": l2, "method_used": method_used, "metric": name, "value": value}
                for name, value in metrics.items()]
    return out


def log_head_run(attribute: str, booster, info: dict, evaluation: list[dict], run_dir: Path) -> str:
    """MLflow run of one head (run name = attribute); returns its run_id. ``model.txt`` is saved in ``run_dir``."""
    conf = ATTRIBUTES[attribute]
    head_l2s = [l2 for l2, entry in specs[attribute]["per_l2"].items() if entry["method_used"] == "head"]
    test = {row["metric"]: row["value"] for row in evaluation if row["l2"] == "all" and row["method_used"] == "head"}
    with mlflow.start_run(run_name=attribute) as run:
        mlflow.set_tags({"release_id": release_id, "attribute": attribute, "kind": "layer2_head"})
        mlflow.log_params({**LGBM_PARAMS, "objective": info["objective"], "num_class": len(conf["values"]),
                           "values": ",".join(conf["values"]), "best_iteration": info["best_iteration"],
                           "min_rows_per_value": conf["min_rows_per_value"], "margin": conf["margin"],
                           "train_l2s": ",".join(info["train_l2s"]), "head_l2s": ",".join(head_l2s),
                           "train_rows": info["train_rows"], "fit_rows": info["fit_rows"],
                           "inner_valid_rows": info["inner_valid_rows"],
                           "inner_valid_months": info["inner_valid_months"], "release_id": release_id,
                           "feature_columns_file": f"{SPEC_L2}/feature_columns.json"})
        metrics = {"inner_valid_logloss": info["inner_valid_logloss"]}
        metrics.update({f"test_{name}": value for name, value in test.items()})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None and np.isfinite(v)})
        booster.save_model(str(run_dir / "model.txt"))
        mlflow.log_artifact(str(run_dir / "model.txt"))
        mlflow.lightgbm.log_model(booster, name="model")
        return run.info.run_id


specs, boosters, head_infos, evaluation_rows, attribute_seconds = {}, {}, {}, [], {}
for attribute, conf in ATTRIBUTES.items():
    t0 = time.time()
    values = list(conf["values"])
    valid_pairs = {(l2, v) for l2 in ALL_L2S for v in VALID[attribute][l2]}
    value_ok = pd.MultiIndex.from_arrays([data["l2"], data[attribute]]).isin(list(valid_pairs))
    if (~value_ok).any():
        print(f"WARNING {attribute}: {(~value_ok).sum():,} purchases have a missing value or one that is not valid "
              f"for their L2 (left out of training and evaluation)")
    plan = l2_plan(attribute, value_ok)
    specs[attribute] = {
        "method": conf["method"], "values": values, "margin": conf.get("margin"),
        "min_rows_per_value": conf.get("min_rows_per_value"), "objective": None, "model_run_id": None,
        "rule": conf.get("rule"), "constraints": modeling.normalize_constraints(conf.get("constraints")),
        "per_l2": plan,
    }
    booster = None
    if any(entry["method_used"] == "head" for entry in plan.values()):
        booster, overrides, head_infos[attribute] = train_head(attribute, value_ok)
        specs[attribute]["objective"] = overrides["objective"]
    elif conf["method"] == "head":
        print(f"{attribute}: every L2 is fixed or falls back to its share; no head is trained")
    boosters[attribute] = booster
    evaluation_rows += evaluate_attribute(attribute, specs[attribute], value_ok, booster)
    attribute_seconds[attribute] = round(time.time() - t0, 1)
    used = pd.Series({l2: entry["method_used"] for l2, entry in plan.items()}).value_counts().to_dict()
    print(f"{attribute}: method_used {used}"
          + (f"; head best_iteration {head_infos[attribute]['best_iteration']}, inner valid log-loss "
             f"{head_infos[attribute]['inner_valid_logloss']:.4f}" if booster is not None else "")
          + f"; {attribute_seconds[attribute]}s")

evaluation = pd.DataFrame(evaluation_rows)

# COMMAND ----------

# %%
# Reasons smoke test: explain_rows with a Layer 1 model of the release on 100 test purchases (nothing is written).
SMOKE_L2 = "product1" if "product1" in RELEASE_L2S else RELEASE_L2S[0]
smoke_booster = lgb.Booster(model_file=str(release_artifact(SMOKE_L2, "model.txt")))
smoke_rows = np.flatnonzero(IS_TEST)[:SMOKE_ROWS]
smoke_X = modeling.prepare_matrix(data.iloc[smoke_rows], FEATURES, feature_specs[SMOKE_L2]["categorical_levels"])
check(smoke_booster.feature_name() == FEATURES, f"{SMOKE_L2}: model features differ from feature_columns.json")
smoke_contributions = modeling.feature_contributions(smoke_booster, smoke_X)
smoke = modeling.explain_rows(smoke_booster, smoke_X, FEATURE_TO_GROUP, GROUPS, TOP_N,
                              contributions=smoke_contributions)
bias = smoke_booster.predict(smoke_X.iloc[:1], pred_contrib=True)[0, -1]
additivity = float(np.max(np.abs(smoke_contributions.sum(axis=1) + bias
                                 - smoke_booster.predict(smoke_X, raw_score=True))))
check(additivity < 1e-6, f"pred_contrib does not add up to the raw score (max error {additivity})")
hidden = {g["name"] for g in GROUPS.values() if g.get("sensitive", True) or not g.get("explainable", False)}
shown = [reason["group"] for reasons in smoke for reason in reasons]
check(not set(shown) & hidden, f"sensitive / non-explainable groups among the reasons: {set(shown) & hidden}")
reason_counts = pd.Series([len(reasons) for reasons in smoke]).value_counts().sort_index().to_dict()
print(f"Reasons smoke test ({SMOKE_L2} Layer 1 model, {len(smoke)} test purchases, top {TOP_N}): reasons per row "
      f"{reason_counts}; contributions add up to the raw score (max error {additivity:.1e})")
for i in range(min(SMOKE_EXAMPLES, len(smoke))):
    row = data.iloc[smoke_rows[i]]
    print(f"  {row['client_id']} ({row['l2']} purchase {row['purchase_id']}):")
    for reason in smoke[i]:
        print(f"    {reason['group']:<26} {reason['contribution']:+.4f}  top feature {reason['top_feature']} = "
              f"{reason['top_feature_value']} ({reason['top_feature_contribution']:+.4f})")

# COMMAND ----------

# %%
# MLflow first (the heads, then the layer2 run whose run_id is layer2_version), then the file bundle, then the
# tables: ml.layer2_models names the layer2_version that notebook 04 loads, so it is written last.
t0 = time.time()
LAYER2_DIR = RELEASE_DIR / "layer2"
staging_dir = common.make_staging_dir(LAYER2_DIR)
try:
    for attribute, booster in boosters.items():
        if booster is None:
            continue
        run_dir = work_dir / f"head_{attribute}"
        run_dir.mkdir()
        attribute_evaluation = [row for row in evaluation_rows if row["attribute"] == attribute]
        specs[attribute]["model_run_id"] = log_head_run(attribute, booster, head_infos[attribute],
                                                        attribute_evaluation, run_dir)
        shutil.copyfile(run_dir / "model.txt", staging_dir / f"{attribute}_model.txt")
    with mlflow.start_run(run_name="layer2") as layer2_run:
        LAYER2_VERSION = layer2_run.info.run_id
        layer2_spec = {
            "release_id": release_id,
            "layer2_version": LAYER2_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "attributes": specs,
            "buying_for_constraints": modeling.normalize_constraints((ATTRIBUTES.get("buying_for") or {})
                                                                     .get("constraints")),
            "feature_columns_file": f"{SPEC_L2}/feature_columns.json",
            "base_feature_spec": LAYER2_FEATURE_SPEC,
            "base_feature_spec_hash": l1_contract.digest(LAYER2_FEATURE_SPEC),
            "training_config_hash": l1_contract.digest(training_conf),
            "l2_column": L2_COLUMN,
            "l2_levels": ALL_L2S,
            "split": {"train_application_end": str(TRAIN_END), "test_application_start": str(TEST_START),
                      "test_application_end": str(TEST_END), "inner_valid_months": INNER_VALID_MONTHS},
        }
        with open(staging_dir / "layer2_spec.json", "w", encoding="utf-8") as f:
            json.dump(layer2_spec, f, indent=2)
        evaluation.to_csv(work_dir / "l2_evaluation.csv", index=False)
        mlflow.set_tags({"release_id": release_id, "kind": "layer2"})
        mlflow.log_params({"release_id": release_id, "attributes": ",".join(ATTRIBUTES), "model_source": MODEL_SOURCE,
                           "feature_columns_file": layer2_spec["feature_columns_file"]})
        mlflow.log_artifact(str(staging_dir / "layer2_spec.json"))
        mlflow.log_artifact(str(work_dir / "l2_evaluation.csv"))
    common.publish_directory(staging_dir, LAYER2_DIR)
except BaseException:
    shutil.rmtree(staging_dir, ignore_errors=True)
    print(f"Layer 2 bundle not published; {LAYER2_DIR} is unchanged")
    raise

with open(MANIFEST_PATH, encoding="utf-8") as f:
    manifest = json.load(f)
manifest.update(layer2_version=LAYER2_VERSION, layer2_spec_file="layer2/layer2_spec.json")
manifest_tmp = MANIFEST_PATH.with_name(f"manifest.json.tmp-{os.getpid()}")
with open(manifest_tmp, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
os.replace(manifest_tmp, MANIFEST_PATH)
publish_seconds = round(time.time() - t0, 1)
print(f"MLflow layer2 run {LAYER2_VERSION}; published {LAYER2_DIR} and updated {MANIFEST_PATH} in {publish_seconds}s")

# COMMAND ----------

# %%
EVALUATION_COLUMNS = [("release_id", "string"), ("attribute", "string"), ("l2", "string"), ("method_used", "string"),
                      ("metric", "string"), ("value", "double")]
MODELS_COLUMNS = [("release_id", "string"), ("attribute", "string"), ("l2", "string"), ("method_used", "string"),
                  ("value", "string"), ("share_prob", "double"), ("model_run_id", "string"), ("rule", "string"),
                  ("layer2_version", "string")]


def spark_frame(pdf: pd.DataFrame, columns: list[tuple[str, str]]):
    """Spark DataFrame of ``pdf`` with exactly ``columns`` (name, Spark type), matched by name."""
    return spark.createDataFrame(pdf[[name for name, _ in columns]],
                                 ", ".join(f"`{name}` {data_type}" for name, data_type in columns))


# ml.layer2_models: one row per attribute x L2 x valid value (one row with value NULL for an L2 without any).
model_rows = []
for attribute, spec in specs.items():
    rule_text = json.dumps(spec["rule"]) if spec["method"] == "rule" else None
    for l2, entry in spec["per_l2"].items():
        method_used = entry["method_used"]
        for value in entry["valid_values"] or [None]:
            model_rows.append({
                "attribute": attribute, "l2": l2, "method_used": method_used, "value": value,
                "share_prob": entry["share"][value] if entry["share"] and value is not None else None,
                "model_run_id": spec["model_run_id"] if method_used == "head" else None,
                "rule": rule_text if method_used == "rule" else None,
            })
models = pd.DataFrame(model_rows).assign(release_id=release_id, layer2_version=LAYER2_VERSION)
evaluation_out = evaluation.assign(release_id=release_id)
predicate = f"release_id = '{release_id}'"
t0 = time.time()
evaluation_table = common.write_table(spark_frame(evaluation_out, EVALUATION_COLUMNS), "ml.l2_evaluation", env_conf,
                                      mode="overwrite", replace_where=predicate)
models_table = common.write_table(spark_frame(models, MODELS_COLUMNS), "ml.layer2_models", env_conf,
                                  mode="overwrite", replace_where=predicate)
written = common.read_table(spark, "ml.layer2_models", env_conf).filter(F.col("release_id") == release_id)
versions = [r[0] for r in written.select("layer2_version").distinct().collect()]
check(written.count() == len(models) and versions == [LAYER2_VERSION],
      f"{models_table} does not hold exactly the {len(models)} rows of layer2_version {LAYER2_VERSION}")
print(f"Wrote {evaluation_table} ({len(evaluation_out)} rows) and {models_table} ({len(models)} rows) in "
      f"{time.time() - t0:.1f}s")
shutil.rmtree(work_dir, ignore_errors=True)

# COMMAND ----------

# %%
def metric(attribute: str, l2: str, method_used: str, name: str):
    found = evaluation[(evaluation["attribute"] == attribute) & (evaluation["l2"] == l2)
                       & (evaluation["method_used"] == method_used) & (evaluation["metric"] == name)]["value"]
    return float(found.iloc[0]) if len(found) else None


def fmt(value, digits: int = 4) -> str:
    return "-" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


header = (f"{'attribute':<12} {'l2':<10} {'method_used':<14} {'n_test':>6} {'logloss':>8} {'share_ll':>8} "
          f"{'accuracy':>8} {'share_acc':>9} {'abstain':>7} {'share_abs':>9}")
print(f"\nLayer 2 test evaluation, release {release_id} (logloss / accuracy of the method used vs the share baseline)")
print(f"{header}\n{'-' * len(header)}")
summary_rows = {}
for attribute in ATTRIBUTES:
    groups = evaluation[evaluation["attribute"] == attribute][["l2", "method_used"]].drop_duplicates()
    for l2, method_used in groups.itertuples(index=False):
        values = {name: metric(attribute, l2, method_used, name)
                  for name in ("n_rows", "logloss", "share_logloss", "accuracy", "share_accuracy", "abstain_rate",
                               "share_abstain_rate")}
        summary_rows.setdefault(attribute, {})[f"{l2}/{method_used}"] = {k: v for k, v in values.items()
                                                                        if v is not None}
        print(f"{attribute:<12} {l2:<10} {method_used:<14} {int(values['n_rows']):>6} {fmt(values['logloss']):>8} "
              f"{fmt(values['share_logloss']):>8} {fmt(values['accuracy']):>8} {fmt(values['share_accuracy']):>9} "
              f"{fmt(values['abstain_rate'], 3):>7} {fmt(values['share_abstain_rate'], 3):>9}")

summary = {
    "env": env,
    "release_id": release_id,
    "layer2_version": LAYER2_VERSION,
    "experiment_id": experiment_id,
    "model_source": MODEL_SOURCE,
    "bundle": str(LAYER2_DIR),
    "output_tables": [evaluation_table, models_table],
    "purchases": {"train": int(IS_TRAIN.sum()), "test": int(IS_TEST.sum()), "unmatched": n_purchases - len(data)},
    "method_used": {attribute: pd.Series({l2: e["method_used"] for l2, e in spec["per_l2"].items()})
                    .value_counts().to_dict() for attribute, spec in specs.items()},
    "heads": {attribute: {k: v for k, v in info.items() if k != "train_l2s"} | {"model_run_id":
                                                                               specs[attribute]["model_run_id"]}
              for attribute, info in head_infos.items()},
    "reasons_smoke": {"l2": SMOKE_L2, "rows": len(smoke), "reasons_per_row": reason_counts},
    "seconds": {"load": load_seconds, "attributes": attribute_seconds, "publish": publish_seconds},
    "elapsed_seconds": round(time.time() - started, 1),
}
print(json.dumps(summary, indent=2, default=str))
