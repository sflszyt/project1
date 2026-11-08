# Databricks notebook source
# %%
# Notebook 04: batch scoring, Top-k recommendations, Layer 2 guidance and reasons (docs/MVP_SPEC.md section 10).
#
# Scoring snapshot: the latest ml.modeling_table snapshot <= run_date (at most scoring.lookback_days before it), all
# clients. Per L2 of the release (ml.model_selection; models, calibrators and feature lists from MLflow or the file
# bundle per env model_source):
#   raw_propensity_score             LightGBM probability; NULL with exclusion_reason 'not_eligible' when elig_{l2} = 0
#   calibrated_purchase_probability  calibrator output when calibration_status starts with 'calibrated', else NULL
#   ranking_score                    ranking field (training.yaml ranking.ranking_field; raw when calibration is
#                                    disabled) x business_weight
#   rank                             per client among the ranking_eligible_flag = 1 rows: 1, 2, ... by descending
#                                    ranking_score (ties by l2 name); NULL for every other row
# Rows with rank <= explain.top_k also get the Layer 2 guidance (<attribute>_guidance / _prob / _status for every
# layer2.yaml attribute, served from the release's layer2_spec.json and heads by modeling.layer2_probabilities) and
# reasons (reason_1..n from modeling.explain_rows with the L2's model).
# Outputs (Delta, partitioned by run_date, replaceWhere run_date = <run_date>): scores.customer_l2_scores,
# scores.customer_top_recommendations (rank <= top_k), scores.customer_reason_contributions (top-k row x reason
# group), scores.scoring_runs (one row, written last: a run_date is committed only when it has its scoring_runs
# row); manifest.json in MLflow experiment scoring (run name = run_date). The partition column run_date is the last
# column of every table (the local in-memory catalog rejects later replaceWhere writes otherwise).
#
# Parameters: env (default local), run_date (default today), release_id (required on Databricks, where the job
# passes mvp-{{job.run_id}} to every task; locally the default is the release with the latest selected_at in
# ml.model_selection), l2_list (pilot | all | comma-separated; default: every L2 of the release).
# Run locally: python notebooks/04_batch_scoring.py --env local --run_date 2026-09-30 --release_id mvp-acceptance
import hashlib
import json
import math
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import date, datetime, timezone
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
run_date_text = common.get_param("run_date", None)
release_id = common.get_param("release_id", None)
l2_list = common.get_param("l2_list", None)
RUN_DATE = date.fromisoformat(run_date_text) if run_date_text else date.today()
started = time.time()
print(f"env={env} run_date={RUN_DATE} release_id={release_id or '(latest in ml.model_selection)'} "
      f"l2_list={l2_list or '(every L2 of the release)'}")

# COMMAND ----------

# %%
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

env_conf = common.load_env(env)
products = common.load_conf("products")
training_conf = common.load_conf("training")
CONFIG_HASH = l1_contract.digest(training_conf)
layer2_conf = common.load_conf("layer2")
reason_conf = common.load_conf("reason_groups")
CLIENT_ID = products["keys"]["client_id"]
SNAPSHOT_DATE = products["keys"]["snapshot_date"]
CATALOG = products["products"]
ELIG_TEMPLATE = products["eligibility_column_template"]
LOOKBACK_DAYS = int(training_conf["scoring"]["lookback_days"])
TOP_K = int(training_conf["explain"]["top_k"])
TOP_N = int(training_conf["explain"]["top_n_reasons"])
STORE_FEATURE_LEVEL = bool(training_conf["explain"].get("store_feature_level", False))
CALIBRATION_ENABLED = bool(training_conf["calibration"].get("enabled", True))
RANKING_FIELD = str(training_conf["ranking"]["ranking_field"]) if CALIBRATION_ENABLED else "raw"
SCORE_COLUMN = {"calibrated": "calibrated_purchase_probability", "raw": "raw_propensity_score"}
MODEL_SOURCE = str(env_conf.get("model_source", "mlflow"))
ATTRIBUTES = layer2_conf["attributes"]
# layer2.yaml keys that notebook 03 baked into layer2_spec.json: scoring refuses a release whose spec differs.
LAYER2_SPEC_KEYS = ("method", "values", "margin", "min_rows_per_value", "rule", "constraints")
CALIBRATOR_CLASS = {"platt": "platt", "intercept": "intercept", "none": "identity", "disabled": "identity"}
# Output files of about this many rows (createDataFrame from pandas makes one partition per 10,000-row Arrow batch).
ROWS_PER_FILE = 1_000_000


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


check(RANKING_FIELD in SCORE_COLUMN, f"ranking.ranking_field must be calibrated or raw, got {RANKING_FIELD!r}")
check(MODEL_SOURCE in ("mlflow", "files"), f"model_source must be mlflow or files, got {MODEL_SOURCE!r}")
check(TOP_K >= 1 and TOP_N >= 1, "explain.top_k and explain.top_n_reasons must be >= 1")
check(release_id is not None or not common.is_databricks(),
      "release_id is required on Databricks (the job passes mvp-{{job.run_id}}); the latest-release default is local")
check(release_id is None or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", release_id) is not None,
      f"release_id {release_id!r} must be letters, digits, '_', '.', '-'")
feature_groups = pd.read_csv(common.repo_root() / "conf" / "feature_groups.csv", dtype=str)
FEATURE_TO_GROUP = dict(zip(feature_groups["feature_name"], feature_groups["reason_group"]))
GROUPS = reason_conf["groups"]
print(f"ranking_field {RANKING_FIELD} (calibration enabled: {CALIBRATION_ENABLED}); top_k {TOP_K}, "
      f"{TOP_N} reasons; model_source {MODEL_SOURCE}")


def git_commit() -> str | None:
    """HEAD of the repository; None when git is unavailable or fails."""
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=common.repo_root(), capture_output=True,
                                text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return (result.stdout.strip() or None) if result.returncode == 0 else None

# COMMAND ----------

# %%
import mlflow  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

spark = common.get_spark(env_conf)
experiment_id = common.mlflow_setup(env_conf, "scoring")


INPUT_TABLES = ["ml.modeling_table", "ml.model_selection", "ml.layer2_models"]
# Pin each actual input before reading any of them. These are per-table versions, not an atomic multi-table batch.
input_versions = {logical: common.delta_version(spark, logical, env_conf) for logical in INPUT_TABLES}
selection = common.to_pandas(common.read_table(spark, "ml.model_selection", env_conf,
                                              version_as_of=input_versions["ml.model_selection"]))
check(len(selection) > 0, "ml.model_selection is empty (run notebook 02)")
if release_id is None:
    release_id = selection.sort_values(["selected_at", "release_id"])["release_id"].iloc[-1]
    print(f"release_id defaults to the latest release in ml.model_selection: {release_id}")
RELEASE = selection[selection["release_id"] == release_id].set_index("l2")
check({"training_config_hash", "provenance_json", "evidence_source", "evidence_event_count"} <= set(RELEASE.columns),
      "Legacy release has no training provenance; retrain and publish a fresh release")
PROVENANCE = {}
for l2, row in RELEASE.iterrows():
    check(row["training_config_hash"] == CONFIG_HASH, f"{l2}: selected YAML differs from frozen release configuration")
    provenance = json.loads(row["provenance_json"])
    l1_contract.validate_provenance(provenance, training_conf, row["model_run_id"])
    check(provenance["calibrator_run_id"] == row["calibrator_run_id"]
          and provenance["calibration_enabled"] == CALIBRATION_ENABLED
          and provenance["calibration_status"] == row["calibration_status"]
          and provenance["calibration_method"] == row["calibration_method"]
          and provenance["ranking_field"] == RANKING_FIELD,
          f"{l2}: release calibration provenance differs from selection")
    PROVENANCE[l2] = provenance
check(len(RELEASE) > 0, f"release {release_id!r} is not in ml.model_selection")
check(RELEASE.index.is_unique, f"release {release_id} has several ml.model_selection rows for one L2")
RELEASE_L2S = [l2 for l2 in CATALOG if l2 in RELEASE.index]
check(len(RELEASE_L2S) == len(RELEASE), f"release {release_id} has L2s that are not in products.yaml")
L2S = modeling.resolve_l2_list(products, l2_list) if l2_list else RELEASE_L2S
outside = [l2 for l2 in L2S if l2 not in RELEASE.index]
check(not outside, f"l2_list has L2s that release {release_id} does not contain: {outside}")
ELIG = {l2: ELIG_TEMPLATE.format(l2=l2) for l2 in L2S}

# The release gate was evaluated on the release's ranking field: scoring on another one is a contradiction, and so
# is ranking on calibrated probabilities for a ranking-eligible L2 that has none.
release_fields = sorted(set(RELEASE.loc[L2S, "ranking_field"]))
check(release_fields == [RANKING_FIELD], f"training.yaml ranks on {RANKING_FIELD}, release {release_id} was "
                                         f"validated on {release_fields}: re-run notebook 02 or fix training.yaml")
for l2 in L2S:
    row = RELEASE.loc[l2]
    check(not (RANKING_FIELD == "calibrated" and bool(row["ranking_eligible_flag"])
               and not str(row["calibration_status"]).startswith("calibrated")),
          f"{l2}: ranking_field calibrated, ranking_eligible_flag 1 but calibration_status "
          f"{row['calibration_status']} has no calibrated probability")

layer2_models = common.to_pandas(common.read_table(spark, "ml.layer2_models", env_conf,
                                                 version_as_of=input_versions["ml.layer2_models"])
                                 .filter(F.col("release_id") == release_id))
layer2_versions = sorted(set(layer2_models["layer2_version"]))
check(len(layer2_versions) == 1, f"ml.layer2_models has layer2 versions {layer2_versions} for release {release_id} "
                                 f"(run notebook 03)")
LAYER2_VERSION = layer2_versions[0]
print(f"Release {release_id}: scoring {L2S} of {RELEASE_L2S}; layer2_version {LAYER2_VERSION}")

# COMMAND ----------

# %%
# Models, calibrators, feature lists and the Layer 2 specification + heads, from MLflow or from the file bundle.
artifacts_dir = Path(env_conf["paths"]["artifacts_dir"])
RELEASE_DIR = (artifacts_dir if artifacts_dir.is_absolute() else common.repo_root() / artifacts_dir) / release_id
work_dir = Path(tempfile.mkdtemp(prefix="prodrec-04-"))


def fetch(run_id: str, name: str, bundle_path: Path) -> tuple[Path, str]:
    """``(local file, source)`` of artifact ``name`` of MLflow run ``run_id`` (model_source mlflow) or of the file
    bundle path (files)."""
    if MODEL_SOURCE == "mlflow":
        target = work_dir / run_id
        target.mkdir(exist_ok=True)
        path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=name, dst_path=str(target))
        return Path(path), f"runs:/{run_id}/{name}"
    check(bundle_path.is_file(), f"{bundle_path} does not exist (notebooks 02 / 03 publish the bundle)")
    return bundle_path, str(bundle_path)


t0 = time.time()
if MODEL_SOURCE == "files":  # the bundle must be the release that the tables describe
    with open(RELEASE_DIR / "manifest.json", encoding="utf-8") as f:
        bundle_manifest = json.load(f)
    check(bundle_manifest.get("release_id") == release_id, f"{RELEASE_DIR} holds another release")
    check(bundle_manifest.get("training_config_hash") == CONFIG_HASH, "bundle has stale configuration provenance")
    check(bundle_manifest.get("layer2_version") == LAYER2_VERSION,
          f"{RELEASE_DIR}/manifest.json has layer2_version {bundle_manifest.get('layer2_version')}, ml.layer2_models "
          f"{LAYER2_VERSION}")
    for l2 in L2S:
        entry = bundle_manifest["l2s"].get(l2) or {}
        check(entry.get("model_run_id") == RELEASE.loc[l2, "model_run_id"]
              and entry.get("calibrator_run_id") == RELEASE.loc[l2, "calibrator_run_id"],
              f"{l2}: the bundle's runs differ from ml.model_selection")
        check(entry.get("provenance") == PROVENANCE[l2], f"{l2}: bundle provenance differs from selection")

models, calibrators, feature_specs, sources = {}, {}, {}, {}
for l2 in L2S:
    row = RELEASE.loc[l2]
    model_path, model_source = fetch(row["model_run_id"], "model.txt", RELEASE_DIR / l2 / "model.txt")
    calibrator_path, calibrator_source = fetch(row["calibrator_run_id"], "calibrator.pkl",
                                               RELEASE_DIR / l2 / "calibrator.pkl")
    spec_path, spec_source = fetch(row["model_run_id"], "feature_columns.json",
                                   RELEASE_DIR / l2 / "feature_columns.json")
    models[l2] = lgb.Booster(model_file=str(model_path))
    with open(calibrator_path, "rb") as f:
        calibrators[l2] = pickle.load(f)
    with open(spec_path, encoding="utf-8") as f:
        feature_specs[l2] = json.load(f)
    provenance = PROVENANCE[l2]
    for name in ("training_provenance.json", "calibration_provenance.json"):
        path, _ = fetch(row["model_run_id"] if name.startswith("training") else row["calibrator_run_id"],
                        name, RELEASE_DIR / l2 / name)
        original = json.loads(path.read_text(encoding="utf-8"))
        check(all(provenance.get(k) == v for k, v in original.items()), f"{l2}: {name} differs from release")
        l1_contract.validate_provenance(original, training_conf, row["model_run_id"])
        if name.startswith("calibration"):
            check(original == provenance, f"{l2}: incomplete calibration provenance")
    l1_contract.validate_feature_spec(feature_specs[l2], provenance, "champion_refit")
    l1_contract.validate_model_file(model_path, provenance)
    check(hashlib.sha256(calibrator_path.read_bytes()).hexdigest() == provenance["calibrator_sha256"],
          f"{l2}: calibrator artifact differs from its original run")
    check(models[l2].feature_name() == feature_specs[l2]["feature_columns"],
          f"{l2}: model features differ from feature_columns.json")
    check(calibrators[l2].method == CALIBRATOR_CLASS.get(row["calibration_method"]),
          f"{l2}: calibrator {calibrators[l2]!r} does not match calibration_method {row['calibration_method']}")
    sources[l2] = {"model": model_source, "calibrator": calibrator_source, "feature_columns": spec_source}

spec_path, layer2_source = fetch(LAYER2_VERSION, "layer2_spec.json", RELEASE_DIR / "layer2" / "layer2_spec.json")
with open(spec_path, encoding="utf-8") as f:
    LAYER2 = json.load(f)
check(LAYER2["release_id"] == release_id and LAYER2["layer2_version"] == LAYER2_VERSION,
      f"layer2_spec.json ({layer2_source}) belongs to {LAYER2['release_id']} / {LAYER2['layer2_version']}")
check(set(LAYER2["attributes"]) == set(ATTRIBUTES),
      f"layer2.yaml attributes {sorted(ATTRIBUTES)} differ from release {release_id}'s layer2_spec.json "
      f"{sorted(LAYER2['attributes'])}: re-run notebook 03")
for attribute, conf in ATTRIBUTES.items():
    served = LAYER2["attributes"][attribute]
    for key in LAYER2_SPEC_KEYS:
        if key == "constraints":  # compared with on_missing filled in (default allow) on both sides
            configured = modeling.normalize_constraints(conf.get(key))
            released = modeling.normalize_constraints(served.get(key))
        else:
            configured, released = conf.get(key), served.get(key)
        check(configured == released, f"layer2.yaml {attribute}.{key} = {configured!r} differs from release "
                                      f"{release_id}'s layer2_spec.json ({released!r}): re-run notebook 03")

# The heads encode the Layer 1 features of feature_columns_file (a release L2's feature_columns.json) + l2.
head_spec_l2 = LAYER2["feature_columns_file"].split("/")[0]
check(head_spec_l2 in RELEASE.index, f"layer2_spec.json feature_columns_file {LAYER2['feature_columns_file']} "
                                     f"is not an L2 of release {release_id}")
head_spec_path, head_spec_source = fetch(RELEASE.loc[head_spec_l2, "model_run_id"], "feature_columns.json",
                                         RELEASE_DIR / LAYER2["feature_columns_file"])
with open(head_spec_path, encoding="utf-8") as f:
    head_spec = json.load(f)
check(LAYER2.get("training_config_hash") == CONFIG_HASH, "Layer2 release has stale training configuration")
layer2_feature_spec = LAYER2.get("base_feature_spec") or {}
check(layer2_feature_spec.get("feature_columns") == head_spec["feature_columns"]
      and layer2_feature_spec.get("fitted_on") == "layer2_train_purchases"
      and l1_contract.digest(layer2_feature_spec) == LAYER2.get("base_feature_spec_hash"),
      "Layer2 fitted feature specification missing or inconsistent; rerun 03")
head_spec = layer2_feature_spec
L2_COLUMN = LAYER2["l2_column"]
HEAD_BASE_FEATURES = list(head_spec["feature_columns"])
HEAD_FEATURES = HEAD_BASE_FEATURES + [L2_COLUMN]
HEAD_LEVELS = {**head_spec["categorical_levels"], L2_COLUMN: list(LAYER2["l2_levels"])}
heads, head_sources = {}, {}
for attribute, served in LAYER2["attributes"].items():
    uses_head = any(entry["method_used"] == "head" for entry in served["per_l2"].values())
    check(uses_head == bool(served["model_run_id"]), f"{attribute}: head L2s and model_run_id disagree")
    if uses_head:
        path, head_sources[attribute] = fetch(served["model_run_id"], "model.txt",
                                              RELEASE_DIR / "layer2" / f"{attribute}_model.txt")
        heads[attribute] = lgb.Booster(model_file=str(path))
        check(heads[attribute].feature_name() == HEAD_FEATURES, f"{attribute}: head features differ from the spec")
LOAD_FEATURES = list(dict.fromkeys([f for l2 in L2S for f in feature_specs[l2]["feature_columns"]]
                                   + HEAD_BASE_FEATURES))
load_seconds = round(time.time() - t0, 1)
print(f"Loaded {len(L2S)} models / calibrators and layer2 {LAYER2_VERSION} ({len(heads)} heads) from {MODEL_SOURCE} "
      f"in {load_seconds}s")

# COMMAND ----------

# %%
# Scoring snapshot and its clients.
t0 = time.time()
modeling_sdf = common.read_table(spark, "ml.modeling_table", env_conf,
                                 version_as_of=input_versions["ml.modeling_table"])
SNAPSHOT = modeling_sdf.filter(F.col(SNAPSHOT_DATE) <= F.lit(RUN_DATE)).agg(F.max(SNAPSHOT_DATE)).collect()[0][0]
check(SNAPSHOT is not None, f"ml.modeling_table has no snapshot on or before run_date {RUN_DATE}")
SNAPSHOT_LAG = (RUN_DATE - SNAPSHOT).days
check(SNAPSHOT_LAG <= LOOKBACK_DAYS, f"the latest snapshot {SNAPSHOT} is {SNAPSHOT_LAG} days before run_date "
                                     f"{RUN_DATE}; scoring.lookback_days is {LOOKBACK_DAYS}")
missing = [c for c in LOAD_FEATURES + list(ELIG.values()) if c not in modeling_sdf.columns]
check(not missing, f"ml.modeling_table lacks columns {missing}")
snap = (common.to_pandas(modeling_sdf.filter(F.col(SNAPSHOT_DATE) == F.lit(SNAPSHOT)),
                         [CLIENT_ID] + LOAD_FEATURES + list(ELIG.values()))
        .sort_values(CLIENT_ID, kind="mergesort").reset_index(drop=True))
check(len(snap) > 0 and snap[CLIENT_ID].is_unique, f"snapshot {SNAPSHOT}: no clients or repeated {CLIENT_ID}")
snapshot_seconds = round(time.time() - t0, 1)
print(f"Scoring snapshot {SNAPSHOT} ({SNAPSHOT_LAG} days before run_date): {len(snap):,} clients in "
      f"{snapshot_seconds}s")

# COMMAND ----------

# %%
# Layer 1 scores: one row per client x L2.
t0 = time.time()
n_clients = len(snap)
CLIENTS = snap[CLIENT_ID].to_numpy(dtype=object)
parts = []
for l2 in L2S:
    row, product = RELEASE.loc[l2], CATALOG[l2]
    eligible = (snap[ELIG[l2]] == 1).to_numpy()
    raw = np.full(n_clients, np.nan)
    calibrated = np.full(n_clients, np.nan)
    if eligible.any():
        X = modeling.prepare_matrix(snap.loc[eligible], feature_specs[l2]["feature_columns"],
                                    feature_specs[l2]["categorical_levels"])
        raw[eligible] = models[l2].predict(X)
        if str(row["calibration_status"]).startswith("calibrated"):
            calibrated[eligible] = calibrators[l2].predict(raw[eligible])
    flag = np.where(eligible, int(bool(row["ranking_eligible_flag"])), 0)
    weight = float(product["business_weight"])
    parts.append(pd.DataFrame({
        "_pos": np.arange(n_clients),
        "snapshot_date": SNAPSHOT,
        "client_id": CLIENTS,
        "l2": l2,
        "l1_group": product["l1"],
        "display_name": product["display_name"],
        "elig": eligible.astype(np.int32),
        "raw_propensity_score": raw,
        "calibrated_purchase_probability": calibrated,
        "calibration_method": row["calibration_method"],
        "calibration_status": row["calibration_status"],
        "calibration_event_count": int(row["calibration_event_count"]),
        "evidence_source": row["evidence_source"], "evidence_event_count": int(row["evidence_event_count"]),
        "training_config_hash": CONFIG_HASH, "model_role": "champion_refit",
        "evidence_tier": row["evidence_tier"],
        "gate_passed": bool(row["gate_passed"]),
        "ranking_eligible_flag": flag.astype(np.int32),
        "exclusion_reason": np.where(~eligible, "not_eligible",
                                     np.where(flag == 1, None, "not_ranking_eligible")).astype(object),
        "business_weight": weight,
        "ranking_field": RANKING_FIELD,
        "ranking_score": (calibrated if RANKING_FIELD == "calibrated" else raw) * weight,
        "model_version": row["model_run_id"],
        "calibrator_version": row["calibrator_run_id"],
        "layer2_version": LAYER2_VERSION,
        "release_id": release_id,
    }))
scores = pd.concat(parts, ignore_index=True)
del parts
l1_seconds = round(time.time() - t0, 1)

# Rank per client among ranking_eligible_flag = 1 rows by descending ranking_score; ties by l2 name.
eligible_rows = scores["ranking_eligible_flag"] == 1
null_scores = int((eligible_rows & scores["ranking_score"].isna()).sum())
check(null_scores == 0, f"{null_scores} ranking-eligible rows have no {RANKING_FIELD} score (configuration "
                        f"contradiction)")
ordered = scores.loc[eligible_rows, ["client_id", "ranking_score", "l2"]].sort_values(
    ["client_id", "ranking_score", "l2"], ascending=[True, False, True], kind="mergesort")
scores["rank"] = pd.Series(pd.NA, index=scores.index, dtype="Int32")
scores.loc[ordered.index, "rank"] = (ordered.groupby("client_id", sort=False).cumcount() + 1).astype("int32")
TOP_ROWS = np.flatnonzero(scores["rank"].le(TOP_K).fillna(False).to_numpy(dtype=bool))
print(f"Layer 1: {len(scores):,} rows, {int(eligible_rows.sum()):,} ranked, {len(TOP_ROWS):,} in the top {TOP_K} "
      f"in {l1_seconds}s")

# COMMAND ----------

# %%
# Layer 2 guidance for the top-k rows (all L2s at once per attribute).
t0 = time.time()
top_l2s = scores["l2"].to_numpy(dtype=object)[TOP_ROWS]
top_frame = snap.iloc[scores["_pos"].to_numpy()[TOP_ROWS]].assign(**{L2_COLUMN: top_l2s})
X_head = modeling.prepare_matrix(top_frame, HEAD_FEATURES, HEAD_LEVELS)
X_layer1 = X_head[HEAD_BASE_FEATURES]
status_counts = {}
for attribute in ATTRIBUTES:
    served = LAYER2["attributes"][attribute]
    probs, methods = modeling.layer2_probabilities(served, top_l2s, X_layer1, X_head, heads.get(attribute))
    index, prob, _, status = modeling.layer2_guidance(probs, methods, served.get("margin"))
    values = np.asarray(served["values"], dtype=object)
    guidance = np.where((index >= 0) & (status != "abstain"), values[np.maximum(index, 0)], None)
    scores[f"{attribute}_guidance"] = pd.Series(None, index=scores.index, dtype=object)
    scores[f"{attribute}_prob"] = np.nan
    scores[f"{attribute}_status"] = pd.Series(None, index=scores.index, dtype=object)
    scores.loc[TOP_ROWS, f"{attribute}_guidance"] = guidance
    scores.loc[TOP_ROWS, f"{attribute}_prob"] = prob
    scores.loc[TOP_ROWS, f"{attribute}_status"] = status
    status_counts[attribute] = pd.Series(status).value_counts().to_dict()
del X_head, X_layer1, top_frame
layer2_seconds = round(time.time() - t0, 1)
print(f"Layer 2 guidance for {len(TOP_ROWS):,} rows in {layer2_seconds}s: {status_counts}")

# COMMAND ----------

# %%
# Reasons for the top-k rows with each L2's model; every reason group's contribution for
# scores.customer_reason_contributions.
t0 = time.time()
group_shown = {g: bool(conf.get("explainable", False)) and not conf.get("sensitive", True)
               for g, conf in GROUPS.items()}
for i in range(1, TOP_N + 1):
    scores[f"reason_{i}"] = pd.Series(None, index=scores.index, dtype=object)
    scores[f"reason_{i}_contribution"] = np.nan
contribution_parts, reasons_missing, group_features = [], {}, {}
for l2 in L2S:
    rows = TOP_ROWS[top_l2s == l2]
    if len(rows) == 0:
        continue
    features = list(feature_specs[l2]["feature_columns"])
    X = modeling.prepare_matrix(snap.iloc[scores["_pos"].to_numpy()[rows]], features,
                                feature_specs[l2]["categorical_levels"])
    contributions = modeling.feature_contributions(models[l2], X)
    reasons = modeling.explain_rows(models[l2], X, FEATURE_TO_GROUP, GROUPS, TOP_N, contributions=contributions)
    for i in range(TOP_N):
        scores.loc[rows, f"reason_{i + 1}"] = np.array([r[i]["group"] if len(r) > i else None for r in reasons],
                                                       dtype=object)
        scores.loc[rows, f"reason_{i + 1}_contribution"] = [r[i]["contribution"] if len(r) > i else np.nan
                                                            for r in reasons]
    group_ids, totals, top_feature, top_contribution = modeling.group_contributions(
        contributions, features, FEATURE_TO_GROUP, list(GROUPS))
    # A row may lack reasons only when no explainable, non-sensitive group contributes positively.
    has_positive = (totals[:, [group_shown[g] for g in group_ids]] > 0).any(axis=1)
    has_reason = np.array([len(r) > 0 for r in reasons])
    check((has_positive == has_reason).all(), f"{l2}: reasons disagree with the group contributions")
    reasons_missing[l2] = int((~has_reason).sum())
    feature_names = np.asarray(features, dtype=object)
    row_clients = scores["client_id"].to_numpy(dtype=object)[rows]
    for j, group in enumerate(group_ids):
        members = np.array([k for k, f in enumerate(features) if FEATURE_TO_GROUP[f] == group])
        name = str(GROUPS[group].get("name", group))
        group_features.setdefault(name, [features[k] for k in members])
        check(group_features[name] == [features[k] for k in members],
              f"reason group {name}: L2s have different features (feature_contributions order)")
        part = {
            "client_id": row_clients,
            "l2": l2,
            "reason_group": name,
            "sensitive": bool(GROUPS[group].get("sensitive", True)),
            "explainable": bool(GROUPS[group].get("explainable", False)),
            "contribution": totals[:, j],
            "top_feature": feature_names[top_feature[:, j]],
            "top_feature_value": modeling.feature_values_as_text(X, np.arange(len(rows)), top_feature[:, j]),
            "top_feature_contribution": top_contribution[:, j],
        }
        if STORE_FEATURE_LEVEL:  # the group's features in feature_columns order (manifest reason_group_features)
            part["feature_contributions"] = contributions[:, members].tolist()
        contribution_parts.append(pd.DataFrame(part))
reason_contributions = pd.concat(contribution_parts, ignore_index=True) if contribution_parts else pd.DataFrame()
del contribution_parts
reasons_seconds = round(time.time() - t0, 1)
print(f"Reasons for {len(TOP_ROWS):,} rows ({len(reason_contributions):,} group contributions) in "
      f"{reasons_seconds}s; rows without a positive explainable group: {reasons_missing}")

# COMMAND ----------

# %%
# Checks before anything is written.
ranked = scores.loc[scores["rank"].notna(), ["client_id", "rank"]]
per_client = ranked.groupby("client_id")["rank"].agg(["min", "max", "count", "nunique"])
check(((per_client["min"] == 1) & (per_client["max"] == per_client["count"])
       & (per_client["nunique"] == per_client["count"])).all(), "rank is not unique / consecutive from 1 per client")
check(scores.loc[(scores["ranking_eligible_flag"] == 0) | (scores["elig"] == 0), "rank"].isna().all(),
      "rows with ranking_eligible_flag = 0 or elig = 0 have a rank")
check(scores.loc[scores["elig"] == 0, ["raw_propensity_score", "calibrated_purchase_probability"]].isna().all().all(),
      "rows with elig = 0 have scores")
check(scores["ranking_field"].nunique() == 1, "ranking_field is not unique")
expected_top = int(per_client["count"].clip(upper=TOP_K).sum())
check(len(TOP_ROWS) == expected_top, f"{len(TOP_ROWS)} top-k rows, expected {expected_top}")
top = scores.iloc[TOP_ROWS]
n_without_reason = int(top["reason_1"].isna().sum())
check(n_without_reason == sum(reasons_missing.values()), "top-k rows without reason_1 are not all explained")
check(top[[f"{a}_status" for a in ATTRIBUTES]].notna().all().all(), "top-k rows without a Layer 2 status")
rank_check_passed = True
print(f"Checks passed: ranks unique and consecutive for {len(per_client):,} clients; {len(top):,} top-k rows, "
      f"{n_without_reason} without a reason")

# COMMAND ----------

# %%
ATTRIBUTE_COLUMNS = [(f"{a}_{suffix}", data_type) for a in ATTRIBUTES
                     for suffix, data_type in (("guidance", "string"), ("prob", "double"), ("status", "string"))]
REASON_COLUMNS = [(f"reason_{i}{suffix}", data_type) for i in range(1, TOP_N + 1)
                  for suffix, data_type in (("", "string"), ("_contribution", "double"))]
# docs/MVP_SPEC.md section 10 order, except run_date (the partition column) last.
SCORES_COLUMNS = [
    ("snapshot_date", "date"), ("client_id", "string"), ("l2", "string"), ("l1_group", "string"),
    ("display_name", "string"), ("elig", "int"), ("raw_propensity_score", "double"),
    ("calibrated_purchase_probability", "double"), ("calibration_method", "string"),
    ("calibration_status", "string"), ("calibration_event_count", "bigint"), ("evidence_tier", "string"),
    ("gate_passed", "boolean"), ("ranking_eligible_flag", "int"), ("exclusion_reason", "string"),
    ("business_weight", "double"), ("ranking_field", "string"), ("ranking_score", "double"), ("rank", "int"),
    *ATTRIBUTE_COLUMNS, *REASON_COLUMNS,
    ("model_version", "string"), ("calibrator_version", "string"), ("layer2_version", "string"),
    ("evidence_source", "string"), ("evidence_event_count", "bigint"),
    ("training_config_hash", "string"), ("model_role", "string"),
    ("release_id", "string"), ("run_date", "date"),
]
CONTRIBUTION_COLUMNS = [
    ("client_id", "string"), ("l2", "string"), ("reason_group", "string"), ("sensitive", "boolean"),
    ("explainable", "boolean"), ("contribution", "double"), ("top_feature", "string"),
    ("top_feature_value", "string"), ("top_feature_contribution", "double"),
    *([("feature_contributions", "array<double>")] if STORE_FEATURE_LEVEL else []), ("run_date", "date"),
]
RUNS_COLUMNS = [
    ("release_id", "string"), ("snapshot_date", "date"), ("env", "string"), ("model_source", "string"),
    ("git_commit", "string"), ("n_customers", "bigint"), ("n_rows", "bigint"), ("n_ranked_rows", "bigint"),
    ("n_top_rows", "bigint"), ("ranking_field", "string"), ("rank_check_passed", "boolean"),
    ("created_at", "timestamp"), ("manifest_json", "string"), ("run_date", "date"),
]


def spark_frame(pdf: pd.DataFrame, columns: list[tuple[str, str]]):
    """Spark DataFrame of ``pdf`` with exactly ``columns`` (name, Spark type), matched by name, in about
    ``len(pdf) / ROWS_PER_FILE`` partitions (one output file each)."""
    sdf = spark.createDataFrame(pdf[[name for name, _ in columns]],
                                ", ".join(f"`{name}` {data_type}" for name, data_type in columns))
    return sdf.coalesce(max(1, math.ceil(len(pdf) / ROWS_PER_FILE)))


scores["run_date"] = RUN_DATE
reason_contributions = reason_contributions.reindex(columns=[name for name, _ in CONTRIBUTION_COLUMNS])
reason_contributions["run_date"] = RUN_DATE
top = scores.iloc[TOP_ROWS]
selected = RELEASE.loc[L2S]
row_counts = {
    "n_customers": n_clients,
    "customer_l2_scores": len(scores),
    "n_ranked_rows": len(ranked),
    "customer_top_recommendations": len(top),
    "customer_reason_contributions": len(reason_contributions),
}
created_at = datetime.now(timezone.utc)
manifest = {
    "run_date": str(RUN_DATE),
    "release_id": release_id,
    "snapshot_date": str(SNAPSHOT),
    "env": env,
    "model_source": MODEL_SOURCE,
    "git_commit": git_commit(),
    "created_at": created_at.isoformat(timespec="seconds"),
    "input_tables": {logical: {"table": common.table_name(logical, env_conf), "delta_version": input_versions[logical]}
                     for logical in INPUT_TABLES},
    "ranking_field": RANKING_FIELD,
    "training_config_hash": CONFIG_HASH,
    "calibration_enabled": CALIBRATION_ENABLED,
    "effective_training_config": json.loads(l1_contract.canonical_json(training_conf)),
    "top_k": TOP_K,
    "top_n_reasons": TOP_N,
    "store_feature_level": STORE_FEATURE_LEVEL,
    "lookback_days": LOOKBACK_DAYS,
    "l2s": {l2: {
        "model_run_id": RELEASE.loc[l2, "model_run_id"],
        "calibrator_run_id": RELEASE.loc[l2, "calibrator_run_id"],
        "calibration_method": RELEASE.loc[l2, "calibration_method"],
        "status": RELEASE.loc[l2, "calibration_status"],
        "tier": RELEASE.loc[l2, "evidence_tier"],
        "gate_passed": bool(RELEASE.loc[l2, "gate_passed"]),
        "ranking_eligible_flag": bool(RELEASE.loc[l2, "ranking_eligible_flag"]),
        "ranking_field": RANKING_FIELD,
        "business_weight": float(CATALOG[l2]["business_weight"]),
        "train_end": str(RELEASE.loc[l2, "train_end"]),
        "calibration_end": str(RELEASE.loc[l2, "calibration_end"]),
        "evaluated_on": str(RELEASE.loc[l2, "evaluated_on"]),
        "loaded_from": sources[l2],
        "provenance": PROVENANCE[l2],
    } for l2 in L2S},
    "layer2": {"layer2_version": LAYER2_VERSION, "spec": layer2_source, "heads": head_sources,
               "feature_columns": head_spec_source},
    "train_end": sorted({str(d) for d in selected["train_end"]}),
    "calibration_end": sorted({str(d) for d in selected["calibration_end"]}),
    "evaluated_on": sorted({str(d) for d in selected["evaluated_on"]}),
    "staleness_days": {"snapshot_after_train_end": (SNAPSHOT - max(selected["train_end"])).days,
                       "snapshot_after_evaluated_on": (SNAPSHOT - max(selected["evaluated_on"])).days},
    "row_counts": row_counts,
    "rank_check_passed": rank_check_passed,
    "top_k_rows_without_reasons": reasons_missing,
    "reason_group_features": group_features,
}

# Commit protocol: a run_date counts as scored only when scores.scoring_runs has its row, and downstream readers
# must check that row. Its old row goes first, then the three data tables are replaced and checked, and the
# scoring_runs row is written last: a failure in between leaves the run_date without a scoring_runs row.
# mergeSchema (on these writers only): a new layer2.yaml attribute, reason or feature-level column adds columns.
predicate = f"run_date = '{RUN_DATE}'"
DATA_TABLES = [("scores.customer_l2_scores", scores, SCORES_COLUMNS),
               ("scores.customer_top_recommendations", top, SCORES_COLUMNS),
               ("scores.customer_reason_contributions", reason_contributions, CONTRIBUTION_COLUMNS)]
t0 = time.time()
with mlflow.start_run(run_name=str(RUN_DATE)) as scoring_run:
    manifest["scoring_run_id"] = scoring_run.info.run_id
    mlflow.set_tags({"release_id": release_id, "run_date": str(RUN_DATE), "kind": "scoring",
                     "model_source": MODEL_SOURCE})
    mlflow.log_params({"release_id": release_id, "run_date": str(RUN_DATE), "snapshot_date": str(SNAPSHOT),
                       "env": env, "model_source": MODEL_SOURCE, "l2s": ",".join(L2S),
                       "ranking_field": RANKING_FIELD, "top_k": TOP_K, "layer2_version": LAYER2_VERSION})
    mlflow.log_metrics({name: float(value) for name, value in row_counts.items()})
    manifest_path = work_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    mlflow.log_artifact(str(manifest_path))

    uncommitted = common.delete_rows(spark, "scores.scoring_runs", env_conf, predicate)
    print(f"run_date {RUN_DATE} uncommitted (scores.scoring_runs row deleted: {uncommitted})")
    written = {}
    for logical, pdf, columns in DATA_TABLES:
        written[logical] = common.write_table(spark_frame(pdf, columns), logical, env_conf, mode="overwrite",
                                              replace_where=predicate, partition_by=["run_date"], merge_schema=True)
    for logical, pdf, _ in DATA_TABLES:
        found = common.read_table(spark, logical, env_conf).filter(F.col("run_date") == F.lit(RUN_DATE)).count()
        check(found == len(pdf), f"{written[logical]} has {found} rows for run_date {RUN_DATE}, expected {len(pdf)}")

    run_row = pd.DataFrame([{
        "release_id": release_id, "snapshot_date": SNAPSHOT, "env": env, "model_source": MODEL_SOURCE,
        "git_commit": manifest["git_commit"], "n_customers": n_clients, "n_rows": len(scores),
        "n_ranked_rows": len(ranked), "n_top_rows": len(top), "ranking_field": RANKING_FIELD,
        "rank_check_passed": rank_check_passed, "created_at": pd.Timestamp(created_at),
        "manifest_json": json.dumps(manifest, default=str), "run_date": RUN_DATE,
    }])
    written["scores.scoring_runs"] = common.write_table(spark_frame(run_row, RUNS_COLUMNS), "scores.scoring_runs",
                                                        env_conf, mode="overwrite", replace_where=predicate,
                                                        partition_by=["run_date"], merge_schema=True)
    committed = common.read_table(spark, "scores.scoring_runs", env_conf).filter(F.col("run_date") == F.lit(RUN_DATE))
    check(committed.count() == 1, f"{written['scores.scoring_runs']} does not have exactly one row for {RUN_DATE}")
write_seconds = round(time.time() - t0, 1)
expected_rows = {logical: len(pdf) for logical, pdf, _ in DATA_TABLES} | {"scores.scoring_runs": 1}
print(f"Wrote {', '.join(f'{name} ({expected_rows[logical]:,})' for logical, name in written.items())} in "
      f"{write_seconds}s; run_date {RUN_DATE} committed; MLflow scoring run {manifest['scoring_run_id']}")
shutil.rmtree(work_dir, ignore_errors=True)

# COMMAND ----------

# %%
def mean_or_none(values) -> float | None:
    values = pd.Series(values).dropna()
    return round(float(values.mean()), 6) if len(values) else None


per_l2 = {}
for l2 in L2S:
    rows = scores[scores["l2"] == l2]
    per_l2[l2] = {
        "eligible_rows": int(rows["elig"].sum()),
        "mean_raw": mean_or_none(rows["raw_propensity_score"]),
        "mean_calibrated": mean_or_none(rows["calibrated_purchase_probability"]),
        "ranking_eligible_flag": bool(RELEASE.loc[l2, "ranking_eligible_flag"]),
        "ranked_rows": int(rows["rank"].notna().sum()),
        "top_k_rows": int((rows["rank"] <= TOP_K).fillna(False).sum()),
        "status": RELEASE.loc[l2, "calibration_status"],
        "tier": RELEASE.loc[l2, "evidence_tier"],
    }
summary = {
    "env": env,
    "run_date": str(RUN_DATE),
    "snapshot_date": str(SNAPSHOT),
    "release_id": release_id,
    "layer2_version": LAYER2_VERSION,
    "model_source": MODEL_SOURCE,
    "experiment_id": experiment_id,
    "scoring_run_id": manifest["scoring_run_id"],
    "output_tables": list(written.values()),
    "customers": n_clients,
    "row_counts": row_counts,
    "l2s": per_l2,
    "rank1_distribution": scores.loc[scores["rank"].eq(1).fillna(False).to_numpy(dtype=bool), "l2"]
    .value_counts().to_dict(),
    "layer2_status": status_counts,
    "reasons": {"top_k_rows": len(top), "with_reason_1": int(top["reason_1"].notna().sum()),
                "reason_1_distribution": top["reason_1"].value_counts().head(10).to_dict()},
    "seconds": {"load_models": load_seconds, "snapshot": snapshot_seconds, "layer1": l1_seconds,
                "layer2": layer2_seconds, "reasons": reasons_seconds, "write": write_seconds},
    "elapsed_seconds": round(time.time() - started, 1),
}
print(json.dumps(summary, indent=2, default=str))
