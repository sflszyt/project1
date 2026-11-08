# Databricks notebook source
# %%
# Notebook 02: calibrate the Layer 1 raw scores and validate them out of time (docs/MVP_SPEC.md section 8).
#
# Per L2, from ml.l1_raw_scores (calibration and test blocks; eligible rows only) and ml.l1_training_runs:
#   evidence  distinct purchase events in the calibration block -> evidence_tier (own / limited / insufficient)
#   calibrate on the calibration block only, by tier: own -> Platt (intercept-only fallback when b <= 0),
#             limited -> intercept-only, insufficient -> none; calibration.enabled = false -> disabled
#   validate  Test block, report only: raw and calibrated metrics, reliability tables, O/E and slope per Test
#             snapshot, client-clustered bootstrap CIs; release gate on the ranking field
#   ranking_eligible_flag = tier in (own, limited) and gate_passed and status != not_calibrated
# Outputs: ml.l1_evaluation, ml.model_selection (replaceWhere release_id), MLflow experiment l1_calibration (one run
# per L2, calibrator_version = run_id, plus a "release" run with the manifest), file bundle
# <artifacts_dir>/<release_id>/<l2>/{model.txt, calibrator.pkl, feature_columns.json} + manifest.json (written to a
# staging directory and swapped in only when complete by common.publish_directory; on failure the previous bundle
# stays as it was).
#
# Parameters: env (default local), l2_list (pilot | all | comma-separated; default pilot), release_id (default
# mvp-<YYYYMMDD-HHMMSS>), allow_overwrite (default false: a release_id that ml.model_selection already holds is
# refused; true replaces that release).
# Run locally: python notebooks/02_calibrate_and_validate.py --env local --l2_list product1,product14 --release_id smoke
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
l2_list = common.get_param("l2_list", "pilot")
release_id = common.get_param("release_id", f"mvp-{datetime.now():%Y%m%d-%H%M%S}")
allow_overwrite_text = str(common.get_param("allow_overwrite", "false")).strip().lower()
started = time.time()
print(f"env={env} l2_list={l2_list} release_id={release_id} allow_overwrite={allow_overwrite_text}")

# COMMAND ----------

# %%
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

env_conf = common.load_env(env)
products = common.load_conf("products")
training_conf = common.load_conf("training")
CONFIG_HASH = l1_contract.digest(training_conf)
L2S = modeling.resolve_l2_list(products, l2_list)
CLIENT_ID = products["keys"]["client_id"]
SNAPSHOT_DATE = products["keys"]["snapshot_date"]
HORIZON = int(training_conf["timeline"]["horizon_months"])
TIERS = training_conf["evidence_tiers"]
CALIBRATION = training_conf["calibration"]
CALIBRATION_ENABLED = bool(CALIBRATION.get("enabled", True))
GATE = training_conf["gate"]
TOP_FRACTIONS = [float(f) for f in training_conf["evaluation"]["top_fractions"]]
BOOTSTRAP = training_conf["evaluation"]["bootstrap"]
N_BINS = 10
LIFT_GATE_METRIC = f"lift_{modeling.fraction_label(0.10)}"  # gate.lift10_min applies to lift at the top 10%
# Per Test snapshot only O/E and slope are reported (with the counts and rates they come from).
SNAPSHOT_METRICS = ["n_rows", "n_pos", "obs_rate", "mean_pred", "oe", "slope", "intercept"]
CI_METRICS = {"auroc": modeling.weighted_auroc, "oe": modeling.weighted_oe}
KIND_PREFIX = {"raw": "raw", "calibrated": "cal"}


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


check(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", release_id) is not None,
      f"release_id {release_id!r} must be letters, digits, '_', '.', '-'")
check(allow_overwrite_text in ("true", "false"), f"allow_overwrite must be true or false, got {allow_overwrite_text!r}")
ALLOW_OVERWRITE = allow_overwrite_text == "true"
check(not CALIBRATION.get("isotonic"), "calibration.isotonic is not implemented in the MVP")
check(not CALIBRATION.get("final_refit_calibrator"), "calibration.final_refit_calibrator is not implemented")
check(0.10 in TOP_FRACTIONS, "evaluation.top_fractions must contain 0.10 (gate.lift10_min)")
RANKING_FIELD = str(training_conf["ranking"]["ranking_field"]) if CALIBRATION_ENABLED else "raw"
check(RANKING_FIELD in ("calibrated", "raw"), f"ranking.ranking_field must be calibrated or raw, got {RANKING_FIELD}")
artifacts_dir = Path(env_conf["paths"]["artifacts_dir"])
RELEASE_DIR = (artifacts_dir if artifacts_dir.is_absolute() else common.repo_root() / artifacts_dir) / release_id
print(f"L2s: {L2S}; ranking_field {RANKING_FIELD} (calibration enabled: {CALIBRATION_ENABLED}); bundle {RELEASE_DIR}")


def finite_or_none(value):
    """JSON / table friendly number: None for None, NaN and infinities."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def pair(values, digits: int) -> str:
    return " -> ".join("N/A" if finite_or_none(v) is None else f"{v:.{digits}f}" for v in values)


def git_output(*args: str) -> str | None:
    """stdout of ``git <args>`` in the repository root; None when git is unavailable or fails."""
    try:
        result = subprocess.run(["git", *args], cwd=common.repo_root(), capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def git_commit() -> str | None:
    output = git_output("rev-parse", "HEAD")
    return (output or "").strip() or None


def git_dirty() -> bool | None:
    """True when notebooks/ or conf/ differ from HEAD (incl. untracked files); None without git."""
    output = git_output("status", "--porcelain", "--", "notebooks", "conf")
    return None if output is None else bool(output.strip())


def code_sha256() -> str:
    """First 16 hex digits of the SHA-256 of the notebooks/*.py contents, concatenated in file name order."""
    digest = hashlib.sha256()
    for path in sorted(_notebook_dir().glob("*.py"), key=lambda p: p.name):
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]

# COMMAND ----------

# %%
import mlflow  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

spark = common.get_spark(env_conf)
experiment_id = common.mlflow_setup(env_conf, "l1_calibration")

# A published release is immutable unless explicitly overwritten.
if common.table_exists(spark, "ml.model_selection", env_conf):
    existing = (common.read_table(spark, "ml.model_selection", env_conf)
                .filter(F.col("release_id") == release_id).limit(1).count())
    check(existing == 0 or ALLOW_OVERWRITE, f"release {release_id} already exists in ml.model_selection; choose a new "
                                            f"release_id or pass allow_overwrite=true to replace it")
    if existing:
        print(f"WARNING release {release_id} exists and is overwritten (allow_overwrite=true)")

INPUT_TABLES = ["ml.l1_training_runs", "ml.l1_raw_scores", "ml.modeling_table", "ml.data_dictionary",
                "raw.l2_purchase_config"]
INPUT_VERSIONS = {name: common.delta_version(spark, name, env_conf) for name in INPUT_TABLES}
training_runs = common.to_pandas(common.read_table(spark, "ml.l1_training_runs", env_conf,
                                                 version_as_of=INPUT_VERSIONS["ml.l1_training_runs"])
                                 .filter(F.col("l2").isin(L2S)))
counts = training_runs["l2"].value_counts()
check(set(counts.index) == set(L2S) and (counts == 1).all(),
      f"ml.l1_training_runs needs exactly one row per L2; found {counts.to_dict()} for {L2S} (run notebook 01)")
training_runs = training_runs.set_index("l2")
check({"training_config_hash", "provenance_json"} <= set(training_runs.columns),
      "Legacy training output has no configuration provenance; rerun notebook 01")
TRAINING_PROVENANCE = {}
for l2, row in training_runs.iterrows():
    check(row["training_config_hash"] == CONFIG_HASH, f"{l2}: stale training configuration; rerun 01")
    provenance = json.loads(row["provenance_json"])
    l1_contract.validate_provenance(provenance, training_conf, row["model_run_id"])
    for name in ("ml.modeling_table", "ml.data_dictionary", "raw.l2_purchase_config"):
        check(provenance["input_tables"][name]["delta_version"] == INPUT_VERSIONS[name],
              f"{l2}: stale upstream {name}; changed since training (rerun 01)")
        check(provenance["input_tables"][name]["table"] == common.table_name(name, env_conf),
              f"{l2}: upstream physical table differs from training provenance")
    TRAINING_PROVENANCE[l2] = provenance

t0 = time.time()
scores = common.to_pandas(common.read_table(spark, "ml.l1_raw_scores", env_conf,
                                          version_as_of=INPUT_VERSIONS["ml.l1_raw_scores"])
                          .filter(F.col("l2").isin(L2S) & F.col("split").isin("calibration", "test")),
                          ["l2", "split", "client_id", "snapshot_date", "y", "raw_score", "raw_margin",
                           "model_run_id", "model_role", "model_artifact", "feature_spec_hash",
                           "training_config_hash", "best_iteration"])
print(f"Loaded ml.l1_raw_scores: {len(scores):,} calibration / test rows in {time.time() - t0:.1f}s")
training_artifacts_dir = Path(tempfile.mkdtemp(prefix="prodrec-02-training-"))
for l2 in L2S:
    provenance = TRAINING_PROVENANCE[l2]
    l1_contract.validate_raw_scores(scores[scores["l2"] == l2], provenance, CONFIG_HASH)
    target = training_artifacts_dir / l2
    target.mkdir()
    for name in ("model.txt", "feature_columns.json", "training_provenance.json"):
        mlflow.artifacts.download_artifacts(run_id=provenance["model_run_id"], artifact_path=name, dst_path=str(target))
    saved = json.loads((target / "training_provenance.json").read_text(encoding="utf-8"))
    check(saved == provenance, f"{l2}: training table differs from original run provenance")
    l1_contract.validate_feature_spec(json.loads((target / "feature_columns.json").read_text(encoding="utf-8")),
                                      provenance, "champion_refit")
    l1_contract.validate_model_file(target / "model.txt", provenance)

# Block boundaries: the split column written by notebook 00 (the same blocks notebook 01 trained on).
split_rows = (common.read_table(spark, "ml.modeling_table", env_conf,
                                version_as_of=INPUT_VERSIONS["ml.modeling_table"])
              .select(SNAPSHOT_DATE, "split").distinct().collect())
BLOCK_DATES = common.derive_split_dates(training_conf, [r[SNAPSHOT_DATE] for r in split_rows])
expected = {d: b for b in ("train", "calibration", "test", "unused", "immature") for d in BLOCK_DATES[b]}
expected.update({d: "gap" for d in BLOCK_DATES["gap1"] + BLOCK_DATES["gap2"]})
check(all(expected[r[SNAPSHOT_DATE]] == r["split"] for r in split_rows),
      "stale date-to-split assignments in modeling_table; regenerate upstream explicitly")
TIMELINE = l1_contract.timeline_manifest(training_conf, BLOCK_DATES)
check(all(p["timeline"] == TIMELINE for p in TRAINING_PROVENANCE.values()), "training timeline differs from input")
CAL_FIRST = BLOCK_DATES["calibration"][0] if BLOCK_DATES["calibration"] else None
CAL_LAST = BLOCK_DATES["calibration"][-1] if BLOCK_DATES["calibration"] else None
EVIDENCE_FIRST, EVIDENCE_LAST = BLOCK_DATES["evidence"][0], BLOCK_DATES["evidence"][-1]
TEST_FIRST, TEST_LAST = BLOCK_DATES["test"][0], BLOCK_DATES["test"][-1]
train_ends = sorted(set(training_runs["train_last"]))
check(train_ends == [BLOCK_DATES["refit"][-1]],
      f"train_last in ml.l1_training_runs {train_ends} differs from refit end {BLOCK_DATES['refit'][-1]}")
TRAIN_END = train_ends[0]
print(f"train end {TRAIN_END}; calibration {CAL_FIRST}..{CAL_LAST}; test {TEST_FIRST}..{TEST_LAST}")

# Event counts use the same client sample as the scores (sample_fraction of the notebook 01 run).
purchases_sdf = common.read_table(spark, "raw.l2_purchase_config", env_conf,
                                  version_as_of=INPUT_VERSIONS["raw.l2_purchase_config"]).filter(F.col("l2").isin(L2S))
PURCHASE_COLUMNS = ["purchase_id", "client_id", "l2", "application_date", "elig_at_feature_snapshot"]
purchases_by_fraction = {}
for fraction in sorted({float(f) for f in training_runs["sample_fraction"]}):
    sampled = common.sample_clients(purchases_sdf, fraction, key=CLIENT_ID) if fraction < 1 else purchases_sdf
    purchases_by_fraction[fraction] = common.to_pandas(sampled, PURCHASE_COLUMNS)
    print(f"Loaded raw.l2_purchase_config (sample_fraction {fraction}): {len(purchases_by_fraction[fraction]):,} "
          f"purchases of the selected L2s")

# COMMAND ----------

# %%
def fit_calibrator(method: str, s: np.ndarray, y: np.ndarray):
    """``(calibrator, method, status)`` for the tier's method, fitted on the calibration block."""
    if method == "platt":
        calibrator = modeling.PlattCalibrator(C=float(CALIBRATION.get("platt_C", 1e6))).fit(s, y)
        if calibrator.b > 0:
            return calibrator, "platt", "calibrated_platt"
        print(f"WARNING Platt slope b = {calibrator.b:.4f} <= 0 would reverse the ranking: intercept-only fallback")
        return modeling.InterceptCalibrator().fit(s, y), "intercept", "calibrated_intercept_fallback"
    if method == "intercept":
        return modeling.InterceptCalibrator().fit(s, y), "intercept", "calibrated_intercept"
    if method == "none":
        return modeling.IdentityCalibrator(), "none", "not_calibrated"
    if method == "disabled":
        return modeling.IdentityCalibrator(), "disabled", "disabled"
    raise ValueError(f"unknown calibration method {method!r}")


def gate_checks(metrics: dict) -> dict:
    """Release gate on one set of Test metrics: {check: {value, min[, max], passed}}."""
    checks = {
        "auroc": {"value": metrics["auroc"], "min": float(GATE["auroc_min"])},
        LIFT_GATE_METRIC: {"value": metrics[LIFT_GATE_METRIC], "min": float(GATE["lift10_min"])},
        "oe": {"value": metrics["oe"], "min": float(GATE["oe_range"][0]), "max": float(GATE["oe_range"][1])},
        "slope": {"value": metrics["slope"], "min": float(GATE["slope_range"][0]),
                  "max": float(GATE["slope_range"][1])},
    }
    for item in checks.values():
        value = finite_or_none(item["value"])
        item["value"] = value
        item["passed"] = value is not None and value >= item["min"] and ("max" not in item or value <= item["max"])
    return checks


def evaluate_l2(l2: str) -> tuple[dict, list[dict], dict, dict, bytes]:
    """Calibrate, validate and log one L2; returns (model_selection row, l1_evaluation rows, manifest entry,
    summary, calibrator pickle). The file bundle is written for all L2s at the end."""
    t0 = time.time()
    run = training_runs.loc[l2]
    model_run_id = run["model_run_id"]
    l2_scores = scores[scores["l2"] == l2]
    found_runs = sorted(set(l2_scores["model_run_id"]))
    check(found_runs == [model_run_id],
          f"{l2}: ml.l1_raw_scores has model runs {found_runs}, ml.l1_training_runs {model_run_id}")
    provenance = TRAINING_PROVENANCE[l2]
    blocks = {}
    for split in ("calibration", "test"):
        frame = l2_scores[l2_scores["split"] == split]
        check(len(frame) > 0 or not BLOCK_DATES[split], f"{l2}: no {split} rows in ml.l1_raw_scores")
        outside = sorted(set(frame["snapshot_date"]) - set(BLOCK_DATES[split]))
        check(not outside, f"{l2}: {split} raw scores on snapshots outside the {split} block: {outside}")
        blocks[split] = frame
    s_cal, y_cal = blocks["calibration"]["raw_score"].to_numpy(), blocks["calibration"]["y"].to_numpy()
    s_test, y_test = blocks["test"]["raw_score"].to_numpy(), blocks["test"]["y"].to_numpy()

    purchases = purchases_by_fraction[float(run["sample_fraction"])]
    calibration_event_count = (modeling.count_events(purchases, l2, CAL_FIRST, CAL_LAST, HORIZON)
                               if CAL_FIRST else 0)
    evidence_event_count = modeling.count_events(purchases, l2, EVIDENCE_FIRST, EVIDENCE_LAST, HORIZON)
    test_event_count = modeling.count_events(purchases, l2, TEST_FIRST, TEST_LAST, HORIZON)
    tier, method = modeling.select_calibrator(evidence_event_count, TIERS, CALIBRATION)
    calibrator, method, status = fit_calibrator(method, s_cal, y_cal)
    calibrated = status.startswith("calibrated")
    kinds = ("raw", "calibrated") if calibrated else ("raw",)
    predictions = {(block, kind): (values if kind == "raw" else calibrator.predict(values))
                   for block, values in (("calibration", s_cal), ("test", s_test)) if len(values)
                   for kind in kinds}
    spearman = None
    if calibrated:
        spearman = float(spearmanr(s_test, predictions[("test", "calibrated")]).statistic)
        check(spearman >= 1 - 1e-9, f"{l2}: Spearman(raw, calibrated) = {spearman} on Test; calibration reorders")

    labels = {"calibration": y_cal, "test": y_test}
    metrics = {key: modeling.binary_metrics(labels[key[0]], p, TOP_FRACTIONS) for key, p in predictions.items()}
    reliability = {kind: modeling.reliability_table(y_test, predictions[("test", kind)], N_BINS)
                   for kind in kinds}
    test_snapshots = blocks["test"]["snapshot_date"].to_numpy()
    by_snapshot = []
    for kind in kinds:
        for snapshot in sorted(set(test_snapshots)):
            rows = test_snapshots == snapshot
            snapshot_metrics = modeling.binary_metrics(y_test[rows], predictions[("test", kind)][rows], TOP_FRACTIONS)
            by_snapshot.append({"snapshot_date": snapshot, "score_kind": kind,
                                **{m: snapshot_metrics[m] for m in SNAPSHOT_METRICS}})
    by_snapshot = pd.DataFrame(by_snapshot)

    # The gate and the CIs use the field that ranks: calibrated when the release ranks on it and this L2 is
    # calibrated, raw otherwise.
    gate_kind = "calibrated" if RANKING_FIELD == "calibrated" and calibrated else "raw"
    clients = blocks["test"]["client_id"].to_numpy()
    ci = {name: modeling.cluster_bootstrap(y_test, predictions[("test", gate_kind)], clients, stat,
                                           n_boot=int(BOOTSTRAP["n_boot"]), seed=int(BOOTSTRAP["seed"]),
                                           ci=float(BOOTSTRAP["ci"]))
          for name, stat in CI_METRICS.items()}
    checks = gate_checks(metrics[("test", gate_kind)])
    gate_passed = all(item["passed"] for item in checks.values())
    ranking_eligible_flag = tier in ("own", "limited") and gate_passed and status != "not_calibrated"
    gate_detail = {"score_kind": gate_kind, "gate_passed": gate_passed, "checks": checks}
    eval_seconds = round(time.time() - t0, 1)

    t1 = time.time()
    params = calibrator.params()
    with mlflow.start_run(run_name=l2) as calibration_run:
        calibrator_run_id = calibration_run.info.run_id
        mlflow.set_tags({"l2": l2, "calibration_method": method, "calibration_status": status,
                         "evidence_tier": tier, "gate_passed": str(gate_passed).lower(),
                         "ranking_eligible_flag": str(ranking_eligible_flag).lower(), "release_id": release_id,
                         "model_run_id": model_run_id})
        mlflow.log_params({
            "evidence_tier": tier, "calibration_method": method, "calibration_status": status,
            "calibration_event_count": calibration_event_count, "test_event_count": test_event_count,
            "own_model_min_events": TIERS["own_model_min_events"], "limited_min_events": TIERS["limited_min_events"],
            "calibration_enabled": CALIBRATION_ENABLED, "platt_C": CALIBRATION.get("platt_C"),
            "gate_auroc_min": GATE["auroc_min"], "gate_lift10_min": GATE["lift10_min"],
            "gate_oe_range": GATE["oe_range"], "gate_slope_range": GATE["slope_range"],
            "gate_score_kind": gate_kind, "ranking_field": RANKING_FIELD, "release_id": release_id,
            "model_run_id": model_run_id, "sample_fraction": float(run["sample_fraction"]),
            "calibrator_a": params["a"], "calibrator_b": params["b"],
            "calibration_rows": len(s_cal), "test_rows": len(s_test),
            "calibration_first": str(CAL_FIRST), "calibration_last": str(CAL_LAST),
            "test_first": str(TEST_FIRST), "test_last": str(TEST_LAST),
        })
        logged = {f"{KIND_PREFIX[kind]}_{name}": value
                  for kind in kinds for name, value in metrics[("test", kind)].items()}
        for name, (low, high) in ci.items():
            logged[f"{KIND_PREFIX[gate_kind]}_{name}_ci_low"] = low
            logged[f"{KIND_PREFIX[gate_kind]}_{name}_ci_high"] = high
        if spearman is not None:
            logged["spearman_raw_calibrated"] = spearman
        mlflow.log_metrics({name: float(v) for name, v in logged.items() if finite_or_none(v) is not None})
        calibrator_pickle = pickle.dumps(calibrator)  # the same bytes go to MLflow and to the file bundle
        release_provenance = {**provenance, "calibration_enabled": CALIBRATION_ENABLED,
            "calibrator_run_id": calibrator_run_id, "calibration_status": status, "calibration_method": method,
            "calibrator_sha256": hashlib.sha256(calibrator_pickle).hexdigest(),
            "evidence_source": TIERS["source"], "evidence_event_count": evidence_event_count,
            "ranking_field": RANKING_FIELD,
            "calibration_input_tables": {name: {"table": common.table_name(name, env_conf), "delta_version": v}
                                         for name, v in INPUT_VERSIONS.items()}}
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "calibrator.pkl").write_bytes(calibrator_pickle)
            reliability["raw"].to_csv(tmp / "reliability_raw.csv", index=False)
            if calibrated:
                reliability["calibrated"].to_csv(tmp / "reliability_calibrated.csv", index=False)
            (tmp / "calibration_provenance.json").write_text(json.dumps(release_provenance, indent=2), encoding="utf-8")
            by_snapshot.to_csv(tmp / "test_by_snapshot.csv", index=False)
            with open(tmp / "gate.json", "w", encoding="utf-8") as f:
                json.dump({"l2": l2, "evidence_tier": tier, "calibration_status": status,
                           "ranking_eligible_flag": ranking_eligible_flag, **gate_detail}, f, indent=2)
            for path in tmp.iterdir():
                mlflow.log_artifact(str(path))
    log_seconds = round(time.time() - t1, 1)

    evaluation_rows = []
    for (split, kind), values in metrics.items():
        for metric, value in values.items():
            low, high = ci[metric] if split == "test" and kind == gate_kind and metric in ci else (None, None)
            evaluation_rows.append({"split": split, "score_kind": kind, "group": "all", "metric": metric,
                                    "value": finite_or_none(value), "ci_low": finite_or_none(low),
                                    "ci_high": finite_or_none(high)})
    for record in by_snapshot.to_dict("records"):
        for metric in SNAPSHOT_METRICS:
            evaluation_rows.append({"split": "test", "score_kind": record["score_kind"],
                                    "group": str(record["snapshot_date"]), "metric": metric,
                                    "value": finite_or_none(record[metric]), "ci_low": None, "ci_high": None})
    for row in evaluation_rows:
        row.update(release_id=release_id, l2=l2)

    selection_row = {
        "release_id": release_id, "l2": l2, "model_run_id": model_run_id, "calibrator_run_id": calibrator_run_id,
        "calibration_method": method, "calibration_status": status,
        "calibration_event_count": calibration_event_count, "test_event_count": test_event_count,
        "evidence_source": TIERS["source"], "evidence_event_count": evidence_event_count,
        "training_config_hash": CONFIG_HASH, "provenance_json": json.dumps(release_provenance),
        "evidence_tier": tier, "gate_passed": gate_passed, "gate_detail": json.dumps(gate_detail),
        "ranking_field": RANKING_FIELD, "ranking_eligible_flag": ranking_eligible_flag,
        "train_end": run["train_last"], "calibration_end": CAL_LAST, "evaluated_on": TEST_LAST,
        "platt_a": finite_or_none(params["a"]), "platt_b": finite_or_none(params["b"]),
    }
    manifest_entry = {
        "model_run_id": model_run_id, "calibrator_run_id": calibrator_run_id, "status": status, "tier": tier,
        "gate_passed": gate_passed, "ranking_eligible_flag": ranking_eligible_flag,
        "feature_columns_file": f"{l2}/feature_columns.json",
        "provenance": release_provenance,
    }
    raw_test = metrics[("test", "raw")]
    cal_test = metrics.get(("test", "calibrated"), {})
    summary = {
        "tier": tier, "method": method, "status": status, "calibration_event_count": calibration_event_count,
        "evidence_source": TIERS["source"], "evidence_event_count": evidence_event_count,
        "test_event_count": test_event_count, "platt_a": params["a"], "platt_b": params["b"],
        "auroc": (raw_test["auroc"], finite_or_none(cal_test.get("auroc"))),
        "oe": (raw_test["oe"], finite_or_none(cal_test.get("oe"))),
        "slope": (raw_test["slope"], finite_or_none(cal_test.get("slope"))),
        "lift10": (raw_test[LIFT_GATE_METRIC], finite_or_none(cal_test.get(LIFT_GATE_METRIC))),
        "gate_kind": gate_kind, "gate_passed": gate_passed, "ranking_eligible_flag": ranking_eligible_flag,
        "auroc_ci": ci["auroc"], "oe_ci": ci["oe"], "spearman": spearman,
        "failed_checks": [name for name, item in checks.items() if not item["passed"]],
        "seconds": {"evaluate": eval_seconds, "log": log_seconds, "total": round(time.time() - t0, 1)},
    }
    print(f"{l2}: {evidence_event_count} evidence events ({TIERS['source']}) -> {tier}, {status}; "
          f"{calibration_event_count} calibration events; Test AUROC {pair(summary['auroc'], 4)}, "
          f"O/E {pair(summary['oe'], 3)}, slope {pair(summary['slope'], 3)}; gate ({gate_kind}) "
          f"{'passed' if gate_passed else 'FAILED ' + str(summary['failed_checks'])}; eligible {ranking_eligible_flag}"
          f"; {summary['seconds']}")
    return selection_row, evaluation_rows, manifest_entry, summary, calibrator_pickle


selection_rows, evaluation_rows, manifest_l2s, summaries, calibrator_pickles = [], [], {}, {}, {}
for l2 in L2S:
    selection_row, rows, manifest_l2s[l2], summaries[l2], calibrator_pickles[l2] = evaluate_l2(l2)
    selection_rows.append(selection_row)
    evaluation_rows.extend(rows)

# COMMAND ----------

# %%
EVALUATION_COLUMNS = [("release_id", "string"), ("l2", "string"), ("split", "string"), ("score_kind", "string"),
                      ("group", "string"), ("metric", "string"), ("value", "double"), ("ci_low", "double"),
                      ("ci_high", "double")]
SELECTION_COLUMNS = [
    ("release_id", "string"), ("l2", "string"), ("model_run_id", "string"), ("calibrator_run_id", "string"),
    ("calibration_method", "string"), ("calibration_status", "string"), ("calibration_event_count", "bigint"),
    ("test_event_count", "bigint"), ("evidence_tier", "string"), ("gate_passed", "boolean"),
    ("gate_detail", "string"), ("ranking_field", "string"), ("ranking_eligible_flag", "boolean"),
    ("train_end", "date"), ("calibration_end", "date"), ("evaluated_on", "date"), ("platt_a", "double"),
    ("platt_b", "double"), ("selected_at", "timestamp"),
    ("evidence_source", "string"), ("evidence_event_count", "bigint"),
    ("training_config_hash", "string"), ("provenance_json", "string"),
]


def spark_frame(pdf: pd.DataFrame, columns: list[tuple[str, str]]):
    """Spark DataFrame of ``pdf`` with exactly ``columns`` (name, Spark type), matched by name."""
    return spark.createDataFrame(pdf[[name for name, _ in columns]],
                                 ", ".join(f"`{name}` {data_type}" for name, data_type in columns))


t0 = time.time()
predicate = f"release_id = '{release_id}'"
selected_at = pd.Timestamp.now(tz="UTC")
selection = pd.DataFrame(selection_rows).assign(selected_at=selected_at)
evaluation_table = common.write_table(spark_frame(pd.DataFrame(evaluation_rows), EVALUATION_COLUMNS),
                                      "ml.l1_evaluation", env_conf, mode="overwrite", replace_where=predicate)
selection_table = common.write_table(spark_frame(selection, SELECTION_COLUMNS), "ml.model_selection", env_conf,
                                     mode="overwrite", replace_where=predicate, merge_schema=True)
n_selection = common.read_table(spark, "ml.model_selection", env_conf).filter(F.col("release_id") == release_id).count()
check(n_selection == len(L2S), f"{selection_table} has {n_selection} rows for {release_id}, expected {len(L2S)}")
print(f"Wrote {evaluation_table} ({len(evaluation_rows)} rows) and {selection_table} ({n_selection} rows) "
      f"in {time.time() - t0:.1f}s")

manifest = {
    "release_id": release_id,
    "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "env": env,
    "git_commit": git_commit(),
    "git_dirty": git_dirty(),
    "code_sha256": code_sha256(),
    "l2s": manifest_l2s,
    "ranking_field": RANKING_FIELD,
    "calibration_enabled": CALIBRATION_ENABLED,
    "training_config_hash": CONFIG_HASH, "effective_training_config": json.loads(l1_contract.canonical_json(training_conf)),
    "timeline": TIMELINE,
    "train_end": str(TRAIN_END),
    "calibration_end": str(CAL_LAST),
    "evaluated_on": str(TEST_LAST),
}


def stage_bundle(staging: Path) -> Path:
    """Write every L2's model.txt / feature_columns.json (from its notebook 01 run) and calibrator.pkl, then
    manifest.json, under ``staging``; returns the manifest path."""
    for l2 in L2S:
        l2_dir = staging / l2
        l2_dir.mkdir()
        model_run_id = manifest_l2s[l2]["model_run_id"]
        for name in ("model.txt", "feature_columns.json", "training_provenance.json"):
            shutil.copy2(training_artifacts_dir / l2 / name, l2_dir / name)
            check((l2_dir / name).is_file(), f"{l2}: {name} was not downloaded from run {model_run_id}")
        (l2_dir / "calibrator.pkl").write_bytes(calibrator_pickles[l2])
        (l2_dir / "calibration_provenance.json").write_text(
            json.dumps(manifest_l2s[l2]["provenance"], indent=2), encoding="utf-8")
    path = staging / "manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path


# The bundle of a release is replaced as a whole and only once it is complete: everything goes to a staging
# directory first; any failure removes the staging directory and leaves the previous bundle untouched.
t0 = time.time()
staging_dir = common.make_staging_dir(RELEASE_DIR)
try:
    staged_manifest = stage_bundle(staging_dir)
    with mlflow.start_run(run_name="release") as release_run:
        mlflow.set_tags({"release_id": release_id, "kind": "release",
                         "git_dirty": "null" if manifest["git_dirty"] is None else str(manifest["git_dirty"]).lower(),
                         "code_sha256": manifest["code_sha256"]})
        mlflow.log_params({"release_id": release_id, "l2s": ",".join(L2S), "ranking_field": RANKING_FIELD,
                           "calibration_enabled": CALIBRATION_ENABLED})
        mlflow.log_artifact(str(staged_manifest))
    common.publish_directory(staging_dir, RELEASE_DIR)
    shutil.rmtree(training_artifacts_dir)
except BaseException:
    shutil.rmtree(staging_dir, ignore_errors=True)
    print(f"Bundle not published; {RELEASE_DIR} is unchanged")
    raise
print(f"Published {RELEASE_DIR} ({len(L2S)} L2s + manifest.json, git_dirty {manifest['git_dirty']}, code_sha256 "
      f"{manifest['code_sha256']}) in {time.time() - t0:.1f}s; MLflow release run {release_run.info.run_id}")

# COMMAND ----------

# %%
header = (f"{'l2':<10} {'tier':<12} {'method':<9} {'status':<29} "
          f"{'evidence_source':<15} {'evidence_ev':>11} {'cal_ev':>6} {'test_ev':>7} "
          f"{'AUROC raw -> cal':<17} {'O/E raw -> cal':<15} {'slope raw -> cal':<15} {'gate':<6} eligible")
print(f"\nRelease {release_id} ({RANKING_FIELD} ranking field)\n{header}\n{'-' * len(header)}")
for l2, s in summaries.items():
    print(f"{l2:<10} {s['tier']:<12} {s['method']:<9} {s['status']:<29} "
          f"{s['evidence_source']:<15} {s['evidence_event_count']:>11} {s['calibration_event_count']:>6} "
          f"{s['test_event_count']:>7} {pair(s['auroc'], 4):<17} {pair(s['oe'], 3):<15} {pair(s['slope'], 3):<15} "
          f"{'pass' if s['gate_passed'] else 'FAIL':<6} {s['ranking_eligible_flag']}")
print(json.dumps({"release_id": release_id, "experiment_id": experiment_id, "bundle": str(RELEASE_DIR),
                  "output_tables": [evaluation_table, selection_table], "l2s": summaries,
                  "elapsed_seconds": round(time.time() - started, 1)}, indent=2, default=str))
