# Databricks notebook source
# %%
# Notebook 01: train the Layer 1 propensity models, one per L2 (docs/MVP_SPEC.md section 7).
#
# ml.modeling_table (train / calibration / test blocks; elig_{l2} = 1 rows only) -> per L2:
#   champion  LightGBM early-stopped on the inner validation block (last snapshots of Train, after an inner gap),
#             then refit on all of Train with best_iteration
#   baseline  logistic regression on (a sample of) the inner fit block; WARNING when it beats LightGBM on inner
#             validation log-loss (LightGBM stays the champion)
#   genuine tuning scores on inner_valid; frozen champion scores on calibration / test
# Outputs: ml.l1_raw_scores (partitioned by l2), ml.l1_training_runs (one row per L2), MLflow experiment
# l1_training (one run per L2, run name = l2; model_version = run_id).
#
# Parameters: env (default local), l2_list (pilot | all | comma-separated; default pilot), sample_fraction (default
# 1.0; client hash sample of ml.modeling_table and raw.l2_purchase_config, whole time series per client).
# Run locally: python notebooks/01_train_layer1.py --env local --l2_list product1,product14 --sample_fraction 0.1
import csv
import json
import sys
import tempfile
import time
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
sample_fraction = float(common.get_param("sample_fraction", 1.0))
started = time.time()
print(f"env={env} l2_list={l2_list} sample_fraction={sample_fraction}")

# COMMAND ----------

# %%
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

env_conf = common.load_env(env)
products = common.load_conf("products")
training_conf = common.load_conf("training")
L2S = modeling.resolve_l2_list(products, l2_list)
CLIENT_ID = products["keys"]["client_id"]
SNAPSHOT_DATE = products["keys"]["snapshot_date"]
LABEL = {l2: products["label_column_template"].format(l2=l2) for l2 in L2S}
ELIG = {l2: products["eligibility_column_template"].format(l2=l2) for l2 in L2S}
FEATURE_EXCLUSIONS = list(products.get("feature_exclusions") or [])
# Never features, whatever the data dictionary says (docs/MVP_SPEC.md section 7).
NON_FEATURES = {CLIENT_ID, SNAPSHOT_DATE, "CLIENT_JOINDATE", "snapshot_month", "split"}
HORIZON = int(training_conf["timeline"]["horizon_months"])
LGBM_PARAMS = dict(training_conf["lightgbm"])
LOGISTIC_CONF = dict(training_conf["logistic"])
BASELINE_ROWS = int(LOGISTIC_CONF["baseline_sample_rows"])
SEED = int(LGBM_PARAMS["seed"])
TOP_FRACTIONS = [float(f) for f in training_conf["evaluation"]["top_fractions"]]
BLOCKS = ["train", "test"] + (["calibration"] if training_conf["split"]["calibration_snapshots"] else [])
CONFIG_HASH = l1_contract.digest(training_conf)
INNER_METRICS = ["auroc", "pr_auc", "logloss", "brier"]
NUMERIC_TYPE_PREFIXES = ("tinyint", "smallint", "int", "bigint", "float", "double", "decimal")
print(f"L2s: {L2S}")


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def as_date(value):
    return pd.Timestamp(value).date()

# COMMAND ----------

# %%
import mlflow  # noqa: E402
import mlflow.lightgbm  # noqa: E402
import mlflow.sklearn  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

spark = common.get_spark(env_conf)
INPUT_TABLES = ["ml.modeling_table", "ml.data_dictionary", "raw.l2_purchase_config"]
INPUT_VERSIONS = {name: common.delta_version(spark, name, env_conf) for name in INPUT_TABLES}
modeling_sdf = common.read_table(spark, "ml.modeling_table", env_conf,
                                 version_as_of=INPUT_VERSIONS["ml.modeling_table"])

# The dictionary is a multi-file Delta table, so its row order is arbitrary: order it by the modeling table schema.
dictionary = common.to_pandas(common.read_table(spark, "ml.data_dictionary", env_conf,
                                               version_as_of=INPUT_VERSIONS["ml.data_dictionary"])
                              .limit(len(modeling_sdf.columns) + 1))
with open(common.repo_root() / "conf" / "feature_groups.csv", encoding="utf-8", newline="") as f:
    feature_names = [row["feature_name"] for row in csv.DictReader(f)]
validated_splits = common.validate_training_input(modeling_sdf, dictionary, products, training_conf, feature_names)
print(f"Validated full ml.modeling_table input; last mature {validated_splits['last_mature']}")
position = {name: i for i, name in enumerate(modeling_sdf.columns)}
unknown = sorted(set(dictionary["column_name"]) - position.keys())
check(not unknown, f"ml.data_dictionary has columns that ml.modeling_table lacks: {unknown}")
dictionary = dictionary.assign(_position=dictionary["column_name"].map(position)).sort_values("_position")
FEATURES = modeling.feature_columns_from_dictionary(dictionary, FEATURE_EXCLUSIONS)
leaked = [c for c in FEATURES if c in NON_FEATURES or c.lower().startswith(modeling.FORBIDDEN_FEATURE_PREFIXES)]
check(not leaked, f"non-feature columns would be features: {leaked}")
data_types = dict(zip(dictionary["column_name"], dictionary["data_type"]))
STRING_FEATURES = [c for c in FEATURES if data_types[c] == "string"]
NUMERIC_FEATURES = [c for c in FEATURES if data_types[c].startswith(NUMERIC_TYPE_PREFIXES)]
untyped = [(c, data_types[c]) for c in FEATURES if c not in STRING_FEATURES and c not in NUMERIC_FEATURES]
check(not untyped, f"features that are neither numeric nor string: {untyped}")
missing = [c for c in [LABEL[l2] for l2 in L2S] + [ELIG[l2] for l2 in L2S] if c not in position]
check(not missing, f"ml.modeling_table lacks label / eligibility columns {missing}")
print(f"{len(FEATURES)} features: {len(NUMERIC_FEATURES)} numeric, {len(STRING_FEATURES)} categorical "
      f"({', '.join(STRING_FEATURES)})")
experiment_id = common.mlflow_setup(env_conf, "l1_training")

# COMMAND ----------

# %%
# One toPandas per block; only keys, features, the selected L2s' labels / eligibility, split and snapshot_month.
source = common.sample_clients(modeling_sdf, sample_fraction, key=CLIENT_ID) if sample_fraction < 1 else modeling_sdf
load_columns = ([CLIENT_ID, SNAPSHOT_DATE] + FEATURES + [LABEL[l2] for l2 in L2S] + [ELIG[l2] for l2 in L2S]
                + ["split", "snapshot_month"])
frames, load_seconds = {}, {}
for block in BLOCKS:
    t0 = time.time()
    frames[block] = common.to_pandas(source.filter(F.col("split") == block), load_columns)
    load_seconds[block] = round(time.time() - t0, 1)
    check(len(frames[block]) > 0, f"no {block} rows in ml.modeling_table")
    print(f"Loaded {block:<12} {len(frames[block]):>10,} rows in {load_seconds[block]:>6.1f}s")

purchases_sdf = common.read_table(spark, "raw.l2_purchase_config", env_conf,
                                  version_as_of=INPUT_VERSIONS["raw.l2_purchase_config"]).filter(F.col("l2").isin(L2S))
if sample_fraction < 1:
    purchases_sdf = common.sample_clients(purchases_sdf, sample_fraction, key=CLIENT_ID)
purchases = common.to_pandas(purchases_sdf, ["purchase_id", "client_id", "l2", "application_date",
                                             "elig_at_feature_snapshot"])
print(f"Loaded raw.l2_purchase_config: {len(purchases):,} purchases of the selected L2s")

# COMMAND ----------

# %%
# Each product fits its own transform on the rows used by that model, in train_l2 below.
t0 = time.time()
meta = {}
for block in BLOCKS:
    block_meta = frames[block][[CLIENT_ID, SNAPSHOT_DATE] + [LABEL[l2] for l2 in L2S] + [ELIG[l2] for l2 in L2S]]
    meta[block] = block_meta.assign(**{SNAPSHOT_DATE: pd.to_datetime(block_meta[SNAPSHOT_DATE])})
prepare_seconds = round(time.time() - t0, 1)

train_snapshots = sorted(meta["train"][SNAPSHOT_DATE].unique())
FIT_DATES, GAP_DATES, VALID_DATES = (validated_splits[k] for k in ("tuning_fit", "inner_gap", "inner_valid"))
TRAIN_FIRST, TRAIN_LAST = as_date(train_snapshots[0]), as_date(train_snapshots[-1])
VALID_FIRST, VALID_LAST = as_date(VALID_DATES[0]), as_date(VALID_DATES[-1])
block_dates = {block: sorted(meta[block][SNAPSHOT_DATE].unique()) for block in BLOCKS}
print(f"Train {TRAIN_FIRST}..{TRAIN_LAST} ({len(train_snapshots)} snapshots): inner fit "
      f"{as_date(FIT_DATES[0])}..{as_date(FIT_DATES[-1])}, gap {as_date(GAP_DATES[0]) if GAP_DATES else '-'}.."
      f"{as_date(GAP_DATES[-1]) if GAP_DATES else '-'}, valid {VALID_FIRST}..{VALID_LAST}")
for block in [b for b in BLOCKS if b != "train"]:
    print(f"{block}: {as_date(block_dates[block][0])}..{as_date(block_dates[block][-1])}")

TIMELINE = l1_contract.timeline_manifest(training_conf, validated_splits)

# COMMAND ----------

# %%
def labels(block: str, l2: str, rows: np.ndarray) -> np.ndarray:
    """0/1 labels of ``l2`` for the block rows selected by a boolean mask or row positions."""
    values = meta[block][LABEL[l2]].to_numpy()[rows]
    check(not pd.isna(values).any(), f"{l2}: NULL labels among eligible {block} rows")
    return values.astype(np.int64)


def log_artifacts(champion, early, baseline, champion_spec, tuning_spec, provenance, run_dir: Path) -> None:
    importance = pd.DataFrame({
        "feature": champion.feature_name(),
        "gain": champion.feature_importance(importance_type="gain"),
        "split": champion.feature_importance(importance_type="split"),
    }).sort_values(["gain", "split"], ascending=False)
    importance.to_csv(run_dir / "feature_importance.csv", index=False)
    with open(run_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(champion_spec, f, indent=2)
    champion.save_model(str(run_dir / "model.txt"))
    tuning_dir = run_dir / "tuning"
    tuning_dir.mkdir()
    early.save_model(str(tuning_dir / "model.txt"))
    (tuning_dir / "feature_columns.json").write_text(json.dumps(tuning_spec, indent=2), encoding="utf-8")
    import hashlib

    provenance["artifact_sha256"] = {name: hashlib.sha256((run_dir / name).read_bytes()).hexdigest()
                                     for name in ("model.txt", "tuning/model.txt")}
    (run_dir / "training_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    mlflow.log_artifacts(str(tuning_dir), artifact_path="tuning")
    mlflow.log_artifact(str(run_dir / "training_provenance.json"))
    for name in ("feature_importance.csv", "feature_columns.json", "model.txt"):
        mlflow.log_artifact(str(run_dir / name))
    mlflow.lightgbm.log_model(champion, name="model")
    # The baseline pipeline references modeling.signed_log1p: ship modeling.py with it.
    mlflow.sklearn.log_model(baseline, name="baseline", code_paths=[str(_notebook_dir() / "modeling.py")])


def train_l2(l2: str) -> tuple[pd.DataFrame, dict, dict]:
    """Champion + baseline + raw scores + MLflow run of one L2; returns (scores, l1_training_runs row, summary)."""
    timings, t0 = {}, time.time()
    all_meta = meta["train"]
    eligible = (all_meta[ELIG[l2]] == 1).to_numpy()
    train_meta = all_meta.loc[eligible].reset_index(drop=True)
    train_frame = frames["train"].loc[eligible].reset_index(drop=True)
    snapshots = train_meta[SNAPSHOT_DATE]
    in_fit = snapshots.dt.date.isin(FIT_DATES).to_numpy()
    in_valid = snapshots.dt.date.isin(VALID_DATES).to_numpy()
    in_refit = snapshots.dt.date.isin(validated_splits["refit"]).to_numpy()
    y_train = labels("train", l2, eligible)
    y_fit, y_valid = y_train[in_fit], y_train[in_valid]
    check(y_fit.sum() > 0 and y_valid.sum() > 0, f"{l2}: no positives in the inner fit or valid block")
    early, champion, tuning_spec, champion_spec, best_iteration = modeling.train_l1_pair(
        train_frame, y_train, np.flatnonzero(in_fit), np.flatnonzero(in_valid), np.flatnonzero(in_refit),
        FEATURES, STRING_FEATURES, LGBM_PARAMS)
    for spec in (tuning_spec, champion_spec):
        spec["exclusions"] = FEATURE_EXCLUSIONS
    X_tuning = modeling.prepare_matrix(train_frame, FEATURES, tuning_spec["categorical_levels"])
    X_valid = X_tuning.loc[in_valid]
    check(champion.feature_name() == FEATURES, f"{l2}: LightGBM renamed feature columns")
    timings["lgbm_tuning_and_refit"] = round(time.time() - t0, 1)

    t1 = time.time()
    fit_rows = np.flatnonzero(in_fit)
    if len(fit_rows) > BASELINE_ROWS:
        fit_rows = np.sort(np.random.default_rng(SEED).choice(fit_rows, size=BASELINE_ROWS, replace=False))
    baseline_heavy = modeling.detect_heavy_tailed(X_tuning.iloc[fit_rows], NUMERIC_FEATURES)
    baseline = modeling.fit_logistic(X_tuning.iloc[fit_rows], y_train[fit_rows], NUMERIC_FEATURES,
                                     STRING_FEATURES, baseline_heavy, LOGISTIC_CONF)
    timings["logistic"] = round(time.time() - t1, 1)
    inner = {
        "lgbm": modeling.binary_metrics(y_valid, early.predict(X_valid, num_iteration=best_iteration), TOP_FRACTIONS),
        "logit": modeling.binary_metrics(y_valid, baseline.predict_proba(X_valid)[:, 1], TOP_FRACTIONS),
    }
    baseline_beats_champion = bool(inner["logit"]["logloss"] < inner["lgbm"]["logloss"])
    if baseline_beats_champion:
        print(f"WARNING {l2}: the logistic baseline beats LightGBM on inner validation log-loss "
              f"({inner['logit']['logloss']:.6f} < {inner['lgbm']['logloss']:.6f}); LightGBM stays the champion")
    refit_first, refit_last = validated_splits["refit"][0], validated_splits["refit"][-1]
    train_event_count = modeling.count_events(purchases, l2, refit_first, refit_last, HORIZON)

    t1 = time.time()
    with mlflow.start_run(run_name=l2) as run:
        run_id = run.info.run_id
        provenance = {
            "contract_version": 1, "training_config_hash": CONFIG_HASH,
            "effective_training_config": json.loads(l1_contract.canonical_json(training_conf)),
            "model_run_id": run_id, "model_role": "champion_refit", "model_artifact": "model.txt",
            "tuning_model_artifact": "tuning/model.txt", "best_iteration": best_iteration,
            "timeline": TIMELINE, "sample_fraction": sample_fraction,
            "score_meaning": "raw_lightgbm_probability", "tuning_metric_role": "iteration_selection_diagnostic",
            "feature_spec_hashes": {"tuning": l1_contract.digest(tuning_spec),
                                    "champion_refit": l1_contract.digest(champion_spec)},
            "input_tables": {name: {"table": common.table_name(name, env_conf), "delta_version": version}
                             for name, version in INPUT_VERSIONS.items()},
        }
        score_parts = []
        score_blocks = [("inner_valid", "train", eligible & all_meta[SNAPSHOT_DATE].dt.date.isin(VALID_DATES))]
        score_blocks += [(b, b, (meta[b][ELIG[l2]] == 1).to_numpy()) for b in BLOCKS if b != "train"]
        for split, block, mask in score_blocks:
            tuning = split == "inner_valid"
            spec, booster = (tuning_spec, early) if tuning else (champion_spec, champion)
            X_block = modeling.prepare_matrix(frames[block].loc[mask], FEATURES, spec["categorical_levels"])
            block_meta = meta[block].loc[mask]
            score_parts.append(pd.DataFrame({
                "split": split,
                "client_id": block_meta[CLIENT_ID].to_numpy(),
                "snapshot_date": block_meta[SNAPSHOT_DATE].dt.date.to_numpy(),
                "y": labels(block, l2, mask).astype(np.int32),
                "raw_score": booster.predict(X_block, num_iteration=best_iteration),
                "raw_margin": booster.predict(X_block, raw_score=True, num_iteration=best_iteration),
                "model_run_id": run_id,
                "model_role": spec["model_role"], "model_artifact": "tuning/model.txt" if tuning else "model.txt",
                "feature_spec_hash": l1_contract.digest(spec), "training_config_hash": CONFIG_HASH,
                "best_iteration": best_iteration,
                "l2": l2,
            }))
        scores = pd.concat(score_parts, ignore_index=True)
        rows = {split: int((scores["split"] == split).sum()) for split in ("inner_valid", "calibration", "test")}

        mlflow.set_tags({"l2": l2, "train_end": str(refit_last), "inner_valid_end": str(VALID_LAST),
                         "champion": "lightgbm", "baseline_beats_champion": str(baseline_beats_champion).lower(),
                         "sample_fraction": str(sample_fraction)})
        mlflow.log_params({
            **LGBM_PARAMS,
            **{f"logistic_{k}": v for k, v in LOGISTIC_CONF.items()},
            "best_iteration": best_iteration,
            "sample_fraction": sample_fraction,
            "n_features": len(FEATURES),
            "train_rows": int(in_refit.sum()),
            "train_pos": int(y_train[in_refit].sum()),
            "fit_rows": int(in_fit.sum()),
            "inner_valid_rows": int(in_valid.sum()),
            "logistic_rows": len(fit_rows),
            "calibration_rows": rows["calibration"],
            "test_rows": rows["test"],
            "train_event_count": train_event_count,
            "train_first": str(refit_first),
            "train_last": str(refit_last),
            "inner_valid_snapshots": ",".join(str(as_date(d)) for d in VALID_DATES),
            "inner_gap_snapshots": ",".join(str(as_date(d)) for d in GAP_DATES),
        })
        mlflow.log_metrics({f"{model}_{metric}": inner[model][metric]
                            for model in ("lgbm", "logit") for metric in INNER_METRICS})
        with tempfile.TemporaryDirectory() as tmp:
            log_artifacts(champion, early, baseline, champion_spec, tuning_spec, provenance, Path(tmp))
    timings["score_and_log"] = round(time.time() - t1, 1)
    timings["total"] = round(time.time() - t0, 1)

    run_row = {
        "l2": l2,
        "model_run_id": run_id,
        "training_config_hash": CONFIG_HASH, "provenance_json": json.dumps(provenance),
        "best_iteration": best_iteration,
        "train_rows": int(in_refit.sum()),
        "train_pos": int(y_train[in_refit].sum()),
        "train_event_count": train_event_count,
        "train_first": refit_first,
        "train_last": refit_last,
        "inner_valid_first": VALID_FIRST,
        "inner_valid_last": VALID_LAST,
        "lgbm_inner_auroc": inner["lgbm"]["auroc"],
        "logit_inner_auroc": inner["logit"]["auroc"],
        "baseline_beats_champion": baseline_beats_champion,
        "sample_fraction": sample_fraction,
        "trained_at": pd.Timestamp.now(tz="UTC"),
    }
    summary = {
        "model_run_id": run_id,
        "best_iteration": best_iteration,
        "train_rows": run_row["train_rows"],
        "train_pos": run_row["train_pos"],
        "train_event_count": train_event_count,
        "score_rows": rows,
        "lgbm_inner_auroc": round(inner["lgbm"]["auroc"], 4),
        "logit_inner_auroc": round(inner["logit"]["auroc"], 4),
        "lgbm_inner_logloss": round(inner["lgbm"]["logloss"], 6),
        "logit_inner_logloss": round(inner["logit"]["logloss"], 6),
        "baseline_beats_champion": baseline_beats_champion,
        "seconds": timings,
    }
    print(f"{l2}: best_iteration {best_iteration}, inner AUROC lgbm {inner['lgbm']['auroc']:.4f} / logit "
          f"{inner['logit']['auroc']:.4f}, train rows {run_row['train_rows']:,} ({run_row['train_pos']:,} positive), "
          f"train events {train_event_count:,}, {timings['total']}s {timings}")
    return scores, run_row, summary


all_scores, run_rows, summaries = {}, [], {}
for l2 in L2S:
    all_scores[l2], run_row, summaries[l2] = train_l2(l2)
    run_rows.append(run_row)

# COMMAND ----------

# %%
# Delta outputs, one replaceWhere slice per L2. The partition column l2 comes last: the local in-memory catalog
# rejects later replaceWhere writes to a partitioned table whose partition columns are not at the end.
RAW_SCORES_COLUMNS = [("split", "string"), ("client_id", "string"), ("snapshot_date", "date"), ("y", "int"),
                      ("raw_score", "double"), ("raw_margin", "double"), ("model_run_id", "string"),
                      ("model_role", "string"), ("model_artifact", "string"), ("feature_spec_hash", "string"),
                      ("training_config_hash", "string"), ("best_iteration", "int"), ("l2", "string")]
TRAINING_RUNS_COLUMNS = [
    ("l2", "string"), ("model_run_id", "string"), ("best_iteration", "int"), ("train_rows", "bigint"),
    ("train_pos", "bigint"), ("train_event_count", "bigint"), ("train_first", "date"), ("train_last", "date"),
    ("inner_valid_first", "date"), ("inner_valid_last", "date"), ("lgbm_inner_auroc", "double"),
    ("logit_inner_auroc", "double"), ("baseline_beats_champion", "boolean"), ("sample_fraction", "double"),
    ("trained_at", "timestamp"),
    ("training_config_hash", "string"), ("provenance_json", "string"),
]


def spark_frame(pdf: pd.DataFrame, columns: list[tuple[str, str]]):
    """Spark DataFrame of ``pdf`` with exactly ``columns`` (name, Spark type), matched by name."""
    return spark.createDataFrame(pdf[[name for name, _ in columns]],
                                 ", ".join(f"`{name}` {data_type}" for name, data_type in columns))


t0 = time.time()
for l2 in L2S:
    predicate = f"l2 = '{l2}'"
    raw_scores_table = common.write_table(spark_frame(all_scores[l2], RAW_SCORES_COLUMNS), "ml.l1_raw_scores",
                                          env_conf, mode="overwrite", replace_where=predicate, partition_by=["l2"],
                                          merge_schema=True)
    run_frame = pd.DataFrame([row for row in run_rows if row["l2"] == l2])
    training_runs_table = common.write_table(spark_frame(run_frame, TRAINING_RUNS_COLUMNS), "ml.l1_training_runs",
                                             env_conf, mode="overwrite", replace_where=predicate, merge_schema=True)
write_seconds = round(time.time() - t0, 1)

written = (common.read_table(spark, "ml.l1_raw_scores", env_conf).filter(F.col("l2").isin(L2S))
           .groupBy("l2", "split").count().collect())
written_counts = {(r["l2"], r["split"]): r["count"] for r in written}
l1_contract.validate_score_counts(written_counts, {l2: summaries[l2]["score_rows"] for l2 in L2S})
print(f"Wrote {raw_scores_table} and {training_runs_table} ({len(L2S)} L2 slices) in {write_seconds}s")

# COMMAND ----------

# %%
summary = {
    "env": env,
    "sample_fraction": sample_fraction,
    "experiment_id": experiment_id,
    "output_tables": [raw_scores_table, training_runs_table],
    "train": {"first": str(TRAIN_FIRST), "last": str(TRAIN_LAST), "inner_valid_first": str(VALID_FIRST),
              "inner_valid_last": str(VALID_LAST)},
    "load_seconds": load_seconds,
    "prepare_seconds": prepare_seconds,
    "write_seconds": write_seconds,
    "l2s": summaries,
    "elapsed_seconds": round(time.time() - started, 1),
}
print(json.dumps(summary, indent=2, default=str))
