"""Small, shared L1 configuration/date/artifact contract. No Spark or model fitting here."""

from __future__ import annotations

import calendar
import hashlib
import json
from datetime import date, datetime

CONFIG_KEYS = {
    "timeline": {"extraction_date", "horizon_months", "maturity_months", "mode", "tuning_fit_as_of",
                 "base_fit_as_of", "calibrator_fit_as_of", "evaluation_as_of"},
    "split": {"test_snapshots", "train_calibration_gap_snapshots", "calibration_test_gap_snapshots",
              "calibration_snapshots", "inner_valid_snapshots", "inner_gap_snapshots", "min_train_snapshots",
              "train_max_snapshots", "refit_scope"},
    "lightgbm": {"objective", "metric", "num_leaves", "learning_rate", "min_data_in_leaf", "feature_fraction",
                 "bagging_fraction", "bagging_freq", "lambda_l2", "max_rounds", "early_stopping_rounds", "seed",
                 "deterministic", "force_row_wise", "num_threads", "verbosity"},
    "logistic": {"solver", "C", "max_iter", "baseline_sample_rows"},
    "evidence_tiers": {"source", "own_model_min_events", "limited_min_events"},
    "calibration": {"enabled", "isotonic", "platt_C", "final_refit_calibrator"},
    "gate": {"auroc_min", "lift10_min", "oe_range", "slope_range"},
    "ranking": {"ranking_field"},
    "evaluation": {"top_fractions", "bootstrap"},
    "explain": {"top_k", "top_n_reasons", "store_feature_level"},
    "scoring": {"lookback_days"},
}


def as_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except (ValueError, TypeError) as error:
        raise ValueError(f"Expected ISO date, got {value!r}") from error


def add_months(d: date, months: int) -> date:
    """Calendar months, day clipped once (the source's H+m maturity convention)."""
    year, month0 = divmod(d.year * 12 + d.month - 1 + months, 12)
    return date(year, month0 + 1, min(d.day, calendar.monthrange(year, month0 + 1)[1]))


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def validate_config(conf: dict) -> dict:
    """Reject unknown/missing policy keys and ambiguous types before any material write."""
    if not isinstance(conf, dict) or set(conf) != set(CONFIG_KEYS):
        raise ValueError("training configuration has missing/unknown sections")
    for section, keys in CONFIG_KEYS.items():
        values = conf[section]
        if not isinstance(values, dict) or set(values) != keys:
            raise ValueError(f"{section}: missing/unknown keys; expected {sorted(keys)}")
    bootstrap = conf["evaluation"]["bootstrap"]
    if not isinstance(bootstrap, dict) or set(bootstrap) != {"n_boot", "seed", "ci"}:
        raise ValueError("evaluation.bootstrap has missing/unknown keys")
    split, timeline = conf["split"], conf["timeline"]
    for name in split.keys() - {"refit_scope"}:
        value = split[name]
        if name == "train_max_snapshots" and value is None:
            continue
        minimum = 0 if "gap" in name or name == "calibration_snapshots" else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f"split.{name} must be an integer >= {minimum}, got {value!r}")
    if split["train_max_snapshots"] is not None and split["train_max_snapshots"] < split["min_train_snapshots"]:
        raise ValueError("train_max_snapshots must be >= min_train_snapshots")
    if split["refit_scope"] not in ("all_train", "tuning_fit"):
        raise ValueError("split.refit_scope must be all_train or tuning_fit")
    for name, expected in (("horizon_months", 3), ("maturity_months", 2)):
        if type(timeline[name]) is not int or timeline[name] != expected:
            raise ValueError(f"timeline.{name} must be {expected}: the input labels use H=3 and m=2")
    if timeline["mode"] != "retrospective_holdout":
        raise ValueError("Only timeline.mode=retrospective_holdout is implemented; no deployment replay claim")
    extraction = as_date(timeline["extraction_date"])
    effective = {}
    for name in ("tuning_fit_as_of", "base_fit_as_of", "calibrator_fit_as_of", "evaluation_as_of"):
        effective[name] = extraction if timeline[name] is None else as_date(timeline[name])
        if effective[name] > extraction:
            raise ValueError(f"{name} exceeds extraction_date; future labels are unavailable")
    calibration = conf["calibration"]
    for name in ("enabled", "isotonic", "final_refit_calibrator"):
        if type(calibration[name]) is not bool:
            raise ValueError(f"calibration.{name} must be boolean")
    if calibration["isotonic"] or calibration["final_refit_calibrator"]:
        raise ValueError("isotonic / final_refit_calibrator are not implemented")
    source = conf["evidence_tiers"]["source"]
    if source not in ("calibration", "inner_valid"):
        raise ValueError("evidence_tiers.source must be calibration or inner_valid; never test")
    if calibration["enabled"] and (not split["calibration_snapshots"] or source != "calibration"):
        raise ValueError("enabled calibration requires a nonempty calibration allocation and calibration evidence")
    if source == "calibration" and not split["calibration_snapshots"]:
        raise ValueError("calibration evidence requires a nonempty allocation; explicitly choose inner_valid for raw")
    if not split["calibration_snapshots"] and split["train_calibration_gap_snapshots"]:
        raise ValueError("zero calibration allocation requires train_calibration_gap_snapshots=0 explicitly")
    if conf["ranking"]["ranking_field"] not in ("raw", "calibrated"):
        raise ValueError("ranking.ranking_field must be raw or calibrated")
    tiers = conf["evidence_tiers"]
    if any(type(tiers[k]) is not int or tiers[k] <= 0 for k in ("limited_min_events", "own_model_min_events")):
        raise ValueError("evidence thresholds must be positive integers")
    if tiers["own_model_min_events"] <= tiers["limited_min_events"]:
        raise ValueError("own evidence threshold must exceed limited")
    if not effective["tuning_fit_as_of"] <= effective["base_fit_as_of"] <= effective["evaluation_as_of"]:
        raise ValueError("tuning/base/evaluation as_of must be ordered")
    if calibration["enabled"] and not (effective["base_fit_as_of"] <= effective["calibrator_fit_as_of"]
                                       <= effective["evaluation_as_of"]):
        raise ValueError("base/calibrator/evaluation as_of must be ordered")
    return effective


def derive_dates(conf: dict, snapshot_dates) -> dict:
    effective = validate_config(conf)
    split, timeline = conf["split"], conf["timeline"]
    extraction = as_date(timeline["extraction_date"])
    h, lag = timeline["horizon_months"], timeline["horizon_months"] + timeline["maturity_months"]
    dates = sorted({as_date(d) for d in snapshot_dates})
    if not dates or any(d.day != calendar.monthrange(d.year, d.month)[1] for d in dates):
        raise ValueError("Snapshot dates must be nonempty month ends")
    mature = [d for d in dates if add_months(d, lag) <= extraction]
    sizes = (("test", split["test_snapshots"]), ("gap2", split["calibration_test_gap_snapshots"]),
             ("calibration", split["calibration_snapshots"]), ("gap1", split["train_calibration_gap_snapshots"]))
    n_train = len(mature) - sum(size for _, size in sizes)
    if n_train < split["min_train_snapshots"]:
        raise ValueError(f"{len(mature)} mature snapshots leave {n_train} train; min_train_snapshots="
                         f"{split['min_train_snapshots']}; requested blocks are never trimmed")
    blocks, end = {}, len(mature)
    for name, size in sizes:
        blocks[name] = mature[end - size:end] if size else []
        end -= size
    cap = split["train_max_snapshots"]
    start = max(0, end - cap) if cap is not None else 0
    blocks.update(train=mature[start:end], unused=mature[:start],
                  immature=[d for d in dates if d not in mature], last_mature=mature[-1])
    valid, gap = split["inner_valid_snapshots"], split["inner_gap_snapshots"]
    fit_end = len(blocks["train"]) - valid - gap
    if fit_end < 1:
        raise ValueError("Train history leaves no tuning fit snapshots after requested validation and gap")
    blocks["tuning_fit"] = blocks["train"][:fit_end]
    blocks["inner_gap"] = blocks["train"][fit_end:fit_end + gap]
    blocks["inner_valid"] = blocks["train"][-valid:]
    blocks["refit"] = blocks["train"] if split["refit_scope"] == "all_train" else blocks["tuning_fit"]
    blocks["evidence"] = blocks[conf["evidence_tiers"]["source"]]
    boundaries = [("tuning_fit", "inner_valid")]
    boundaries += ([("train", "calibration"), ("calibration", "test")] if blocks["calibration"]
                   else [("train", "test")])
    for previous, following in boundaries:
        label_end = add_months(blocks[previous][-1], h)
        if label_end > blocks[following][0]:
            raise ValueError(f"label overlap {previous}->{following}: {label_end} > {blocks[following][0]}")
    availability = dict(effective)
    requirements = {"tuning": ("inner_valid", "tuning_fit_as_of"),
                    "base": ("refit", "base_fit_as_of"),
                    "evaluation": ("test", "evaluation_as_of"),
                    "evidence": ("evidence", "evaluation_as_of")}
    if conf["calibration"]["enabled"]:
        requirements["calibrator"] = ("calibration", "calibrator_fit_as_of")
    for stage, (block, as_of) in requirements.items():
        ready = add_months(blocks[block][-1], lag)
        availability[f"{stage}_ready_on"] = ready
        if ready > effective[as_of]:
            raise ValueError(f"{stage} labels not available: ready_on={ready} > {as_of}={effective[as_of]}")
    blocks["availability"] = availability
    return blocks


def timeline_manifest(conf: dict, blocks: dict) -> dict:
    """Exact observed dates and readiness (no invented regular grid or historical deployment claim)."""
    return json.loads(canonical_json({"mode": conf["timeline"]["mode"], "timeline": conf["timeline"],
        "refit_scope": conf["split"]["refit_scope"], "evidence_source": conf["evidence_tiers"]["source"],
        "blocks": blocks}))


def validate_provenance(provenance: dict, conf: dict, model_run_id: str | None = None) -> None:
    if provenance.get("contract_version") != 1 or provenance.get("training_config_hash") != digest(conf):
        raise ValueError("stale/missing training configuration provenance; rerun 01/02 with the selected YAML")
    if provenance.get("effective_training_config") != json.loads(canonical_json(conf)):
        raise ValueError("effective training configuration differs from its hash")
    if model_run_id is not None and provenance.get("model_run_id") != model_run_id:
        raise ValueError("model run identity differs from provenance")
    if provenance.get("model_role") != "champion_refit" or provenance.get("model_artifact") != "model.txt":
        raise ValueError("serving requires champion_refit model.txt, never the tuning model")


def validate_feature_spec(spec: dict, provenance: dict, role: str) -> None:
    expected = provenance["feature_spec_hashes"].get(role)
    if spec.get("model_role") != role or digest(spec) != expected:
        raise ValueError(f"{role} feature spec differs from training artifact provenance")


def validate_model_file(path, provenance: dict, artifact: str = "model.txt") -> None:
    if hashlib.sha256(path.read_bytes()).hexdigest() != provenance["artifact_sha256"].get(artifact):
        raise ValueError(f"model artifact {artifact} differs from its original training provenance")


def validate_raw_scores(frame, provenance: dict, config_hash: str) -> None:
    """Validate row-level identity and scope before any calibrator fitting or release writes."""
    required = {"model_run_id", "training_config_hash", "model_role", "model_artifact", "best_iteration",
                "feature_spec_hash", "split", "client_id", "snapshot_date", "y", "raw_score", "raw_margin"}
    if not required <= set(frame.columns) or frame.empty:
        raise ValueError("raw scores missing rows or provenance columns; rerun 01")
    if frame.duplicated(["split", "client_id", "snapshot_date"]).any():
        raise ValueError("duplicate raw score keys")
    expected = {"model_run_id": provenance["model_run_id"], "training_config_hash": config_hash,
                "model_role": "champion_refit", "model_artifact": "model.txt",
                "best_iteration": provenance["best_iteration"],
                "feature_spec_hash": provenance["feature_spec_hashes"]["champion_refit"]}
    for key, value in expected.items():
        if not (frame[key] == value).all():
            raise ValueError(f"raw scores have stale/mixed {key}")
    blocks = provenance["timeline"]["blocks"]
    for split in ("calibration", "test"):
        actual = {str(as_date(d)) for d in frame.loc[frame["split"] == split, "snapshot_date"]}
        if actual != set(blocks[split]):
            raise ValueError(f"raw score {split} snapshots differ from training timeline")
    if not frame["split"].isin(["calibration", "test"]).all():
        raise ValueError("calibration/evaluation may only consume frozen champion holdout scores")
    if not frame["y"].isin([0, 1]).all() or not frame["raw_score"].between(0, 1).all():
        raise ValueError("raw scores contain invalid labels or probabilities")


def validate_score_counts(observed: dict, expected: dict) -> None:
    """A zero allocation has no groupBy row; missing positive counts still fail."""
    for l2, blocks in expected.items():
        for block, count in blocks.items():
            if observed.get((l2, block), 0) != count:
                raise ValueError(f"raw score count {l2}/{block}: {observed.get((l2, block), 0)} != {count}")
