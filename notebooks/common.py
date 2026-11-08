"""Shared helpers for the product-recommendation notebooks (imported through sys.path; there is no package).

Every notebook adds its own directory (``notebooks/``) to ``sys.path`` and runs ``import common``. All differences
between local runs and Databricks live here and in ``conf/env/<env>.yaml``:

- Detection: :func:`is_databricks` checks ``DATABRICKS_RUNTIME_VERSION``. ``platform`` in the env file must match.
- Spark: on Databricks :func:`get_spark` returns the cluster's session. Locally it builds ``local[*]`` with Delta
  Lake, ANSI SQL, Arrow, UTC and warehouse ``<repo>/spark-warehouse``, sets JAVA_HOME / HADOOP_HOME / PYSPARK_PYTHON
  from ``tools/`` when they are missing, and uses an in-memory catalog instead of a Hive metastore (so several
  processes can share the warehouse). That catalog is rebuilt ("rehydrated") at start from the Delta tables on disk:
  every ``<db>.db/<table>/_delta_log`` becomes ``CREATE DATABASE IF NOT EXISTS`` + ``CREATE TABLE IF NOT EXISTS ...
  USING DELTA LOCATION``. Tables created by another process later are registered on first read or write.
  Registered tables are external: ``DROP TABLE`` keeps their files, so delete the directory to remove one locally.
- Table names: notebooks use logical names ``"<schema>.<table>"``. ``unity_catalog: false`` (local) maps them to
  ``{catalog}__{schema}.{table}``, ``unity_catalog: true`` (Databricks) to ``{catalog}.{schema}.{table}``;
  ``table_overrides`` in the env file win.
- Parameters (:func:`get_param`): Databricks widgets / job parameters; locally ``--<name> <value>`` on the command
  line, then the environment variable ``PRODREC_<NAME>``, then the default.
- MLflow (:func:`mlflow_setup`): local SQLite store under ``mlruns/`` (artifacts in ``mlruns/artifacts``);
  Databricks workspace tracking.
- File bundles (:func:`make_staging_dir`, :func:`publish_directory`): a directory is built under a staging name and
  swapped in only when complete, repairing what an interrupted earlier swap left behind.
"""

from __future__ import annotations

import argparse
import calendar
import glob
import os
import re
import shutil
import sys
import tempfile
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import l1_contract
import yaml

NOTEBOOKS_DIR = Path(__file__).resolve().parent

# Local tool layout under paths.tools_dir (provisioned by the archived scripts/setup_local_env.ps1).
_JDK_DIR = "jdk-21"
_HADOOP_DIR = "hadoop"
# Delta replays a table's log with 50 partitions by default; on local[*] that dominates small-table operations.
_DELTA_SNAPSHOT_PARTITIONS = 4
_SCRATCH_ROOT = "prodrec-spark"
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


# ---------------------------------------------------------------------------------------------------------------
# Environment, configuration and parameters
# ---------------------------------------------------------------------------------------------------------------

def is_databricks() -> bool:
    """True inside a Databricks runtime."""
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """The repository root: the first directory upwards from this file, then from the cwd, with conf/products.yaml."""
    for start in (NOTEBOOKS_DIR, Path.cwd().resolve()):
        for candidate in (start, *start.parents):
            if (candidate / "conf" / "products.yaml").is_file():
                return candidate
    raise FileNotFoundError(f"No conf/products.yaml above {NOTEBOOKS_DIR} or {Path.cwd()}")


def _read_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_conf(name: str) -> dict:
    """``conf/<name>.yaml`` as a dict (e.g. ``"products"``, ``"training"``, ``"layer2"``)."""
    if name == "training":
        path = Path(get_param("training_config", "conf/training.yaml"))
        conf = _read_yaml(path if path.is_absolute() else repo_root() / path)
        l1_contract.validate_config(conf)
        return conf
    return _read_yaml(repo_root() / "conf" / f"{name}.yaml")


def load_env(env: str) -> dict:
    """``conf/env/<env>.yaml`` as a dict."""
    conf = _read_yaml(repo_root() / "conf" / "env" / f"{env}.yaml")
    if conf.get("env") != env:
        raise ValueError(f"conf/env/{env}.yaml declares env {conf.get('env')!r}")
    return conf


def _dbutils():
    try:
        from IPython import get_ipython

        namespace = get_ipython().user_ns
        if "dbutils" in namespace:
            return namespace["dbutils"]
    except Exception:  # not in an IPython kernel
        pass
    from pyspark.dbutils import DBUtils
    from pyspark.sql import SparkSession

    return DBUtils(SparkSession.builder.getOrCreate())


def get_param(name: str, default=None):
    """A notebook parameter as a string, or ``default`` when it is not given (or given as an empty string).

    Databricks: registers a text widget with the default, then reads the widget (job parameters win); a failure to
    register is tolerated only when the widget already exists (e.g. as another widget type), and a failure to read
    raises. Locally: ``--<name> <value>`` / ``--<name>=<value>`` on the command line, then ``PRODREC_<NAME>``, then
    ``default``. Callers convert the type (``float(get_param("sample_fraction", 1.0))``).
    """
    if is_databricks():
        dbutils = _dbutils()
        try:
            dbutils.widgets.text(name, "" if default is None else str(default))
        except Exception as error:
            try:
                dbutils.widgets.get(name)  # exists already (possibly as another type): keep it
            except Exception:
                raise RuntimeError(f"cannot define widget {name!r}") from error
        try:
            value = dbutils.widgets.get(name)
        except Exception as error:
            raise RuntimeError(f"cannot read widget {name!r}") from error
    else:
        parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        parser.add_argument(f"--{name}", dest="value", default=None)
        args, _ = parser.parse_known_args(sys.argv[1:])
        value = args.value if args.value is not None else os.environ.get(f"PRODREC_{name.upper()}")
    return default if value is None or value == "" else value


# ---------------------------------------------------------------------------------------------------------------
# Spark session (local: Delta + in-memory catalog rehydrated from the warehouse)
# ---------------------------------------------------------------------------------------------------------------

def get_spark(env_conf: dict):
    """The SparkSession: the cluster's on Databricks; locally created on first call (see the module docstring)."""
    platform = env_conf.get("platform")
    if is_databricks():
        if platform != "databricks":
            raise RuntimeError(f"env {env_conf.get('env')!r} has platform {platform!r}, but this is Databricks")
        from pyspark.sql import SparkSession

        return SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    if platform != "local":
        raise RuntimeError(f"env {env_conf.get('env')!r} has platform {platform!r}; locally use a local env file")
    return _get_local_spark(env_conf)


def _is_local(env_conf: dict) -> bool:
    return env_conf.get("platform") == "local"


def _prepend_env_path(var: str, directory: Path) -> None:
    entries = [e for e in os.environ.get(var, "").split(os.pathsep) if e]
    if os.path.normcase(os.path.normpath(str(directory))) not in {os.path.normcase(os.path.normpath(e))
                                                                  for e in entries}:
        os.environ[var] = os.pathsep.join([str(directory), *entries])


def _prepare_local_environment(env_conf: dict) -> None:
    """Set JAVA_HOME, HADOOP_HOME, PATH, PYSPARK_PYTHON, PYTHONPATH and SPARK_LOCAL_IP when they are missing.

    Values already set (e.g. by scripts/env.ps1) are kept. PYTHONPATH gets notebooks/ so the Python workers can import
    ``spark_worker``. SPARK_LOCAL_IP avoids a slow host name lookup on Windows (about 15 s per JVM start).
    """
    tools = repo_root() / env_conf["paths"]["tools_dir"]
    if not os.environ.get("JAVA_HOME"):
        jdk = tools / _JDK_DIR
        if (jdk / "bin").is_dir():
            os.environ["JAVA_HOME"] = str(jdk)
        elif shutil.which("java") is None:
            raise RuntimeError(f"No JDK: JAVA_HOME is not set, {jdk} does not exist and java is not on PATH")
    if not os.environ.get("HADOOP_HOME"):
        hadoop = tools / _HADOOP_DIR
        if (hadoop / "bin").is_dir():
            os.environ["HADOOP_HOME"] = str(hadoop)
        elif os.name == "nt":
            raise RuntimeError(f"HADOOP_HOME is not set and {hadoop} (winutils.exe, hadoop.dll) does not exist")
    for var in ("HADOOP_HOME", "JAVA_HOME"):  # JDK bin ends up first, as in scripts/env.ps1
        if os.environ.get(var):
            _prepend_env_path("PATH", Path(os.environ[var]) / "bin")
    if not os.environ.get("PYSPARK_PYTHON"):
        os.environ["PYSPARK_PYTHON"] = sys.executable
    _prepend_env_path("PYTHONPATH", NOTEBOOKS_DIR)
    if not os.environ.get("SPARK_LOCAL_IP"):
        os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"


def _get_local_spark(env_conf: dict):
    from delta import configure_spark_with_delta_pip
    from pyspark.sql import SparkSession

    _prepare_local_environment(env_conf)
    spark_conf = env_conf.get("spark") or {}
    warehouse = (repo_root() / env_conf["paths"]["warehouse_dir"]).resolve()
    warehouse.mkdir(parents=True, exist_ok=True)
    scratch = _process_scratch_dir()
    builder = (
        SparkSession.builder.master(str(spark_conf.get("master", "local[*]")))
        .appName(f"prodrec-{env_conf['env']}")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.catalogImplementation", "in-memory")
        .config("spark.sql.warehouse.dir", warehouse.as_uri())
        .config("spark.driver.memory", str(spark_conf.get("driver_memory", "4g")))
        .config("spark.sql.shuffle.partitions", str(spark_conf.get("shuffle_partitions", 32)))
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.ansi.enabled", "true")
        .config("spark.databricks.delta.snapshotPartitions", str(_DELTA_SNAPSHOT_PARTITIONS))
        .config("spark.ui.showConsoleProgress", "false")
        # JVM temp files (JAR copies, native libraries) go to a per-process directory: Windows keeps them locked
        # until the JVM exits, so Spark cannot delete them; a later session removes the directory instead.
        .config("spark.driver.extraJavaOptions", f"-Djava.io.tmpdir={_java_option_value(str(scratch))}")
    )
    if spark_conf.get("driver_max_result_size") is not None:  # absent: Spark's default (1g)
        builder = builder.config("spark.driver.maxResultSize", str(spark_conf["driver_max_result_size"]))
    if os.name == "nt":
        builder = builder.config("spark.python.worker.module", "spark_worker")  # see notebooks/spark_worker.py
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    # getOrCreate() reuses a session already running in this process and ignores static settings for it.
    # Default None, not "": Spark validates a non-null default against the option list of catalogImplementation.
    extensions = spark.conf.get("spark.sql.extensions", None) or ""
    catalog_implementation = spark.conf.get("spark.sql.catalogImplementation", None)
    if "DeltaSparkSessionExtension" not in extensions or catalog_implementation != "in-memory":
        raise RuntimeError(f"A Spark session without the Delta / in-memory catalog setup is already running in this "
                           f"process (spark.sql.extensions={extensions!r}, "
                           f"spark.sql.catalogImplementation={catalog_implementation!r}); restart the process")
    if _warehouse_path(spark) != warehouse:
        raise RuntimeError(f"A Spark session with warehouse {_warehouse_path(spark)} is already running in this "
                           f"process; it cannot switch to {warehouse}")
    if os.name == "nt":
        _silence_temp_dir_cleanup_errors(spark)
    registered = _rehydrate_catalog(spark, warehouse)
    print(f"Local Spark {spark.version}: {len(registered)} Delta tables registered from {warehouse}")
    return spark


def _java_option_value(value: str) -> str:
    # Spark's launcher splits extraJavaOptions like a shell: backslash escapes, double quotes group.
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"' if any(ch.isspace() for ch in value) else escaped


def _silence_temp_dir_cleanup_errors(spark) -> None:
    # On Windows, Spark's own cleanup of its temp directory fails at stop and at JVM exit (locked JARs) and logs a
    # stack trace each time; _process_scratch_dir removes the directory once the process is gone.
    jvm = spark.sparkContext._jvm
    off = jvm.org.apache.logging.log4j.Level.OFF
    for name in ("org.apache.spark.SparkEnv", "org.apache.spark.util.ShutdownHookManager"):
        jvm.org.apache.logging.log4j.core.config.Configurator.setLevel(name, off)


def _process_scratch_dir() -> Path:
    """``<temp>/prodrec-spark/<pid>`` for this process, after removing directories of processes that have exited."""
    root = Path(tempfile.gettempdir()) / _SCRATCH_ROOT
    root.mkdir(parents=True, exist_ok=True)
    for child in root.iterdir():
        owner = int(child.name) if child.is_dir() and child.name.isdigit() else None
        if owner is not None and owner != os.getpid() and not _pid_alive(owner):
            shutil.rmtree(child, ignore_errors=True)  # leftovers are retried at the next start
    mine = root / str(os.getpid())
    mine.mkdir(exist_ok=True)
    return mine


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        process_query_limited_information, error_access_denied, still_active = 0x1000, 5, 259
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return ctypes.get_last_error() == error_access_denied  # exists but belongs to someone else
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _warehouse_path(spark) -> Path:
    """The session's ``spark.sql.warehouse.dir`` as a local path."""
    uri = spark.conf.get("spark.sql.warehouse.dir")
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(url2pathname(parsed.path))
    if len(parsed.scheme) <= 1:  # a plain path; a one-letter "scheme" is a Windows drive
        return Path(uri)
    raise ValueError(f"spark.sql.warehouse.dir is not a local path: {uri}")


def _quote(name: str) -> str:
    """Backtick-quote a dotted SQL name part by part."""
    return ".".join("`" + part.strip("`").replace("`", "``") + "`" for part in name.split("."))


def _is_delta_table(path: Path) -> bool:
    log = path / "_delta_log"
    return log.is_dir() and any(e.suffix == ".json" or e.name == "_last_checkpoint" for e in log.iterdir())


def _register(spark, database: str, table: str, location: Path) -> None:
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {_quote(database)}")
    spark.sql(f"CREATE TABLE IF NOT EXISTS {_quote(database)}.{_quote(table)} "
              f"USING DELTA LOCATION '{location.resolve().as_uri()}'")


def _rehydrate_catalog(spark, warehouse: Path) -> list[str]:
    """Register every database directory and Delta table of the warehouse; returns ``"db.table"`` names."""
    registered: list[str] = []
    for db_dir in sorted(p for p in warehouse.iterdir() if p.is_dir() and p.name.endswith(".db")):
        database = db_dir.name[: -len(".db")]
        spark.sql(f"CREATE DATABASE IF NOT EXISTS {_quote(database)}")
        for table_dir in sorted(p for p in db_dir.iterdir() if _is_delta_table(p)):
            _register(spark, database, table_dir.name, table_dir)
            registered.append(f"{database}.{table_dir.name}")
    return registered


def register_local_table(spark, physical_name: str) -> bool:
    """Register local ``db.table`` when it exists on disk but not in this session's catalog (created by another
    process after this session started). Returns True when it registered."""
    parts = physical_name.split(".")
    if len(parts) != 2 or spark.catalog.tableExists(physical_name):
        return False
    database, table = parts
    location = _warehouse_path(spark) / f"{database}.db" / table
    if not _is_delta_table(location):
        return False
    _register(spark, database, table, location)
    return True


# ---------------------------------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------------------------------

def table_name(logical: str, env_conf: dict) -> str:
    """Physical name of ``"<schema>.<table>"``: local ``{catalog}__{schema}.{table}``, Unity Catalog
    ``{catalog}.{schema}.{table}``; ``table_overrides`` win."""
    overrides = env_conf.get("table_overrides") or {}
    parts = logical.split(".") if isinstance(logical, str) else []
    if len(parts) != 2 or not all(_IDENTIFIER.fullmatch(p) for p in parts):
        raise ValueError(f"Invalid logical table name {logical!r}: expected '<schema>.<table>'")
    schema, table = parts
    if schema == "synthetic_truth" and not env_conf.get("allow_synthetic_data"):
        raise ValueError("synthetic_truth tables are not available: allow_synthetic_data is false")
    if logical in overrides:
        return overrides[logical]
    catalog = str(env_conf.get("catalog"))
    if not _IDENTIFIER.fullmatch(catalog):
        raise ValueError(f"catalog {catalog!r} of env {env_conf.get('env')!r} is not set to a valid name")
    return f"{catalog}.{schema}.{table}" if env_conf.get("unity_catalog") else f"{catalog}__{schema}.{table}"


def read_table(spark, logical: str, env_conf: dict, version_as_of: int | None = None):
    """Spark DataFrame of ``logical``, optionally pinned to a Delta version (never falls back to latest)."""
    physical = table_name(logical, env_conf)
    if _is_local(env_conf):
        register_local_table(spark, physical)
    if version_as_of is not None:
        if isinstance(version_as_of, bool) or not isinstance(version_as_of, int) or version_as_of < 0:
            raise ValueError(f"Delta version must be a non-negative integer, got {version_as_of!r}")
        return spark.read.option("versionAsOf", version_as_of).table(_quote(physical))
    return spark.table(physical)


def delta_version(spark, logical: str, env_conf: dict) -> int:
    """Capture the latest committed Delta version after local catalog registration; history errors propagate."""
    physical = table_name(logical, env_conf)
    if _is_local(env_conf):
        register_local_table(spark, physical)
    history = spark.sql(f"DESCRIBE HISTORY {_quote(physical)} LIMIT 1").collect()
    if not history or history[0]["version"] is None:
        raise RuntimeError(f"No Delta version found for {physical}")
    return int(history[0]["version"])


def table_exists(spark, logical: str, env_conf: dict) -> bool:
    """True when the table with logical name ``logical`` exists (locally: after registering it from disk)."""
    physical = table_name(logical, env_conf)
    if _is_local(env_conf):
        register_local_table(spark, physical)
    return bool(spark.catalog.tableExists(physical))


def delete_rows(spark, logical: str, env_conf: dict, predicate: str) -> bool:
    """``DELETE FROM <table> WHERE <predicate>`` on a Delta table; returns False (and does nothing) when the table
    does not exist."""
    if not table_exists(spark, logical, env_conf):
        return False
    spark.sql(f"DELETE FROM {_quote(table_name(logical, env_conf))} WHERE {predicate}")
    return True


def write_table(sdf, logical: str, env_conf: dict, mode: str = "overwrite", replace_where: str | None = None,
                partition_by: list[str] | None = None, merge_schema: bool = False) -> str:
    """Write ``sdf`` to the Delta table ``logical`` (saveAsTable); returns the physical name.

    ``mode="overwrite"`` without ``replace_where`` replaces content and schema (``overwriteSchema``); with
    ``replace_where`` (SQL predicate, e.g. ``"run_date = '2026-09-30'"``) only that slice is replaced and the schema
    must match, unless ``merge_schema`` lets this write add new columns (``mergeSchema`` on this writer only);
    ``mode="append"`` appends.
    """
    if mode not in ("overwrite", "append"):
        raise ValueError(f"mode must be 'overwrite' or 'append', got {mode!r}")
    if replace_where is not None and mode != "overwrite":
        raise ValueError("replace_where requires mode='overwrite'")
    spark = sdf.sparkSession
    physical = table_name(logical, env_conf)
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {_quote(physical.rsplit('.', 1)[0])}")
    if _is_local(env_conf):  # a table another process created must be overwritten in place, not re-created
        register_local_table(spark, physical)
    writer = sdf.write.format("delta").mode(mode)
    if merge_schema:
        writer = writer.option("mergeSchema", "true")
    if replace_where is not None:
        writer = writer.option("replaceWhere", replace_where)
    elif mode == "overwrite":
        writer = writer.option("overwriteSchema", "true")
    if partition_by:
        writer = writer.partitionBy(*partition_by)
    writer.saveAsTable(physical)
    if _is_local(env_conf) and not spark.catalog.tableExists(physical):
        raise RuntimeError(f"{physical} was written but is not in the local catalog")
    return physical


def to_pandas(sdf, columns: list[str] | None = None):
    """pandas DataFrame of ``sdf`` (optionally only ``columns``), through Arrow."""
    return (sdf.select(*columns) if columns is not None else sdf).toPandas()


def sample_clients(sdf, fraction: float, key: str = "client_id"):
    """Keep the clients with ``pmod(crc32(key), 10000) < fraction * 10000`` (whole time series per client)."""
    from pyspark.sql import functions as F

    fraction = float(fraction)
    if not 0 < fraction <= 1:
        raise ValueError(f"sample fraction must be in (0, 1], got {fraction}")
    if fraction >= 1:
        return sdf
    bucket = F.pmod(F.crc32(F.col(key).cast("string").cast("binary")), F.lit(10000))
    return sdf.filter(bucket < F.lit(fraction * 10000))


# ---------------------------------------------------------------------------------------------------------------
# File bundles (notebooks 02 and 03): build in a staging directory, then swap it in
# ---------------------------------------------------------------------------------------------------------------

def make_staging_dir(target_dir) -> Path:
    """An empty ``<target>.staging-<pid>`` directory next to ``target_dir`` (parents created)."""
    target_dir = Path(target_dir)
    staging = target_dir.with_name(f"{target_dir.name}.staging-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    return staging


def publish_directory(staging_dir, target_dir) -> None:
    """Replace ``target_dir`` by the complete ``staging_dir``.

    First repairs what an interrupted earlier publish left behind: ``target`` missing but ``<target>.old`` present
    (stopped between the two renames) -> ``.old`` is renamed back to ``target``; ``<target>.staging-*`` directories of
    any process (other than ``staging_dir``) are deleted; a leftover ``.old`` next to an existing ``target`` is
    deleted. Then ``target`` -> ``.old``, ``staging`` -> ``target``, ``.old`` deleted. When the second rename fails,
    ``.old`` is renamed back to ``target`` and the error is raised. One publisher per target at a time: publishing
    the same target (release) concurrently is not supported.
    """
    staging_dir, target_dir = Path(staging_dir), Path(target_dir)
    old = target_dir.with_name(f"{target_dir.name}.old")
    if not target_dir.exists() and old.exists():
        old.rename(target_dir)
    for stale in target_dir.parent.glob(f"{glob.escape(target_dir.name)}.staging-*"):
        if stale.is_dir() and stale.resolve() != staging_dir.resolve():
            shutil.rmtree(stale)
    if old.exists():
        shutil.rmtree(old)
    if target_dir.exists():
        target_dir.rename(old)
    try:
        staging_dir.rename(target_dir)
    except BaseException:
        if old.exists() and not target_dir.exists():
            old.rename(target_dir)
        raise
    shutil.rmtree(old, ignore_errors=True)


# ---------------------------------------------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------------------------------------------

def _resolve_tracking_uri(uri: str) -> str:
    """A ``sqlite:///<relative path>`` URI resolved against the repository root (parent directory created)."""
    prefix = "sqlite:///"
    if not uri.startswith(prefix):
        return uri
    path = Path(uri[len(prefix):])
    if not path.is_absolute():
        path = repo_root() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return prefix + path.resolve().as_posix()


def mlflow_setup(env_conf: dict, experiment_logical: str) -> str:
    """Point MLflow at the env's tracking store, create/select ``<experiment_root>/<experiment_logical>``; returns
    the experiment id. Local SQLite experiments store artifacts in ``mlruns/artifacts`` next to the database."""
    import mlflow

    conf = env_conf["mlflow"]
    uri = _resolve_tracking_uri(str(conf["tracking_uri"]))
    mlflow.set_tracking_uri(uri)
    name = str(conf["experiment_root"]).rstrip("/") + "/" + experiment_logical
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is not None:
        experiment_id = experiment.experiment_id
    else:
        artifact_location = None
        if uri.startswith("sqlite:///"):
            artifact_location = (Path(uri[len("sqlite:///"):]).parent / "artifacts").as_uri()
        elif is_databricks():  # workspace experiments need their parent folder
            from databricks.sdk import WorkspaceClient

            WorkspaceClient().workspace.mkdirs(name.rsplit("/", 1)[0])
        experiment_id = mlflow.create_experiment(name, artifact_location=artifact_location)
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


# ---------------------------------------------------------------------------------------------------------------
# Time blocks (docs/MVP_SPEC.md section 5)
# ---------------------------------------------------------------------------------------------------------------

def _as_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _add_months(d: date, months: int) -> date:
    """Calendar months with day clipping (Spark ``add_months``, pandas ``DateOffset(months=...)``)."""
    year, month0 = divmod(d.year * 12 + d.month - 1 + months, 12)
    return date(year, month0 + 1, min(d.day, calendar.monthrange(year, month0 + 1)[1]))


def derive_split_dates(training_conf: dict, snapshot_dates: list) -> dict:
    """Validated observed L1 dates, allocations and actual label availability; shared by 00/01/02."""
    return l1_contract.derive_dates(training_conf, snapshot_dates)


def validate_training_input(sdf, dictionary, products_conf: dict, training_conf: dict,
                            feature_names: list[str]) -> dict:
    """Validate 01's full, unsampled modeling input using schema metadata and Spark statistics.

    ``dictionary`` is bounded column metadata, not customer data. Returns the configured time blocks; upstream
    owns creating the table and must regenerate its split column when the timeline changes.
    """
    from pyspark.sql import functions as F

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def unique_names(names, source):
        lowered = [name.lower() for name in names]
        require(len(lowered) == len(set(lowered)), f"{source} repeats column names (ignoring case)")

    keys = list(products_conf["keys"].values())
    snapshot = products_conf["keys"]["snapshot_date"]
    labels = [products_conf["label_column_template"].format(l2=l2) for l2 in products_conf["products"]]
    eligibility = [products_conf["eligibility_column_template"].format(l2=l2) for l2 in products_conf["products"]]
    excluded = set(products_conf.get("feature_exclusions") or [])
    required = keys + labels + eligibility + ["split", "snapshot_month"]
    unique_names(sdf.columns, "ml.modeling_table")
    unique_names(feature_names, "feature_groups.csv")
    require(len(feature_names) == 143, f"feature_groups.csv has {len(feature_names)} features, expected 143")
    missing = sorted(set(required) - set(sdf.columns))
    require(not missing, f"ml.modeling_table lacks required columns {missing}")
    found_features = set(sdf.columns) - set(required)
    require({name.lower() for name in found_features} == {name.lower() for name in feature_names},
            "ml.modeling_table feature columns differ from feature_groups.csv")

    metadata_columns = ["column_name", "data_type", "role"]
    require(set(metadata_columns) <= set(dictionary.columns), "ml.data_dictionary lacks required metadata columns")
    require(not dictionary[metadata_columns].isna().any().any(), "ml.data_dictionary has NULL metadata")
    require(all(isinstance(value, str) for c in metadata_columns for value in dictionary[c]),
            "ml.data_dictionary metadata must be strings")
    unique_names(dictionary["column_name"].tolist(), "ml.data_dictionary")
    require(set(dictionary["column_name"]) == set(sdf.columns),
            "ml.data_dictionary columns must match ml.modeling_table exactly")
    metadata = dictionary.set_index("column_name")
    numeric = ("tinyint", "smallint", "int", "bigint", "float", "double", "decimal")
    for field in sdf.schema.fields:
        name, dtype = field.name, field.dataType.simpleString()
        role = ("key" if name in keys else "label" if name in labels else
                "eligibility" if name in eligibility else "split" if name == "split" else
                "aux" if name == "snapshot_month" else "excluded_feature" if name in excluded else "feature")
        require(metadata.loc[name, "data_type"] == dtype, f"ml.data_dictionary type differs for {name}: {dtype}")
        require(metadata.loc[name, "role"] == role, f"ml.data_dictionary role differs for {name}: expected {role}")
        if role == "feature":
            require(name.lower() not in {c.lower() for c in keys + ["CLIENT_JOINDATE"]}
                    and not name.lower().startswith(("y_", "elig_", "p_true_")),
                    f"non-feature column {name} has feature role")
            require(dtype == "string" or dtype.startswith(numeric), f"unsupported feature type {name}: {dtype}")
        if role in ("label", "eligibility"):
            require(dtype == "boolean" or dtype.startswith(numeric), f"unsupported {role} type {name}: {dtype}")
        if name == snapshot:
            require(dtype == "date", f"{snapshot} must have date type, got {dtype}")
        if name == "split":
            require(dtype == "string", f"split must have string type, got {dtype}")

    null_keys = F.col(keys[0]).isNull() | F.col(keys[1]).isNull()
    require(not sdf.filter(null_keys).limit(1).count(), f"ml.modeling_table has NULL keys {keys}")
    dates = [r[0] for r in sdf.select(snapshot).distinct().collect()]
    splits = derive_split_dates(training_conf, dates)

    expected = F.when(F.col(snapshot).isin(splits["train"]), "train")
    for block, block_dates in (("gap", splits["gap1"] + splits["gap2"]),
                               ("calibration", splits["calibration"]), ("test", splits["test"]),
                               ("immature", splits["immature"]), ("unused", splits["unused"])):
        expected = expected.when(F.col(snapshot).isin(block_dates), block)
    mature = F.col(snapshot) <= F.lit(splits["last_mature"])
    def binary_column(name):
        column = F.col(name)
        return column.cast("int") if sdf.schema[name].dataType.simpleString() == "boolean" else column

    checks = [("date-to-split agreement", ~F.col("split").eqNullSafe(expected))]
    checks += [(f"labels {name}", (mature & (F.col(name).isNull() | ~binary_column(name).isin(0, 1)))
                | (~mature & F.col(name).isNotNull())) for name in labels]
    checks += [(f"eligibility {name}", F.col(name).isNull() | ~binary_column(name).isin(0, 1)) for name in eligibility]
    stats = sdf.agg(*[F.sum(F.when(condition, 1).otherwise(0)).alias(f"invalid_{i}")
                      for i, (_, condition) in enumerate(checks)]).first()
    failures = {name: stats[f"invalid_{i}"] for i, (name, _) in enumerate(checks) if stats[f"invalid_{i}"]}
    require(not failures, f"ml.modeling_table invalid input: {failures}")
    require(not sdf.groupBy(*keys).count().filter(F.col("count") > 1).limit(1).count(),
            f"ml.modeling_table has duplicate {keys} keys")
    return splits
