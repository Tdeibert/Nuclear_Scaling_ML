"""
nsdb -- read access to the nuclear scaling database.

Intended use from a notebook:

    import nsdb
    df = nsdb.nuclei(qc="PASS")

Connections default to read-only, so nothing in an analysis notebook can
mutate the imported data. Writing is the import pipeline's job.

Configuration:
    NUCLEAR_SCALING_DB    path to the .db file
    NUCLEAR_SCALING_ROOT  local data root, used to remap absolute paths that
                          were recorded on Cheaha (see resolve_path)
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

# Layout is <root>/src/nsdb.py, so the project root is one level up. Deriving
# the defaults from __file__ rather than $HOME means the tree can be moved or
# cloned to another machine without editing code.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT_ROOT / "data" / "db" / "nuclear_scaling.db"
DEFAULT_VIEWS_SQL = PROJECT_ROOT / "sql" / "views.sql"
VIEWS = ("v_nuclei", "v_nuclei_intensity", "v_intensity_long", "v_zstack")


# ---------------------------------------------------------------------------
# connection
# ---------------------------------------------------------------------------

def db_path(path: str | Path | None = None) -> Path:
    """Resolve the database path: explicit arg > env var > default."""
    p = Path(path or os.environ.get("NUCLEAR_SCALING_DB") or DEFAULT_DB)
    if not p.exists():
        raise FileNotFoundError(
            f"No database at {p}. Pass path= or set NUCLEAR_SCALING_DB."
        )
    return p


def connect(path: str | Path | None = None, read_only: bool = True) -> sqlite3.Connection:
    """Open the database. Read-only by default.

    SQLite will not create a missing file in ro mode, so a typo in the path
    raises instead of silently handing back an empty database -- which is the
    failure mode that costs you an afternoon.
    """
    p = db_path(path)
    if read_only:
        con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    else:
        con = sqlite3.connect(p)
    con.execute("PRAGMA foreign_keys = ON")
    return con


def query(sql: str, params: tuple | dict = (), path=None, con=None) -> pd.DataFrame:
    """Run SQL, return a DataFrame.

    Always pass values through `params` rather than f-stringing them into
    `sql` -- it is the difference between a quoted string and a syntax error
    the first time an experiment_id contains an apostrophe.
    """
    if con is not None:
        return pd.read_sql_query(sql, con, params=params)
    with closing(connect(path)) as c:
        return pd.read_sql_query(sql, c, params=params)


# ---------------------------------------------------------------------------
# schema setup (the one function that needs write access)
# ---------------------------------------------------------------------------

def install_views(path=None, sql_file: str | Path | None = None) -> list[str]:
    """(Re)create the analysis views. Run once per database file."""
    script = Path(sql_file or DEFAULT_VIEWS_SQL).read_text()
    with closing(connect(path, read_only=False)) as con:
        con.executescript(script)
        con.commit()
        found = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='view' ORDER BY name")]
    missing = set(VIEWS) - set(found)
    if missing:
        raise RuntimeError(f"views failed to install: {sorted(missing)}")
    return found


def check_views(path=None) -> bool:
    """True if the analysis views are present in this database file."""
    found = query(
        "SELECT name FROM sqlite_master WHERE type='view'", path=path
    )["name"].tolist()
    return set(VIEWS).issubset(found)


# ---------------------------------------------------------------------------
# the queries you will actually call
# ---------------------------------------------------------------------------

def experiments(path=None) -> pd.DataFrame:
    """Every (experiment, fov, droplet) in the database with its metadata.

    Run this first in any notebook -- it is the inventory of what you have,
    and it shows immediately whether condition columns are populated.
    """
    return query("SELECT * FROM experimental_cfg ORDER BY experiment_id, fov_id, droplet_id",
                 path=path)


def nuclei(
    experiment: str | list[str] | None = None,
    group: str | list[str] | None = None,
    qc: str | list[str] | None = "PASS",
    intensity: bool = False,
    path=None,
) -> pd.DataFrame:
    """Nuclei rows with experimental metadata attached.

    experiment  one or more experiment_id values; None = all
    group       one or more experimental_group values; None = all
    qc          qc_flag filter; default "PASS". Pass None to get everything,
                including FAIL -- useful when auditing, wrong for results.
    intensity   include the halo intensity columns
    """
    view = "v_nuclei_intensity" if intensity else "v_nuclei"
    where, params = _filters(
        {"experiment_id": experiment, "experimental_group": group, "qc_flag": qc}
    )
    return query(f"SELECT * FROM {view}{where} ORDER BY experiment_id, nucleus_id, time_frame",
                 params, path=path)


def zstack(experiment=None, qc="PASS", selected_only: bool = False, path=None) -> pd.DataFrame:
    """Per-plane z-stack rows with metadata."""
    filt = {"experiment_id": experiment, "qc_flag": qc}
    if selected_only:
        filt["is_selected_max"] = 1
    where, params = _filters(filt)
    return query(f"SELECT * FROM v_zstack{where} ORDER BY nucleus_id, time_frame, slice_id",
                 params, path=path)


def intensity_long(experiment=None, channel=None, qc="PASS", path=None) -> pd.DataFrame:
    """Tidy intensity table joined to nucleus metadata.

    Background-subtracted values are left to the caller: subtracting a NULL
    background should be a visible decision, not a silent one.
    """
    where, params = _filters(
        {"v.experiment_id": experiment, "v.qc_flag": qc, "i.channel": channel}
    )
    sql = f"""
        SELECT i.*, v.experiment_id, v.fov_id, v.droplet_id,
               v.experimental_group, v.treatment, v.biological_replicate,
               v.time_min, v.cross_sectional_area_um2
        FROM v_intensity_long i
        JOIN v_nuclei v
          ON i.nucleus_id = v.nucleus_id AND i.time_frame = v.time_frame
        {where}
        ORDER BY i.nucleus_id, i.time_frame, i.channel, i.halo
    """
    return query(sql, params, path=path)


# ---------------------------------------------------------------------------
# grouping by experimental condition
# ---------------------------------------------------------------------------

# What defines "a condition". Order matters: these become the group key order.
CONDITION_KEYS = ("experimental_group", "treatment", "concentration", "concentration_units")

# What defines "a replicate within a condition". These are the unit of
# biological replication -- see replicate_means().
REPLICATE_KEYS = ("biological_replicate", "extract_batch", "technical_replicate")

# Identity of a single imaged field. Falls back to this when nothing else is
# populated, so grouping never silently collapses distinct acquisitions.
ACQUISITION_KEYS = ("experiment_id", "fov_id", "droplet_id")


def condition_keys(df: pd.DataFrame, replicate: bool = False,
                   time: bool = False, strict: bool = False) -> list[str]:
    """Resolve which columns to group by, given what is actually populated.

    A column that exists but is entirely NULL is dropped: grouping on it adds
    nothing and, with pandas' default dropna=True, would silently discard every
    row. If no condition column is populated at all, falls back to
    experiment_id so you get per-acquisition groups rather than one lump.

    strict=True raises instead of falling back -- use it in a results notebook
    where an unlabelled dataset means someone forgot to fill in the importer.
    """
    keys = [k for k in CONDITION_KEYS if k in df.columns and df[k].notna().any()]
    if not keys:
        if strict:
            raise ValueError(
                "No condition columns populated in experimental_cfg "
                f"({', '.join(CONDITION_KEYS)}). Cannot group by condition."
            )
        keys = [k for k in ACQUISITION_KEYS if k in df.columns][:1]
    if replicate:
        keys += [k for k in REPLICATE_KEYS if k in df.columns and df[k].notna().any()]
    if time:
        keys += [k for k in ("time_frame",) if k in df.columns]
    return keys


def by_condition(df: pd.DataFrame, replicate: bool = False, time: bool = False,
                 strict: bool = False, **kwargs):
    """GroupBy over experimental condition.

    dropna=False and observed=True are set deliberately: a NULL treatment is a
    real group you need to see, not rows to throw away.
    """
    keys = condition_keys(df, replicate=replicate, time=time, strict=strict)
    return df.groupby(keys, dropna=False, observed=True, **kwargs)


def summarise(df: pd.DataFrame, value: str = "cross_sectional_area_um2",
              replicate: bool = False, time: bool = True,
              strict: bool = False) -> pd.DataFrame:
    """Per-condition summary of one measurement.

    Returns n, mean, sd, sem, median and quartiles. n counts non-null values of
    `value`, which is not always the number of rows -- volume_3d_um3 is
    currently NULL everywhere, and a summary that reported n=1096 for it would
    be lying.
    """
    if value not in df.columns:
        raise KeyError(f"{value!r} not in dataframe; have {list(df.columns)}")
    keys = condition_keys(df, replicate=replicate, time=time, strict=strict)
    g = df.groupby(keys, dropna=False, observed=True)[value]
    out = g.agg(
        n="count",
        mean="mean",
        sd="std",
        median="median",
        q25=lambda s: s.quantile(0.25),
        q75=lambda s: s.quantile(0.75),
        min="min",
        max="max",
    ).reset_index()
    out.insert(out.columns.get_loc("sd") + 1, "sem", out["sd"] / out["n"] ** 0.5)
    out.attrs["value"] = value
    out.attrs["group_keys"] = keys
    return out


def replicate_means(df: pd.DataFrame, value: str = "cross_sectional_area_um2",
                    time: bool = True, strict: bool = False) -> pd.DataFrame:
    """Collapse nuclei to one value per replicate, per condition, per timepoint.

    Two-stage aggregation. Nuclei within one extract prep are not independent
    samples of "what this treatment does" -- they share a droplet population, a
    batch, and a segmentation run. Statistics computed on the output of this
    function have the extract as the unit of replication, which is almost
    always the claim you actually want to make.

    Use summarise() to describe the nucleus-level distribution; use this as the
    input to a test comparing conditions.
    """
    rep = [k for k in REPLICATE_KEYS if k in df.columns and df[k].notna().any()]
    if not rep:
        if strict:
            raise ValueError(
                "No replicate column populated -- cannot collapse to replicate "
                "means. Populate biological_replicate in experimental_cfg."
            )
        rep = [k for k in ACQUISITION_KEYS if k in df.columns][:1]

    cond = condition_keys(df, time=time, strict=strict)
    keys = cond + [k for k in rep if k not in cond]
    per_rep = (
        df.groupby(keys, dropna=False, observed=True)[value]
        .agg(n_nuclei="count", replicate_mean="mean")
        .reset_index()
    )
    per_rep.attrs["value"] = value
    per_rep.attrs["condition_keys"] = cond
    per_rep.attrs["replicate_keys"] = rep
    return per_rep


def across_replicates(per_rep: pd.DataFrame) -> pd.DataFrame:
    """Summarise replicate_means() output across replicates.

    n here is the number of replicates, not the number of nuclei. That is the
    n you should be reporting in a figure legend.
    """
    cond = per_rep.attrs.get("condition_keys")
    if cond is None:
        raise ValueError("pass the output of replicate_means()")
    g = per_rep.groupby(cond, dropna=False, observed=True)["replicate_mean"]
    out = g.agg(n_replicates="count", mean="mean", sd="std").reset_index()
    out["sem"] = out["sd"] / out["n_replicates"] ** 0.5
    out["n_nuclei"] = (
        per_rep.groupby(cond, dropna=False, observed=True)["n_nuclei"].sum().values
    )
    return out


# ---------------------------------------------------------------------------
# radial profiles: pointer in SQLite, bulk rows in parquet
# ---------------------------------------------------------------------------

def resolve_path(recorded: str) -> Path:
    """Map a path recorded at import time onto this machine.

    Paths in radial_profile_files are absolute and were written wherever the
    import ran. NUCLEAR_SCALING_ROOT lets the same database work on StarForge,
    Starkillerbase, and Cheaha without rewriting rows.
    """
    p = Path(recorded)
    root = os.environ.get("NUCLEAR_SCALING_ROOT")
    if root and not p.exists():
        for anchor in ("Nuclear_Scaling", "derived", "radial_profiles"):
            if anchor in p.parts:
                idx = p.parts.index(anchor)
                candidate = Path(root).joinpath(*p.parts[idx + 1:])
                if candidate.exists():
                    return candidate
    return p


def radial_files(experiment=None, path=None) -> pd.DataFrame:
    """The parquet sidecar registry. Row counts and sizes without loading."""
    where, params = _filters({"experiment_id": experiment})
    return query(f"SELECT * FROM radial_profile_files{where} ORDER BY created_at",
                 params, path=path)


def radial(experiment: str, fov: str | None = None, columns=None,
           filters=None, path=None) -> pd.DataFrame:
    """Load radial profile rows from parquet for one experiment.

    These files are millions of rows. Always pass `columns` to limit width and
    `filters` (pyarrow predicate list) to push row selection down into the
    parquet reader rather than loading everything and subsetting in pandas.

        nsdb.radial("control_extract_1.1",
                    columns=["nucleus_id", "time_frame", "theta_deg",
                             "rho_normalized", "intensity"],
                    filters=[("channel", "==", "npc"), ("time_frame", "==", 9)])
    """
    reg = radial_files(experiment, path=path)
    if fov is not None:
        reg = reg[reg["fov_id"] == fov]
    if reg.empty:
        raise LookupError(f"no radial profile files registered for {experiment!r}")

    frames = []
    for _, row in reg.iterrows():
        pq = resolve_path(row["parquet_path"])
        if not pq.exists():
            raise FileNotFoundError(
                f"{pq} is registered in the database but missing on disk. "
                "Set NUCLEAR_SCALING_ROOT or re-sync the derived data."
            )
        part = pd.read_parquet(pq, columns=columns, filters=filters)
        part["experiment_id"] = row["experiment_id"]
        part["fov_id"] = row["fov_id"]
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# integrity checks -- run these before you trust a run
# ---------------------------------------------------------------------------

def audit(path=None) -> pd.DataFrame:
    """Cheap structural checks. Every count should be zero.

    Some will not be zero on data exported before pipeline v18.2 -- the
    z-stack provenance and area-mismatch checks report a known upstream gap,
    not a corrupt import. They are here so the gap stays visible.
    """
    checks = {
        "nuclei with no z-stack rows":
            """SELECT COUNT(*) FROM nuclei n WHERE NOT EXISTS
               (SELECT 1 FROM nucleus_z_stack z
                WHERE z.nucleus_id=n.nucleus_id AND z.time_frame=n.time_frame)""",
        "selected_slice_id absent from z-stack":
            "SELECT COUNT(*) FROM nuclei WHERE zstack_consistency = 'plane_missing'",
        "z-stack rows still marked unrepaired_stale":
            "SELECT COUNT(*) FROM nucleus_z_stack WHERE repair_source = 'unrepaired_stale'",
        "nuclei with no tile assignment":
            "SELECT COUNT(*) FROM nuclei WHERE tile_index IS NULL",
        "nucleus-frames with no is_selected_max flag":
            """SELECT COUNT(*) FROM nuclei n WHERE NOT EXISTS
               (SELECT 1 FROM nucleus_z_stack z
                WHERE z.nucleus_id=n.nucleus_id AND z.time_frame=n.time_frame
                  AND z.is_selected_max=1)""",
        "nuclei with no intensity row":
            """SELECT COUNT(*) FROM nuclei n WHERE NOT EXISTS
               (SELECT 1 FROM raw_intensities r
                WHERE r.nucleus_id=n.nucleus_id AND r.time_frame=n.time_frame)""",
        "nuclei whose selected slice is not flagged is_selected_max":
            """SELECT COUNT(*) FROM nuclei n JOIN nucleus_z_stack z
               ON z.nucleus_id=n.nucleus_id AND z.time_frame=n.time_frame
                  AND z.slice_id=n.selected_slice_id
               WHERE z.is_selected_max != 1""",
        "area disagrees between nuclei and its selected z-stack plane":
            "SELECT COUNT(*) FROM nuclei WHERE zstack_consistency = 'area_mismatch'",
        "nucleus_id reused across acquisitions":
            """SELECT COUNT(*) FROM (SELECT nucleus_id FROM nuclei
               GROUP BY nucleus_id
               HAVING COUNT(DISTINCT experiment_id || '|' || fov_id || '|' || droplet_id) > 1)""",
        "nucleus_id prefix disagrees with its metadata columns":
            """SELECT COUNT(*) FROM nuclei
               WHERE nucleus_id NOT LIKE experiment_id || '|' || fov_id || '|%'""",
        "config rows missing experimental_group":
            "SELECT COUNT(*) FROM experimental_cfg WHERE experimental_group IS NULL",
        "config rows missing biological_replicate":
            "SELECT COUNT(*) FROM experimental_cfg WHERE biological_replicate IS NULL",
        "nuclei still UNASSIGNED to a droplet":
            "SELECT COUNT(*) FROM nuclei WHERE droplet_assignment_status='UNASSIGNED'",
    }
    with closing(connect(path)) as con:
        have = {r[1] for r in con.execute("PRAGMA table_info(nuclei)")}
        need = {"zstack_consistency", "tile_index", "repair_status"}
        if not need.issubset(have):
            raise RuntimeError(
                f"{db_path(path)} was built with an older schema (missing "
                f"{sorted(need - have)}). Rebuild it:\n"
                "  python build_db.py init --db <db> --force\n"
                "  python build_db.py import <export_dir> --config <sidecar.json> "
                "--db <db> --parquet-dir <dir>")
        rows = [(name, con.execute(sql).fetchone()[0]) for name, sql in checks.items()]
    out = pd.DataFrame(rows, columns=["check", "n"])
    out["ok"] = out["n"] == 0
    return out


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------

def _filters(spec: dict) -> tuple[str, list]:
    """Build a parametrised WHERE clause. None values are skipped entirely."""
    clauses, params = [], []
    for col, val in spec.items():
        if val is None:
            continue
        if isinstance(val, (list, tuple, set)):
            val = list(val)
            if not val:
                continue
            clauses.append(f"{col} IN ({','.join('?' * len(val))})")
            params.extend(val)
        else:
            clauses.append(f"{col} = ?")
            params.append(val)
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params
