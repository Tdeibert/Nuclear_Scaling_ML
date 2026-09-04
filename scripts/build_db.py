#!/usr/bin/env python3
"""
build_db -- rebuild the nuclear scaling database from a CSV export directory.

    python scripts/build_db.py inspect data/db/exports

    python scripts/build_db.py build data/db/exports \\
        --config configs/control_extract_1.1.json \\
        --db     data/db/nuclear_scaling.db \\
        --parquet-dir data/derived/radial_profiles

Transformations applied on the way in, each of them deliberate:

* Centroid_XYZ_px is exported as the string "(x, y, z)". It is split into
  centroid_x_px / y / z. Packed coordinates cannot be filtered, indexed or
  aggregated in SQL, so they are unpacked once here rather than re-parsed in
  every downstream query.

* Nucleus_ID is exported as a bare integer that is unique per tracked nucleus,
  not per detection -- 854 distinct values across 1,269 rows in the reference
  export. It becomes:
      nucleus_id           "<experiment_id>|<fov_id>|N000001"  (globally unique)
      nucleus_track_id     the cross-frame identity
      source_detection_id  the raw integer, as the link back to the CSV
  The qualified nucleus_id is what keeps IDs unique across replicates. That
  uniqueness is produced HERE; it is not a property inherited from the export.

* Droplet_ID is entirely null in the reference export. Null becomes the
  'UNASSIGNED' sentinel with droplet_assignment_status to match, so "not
  assigned yet" stays distinguishable from "assigned to nothing".

* Time_Real_Minutes is KEPT and is the primary time axis. The FOV is a
  stitched mosaic scanned tile by tile, so nuclei in different tiles of the
  same frame are imaged minutes apart -- a 3x2 serpentine grid at 1 min/tile
  in the reference export. Deriving time from time_frame would discard that.

* The 81 nuclei whose Selected_Slice_ID is absent from NucleusZStack.csv are
  KEPT, not quarantined. The v17 fragmentation repair re-selects z and
  measures planes that produced no original detection, but writes back only
  to best_z_df -- NucleusZStack is built from the unrepaired grouped_z_df.
  The nuclei row holds the better measurement; the z-stack is stale.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from experiment_config import ExperimentConfig  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = PROJECT_ROOT / "sql" / "schema_v2.sql"
VIEWS = PROJECT_ROOT / "sql" / "views.sql"

CSV_FILES = {
    "experimental_cfg": "Experimental_Cfg.csv",
    "nuclei": "Nuclei.csv",
    "nucleus_z_stack": "NucleusZStack.csv",
    "raw_intensities": "RawIntensities.csv",
    "radial_profile": "RadialProfile.csv",
}

RADIAL_COLS = {"nucleus_id", "time_frame", "theta_deg", "rho_normalized",
               "w_wall_proximity_index", "channel", "intensity",
               "is_ridge_point", "cluster_id"}

# Headers no automatic rule gets right. Digit boundaries are ambiguous:
# "Volume_3D_um3" must become volume_3d_um3, but the same rule on "Halo1_NPC"
# would give halo_1_npc. Map those by hand.
ALIASES = {
    "volume_3_d_um3": "volume_3d_um3",
    "volume3d_um3": "volume_3d_um3",
    "date": "experiment_date",
    "n_c_ratio": "nc_ratio",
}

# Present in the CSV, deliberately not stored.
DROPPED: dict[str, dict[str, str]] = {}

# Present in the CSV, stored in a different shape.
TRANSFORMED = {
    "nuclei": {"centroid_xyz_px": "split into centroid_x_px / y_px / z_px"},
    "nucleus_z_stack": {"centroid_xyz_px": "split into centroid_x_px / y_px / z_px"},
}

NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _norm(name: str) -> str:
    """CamelCase / spaced / dashed header -> snake_case."""
    s = str(name).strip().replace(" ", "_").replace("-", "_")
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", s)
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", s)
    s = re.sub(r"__+", "_", s).lower().strip("_")
    return ALIASES.get(s, s)


def table_columns(con: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in con.execute(f"PRAGMA table_info({table})")]


# ---------------------------------------------------------------------------
# transformations
# ---------------------------------------------------------------------------

def split_xyz(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Unpack a "(x, y, z)" string column into three float columns."""
    if "centroid_xyz_px" not in df.columns:
        return df
    parsed, bad = [], []
    for i, v in df["centroid_xyz_px"].items():
        if pd.isna(v):
            parsed.append((None, None, None))
            continue
        nums = NUM_RE.findall(str(v))
        if len(nums) != 3:
            bad.append((i, v))
            parsed.append((None, None, None))
        else:
            parsed.append(tuple(float(x) for x in nums))
    if bad:
        raise ValueError(
            f"{source}: {len(bad)} Centroid_XYZ_px values did not contain exactly three "
            f"numbers, e.g. row {bad[0][0]}: {bad[0][1]!r}")
    xyz = pd.DataFrame(parsed, index=df.index,
                       columns=["centroid_x_px", "centroid_y_px", "centroid_z_px"])
    return pd.concat([df.drop(columns=["centroid_xyz_px"]), xyz], axis=1)


def qualify_nucleus_id(df: pd.DataFrame, cfg: ExperimentConfig,
                       keep_track: bool) -> pd.DataFrame:
    """Turn a bare integer Nucleus_ID into a globally-unique qualified ID."""
    raw = df["nucleus_id"]
    if raw.dtype == object and raw.astype(str).str.contains(r"\|", regex=True).any():
        return df  # already qualified
    ints = pd.to_numeric(raw, errors="coerce")
    if ints.isna().any():
        raise ValueError(f"{int(ints.isna().sum())} non-numeric Nucleus_ID values")
    ints = ints.astype("int64")
    df = df.copy()
    df["nucleus_id"] = f"{cfg.experiment_id}|{cfg.fov_id}|N" + ints.astype(str).str.zfill(6)
    if keep_track:
        df["source_detection_id"] = ints.astype(str)
        df["nucleus_track_id"] = (f"{cfg.experiment_id}|{cfg.fov_id}|T"
                                  + ints.astype(str).str.zfill(6))
    return df


def derive_tiles(df: pd.DataFrame, cfg: ExperimentConfig, issues: list) -> pd.DataFrame:
    """Recover tile identity by inverting time_real_minutes.

        true_time_min = t * tiles_per_frame + tile_index * minutes_per_tile

    so tile_index = (time_real_minutes - t * tiles_per_frame) / minutes_per_tile.
    This needs no image dimensions, unlike the pipeline's centroid-based
    assignment, so it works from the CSV alone.
    """
    span = cfg.tiles_per_frame
    if span is None or "time_real_minutes" not in df.columns:
        return df
    df = df.copy()
    offset = df["time_real_minutes"] - df["time_frame"] * span
    idx = offset / cfg.minutes_per_tile
    n_tiles = cfg.tile_rows * cfg.tile_cols

    bad = (~idx.between(0, n_tiles - 1)) | ((idx - idx.round()).abs() > 1e-6)
    if bad.any():
        issues.append(("WARN", "tile index not recoverable from time_real_minutes",
                       int(bad.sum()),
                       f"offset outside [0, {n_tiles - 1}] or non-integral. Left NULL. "
                       "Check tile_rows/tile_cols/minutes_per_tile in the sidecar."))
    idx = idx.round().where(~bad)
    row = (idx // cfg.tile_cols)
    base = idx - row * cfg.tile_cols
    col = base if not cfg.serpentine_scan else base.where(row % 2 == 0,
                                                          cfg.tile_cols - 1 - base)
    df["tile_index"] = idx.astype("Int64")
    df["tile_row"] = row.astype("Int64")
    df["tile_col"] = col.astype("Int64")
    return df


def coerce(df: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    """Type and sentinel fixes the SQLite CHECK constraints depend on."""
    df = df.copy()
    for c in ("is_selected_max", "is_ridge_point"):
        if c in df.columns:
            df[c] = df[c].astype(bool).astype(int)
    if "droplet_id" in df.columns:
        assigned = df["droplet_id"].notna()
        df["droplet_id"] = df["droplet_id"].where(assigned, "UNASSIGNED").astype(str)
        df["droplet_assignment_status"] = ["ASSIGNED" if a else "UNASSIGNED" for a in assigned]
    for c, v in (("experiment_id", cfg.experiment_id), ("fov_id", cfg.fov_id)):
        if c in df.columns:
            df[c] = df[c].fillna(v).astype(str)
    return df


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------

def inspect(export_dir: Path) -> int:
    tmp = sqlite3.connect(":memory:")
    tmp.executescript(SCHEMA.read_text())

    for table, fname in CSV_FILES.items():
        path = export_dir / fname
        print("=" * 72)
        print(fname)
        if not path.exists():
            print("  MISSING")
            continue
        print(f"  {path.stat().st_size / 1e6:.1f} MB")
        head = pd.read_csv(path, nrows=500)
        known = RADIAL_COLS if table == "radial_profile" else set(table_columns(tmp, table))
        dropped, transformed = DROPPED.get(table, {}), TRANSFORMED.get(table, {})

        for col in head.columns:
            n, filled = _norm(col), head[col].notna().sum()
            if n in transformed:
                mark, note = "~~ ", f"  <-- {transformed[n]}"
            elif n in dropped:
                mark, note = "-- ", f"  <-- dropped: {dropped[n]}"
            elif n in known:
                mark, note = "ok ", ""
            else:
                mark, note = "?? ", "  <-- NOT IN SCHEMA"
            if filled == 0:
                note += "   [EMPTY in first 500 rows]"
            print(f"    {mark}{col:<32} {str(head[col].dtype):<9}{note}")

        derived = {"centroid_x_px", "centroid_y_px", "centroid_z_px", "nucleus_track_id",
                   "source_detection_id", "droplet_assignment_status", "import_run_id"}
        missing = known - {_norm(c) for c in head.columns} - derived
        if missing:
            print(f"  absent from CSV, will be NULL: {sorted(missing)}")
    print("=" * 72)
    print("ok = maps to schema   ~~ = reshaped   -- = dropped   ?? = unrecognised")
    return 0


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def _read_mapped(path: Path, table: str, con: sqlite3.Connection,
                 cfg: ExperimentConfig, strict: bool, issues: list) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [_norm(c) for c in df.columns]
    df = split_xyz(df, path.name)
    df = qualify_nucleus_id(df, cfg, keep_track=(table == "nuclei"))
    df = coerce(df, cfg)
    if table == "nuclei":
        df = derive_tiles(df, cfg, issues)

    schema_cols = table_columns(con, table)
    known = set(schema_cols) | set(DROPPED.get(table, {}))
    unknown = [c for c in df.columns if c not in known]
    if unknown:
        msg = f"{path.name}: columns not in schema: {unknown}"
        if strict:
            raise ValueError(msg + "  (--allow-unknown-columns to drop them)")
        print(f"  WARNING {msg} -- dropping", file=sys.stderr)
    return df[[c for c in df.columns if c in schema_cols]].copy()


def _reject(rows: pd.DataFrame, table: str, reason: str, sink: list) -> None:
    for _, r in rows.iterrows():
        sink.append((table, reason, json.dumps(r.to_dict(), default=str)))


def _crosscheck_cfg_csv(path: Path, cfg: ExperimentConfig, issues: list) -> None:
    """Compare the sidecar against the exported config CSV."""
    if not path.exists():
        return
    row = pd.read_csv(path).iloc[0]
    row.index = [_norm(c) for c in row.index]
    for f in ("pixel_size_um", "z_step_um", "num_z_planes", "frame_interval_min"):
        if f not in row or pd.isna(row[f]):
            continue
        if abs(float(row[f]) - float(getattr(cfg, f))) > 1e-9:
            issues.append(("ERROR", f"sidecar disagrees with export on {f}", 1,
                           f"sidecar={getattr(cfg, f)} export={row[f]}. The sidecar was "
                           "used. Reconcile before trusting any measurement."))


def init_db(db_path: Path, force: bool = False) -> int:
    """Create an empty database with the schema and analysis views."""
    db_path = Path(db_path)
    if db_path.exists() and not force:
        print(f"{db_path} already exists. Use --force to replace it, or "
              f"'import' to add an experiment to it.")
        return 1
    if db_path.exists():
        bak = db_path.with_suffix(db_path.suffix + ".bak")
        shutil.move(str(db_path), str(bak))
        print(f"existing database moved to {bak.name}")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA.read_text())
    con.executescript(VIEWS.read_text())
    con.commit()
    con.close()
    print(f"initialised {db_path}")
    return 0


def _clear_experiment(con: sqlite3.Connection, cfg: ExperimentConfig) -> int:
    """Remove any existing rows for this (experiment, fov, droplet).

    Makes re-import idempotent: fixing a sidecar typo and re-running replaces
    the experiment rather than erroring or duplicating. Order matters because
    nuclei RESTRICTs the config row.
    """
    key = (cfg.experiment_id, cfg.fov_id, cfg.droplet_id)
    n = con.execute("SELECT COUNT(*) FROM nuclei WHERE experiment_id=? AND fov_id=? "
                    "AND droplet_id=?", key).fetchone()[0]
    if not n and not con.execute("SELECT COUNT(*) FROM experimental_cfg WHERE "
                                 "experiment_id=? AND fov_id=? AND droplet_id=?",
                                 key).fetchone()[0]:
        return 0
    con.execute("""DELETE FROM nucleus_z_stack WHERE nucleus_id IN
                   (SELECT nucleus_id FROM nuclei WHERE experiment_id=? AND fov_id=?
                    AND droplet_id=?)""", key)
    con.execute("DELETE FROM nuclei WHERE experiment_id=? AND fov_id=? AND droplet_id=?", key)
    con.execute("DELETE FROM radial_profile_files WHERE experiment_id=? AND fov_id=? "
                "AND droplet_id=?", key)
    con.execute("DELETE FROM experimental_cfg WHERE experiment_id=? AND fov_id=? "
                "AND droplet_id=?", key)
    return n


def import_export(export_dir: Path, config_path: Path, db_path: Path, parquet_dir: Path,
                  strict: bool = True, chunk: int = 1_000_000) -> int:
    export_dir, db_path, parquet_dir = map(Path, (export_dir, db_path, parquet_dir))
    parquet_dir.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    cfg = ExperimentConfig.from_json(config_path).validate()
    print(f"config OK: {cfg.experiment_id}/{cfg.fov_id}  "
          f"[{cfg.experimental_group} / {cfg.treatment} / {cfg.biological_replicate}]")

    if not db_path.exists():
        print(f"{db_path} does not exist -- run 'init' first.")
        return 1

    # Work on a copy; swap in only if the whole import and its checks succeed.
    fd, tmp_path = tempfile.mkstemp(suffix=".db", dir=db_path.parent)
    os.close(fd)
    shutil.copy2(db_path, tmp_path)
    con = sqlite3.connect(tmp_path)
    con.execute("PRAGMA foreign_keys = ON")
    rejected: list[tuple] = []
    issues: list[tuple] = []
    _crosscheck_cfg_csv(export_dir / CSV_FILES["experimental_cfg"], cfg, issues)

    try:
        con.execute("BEGIN")
        replaced = _clear_experiment(con, cfg)
        if replaced:
            print(f"  replacing {replaced} existing rows for this experiment/fov")
        con.execute(
            """INSERT INTO import_runs (started_at, source_directory, config_path,
               experiment_id, fov_id, status) VALUES (?,?,?,?,?,'RUNNING')""",
            (started, str(export_dir), str(config_path), cfg.experiment_id, cfg.fov_id))
        run_id = con.execute("SELECT last_insert_rowid()").fetchone()[0]
        cfg.write(con, validate=False)

        # -- z-stack first; nuclei.selected_slice_id has a deferred FK into it
        z = _read_mapped(export_dir / CSV_FILES["nucleus_z_stack"], "nucleus_z_stack",
                         con, cfg, strict, issues)
        csv_flag = z.set_index(["nucleus_id", "time_frame", "slice_id"])["is_selected_max"]
        if "repair_source" not in z.columns:
            # Export predates the v18.2 write-back: these areas are pre-repair
            # for every repaired nucleus, not just the ones with a missing plane.
            z["repair_source"] = "unrepaired_stale"
            issues.append(("WARN", "z-stack rows carry no repair provenance", len(z),
                           "Export has no Repair_Source column, so per-plane areas are "
                           "pre-repair wherever a nucleus was repaired. Marked "
                           "'unrepaired_stale'. Use nuclei.cross_sectional_area_um2 for "
                           "area; do not measure from nucleus_z_stack."))
        z["import_run_id"] = run_id
        z.to_sql("nucleus_z_stack", con, if_exists="append", index=False)
        print(f"  nucleus_z_stack: {len(z)} rows")

        idx = z["slice_id"].str.extract(r"(\d+)")[0].astype(int)
        if idx.max() > cfg.num_z_planes or idx.min() < 0:
            issues.append(("WARN", "slice index outside declared z range",
                           int(((idx > cfg.num_z_planes) | (idx < 0)).sum()),
                           f"slice_id spans {idx.min()}-{idx.max()} but "
                           f"num_z_planes={cfg.num_z_planes}. Check 0- vs 1-indexing."))

        # -- nuclei
        nuc = _read_mapped(export_dir / CSV_FILES["nuclei"], "nuclei", con, cfg, strict, issues)
        # Selected plane absent from the z-stack: keep the nucleus, annotate it.
        # The repair re-selected z onto a plane grouped_z_df never held, so the
        # nuclei row is the repaired (better) measurement and the z-stack is the
        # stale table. Dropping the nucleus would discard the better data.
        valid = set(map(tuple, z[["nucleus_id", "time_frame", "slice_id"]].values))
        orphan = pd.Series(
            [k not in valid for k in zip(nuc.nucleus_id, nuc.time_frame, nuc.selected_slice_id)],
            index=nuc.index)
        if orphan.any():
            late = int((nuc.loc[orphan, "time_frame"] >= 6).sum())
            note = ("selected_slice_id absent from NucleusZStack (v17 repair re-selected z "
                    "without updating grouped_z_df); nucleus measurement is the repaired "
                    "value, z-profile unavailable for this row")
            nuc.loc[orphan, "qc_notes"] = (
                nuc.loc[orphan, "qc_notes"].fillna("").astype(str).str.strip()
                .apply(lambda s: f"{s}; {note}" if s else note))
            issues.append(("WARN", "selected slice absent from z-stack", int(orphan.sum()),
                           f"{late} at time_frame >= 6. Rows kept and annotated in qc_notes. "
                           "Fix upstream by writing repaired planes back into grouped_z_df "
                           "before building NucleusZStack."))
            print(f"  nuclei: {int(orphan.sum())} rows have no matching z-stack plane "
                  f"(kept, annotated)")
        nuc = nuc.copy()
        nuc["import_run_id"] = run_id
        nuc.to_sql("nuclei", con, if_exists="append", index=False)
        print(f"  nuclei: {len(nuc)} rows")

        # -- recompute is_selected_max from selected_slice_id; report disagreement
        con.execute("UPDATE nucleus_z_stack SET is_selected_max = 0")
        con.execute(
            """UPDATE nucleus_z_stack SET is_selected_max = 1 WHERE EXISTS
               (SELECT 1 FROM nuclei n WHERE n.nucleus_id = nucleus_z_stack.nucleus_id
                AND n.time_frame = nucleus_z_stack.time_frame
                AND n.selected_slice_id = nucleus_z_stack.slice_id)""")
        after = pd.read_sql(
            "SELECT nucleus_id, time_frame, slice_id, is_selected_max FROM nucleus_z_stack",
            con).set_index(["nucleus_id", "time_frame", "slice_id"])["is_selected_max"]
        drift = int((csv_flag.reindex(after.index).fillna(0).astype(int) != after).sum())
        if drift:
            issues.append(("WARN", "Is_Selected_Max in CSV disagrees with selected_slice_id",
                           drift, "Recomputed from nuclei.selected_slice_id as the single "
                                  "source of truth; the CSV flag was not used."))

        con.execute(
            """UPDATE nuclei SET zstack_consistency = CASE
                 WHEN NOT EXISTS (SELECT 1 FROM nucleus_z_stack z
                     WHERE z.nucleus_id=nuclei.nucleus_id
                       AND z.time_frame=nuclei.time_frame
                       AND z.slice_id=nuclei.selected_slice_id) THEN 'plane_missing'
                 WHEN EXISTS (SELECT 1 FROM nucleus_z_stack z
                     WHERE z.nucleus_id=nuclei.nucleus_id
                       AND z.time_frame=nuclei.time_frame
                       AND z.slice_id=nuclei.selected_slice_id
                       AND abs(z.cross_sectional_area_um2
                               - nuclei.cross_sectional_area_um2) > 1e-6)
                     THEN 'area_mismatch'
                 ELSE 'consistent' END
               WHERE import_run_id = ?""", (run_id,))

        mism = con.execute(
            """SELECT COUNT(*) FROM nuclei n JOIN nucleus_z_stack z
               ON z.nucleus_id=n.nucleus_id AND z.time_frame=n.time_frame
                  AND z.slice_id=n.selected_slice_id
               WHERE abs(n.cross_sectional_area_um2 - z.cross_sectional_area_um2) > 1e-6"""
        ).fetchone()[0]
        if mism:
            issues.append(("WARN", "area disagrees between nuclei and z-stack", mism,
                           "Same nucleus and plane, different area. The repair overwrites "
                           "area_px in best_z_df but NucleusZStack is built from the "
                           "unrepaired grouped_z_df. Trust nuclei.cross_sectional_area_um2."))

        stale = con.execute(
            """SELECT COUNT(*) FROM nucleus_z_stack z WHERE NOT EXISTS
               (SELECT 1 FROM nuclei n WHERE n.nucleus_id=z.nucleus_id
                AND n.time_frame=z.time_frame)""").fetchone()[0]
        if stale:
            issues.append(("WARN", "z-stack rows with no surviving nucleus", stale,
                           "They belong to quarantined detections; left in place."))

        # -- intensities
        ri = _read_mapped(export_dir / CSV_FILES["raw_intensities"], "raw_intensities",
                          con, cfg, strict, issues)
        live = set(map(tuple, nuc[["nucleus_id", "time_frame"]].values))
        keep = pd.Series([tuple(k) in live for k in ri[["nucleus_id", "time_frame"]].values],
                         index=ri.index)
        if (~keep).any():
            _reject(ri[~keep], "raw_intensities", "no matching nucleus row", rejected)
        ri[keep].to_sql("raw_intensities", con, if_exists="append", index=False)
        print(f"  raw_intensities: {int(keep.sum())} rows")

        empty = [c for c in ri.columns
                 if c.startswith(("halo", "background")) and ri[c].notna().sum() == 0]
        if empty:
            issues.append(("ERROR", "intensity columns empty in the export", len(empty),
                           "Empty in the CSV, so the pipeline never wrote them: "
                           + ", ".join(empty)
                           + ". nc_ratio is therefore computed without background subtraction."))
            print(f"  raw_intensities: {len(empty)} columns EMPTY in the export")

        # -- radial profile -> parquet
        rows, pq_path = _stream_radial(export_dir / CSV_FILES["radial_profile"],
                                       parquet_dir, cfg, chunk)
        con.execute(
            """INSERT INTO radial_profile_files (experiment_id, fov_id, droplet_id,
               source_csv_path, parquet_path, row_count, parquet_size_bytes, compression,
               created_at, import_run_id) VALUES (?,?,?,?,?,?,?,'zstd',?,?)""",
            (cfg.experiment_id, cfg.fov_id, cfg.droplet_id,
             str(export_dir / CSV_FILES["radial_profile"]), str(pq_path), rows,
             pq_path.stat().st_size,
             datetime.now(timezone.utc).isoformat(timespec="seconds"), run_id))
        print(f"  radial_profile: {rows:,} rows -> {pq_path.name} "
              f"({pq_path.stat().st_size / 1e6:.0f} MB)")

        con.executemany("INSERT INTO rejected_rows (import_run_id, source_table, reason, "
                        "payload) VALUES (?,?,?,?)", [(run_id, *r) for r in rejected])
        for sev, name, n, detail in issues:
            con.execute("INSERT INTO import_issues (import_run_id, severity, check_name, n, "
                        "detail) VALUES (?,?,?,?,?)", (run_id, sev, name, n, detail))
        con.execute(
            """UPDATE import_runs SET completed_at=?, status='COMPLETED', nuclei_rows=?,
               z_stack_rows=?, raw_intensity_rows=?, radial_rows=?, rejected_rows=?
               WHERE import_run_id=?""",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"), len(nuc), len(z),
             int(keep.sum()), rows, len(rejected), run_id))

        con.execute("COMMIT")          # deferred FKs are checked here
        con.executescript(VIEWS.read_text())
        con.execute("ANALYZE")
        con.commit()
        con.close()
    except Exception:
        con.close()
        Path(tmp_path).unlink(missing_ok=True)
        raise

    bak = db_path.with_suffix(db_path.suffix + ".bak")
    shutil.copy2(db_path, bak)
    shutil.move(tmp_path, str(db_path))

    print(f"\nupdated {db_path} (previous state in {bak.name})")
    for sev, name, n, _ in sorted(issues):
        print(f"  {sev:5} {name}: {n}")
    return 0


def _stream_radial(csv_path: Path, parquet_dir: Path, cfg: ExperimentConfig,
                   chunk: int) -> tuple[int, Path]:
    """Convert the radial CSV to one zstd parquet without loading it all.

    nucleus_id is qualified per chunk. Parquet dictionary-encodes the repeated
    prefix, so the qualified string costs little over the bare integer.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = parquet_dir / f"{cfg.experiment_id}__{cfg.fov_id}.parquet"
    writer, total = None, 0
    try:
        for block in pd.read_csv(csv_path, chunksize=chunk):
            block.columns = [_norm(c) for c in block.columns]
            block = qualify_nucleus_id(block, cfg, keep_track=False)
            if "is_ridge_point" in block.columns:
                block["is_ridge_point"] = block["is_ridge_point"].astype(bool)
            block["experiment_id"] = cfg.experiment_id
            block["fov_id"] = cfg.fov_id
            tbl = pa.Table.from_pandas(block, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out, tbl.schema, compression="zstd")
            writer.write_table(tbl)
            total += len(block)
            print(f"    radial: {total:,} rows", end="\r", flush=True)
    finally:
        if writer is not None:
            writer.close()
    print()
    return total, out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("inspect", help="show how a CSV export maps onto the schema")
    i.add_argument("export_dir", type=Path)

    n = sub.add_parser("init", help="create an empty database")
    n.add_argument("--db", type=Path, required=True)
    n.add_argument("--force", action="store_true", help="replace an existing file")

    b = sub.add_parser("import", help="add or replace one experiment/FOV")
    b.add_argument("export_dir", type=Path)
    b.add_argument("--config", type=Path, required=True)
    b.add_argument("--db", type=Path, required=True)
    b.add_argument("--parquet-dir", type=Path, required=True)
    b.add_argument("--allow-unknown-columns", action="store_true")
    b.add_argument("--chunk", type=int, default=1_000_000)

    d = sub.add_parser("audit", help="run structural checks on a database")
    d.add_argument("--db", type=Path, required=True)

    a = p.parse_args(argv)
    if a.cmd == "inspect":
        return inspect(a.export_dir)
    if a.cmd == "init":
        return init_db(a.db, a.force)
    if a.cmd == "audit":
        import nsdb
        rep = nsdb.audit(a.db)
        print(rep.to_string(index=False))
        return 0 if rep.ok.all() else 1
    return import_export(a.export_dir, a.config, a.db, a.parquet_dir,
                         strict=not a.allow_unknown_columns, chunk=a.chunk)


if __name__ == "__main__":
    raise SystemExit(main())
