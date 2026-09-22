#!/usr/bin/env python3
"""Import a segmentation database_export directory using hybrid storage."""

from __future__ import annotations

import argparse
import ast
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit("pandas is required: python -m pip install pandas") from exc

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:
    raise SystemExit("pyarrow is required: python -m pip install pyarrow") from exc

from create_database import create_database


# This file lives in <root>/scripts/database_tools/, hence parents[2].
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT = PROJECT_ROOT / "data" / "db" / "exports"
DEFAULT_TEST_DATABASE = PROJECT_ROOT / "data" / "db" / "nuclear_scaling_test.db"
DEFAULT_PARQUET_DIR = PROJECT_ROOT / "data" / "derived" / "radial_profiles"
MIGRATION_SQL = Path(__file__).with_name("hybrid_migration.sql")
REQUIRED_FILES = (
    "Experimental_Cfg.csv",
    "Nuclei.csv",
    "NucleusZStack.csv",
    "RawIntensities.csv",
    "RadialProfile.csv",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def python_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def parse_centroid(value: Any) -> tuple[float | None, float | None, float | None]:
    if pd.isna(value):
        return None, None, None
    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, (tuple, list)) or len(parsed) != 3:
        raise ValueError(f"Invalid centroid: {value!r}")
    return tuple(float(component) for component in parsed)


def selected_num_z_planes(z_stack: pd.DataFrame) -> int:
    values = z_stack["Slice_ID"].astype(str).str.extract(r"(\d+)$", expand=False)
    numbers = pd.to_numeric(values, errors="coerce").dropna()
    if numbers.empty:
        raise ValueError("Num_Z_Planes is missing and could not be inferred from Slice_ID")
    return int(numbers.max())


def ensure_column(connection: sqlite3.Connection, table: str, definition: str) -> None:
    column = definition.split()[0]
    existing = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {definition}")


def migrate_hybrid_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(MIGRATION_SQL.read_text(encoding="utf-8"))
    ensure_column(connection, "nuclei", "source_detection_id TEXT")
    ensure_column(connection, "nuclei", "nucleus_track_id TEXT")
    ensure_column(
        connection,
        "nuclei",
        "droplet_assignment_status TEXT NOT NULL DEFAULT 'ASSIGNED' "
        "CHECK (droplet_assignment_status IN ('ASSIGNED', 'UNASSIGNED'))",
    )


def upsert_many(
    connection: sqlite3.Connection,
    table: str,
    columns: tuple[str, ...],
    conflict: tuple[str, ...],
    rows: list[tuple[Any, ...]],
) -> None:
    if not rows:
        return
    updates = [column for column in columns if column not in conflict]
    action = "DO UPDATE SET " + ", ".join(
        f"{column}=excluded.{column}" for column in updates
    )
    sql = (
        f"INSERT INTO {table} ({', '.join(columns)}) "
        f"VALUES ({', '.join('?' for _ in columns)}) "
        f"ON CONFLICT ({', '.join(conflict)}) {action}"
    )
    connection.executemany(sql, rows)


def global_nucleus_id(experiment_id: str, fov_id: str, source_id: Any) -> str:
    source = str(python_value(source_id))
    if re.fullmatch(r"\d+(?:\.0+)?", source):
        source = f"N{int(float(source)):06d}"
    return f"{experiment_id}|{fov_id}|{source}"


def convert_radial_to_parquet(
    csv_path: Path,
    parquet_path: Path,
    chunk_rows: int,
    overwrite: bool,
) -> int:
    if parquet_path.exists() and not overwrite:
        return pq.ParquetFile(parquet_path).metadata.num_rows

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = parquet_path.with_suffix(parquet_path.suffix + ".partial")
    writer: pq.ParquetWriter | None = None
    total = 0
    dtypes = {
        "Nucleus_ID": "string",
        "Time_Frame": "int64",
        "Theta_deg": "float64",
        "Rho_Normalized": "float64",
        "W_Wall_Proximity_Index": "float64",
        "Channel": "string",
        "Intensity": "float64",
        "Is_Ridge_Point": "boolean",
        "Cluster_ID": "string",
    }
    try:
        for chunk in pd.read_csv(csv_path, chunksize=chunk_rows, dtype=dtypes):
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    partial_path,
                    table.schema,
                    compression="zstd",
                    use_dictionary=True,
                )
            writer.write_table(table)
            total += len(chunk)
            print(f"  radial rows converted: {total:,}", flush=True)
        if writer is None:
            raise ValueError(f"Radial profile file contains no rows: {csv_path}")
        writer.close()
        writer = None
        partial_path.replace(parquet_path)
        return total
    except Exception:
        if writer is not None:
            writer.close()
        if partial_path.exists():
            partial_path.unlink()
        raise


def import_export(
    export_dir: Path,
    database_path: Path,
    parquet_dir: Path,
    chunk_rows: int,
    overwrite_parquet: bool,
) -> dict[str, Any]:
    export_dir = export_dir.expanduser().resolve()
    database_path = database_path.expanduser().resolve()
    parquet_dir = parquet_dir.expanduser().resolve()
    missing = [name for name in REQUIRED_FILES if not (export_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required export files: {', '.join(missing)}")

    config = pd.read_csv(export_dir / "Experimental_Cfg.csv")
    if len(config) != 1:
        raise ValueError("Current importer requires exactly one Experimental_Cfg row per export")
    cfg = config.iloc[0]
    experiment_id = str(cfg["Experiment_ID"])
    fov_id = str(cfg["FOV_ID"])
    source_droplet = python_value(cfg.get("Droplet_ID"))
    droplet_id = str(source_droplet) if source_droplet is not None else "UNASSIGNED"
    droplet_status = "ASSIGNED" if source_droplet is not None else "UNASSIGNED"

    nuclei = pd.read_csv(export_dir / "Nuclei.csv")
    z_stack = pd.read_csv(export_dir / "NucleusZStack.csv")
    raw = pd.read_csv(export_dir / "RawIntensities.csv")
    num_z_planes = python_value(cfg.get("Num_Z_Planes"))
    if num_z_planes is None:
        num_z_planes = selected_num_z_planes(z_stack)

    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{experiment_id}__{fov_id}")
    radial_csv = export_dir / "RadialProfile.csv"
    parquet_path = parquet_dir / f"{safe_name}.parquet"
    radial_rows = convert_radial_to_parquet(
        radial_csv, parquet_path, chunk_rows, overwrite_parquet
    )

    create_database(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        migrate_hybrid_schema(connection)
        cursor = connection.execute(
            "INSERT INTO import_runs "
            "(started_at, source_directory, experiment_id, fov_id, status) "
            "VALUES (?, ?, ?, ?, 'RUNNING')",
            (utc_now(), str(export_dir), experiment_id, fov_id),
        )
        import_run_id = cursor.lastrowid
        try:
            config_columns = (
                "experiment_id", "fov_id", "droplet_id", "experiment_date",
                "pixel_size_um", "z_step_um", "num_z_planes", "frame_interval_min",
                "channel0_label", "channel1_label", "channel2_label", "microscope",
                "objective", "operator", "segmentation_pipeline_version",
                "model_version", "raw_file_path", "notes",
            )
            config_row = tuple(python_value(value) for value in (
                experiment_id, fov_id, droplet_id, cfg.get("Date"),
                cfg.get("Pixel_Size_um"), cfg.get("Z_Step_um"), num_z_planes,
                cfg.get("Frame_Interval_min"), cfg.get("Channel0_Label"),
                cfg.get("Channel1_Label"), cfg.get("Channel2_Label"),
                cfg.get("Microscope"), cfg.get("Objective"), cfg.get("Operator"),
                cfg.get("Segmentation_Pipeline_Version"), cfg.get("Model_Version"),
                cfg.get("Raw_File_Path"), cfg.get("Notes"),
            ))
            upsert_many(connection, "experimental_cfg", config_columns,
                        ("experiment_id", "fov_id", "droplet_id"), [config_row])

            nucleus_columns = (
                "nucleus_id", "droplet_id", "fov_id", "experiment_id", "time_frame",
                "time_real_minutes", "stage_classification", "selected_slice_id",
                "centroid_x_px", "centroid_y_px", "centroid_z_px",
                "cross_sectional_area_um2", "volume_3d_um3", "nc_ratio",
                "segmentation_pipeline_version", "model_version", "qc_flag",
                "qc_notes", "source_file_path", "source_detection_id",
                "nucleus_track_id", "droplet_assignment_status",
            )
            nucleus_rows = []
            for _, row in nuclei.iterrows():
                x, y, z = parse_centroid(row.get("Centroid_XYZ_px"))
                source_id = python_value(row["Nucleus_ID"])
                nucleus_rows.append(tuple(python_value(value) for value in (
                    global_nucleus_id(experiment_id, fov_id, source_id), droplet_id,
                    fov_id, experiment_id, row["Time_Frame"], row.get("Time_Real_Minutes"),
                    row.get("Stage_Classification"), row.get("Selected_Slice_ID"), x, y, z,
                    row.get("Cross_Sectional_Area_um2"), row.get("Volume_3D_um3"),
                    row.get("NC_Ratio"), row.get("Segmentation_Pipeline_Version"),
                    row.get("Model_Version"), row.get("QC_Flag", "UNREVIEWED"),
                    row.get("QC_Notes"), row.get("Source_File_Path"), str(source_id),
                    None, droplet_status,
                )))
            upsert_many(connection, "nuclei", nucleus_columns,
                        ("nucleus_id", "time_frame"), nucleus_rows)

            z_columns = (
                "nucleus_id", "time_frame", "slice_id", "centroid_x_px",
                "centroid_y_px", "centroid_z_px", "cross_sectional_area_um2",
                "is_selected_max",
            )
            z_rows = []
            for _, row in z_stack.iterrows():
                x, y, z = parse_centroid(row.get("Centroid_XYZ_px"))
                z_rows.append(tuple(python_value(value) for value in (
                    global_nucleus_id(experiment_id, fov_id, row["Nucleus_ID"]),
                    row["Time_Frame"], row["Slice_ID"], x, y, z,
                    row.get("Cross_Sectional_Area_um2"), int(bool(row["Is_Selected_Max"])),
                )))
            upsert_many(connection, "nucleus_z_stack", z_columns,
                        ("nucleus_id", "time_frame", "slice_id"), z_rows)

            raw_map = {
                "Nucleus_ID": "nucleus_id", "Time_Frame": "time_frame",
                **{f"Halo{h}_{c}": f"halo{h}_{c.lower()}"
                   for h in range(1, 5) for c in ("Mcherry", "NPC", "Membrane")},
                "Background_Mcherry": "background_mcherry",
                "Background_NPC": "background_npc",
                "Background_Membrane": "background_membrane",
            }
            raw_columns = tuple(raw_map.values())
            raw_rows = []
            for _, row in raw.iterrows():
                values = []
                for source, target in raw_map.items():
                    value = row.get(source)
                    if target == "nucleus_id":
                        value = global_nucleus_id(experiment_id, fov_id, value)
                    values.append(python_value(value))
                raw_rows.append(tuple(values))
            upsert_many(connection, "raw_intensities", raw_columns,
                        ("nucleus_id", "time_frame"), raw_rows)

            connection.execute(
                "INSERT INTO radial_profile_files "
                "(experiment_id, fov_id, droplet_id, source_csv_path, parquet_path, "
                "row_count, parquet_size_bytes, compression, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'zstd', ?) "
                "ON CONFLICT (experiment_id, fov_id, droplet_id, parquet_path) "
                "DO UPDATE SET source_csv_path=excluded.source_csv_path, "
                "row_count=excluded.row_count, parquet_size_bytes=excluded.parquet_size_bytes, "
                "created_at=excluded.created_at",
                (experiment_id, fov_id, droplet_id, str(radial_csv), str(parquet_path),
                 radial_rows, parquet_path.stat().st_size, utc_now()),
            )
            violations = connection.execute("PRAGMA foreign_key_check").fetchall()
            if violations:
                raise RuntimeError(f"Foreign-key violations: {violations[:10]}")
            connection.execute(
                "UPDATE import_runs SET completed_at=?, status='COMPLETED', nuclei_rows=?, "
                "z_stack_rows=?, raw_intensity_rows=?, radial_rows=? WHERE import_run_id=?",
                (utc_now(), len(nuclei), len(z_stack), len(raw), radial_rows, import_run_id),
            )
            connection.commit()
        except Exception as exc:
            connection.rollback()
            connection.execute(
                "INSERT INTO import_runs "
                "(started_at, completed_at, source_directory, experiment_id, fov_id, status, message) "
                "VALUES (?, ?, ?, ?, ?, 'FAILED', ?)",
                (utc_now(), utc_now(), str(export_dir), experiment_id, fov_id, str(exc)),
            )
            connection.commit()
            raise

    return {
        "database": database_path,
        "parquet": parquet_path,
        "experiment_id": experiment_id,
        "fov_id": fov_id,
        "droplet_status": droplet_status,
        "num_z_planes": int(num_z_planes),
        "nuclei": len(nuclei),
        "z_stack": len(z_stack),
        "raw_intensities": len(raw),
        "radial_rows": radial_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--database", type=Path, default=DEFAULT_TEST_DATABASE)
    parser.add_argument("--parquet-dir", type=Path, default=DEFAULT_PARQUET_DIR)
    parser.add_argument("--chunk-rows", type=int, default=250_000)
    parser.add_argument("--overwrite-parquet", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = import_export(
        args.export_dir, args.database, args.parquet_dir,
        args.chunk_rows, args.overwrite_parquet,
    )
    print("\nHybrid import completed:")
    for key, value in result.items():
        print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")
