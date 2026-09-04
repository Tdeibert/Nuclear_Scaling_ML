#!/usr/bin/env python3
"""Create the Nuclear Scaling SQLite database from schema.sql."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


# This file lives in <root>/scripts/database_tools/, hence parents[2].
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE = DEFAULT_PROJECT_ROOT / "data" / "db" / "nuclear_scaling.db"
SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def create_database(database_path: Path) -> None:
    database_path = database_path.expanduser().resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        result = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if result != "ok":
            raise RuntimeError(f"SQLite integrity check failed: {result}")

    print(f"Database ready: {database_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE,
        help=f"Database path (default: {DEFAULT_DATABASE})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    create_database(arguments.database)
