"""
experiment_config -- the condition metadata contract.

The problem this solves: experimental_cfg has treatment, concentration,
biological_replicate, extract_batch and experimental_group columns, and in the
first test export all of them were NULL. Analysis code that groups by condition
then has nothing to group on, and the failure surfaces weeks later in a
notebook rather than at the moment the data was written.

The fix is to make condition metadata a required input to export, validated
before anything is written. An export that cannot say what condition it
represents should not produce a file.

Usage in the pipeline, before writing any CSV:

    from experiment_config import ExperimentConfig

    cfg = ExperimentConfig.from_json("configs/control_extract_1.1.json")
    cfg.validate()                      # raises with every problem listed
    cfg.write(con)                      # upsert into experimental_cfg

To start a new one:

    ExperimentConfig.write_template("configs/new_run.json")
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path

# experiment_id format: <base>.<replicate index>, e.g. "control_extract_1.1"
# is replicate 1. The base is deliberately unconstrained -- only the replicate
# suffix is load-bearing, so renaming schemes upstream will not break this.
EXPERIMENT_ID_RE = re.compile(r"^(?P<base>.+)\.(?P<replicate>\d+)$")


def parse_replicate(experiment_id: str) -> tuple[str, str]:
    """Split an experiment_id into (base, canonical replicate label).

    "control_extract_1.1" -> ("control_extract_1", "R1")

    This parse happens exactly once, at export, to populate a column. Analysis
    code groups on the column and never touches the string -- the whole point
    of storing it is that nothing downstream has to parse an identifier.
    """
    m = EXPERIMENT_ID_RE.match(experiment_id or "")
    if not m:
        raise ValueError(
            f"experiment_id={experiment_id!r} does not end in a replicate index. "
            "Expected <base>.<n>, e.g. 'control_extract_1.1' for replicate 1."
        )
    return m["base"], f"R{int(m['replicate'])}"

# ---------------------------------------------------------------------------
# Design decisions, stated explicitly so they can be argued with later.
#
# 1. Condition metadata lives in columns, never inside identifier strings.
#    Anything you would have to parse back out of an ID is a column.
#
# 2. NULL and "deliberately none" are different states and must look different.
#    A vehicle-only control gets treatment="none", concentration=0.0 -- NOT
#    NULL. NULL is reserved to mean "nobody filled this in", which is a bug.
#    Without this distinction you cannot tell an untreated control from an
#    unlabelled export, and both look identical in a GROUP BY.
#
# 3. Validation fails loudly at export, not silently at analysis. Every
#    problem is reported at once rather than one per run.
#
# 4. The replicate is folded into experiment_id AND kept as its own column.
#    The first makes nucleus_id and the experimental_cfg primary key unique for
#    free; the second is what you group on.
# ---------------------------------------------------------------------------

# Must be present and non-empty before an export is allowed to proceed.
REQUIRED = (
    "experiment_id",
    "fov_id",
    "droplet_id",
    "experiment_date",
    "experimental_group",
    "treatment",
    "extract_batch",
    "operator",
    "pixel_size_um",
    "z_step_um",
    "num_z_planes",
    "frame_interval_min",
    "tile_rows",
    "tile_cols",
    "minutes_per_tile",
)

# Treatments that mean "no compound applied". These skip the concentration
# requirement. Extend this set rather than leaving concentration NULL.
NO_COMPOUND = {"none", "untreated", "vehicle", "buffer"}


@dataclass
class ExperimentConfig:
    # identity -- primary key of experimental_cfg
    experiment_id: str
    fov_id: str
    droplet_id: str = "UNASSIGNED"

    # condition: the columns analysis groups by
    experimental_group: str | None = None
    treatment: str | None = None
    concentration: float | None = None
    concentration_units: str | None = None
    biological_replicate: str | None = None
    technical_replicate: str | None = None
    extract_batch: str | None = None

    # acquisition
    experiment_date: str | None = None
    pixel_size_um: float | None = None
    z_step_um: float | None = None
    num_z_planes: int | None = None
    frame_interval_min: float | None = None
    # Mosaic scan geometry -- required, because time_real_minutes cannot be
    # interpreted without it (a nucleus's time depends on which tile it is in).
    tile_rows: int | None = None
    tile_cols: int | None = None
    minutes_per_tile: float | None = None
    serpentine_scan: bool | None = None
    channel0_label: str = "Membrane"
    channel1_label: str = "NLS"
    channel2_label: str = "NPC"
    microscope: str | None = None
    objective: str | None = None
    operator: str | None = None

    # provenance
    segmentation_pipeline_version: str | None = None
    model_version: str | None = None
    raw_file_path: str | None = None
    notes: str | None = None

    def __post_init__(self):
        """Derive biological_replicate from experiment_id when not given.

        The replicate index already lives in the ID suffix. Asking for it a
        second time in the sidecar only creates a way for the two to disagree,
        so it is derived by default and an explicit value is treated as an
        assertion to be checked, not as the source of truth.
        """
        if self.biological_replicate is None:
            try:
                _, self.biological_replicate = parse_replicate(self.experiment_id)
            except ValueError:
                pass  # reported by problems(), not raised in the constructor

    @property
    def tiles_per_frame(self) -> float | None:
        """Minutes to scan the whole mosaic once -- the true frame interval."""
        if None in (self.tile_rows, self.tile_cols, self.minutes_per_tile):
            return None
        return self.tile_rows * self.tile_cols * self.minutes_per_tile

    @property
    def experiment_base(self) -> str | None:
        """The ID with the replicate suffix stripped -- what this is a replicate of."""
        try:
            return parse_replicate(self.experiment_id)[0]
        except ValueError:
            return None

    # ---------------------------------------------------------------- checks

    def problems(self) -> list[str]:
        """Every validation failure, not just the first."""
        out: list[str] = []

        for name in REQUIRED:
            val = getattr(self, name)
            if val is None or (isinstance(val, str) and not val.strip()):
                out.append(f"{name} is required but missing")

        treat = (self.treatment or "").strip().lower()
        if treat and treat not in NO_COMPOUND:
            if self.concentration is None:
                out.append(
                    f"treatment={self.treatment!r} requires a concentration "
                    f"(use treatment='none' for untreated controls)"
                )
            if not self.concentration_units:
                out.append("concentration_units is required when a concentration is given")
        elif treat in NO_COMPOUND and self.concentration not in (None, 0, 0.0):
            out.append(
                f"treatment={self.treatment!r} means no compound, but "
                f"concentration={self.concentration}"
            )

        for name in ("pixel_size_um", "z_step_um", "frame_interval_min"):
            val = getattr(self, name)
            if val is not None and val <= 0:
                out.append(f"{name} must be > 0 (got {val}) -- the CHECK constraint will reject it")
        if self.num_z_planes is not None and self.num_z_planes <= 0:
            out.append(f"num_z_planes must be > 0 (got {self.num_z_planes})")

        # The mosaic takes one frame interval to scan, so the tile grid and the
        # frame interval are not independent. If they disagree, time_real_minutes
        # cannot be decomposed into (frame, tile) and every derived tile index
        # is wrong -- silently, since the arithmetic still produces a number.
        if None not in (self.tile_rows, self.tile_cols, self.minutes_per_tile,
                        self.frame_interval_min):
            span = self.tile_rows * self.tile_cols * self.minutes_per_tile
            if abs(span - self.frame_interval_min) > 1e-9:
                out.append(
                    f"tile grid implies {self.tile_rows}x{self.tile_cols} tiles at "
                    f"{self.minutes_per_tile} min = {span} min per frame, but "
                    f"frame_interval_min={self.frame_interval_min}. One of them is wrong.")

        # The replicate index lives in the experiment_id suffix. Because
        # nucleus_id is prefixed with experiment_id, this makes nucleus IDs
        # unique across replicates with no change to the ID scheme -- and makes
        # two replicates sharing an experiment_id impossible rather than merely
        # discouraged.
        try:
            _, derived = parse_replicate(self.experiment_id)
        except ValueError as e:
            out.append(str(e))
        else:
            stated = (self.biological_replicate or "").strip()
            if stated and stated.upper() != derived:
                out.append(
                    f"biological_replicate={stated!r} contradicts "
                    f"experiment_id={self.experiment_id!r}, which is {derived}. "
                    "Leave it unset to derive it, or fix the experiment_id."
                )

        if "|" in self.experiment_id or "|" in self.fov_id:
            out.append("'|' is the nucleus_id delimiter and cannot appear in experiment_id or fov_id")

        return out

    def validate(self) -> "ExperimentConfig":
        """Raise if anything is wrong. Returns self so it can be chained."""
        probs = self.problems()
        if probs:
            raise ValueError(
                f"experiment config for {self.experiment_id}/{self.fov_id} is incomplete:\n"
                + "\n".join(f"  - {p}" for p in probs)
            )
        return self

    # ------------------------------------------------------------------- io

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        data = json.loads(Path(path).read_text())
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known - {"_comment"}
        if unknown:
            raise ValueError(
                f"{path}: unrecognised keys {sorted(unknown)}. "
                "A typo here silently becomes a missing column."
            )
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_json(self, path: str | Path) -> Path:
        p = Path(path)
        p.write_text(json.dumps(asdict(self), indent=2, sort_keys=False) + "\n")
        return p

    @classmethod
    def write_template(cls, path: str | Path) -> Path:
        """Write a blank sidecar with every field present, so nothing is
        forgotten by omission. Fill it in before the run, not after."""
        blank = {f.name: None for f in fields(cls)}
        blank["_comment"] = (
            "Fill in before export. Required: "
            + ", ".join(REQUIRED)
            + ". Untreated controls use treatment='none', concentration=0."
        )
        blank["droplet_id"] = "UNASSIGNED"
        blank["channel0_label"], blank["channel1_label"], blank["channel2_label"] = (
            "Membrane", "NLS", "NPC")
        p = Path(path)
        p.write_text(json.dumps(blank, indent=2) + "\n")
        return p

    # ---------------------------------------------------------------- write

    def write(self, con: sqlite3.Connection, validate: bool = True) -> None:
        """Upsert into experimental_cfg. Validates first by default.

        Upsert rather than insert: re-running an export after fixing a typo in
        the sidecar should correct the row, not raise a uniqueness error and
        leave the wrong metadata in place.
        """
        if validate:
            self.validate()
        d = asdict(self)
        cols = list(d)
        placeholders = ",".join("?" * len(cols))
        updates = ",".join(
            f"{c}=excluded.{c}" for c in cols
            if c not in ("experiment_id", "fov_id", "droplet_id")
        )
        con.execute(
            f"INSERT INTO experimental_cfg ({','.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT (experiment_id, fov_id, droplet_id) DO UPDATE SET {updates}",
            [d[c] for c in cols],
        )
        con.commit()


# ---------------------------------------------------------------------------
# backfill / repair
# ---------------------------------------------------------------------------

def unlabelled(con: sqlite3.Connection) -> list[tuple]:
    """Config rows in the database that would fail validation today.

    Run this against an existing DB to find what needs backfilling.
    """
    rows = con.execute(
        """SELECT experiment_id, fov_id, droplet_id, experimental_group,
                  treatment, biological_replicate, extract_batch
           FROM experimental_cfg
           WHERE experimental_group IS NULL OR treatment IS NULL
              OR biological_replicate IS NULL OR extract_batch IS NULL"""
    ).fetchall()
    return rows
