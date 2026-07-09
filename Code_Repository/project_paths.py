"""
project_paths.py — single source of truth for every filesystem location.

Drop this at the top of the pipeline notebook. It reads structure.json and
materialises the tree under two physical roots, then hands back a registry of
named paths so downstream code never hard-codes a directory again.

Design goals
------------
1. One layout definition (structure.json), not paths scattered across a config.
2. Two decoupled roots — code_root (small, $HOME) and data_root (large,
   /data/user/$USER) — because on Cheaha they live on different filesystems.
3. Run-scoped outputs: every run writes under Runs/<run_id>/, so a stale pickle
   from a different run/config can never be loaded by accident. This is the
   structural fix for the RUN_* staleness class of bug.
4. A config snapshot per run + a guard that refuses to load cached stages whose
   snapshot disagrees with the live config.

Typical use
-----------
    paths = ProjectPaths.for_cheaha("structure.json", project_name="Nuclear_Scaling")
    run   = paths.for_run(ProjectPaths.make_run_id("control_1.1", cfg.to_serializable_dict()))
    run.snapshot_config(cfg.to_serializable_dict())
    ...
    run.seg_dir / "segmentation_index.pkl"   # instead of cfg.seg_dir
"""
from __future__ import annotations

import datetime as _dt
import getpass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


# Stage directory (as named in run_template) -> short attribute the pipeline uses.
_RUN_STAGE_ATTRS: Dict[str, str] = {
    "segmentation": "seg_dir",
    "objects":      "obj_dir",
    "tracking":     "track_dir",
    "analysis":     "analysis_dir",
    "qc":           "qc_dir",
    "mask_tifs":    "mask_tif_dir",
    "exports":      "exports_dir",
}


def _build_tree(base: Path, spec: Optional[dict], created: List[Path]) -> None:
    """Recursively create directories described by a nested dict (None = leaf)."""
    base.mkdir(parents=True, exist_ok=True)
    created.append(base)
    if not spec:
        return
    for name, child in spec.items():
        if str(name).startswith("_"):        # skip _comment and similar meta keys
            continue
        _build_tree(base / name, child, created)


class ProjectPaths:
    """Materialises the static tree and vends run-scoped views."""

    def __init__(self, structure_path, code_root, data_root, create: bool = True):
        self.structure_path = Path(structure_path)
        self.spec: Dict[str, Any] = json.loads(self.structure_path.read_text())
        self.code_root = Path(code_root).expanduser().resolve()
        self.data_root = Path(data_root).expanduser().resolve()
        self.run_template: Dict[str, Any] = self.spec.get("run_template", {})

        # Commonly referenced static leaves.
        self.raw_images_dir    = self.data_root / "Inputs" / "Raw_Images"
        self.models_dir        = self.data_root / "Inputs" / "Models"
        self.training_data_dir = self.data_root / "Inputs" / "Training_Data"
        self.runs_dir          = self.data_root / "Runs"
        self.legacy_dir        = self.data_root / "Legacy"

        if create:
            self.materialise_static()

    # ── convenience constructor encoding the Cheaha storage split ─────────
    @classmethod
    def for_cheaha(cls, structure_path, project_name: str,
                   code_subdir: str = "Projects", create: bool = True) -> "ProjectPaths":
        user = getpass.getuser()
        home = Path(os.environ.get("HOME", f"/home/{user}"))
        # $USER_DATA is the canonical 5TB space; fall back to the standard path.
        user_data = Path(os.environ.get("USER_DATA", f"/data/user/{user}"))
        return cls(
            structure_path=structure_path,
            code_root=home / code_subdir,
            data_root=user_data / project_name,
            create=create,
        )

    # ── static tree ──────────────────────────────────────────────────────
    def materialise_static(self) -> List[Path]:
        created: List[Path] = []
        _build_tree(self.code_root, self.spec.get("code_root"), created)
        _build_tree(self.data_root, self.spec.get("data_root"), created)
        return created

    # ── run identity ─────────────────────────────────────────────────────
    @staticmethod
    def make_run_id(label: str, config_dict: dict) -> str:
        """Hybrid id: <label>__<timestamp>__cfg-<hash>.

        The config hash makes staleness structural — change any config value and
        the id changes, so a new run can never land in an old run's directory.
        The timestamp keeps successive runs of the *same* config distinct; drop
        it if you'd rather have same-config runs resume in place.
        """
        h = hashlib.sha1(
            json.dumps(config_dict, sort_keys=True, default=str).encode()
        ).hexdigest()[:8]
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M")
        safe = "".join(c if (c.isalnum() or c in "-._") else "_" for c in label)
        return f"{safe}__{stamp}__cfg-{h}"

    def for_run(self, run_id: str, create: bool = True) -> "RunPaths":
        run_dir = self.runs_dir / run_id
        if create:
            _build_tree(run_dir, self.run_template, [])
        return RunPaths(self, run_id, run_dir)

    def list_runs(self) -> List[str]:
        if not self.runs_dir.exists():
            return []
        return sorted(p.name for p in self.runs_dir.iterdir() if p.is_dir())

    def registry(self) -> Dict[str, Path]:
        """Flat name -> Path map of the static locations, for QC printing."""
        return {
            "code_root": self.code_root, "data_root": self.data_root,
            "raw_images_dir": self.raw_images_dir, "models_dir": self.models_dir,
            "training_data_dir": self.training_data_dir,
            "runs_dir": self.runs_dir, "legacy_dir": self.legacy_dir,
        }


class RunPaths:
    """Every stage directory + provenance files for a single run_id."""

    def __init__(self, project: ProjectPaths, run_id: str, run_dir: Path):
        self.project = project
        self.run_id = run_id
        self.run_dir = run_dir
        for stage, attr in _RUN_STAGE_ATTRS.items():
            setattr(self, attr, run_dir / stage)
        self.config_snapshot_path = run_dir / "config.json"
        self.manifest_path        = run_dir / "manifest.json"

    # ── provenance / stale-guard ─────────────────────────────────────────
    def snapshot_config(self, config_dict: dict) -> None:
        self.config_snapshot_path.write_text(
            json.dumps(config_dict, indent=2, sort_keys=True, default=str))

    def load_snapshot(self) -> Optional[dict]:
        if self.config_snapshot_path.exists():
            return json.loads(self.config_snapshot_path.read_text())
        return None

    def assert_config_matches(self, config_dict: dict, strict: bool = True) -> None:
        """Refuse to trust cached stages if the live config drifted from the
        snapshot that produced this run's data. Call before any RUN_*=False load."""
        snap = self.load_snapshot()
        if snap is None:
            return
        diffs = {k: (snap.get(k), config_dict.get(k))
                 for k in set(snap) | set(config_dict)
                 if snap.get(k) != config_dict.get(k)}
        if diffs:
            msg = (f"[stale-guard] live config differs from snapshot in "
                   f"run '{self.run_id}':\n" +
                   "\n".join(f"    {k}: snapshot={s!r} live={l!r}"
                             for k, (s, l) in diffs.items()))
            if strict:
                raise RuntimeError(msg)
            print("WARNING:", msg)

    def write_manifest(self, **fields) -> None:
        """Append/update run provenance (stage completions, input hashes, etc.)."""
        manifest = {}
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text())
        manifest.update(fields)
        manifest["updated_at"] = _dt.datetime.now().isoformat(timespec="seconds")
        self.manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    def registry(self) -> Dict[str, Path]:
        reg = {attr: getattr(self, attr) for attr in _RUN_STAGE_ATTRS.values()}
        reg["run_dir"] = self.run_dir
        reg["config_snapshot_path"] = self.config_snapshot_path
        reg["manifest_path"] = self.manifest_path
        return reg


if __name__ == "__main__":
    # Smoke test against a temp root — safe to run anywhere.
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    pp = ProjectPaths("structure.json", code_root=tmp / "code", data_root=tmp / "data")
    demo_cfg = {"nucleus_threshold": 0.5, "focus_min_z": 6}
    rp = pp.for_run(ProjectPaths.make_run_id("demo", demo_cfg))
    rp.snapshot_config(demo_cfg)
    print("static:")
    for k, v in pp.registry().items():
        print(f"  {k:18s} {v}")
    print("run:")
    for k, v in rp.registry().items():
        print(f"  {k:20s} {v}")
    rp.assert_config_matches(demo_cfg)                       # passes
    try:
        rp.assert_config_matches({**demo_cfg, "focus_min_z": 3})  # trips guard
    except RuntimeError as e:
        print("\nguard correctly fired:\n", e)
