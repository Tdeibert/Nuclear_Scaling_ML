"""
paths_config.py
---------------
Environment-aware path configuration for Nuclear_Scaling_ML.
Auto-detects Cheaha vs. local environment and sets all project paths accordingly.

Usage (in any notebook or script):
    from paths_config import paths
    print(paths.patches)
"""

import os
import socket
from pathlib import Path
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

def _detect_environment() -> str:
    """
    Returns 'cheaha' or 'local'.

    Detection order:
      1. CHEAHA env var (set system-wide on Cheaha login + compute nodes)
      2. Hostname pattern  (*.rc.uab.edu  or  ampere* / login*)
      3. PROJECT_ROOT env var override (useful for testing)
      4. Default → local
    """
    if os.environ.get("CHEAHA"):
        return "cheaha"

    hostname = socket.gethostname().lower()
    cheaha_patterns = ("rc.uab.edu", "ampere", "login", "cheaha")
    if any(p in hostname for p in cheaha_patterns):
        return "cheaha"

    return "local"


ENV = _detect_environment()
USER = os.environ.get("USER", os.environ.get("USERNAME", "tdeibert"))


# ---------------------------------------------------------------------------
# Path dataclass
# ---------------------------------------------------------------------------

@dataclass
class ProjectPaths:
    """
    All canonical paths for the Nuclear_Scaling_ML project.
    Attributes are Path objects; call str(paths.X) if a string is needed.
    """

    env: str

    # --- Roots ---
    project_root: Path = field(init=False)
    scratch_root: Path = field(init=False)   # fast I/O workspace (scratch on Cheaha, tmp locally)

    # --- Data inputs ---
    raw_data: Path = field(init=False)       # original .tif stacks
    patches: Path = field(init=False)        # extracted image patches

    # --- Labels ---
    labels: Path = field(init=False)         # current label .npy / .tif files
    label_qc: Path = field(init=False)       # QC outputs (CSVs, overlay images)

    # --- Models ---
    checkpoints: Path = field(init=False)    # Keras .keras / .h5 checkpoints
    logs: Path = field(init=False)           # TensorBoard / training logs

    # --- Outputs ---
    inference: Path = field(init=False)      # model prediction outputs
    figures: Path = field(init=False)        # plots and figures

    def __post_init__(self):
        if self.env == "cheaha":
            self.project_root = Path(f"/home/{USER}/Nuclear_Scaling_ML")
            self.scratch_root  = Path(f"/scratch/{USER}/Nuclear_Scaling_ML")
        else:
            # Local: project root inferred from this file's location.
            # This file lives in <root>/src/, hence parents[1].
            self.project_root = Path(__file__).resolve().parents[1]
            self.scratch_root  = self.project_root / "_scratch"  # gitignored local scratch

        # ---- Stable inputs live under project_root (version-controlled or NFS-safe) ----
        self.raw_data   = self.project_root / "data" / "raw"
        self.labels     = self.project_root / "data" / "labels"
        self.label_qc   = self.project_root / "data" / "label_qc"
        self.figures    = self.project_root / "outputs" / "figures"

        # ---- High-I/O paths live under scratch_root ----
        self.patches     = self.scratch_root / "patches"
        self.checkpoints = self.scratch_root / "checkpoints"
        self.logs        = self.scratch_root / "logs"
        self.inference   = self.scratch_root / "inference"

    def make_all(self) -> None:
        """Create all directories (safe to call every session)."""
        for attr, value in self.__dict__.items():
            if isinstance(value, Path) and attr not in ("project_root", "scratch_root"):
                value.mkdir(parents=True, exist_ok=True)

    def summary(self) -> str:
        lines = [f"Environment : {self.env.upper()}", ""]
        for attr, value in self.__dict__.items():
            if isinstance(value, Path):
                exists = "✓" if value.exists() else "✗"
                lines.append(f"  {exists}  {attr:<18}  {value}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Singleton — import this in notebooks / scripts
# ---------------------------------------------------------------------------

paths = ProjectPaths(env=ENV)
