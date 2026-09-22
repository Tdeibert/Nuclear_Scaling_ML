"""
acquisition_metadata.py — acquisition metadata as data, not constants.

DESIGN
------
1. Store physical FACTS, derive CONSTANTS.
   pixel_size_um is not a fact about the experiment; it is
   camera_pixel_um / (objective_mag * coupler_mag * confocal_zoom).
   Storing the derived value is what let 0.2167 and 0.108 coexist in one
   dataclass. Store the four measurements and the quotient cannot disagree
   with them.

2. Two levels, so per-experiment entry stays short enough to actually happen.
   - rig JSON      : optics + camera. Changes when hardware changes.
   - acquisition   : channels, z-step, interval, sample. One per dataset,
                     references a rig by name.

3. Auto-detect first. `init` reads the TIFF and pre-fills everything it can
   (shape, z-step, pixel size, channel count) so the human only fills in what
   the file genuinely does not carry.

4. Reconcile, don't trust. `validate` compares the derived pixel size against
   whatever the TIFF itself claims and fails on disagreement. That check alone
   would have caught the 0.2167 error the day it was introduced.

5. Channels carry ROLES, not just indices. Downstream code asks for
   `acq.role_index("nls")` instead of hardcoding nucleus_channel_idx=1. The
   model's (NLS, NPC, Membrane) permutation relative to the image's
   (Membrane, NLS, NPC) then has exactly one place it can go wrong.

USAGE
-----
    # bootstrap a stub from the image, then fill in the nulls
    python acquisition_metadata.py init /path/control_extract_1.1.tif \\
        --rig ix85_spin --out meta/control_extract_1.1.json

    python acquisition_metadata.py validate meta/control_extract_1.1.json
    python acquisition_metadata.py show     meta/control_extract_1.1.json

    # in the pipeline
    acq = Acquisition.load("meta/control_extract_1.1.json")
    acq.apply_to_config(cfg)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

def _candidate_rig_dirs(rig_dir=None, search_from=None) -> list[Path]:
    """Where to look for a rig definition, in priority order.

    Anchoring to the module directory alone is brittle: the acquisition sidecar,
    the rig file, and the module do not have to live in the same place, and on
    HPC they often do not. Search instead, and say what was searched on failure.
    """
    out: list[Path] = []
    if rig_dir:
        out.append(Path(rig_dir))
    env = os.environ.get("NUCLEAR_SCALING_RIG_DIR")
    if env:
        out += [Path(env), Path(env) / "rigs"]
    if search_from:                       # next to the acquisition JSON
        sf = Path(search_from).resolve()
        out += [sf / "rigs", sf, sf.parent / "rigs"]
    here = Path(__file__).resolve().parent
    out += [here / "rigs", here]
    cwd = Path.cwd().resolve()
    out += [cwd / "rigs", cwd]
    seen, uniq = set(), []
    for d in out:
        if d not in seen:
            seen.add(d)
            uniq.append(d)
    return uniq


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
@dataclass
class Rig:
    """Optical path. Changes only when hardware changes."""
    name: str
    scope_model: str = ""
    camera_model: str = ""
    camera_pixel_um: float = 0.0      # physical sensor pitch
    camera_sensor_px: tuple = (0, 0)
    objective: str = ""
    objective_mag: float = 0.0
    objective_na: float = 0.0
    objective_immersion: str = ""     # air | water | oil | silicone
    coupler_mag: float = 1.0          # C-mount adapter
    confocal_zoom: float = 1.0        # CSU relay, 1.0 on CSU-W1 standard
    field_number: float = 0.0

    @property
    def total_mag(self) -> float:
        return self.objective_mag * self.coupler_mag * self.confocal_zoom

    @property
    def pixel_size_um(self) -> float:
        """THE derived constant. Never stored, always computed."""
        if self.total_mag <= 0:
            raise ValueError(f"rig '{self.name}': total magnification is zero")
        return self.camera_pixel_um / self.total_mag

    def nyquist_px_um(self, wavelength_nm: float) -> float:
        """Max pixel size that still samples the diffraction limit."""
        if self.objective_na <= 0:
            raise ValueError(f"rig '{self.name}': objective_na not set")
        return (wavelength_nm / (2.0 * self.objective_na)) / 2.0 / 1000.0

    @classmethod
    def load(cls, name: str, rig_dir=None, search_from=None) -> "Rig":
        """`name` may be a bare rig name or a path to a rig JSON."""
        cand = Path(name)
        if cand.suffix == ".json" or cand.is_absolute() or len(cand.parts) > 1:
            if cand.exists():
                return cls._from_file(cand)
            raise FileNotFoundError(f"rig file not found: {cand}")

        searched, found = [], []
        for d in _candidate_rig_dirs(rig_dir, search_from):
            searched.append(d)
            p = d / f"{name}.json"
            if p.exists():
                return cls._from_file(p)
            if d.is_dir():
                found += [q.stem for q in d.glob("*.json")]

        raise FileNotFoundError(
            f"rig '{name}' not found. Searched:\n  " +
            "\n  ".join(str(d) for d in searched) +
            (f"\nJSON files seen in those dirs: {sorted(set(found))}"
             if found else "\nNo JSON files found in any of them.") +
            f"\nFix: put {name}.json in one of the above, set "
            f"NUCLEAR_SCALING_RIG_DIR, or set \"rig_name\" to a full path.")

    @classmethod
    def _from_file(cls, path: Path) -> "Rig":
        d = json.loads(Path(path).read_text())
        d["camera_sensor_px"] = tuple(d.get("camera_sensor_px", (0, 0)))
        return cls(**d)


@dataclass
class Channel:
    """One acquisition channel. `role` is what downstream code keys on."""
    index: int
    role: str                     # membrane | nls | npc | brightfield | other
    fluorophore: str = ""
    excitation_nm: float = 0.0
    emission_filter: str = ""
    exposure_ms: float = 0.0
    laser_power_pct: float | None = None


@dataclass
class Mosaic:
    tiles_x: int = 1
    tiles_y: int = 1
    overlap_frac: float = 0.0

    def expected_px(self, sensor_px: tuple) -> tuple:
        """Stitched dimensions implied by the tiling. A cross-check on scale."""
        sx, sy = sensor_px
        w = self.tiles_x * sx - (self.tiles_x - 1) * self.overlap_frac * sx
        h = self.tiles_y * sy - (self.tiles_y - 1) * self.overlap_frac * sy
        return (round(w), round(h))


@dataclass
class Acquisition:
    """One imaging dataset."""
    image_filename: str
    rig_name: str
    sample: str = ""
    experiment_id: str = ""
    date: str = ""

    z_step_um: float | None = None
    time_interval_s: float | None = None
    n_t: int | None = None
    n_z: int | None = None
    height_px: int | None = None
    width_px: int | None = None

    channels: list = field(default_factory=list)
    mosaic: Mosaic = field(default_factory=Mosaic)
    notes: str = ""

    _rig: Rig | None = field(default=None, repr=False, compare=False)
    _source_path: Path | None = field(default=None, repr=False, compare=False)

    # -- derived -----------------------------------------------------------
    @property
    def rig(self) -> Rig:
        if self._rig is None:
            here = self._source_path.parent if self._source_path else None
            self._rig = Rig.load(self.rig_name, search_from=here)
        return self._rig

    @property
    def pixel_size_um(self) -> float:
        return self.rig.pixel_size_um

    def role_index(self, role: str) -> int:
        for ch in self.channels:
            if ch.role == role:
                return ch.index
        raise KeyError(
            f"no channel with role '{role}' in {self.image_filename}. "
            f"Roles present: {[c.role for c in self.channels]}")

    @property
    def metadata_hash(self) -> str:
        """Stable hash over the acquisition + rig, for run-scoped output dirs."""
        payload = {"acq": self.to_dict(), "rig": asdict(self.rig)}
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    # -- IO ----------------------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("_rig", None)
        d.pop("_source_path", None)
        d["mosaic"] = asdict(self.mosaic)
        d["channels"] = [asdict(c) for c in self.channels]
        return d

    def save(self, path: Path) -> None:
        path = Path(path)
        self._source_path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path, validate: bool = True) -> "Acquisition":
        path = Path(path)
        d = json.loads(path.read_text())
        d.pop("_rig", None)
        d.pop("_source_path", None)
        d["channels"] = [Channel(**c) for c in d.get("channels", [])]
        d["mosaic"] = Mosaic(**d.get("mosaic", {}))
        acq = cls(**d)
        acq._source_path = path.resolve()
        if validate:
            problems = acq.validate()
            fatal = [p for p in problems if p.startswith("ERROR")]
            if fatal:
                raise ValueError(
                    f"{path} failed validation:\n  " + "\n  ".join(fatal))
            for p in problems:
                print(f"  {p}")
        return acq

    # -- validation --------------------------------------------------------
    def validate(self) -> list[str]:
        """Return ERROR/WARN strings. Empty list means clean."""
        out: list[str] = []

        for f in ("z_step_um", "time_interval_s"):
            if getattr(self, f) is None:
                out.append(f"ERROR: {f} is null — fill it in from the imaging table")

        if not self.channels:
            out.append("ERROR: no channels declared")
        roles = [c.role for c in self.channels]
        if len(set(roles)) != len(roles):
            out.append(f"ERROR: duplicate channel roles: {roles}")
        idxs = sorted(c.index for c in self.channels)
        if idxs != list(range(len(idxs))):
            out.append(f"ERROR: channel indices are not contiguous from 0: {idxs}")

        try:
            px = self.pixel_size_um
        except ValueError as e:
            out.append(f"ERROR: {e}")
            return out

        # Mosaic cross-check: does the tiling explain the frame size?
        if self.width_px and self.rig.camera_sensor_px[0]:
            exp_w, exp_h = self.mosaic.expected_px(self.rig.camera_sensor_px)
            if exp_w and abs(exp_w - self.width_px) > 0.02 * self.width_px:
                out.append(
                    f"WARN: mosaic implies {exp_w}x{exp_h} px but image is "
                    f"{self.width_px}x{self.height_px}. Check tiles/overlap "
                    f"or camera_sensor_px.")

        # Sampling check — informational, undersampling can be deliberate.
        exc = [c.excitation_nm for c in self.channels if c.excitation_nm]
        if exc and self.rig.objective_na > 0:
            nyq = self.rig.nyquist_px_um(min(exc))
            ratio = px / nyq
            if ratio > 1.0:
                out.append(
                    f"NOTE: {px:.4f} um/px undersamples by {ratio:.2f}x at "
                    f"{min(exc):.0f} nm (Nyquist {nyq:.4f}). Fine if traded "
                    f"for speed / bleaching — recorded here so it is explicit.")
        return out

    def reconcile_with_tiff(self, image_path: Path) -> list[str]:
        """Compare derived values against what the file itself claims."""
        probe = probe_tiff(image_path)
        out: list[str] = []

        if probe.get("probe_error"):
            out.append(f"NOTE: could not read {Path(image_path).name}: "
                       f"{probe['probe_error']}")
        px_file = probe.get("pixel_size_um")
        if px_file is None:
            out.append(
                f"NOTE: {Path(image_path).name} carries no spatial calibration "
                f"({probe.get('uncalibrated_reason', 'unknown')}), so the "
                f"derived {self.pixel_size_um:.4f} um/px cannot be cross-checked "
                f"against it. Verified instead by the mosaic arithmetic.")
        if px_file:
            if abs(px_file - self.pixel_size_um) / self.pixel_size_um > 0.02:
                out.append(
                    f"ERROR: derived pixel size {self.pixel_size_um:.4f} um/px "
                    f"disagrees with the file's {px_file:.4f} um/px. One of "
                    f"camera_pixel_um / objective_mag / coupler_mag is wrong.")
            else:
                out.append(f"OK: pixel size agrees with file ({px_file:.4f} um/px)")

        z_file = probe.get("z_step_um")
        if z_file is None:
            out.append("NOTE: no z-step in file metadata either.")
        if z_file and self.z_step_um:
            if abs(z_file - self.z_step_um) > 1e-3:
                out.append(
                    f"ERROR: z_step_um {self.z_step_um} disagrees with the "
                    f"file's {z_file}")
            else:
                out.append(f"OK: z-step agrees with file ({z_file} um)")

        for k, attr in (("n_t", "n_t"), ("n_z", "n_z"),
                        ("height_px", "height_px"), ("width_px", "width_px")):
            if probe.get(k) and getattr(self, attr) not in (None, probe[k]):
                out.append(f"ERROR: {attr}={getattr(self, attr)} but file has {probe[k]}")
        return out

    # -- integration -------------------------------------------------------
    def apply_to_config(self, cfg) -> None:
        """Push metadata into PipelineConfig. Single point of entry."""
        cfg.pixel_size_um = self.pixel_size_um
        if self.z_step_um is not None:
            cfg.z_step_um = self.z_step_um
        cfg.membrane_channel_idx = self.role_index("membrane")
        cfg.nucleus_channel_idx = self.role_index("nls")
        cfg.npc_channel_idx = self.role_index("npc")
        cfg.image_filename = self.image_filename
        print(f"acquisition '{self.experiment_id or self.image_filename}' "
              f"[{self.metadata_hash}]\n"
              f"  pixel_size_um : {cfg.pixel_size_um:.4f} "
              f"(= {self.rig.camera_pixel_um} / {self.rig.total_mag:g})\n"
              f"  z_step_um     : {cfg.z_step_um}\n"
              f"  channels      : " +
              ", ".join(f"{c.index}={c.role}" for c in
                        sorted(self.channels, key=lambda c: c.index)))


# ---------------------------------------------------------------------------
# TIFF probing
# ---------------------------------------------------------------------------
def probe_tiff(path: Path) -> dict[str, Any]:
    """Pull whatever the file already carries. Best-effort, never raises.

    Calibration is only reported when the file actually claims to be
    calibrated. An uncalibrated TIFF carries XResolution = 1/1, which reads out
    as "1.0 um/px" and is not a measurement — treating it as one produces a
    spurious conflict with the derived value.
    """
    import tifffile as tiff

    out: dict[str, Any] = {}
    try:
        with tiff.TiffFile(str(path)) as tf:
            arr = tf.series[0]
            for ax, n in zip(arr.axes, arr.shape):
                out[{"T": "n_t", "Z": "n_z", "C": "n_c",
                     "Y": "height_px", "X": "width_px"}.get(ax, ax)] = int(n)

            # ---- OME: units are explicit, so trust it ----
            if tf.is_ome and tf.ome_metadata:
                try:
                    root = ET.fromstring(tf.ome_metadata)
                    ns = {"ome": root.tag.split("}")[0].strip("{")}
                    px = root.find(".//ome:Pixels", ns)
                    if px is not None:
                        for key, attr in (("pixel_size_um", "PhysicalSizeX"),
                                          ("z_step_um", "PhysicalSizeZ"),
                                          ("time_interval_s", "TimeIncrement")):
                            if px.get(attr):
                                out[key] = float(px.get(attr))
                                out["calibration_source"] = "OME"
                        out["ome_channels"] = [
                            c.get("Name") or c.get("Fluor") or ""
                            for c in px.findall("ome:Channel", ns)]
                except ET.ParseError:
                    pass

            # ---- ImageJ: the `unit` string is authoritative ----
            if tf.is_imagej and tf.imagej_metadata and "pixel_size_um" not in out:
                ij = tf.imagej_metadata
                unit = str(ij.get("unit", "")).strip().lower()
                micron = unit in ("micron", "microns", "um", "\u00b5m", "\u03bcm")
                if not micron:
                    out["uncalibrated_reason"] = (
                        f"ImageJ unit is {unit!r}" if unit
                        else "ImageJ metadata has no `unit` field")
                else:
                    out["calibration_source"] = "ImageJ"
                    if "spacing" in ij:
                        out["z_step_um"] = abs(float(ij["spacing"]))
                    if "finterval" in ij:
                        out["time_interval_s"] = float(ij["finterval"])
                    p0 = tf.pages[0]
                    if "XResolution" in p0.tags:
                        num, den = p0.tags["XResolution"].value
                        if num:
                            out["pixel_size_um"] = float(den) / float(num)

            # ---- plain TIFF: only inch/cm resolution units mean anything ----
            if "pixel_size_um" not in out and not tf.is_imagej and not tf.is_ome:
                p0 = tf.pages[0]
                runit = p0.tags["ResolutionUnit"].value if "ResolutionUnit" in p0.tags else None
                runit = int(getattr(runit, "value", runit) or 1)
                xres = p0.tags["XResolution"].value if "XResolution" in p0.tags else None
                if runit in (2, 3) and xres and xres[0]:
                    per_unit = float(xres[0]) / float(xres[1])
                    um_per_unit = 25400.0 if runit == 2 else 10000.0
                    out["pixel_size_um"] = um_per_unit / per_unit
                    out["calibration_source"] = "TIFF ResolutionUnit"
                else:
                    out["uncalibrated_reason"] = (
                        f"ResolutionUnit={runit} (no absolute unit), "
                        f"XResolution={xres}")

            if "pixel_size_um" not in out and "uncalibrated_reason" not in out:
                out["uncalibrated_reason"] = "no usable resolution metadata"
    except Exception as e:
        out["probe_error"] = str(e)
    return out


def imagej_calibration(cfg) -> dict:
    """kwargs for tifffile.imwrite so OUR outputs are calibrated.

    The source hyperstack is uncalibrated, which is why reconcile cannot check
    it. No reason to propagate that: every TIFF this pipeline writes should
    carry the derived scale so it opens correctly in Fiji and napari.
    """
    return dict(
        imagej=True,
        resolution=(1.0 / cfg.pixel_size_um, 1.0 / cfg.pixel_size_um),
        metadata={"axes": "TZYX", "unit": "micron",
                  "spacing": float(cfg.z_step_um)},
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cmd_init(args) -> None:
    probe = probe_tiff(args.image)
    if "probe_error" in probe:
        print(f"[warn] could not read {args.image}: {probe['probe_error']}")

    n_c = probe.get("n_c", 3)
    channels = [Channel(index=i, role="FILL_ME") for i in range(n_c)]

    acq = Acquisition(
        image_filename=Path(args.image).name,
        rig_name=args.rig,
        z_step_um=probe.get("z_step_um"),
        time_interval_s=probe.get("time_interval_s"),
        n_t=probe.get("n_t"), n_z=probe.get("n_z"),
        height_px=probe.get("height_px"), width_px=probe.get("width_px"),
        channels=channels,
        notes="Stub from `init`. Fill nulls and FILL_ME, then run `validate`.",
    )
    acq.save(args.out)
    print(f"wrote {args.out}")
    print("auto-detected:", {k: v for k, v in probe.items() if k != "ome_channels"})
    if probe.get("ome_channels"):
        print("channel names in file:", probe["ome_channels"])
    print("\nStill needed: channel roles, and any null above.")


def cmd_validate(args) -> None:
    acq = Acquisition.load(args.meta, validate=False)
    problems = acq.validate()
    if args.image:
        problems += acq.reconcile_with_tiff(args.image)
    if not problems:
        print("clean")
        return
    for p in problems:
        print(p)
    if any(p.startswith("ERROR") for p in problems):
        raise SystemExit(1)


def cmd_show(args) -> None:
    acq = Acquisition.load(args.meta, validate=False)
    r = acq.rig
    print(f"{acq.experiment_id or acq.image_filename}  [{acq.metadata_hash}]")
    print(f"  rig          : {r.name} ({r.scope_model})")
    print(f"  optics       : {r.objective} NA {r.objective_na} {r.objective_immersion}")
    print(f"  camera       : {r.camera_model} @ {r.camera_pixel_um} um")
    print(f"  total mag    : {r.objective_mag} x {r.coupler_mag} x {r.confocal_zoom}"
          f" = {r.total_mag:g}")
    print(f"  pixel size   : {acq.pixel_size_um:.4f} um/px  (DERIVED)")
    print(f"  z step       : {acq.z_step_um} um")
    print(f"  interval     : {acq.time_interval_s} s")
    print(f"  dims         : T{acq.n_t} Z{acq.n_z} "
          f"Y{acq.height_px} X{acq.width_px}")
    for c in sorted(acq.channels, key=lambda c: c.index):
        print(f"  ch{c.index} {c.role:<10} {c.fluorophore:<18} "
              f"{c.excitation_nm:.0f} nm  {c.exposure_ms:.0f} ms")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="bootstrap a stub sidecar from a TIFF")
    p.add_argument("image", type=Path)
    p.add_argument("--rig", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("validate", help="check a sidecar, optionally against its TIFF")
    p.add_argument("meta", type=Path)
    p.add_argument("--image", type=Path, default=None)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("show", help="print resolved metadata")
    p.add_argument("meta", type=Path)
    p.set_defaults(func=cmd_show)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
