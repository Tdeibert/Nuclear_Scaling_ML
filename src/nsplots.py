"""
nsplots -- figures for the nuclear scaling analysis, all sourced from the
database and its radial-profile parquet sidecars.

WHAT CAN AND CANNOT BE REPRODUCED FROM THE EXPORT
-------------------------------------------------
The v18.1 RadialProfile export writes Rho_Normalized = distance_px /
ray_length_px, so 0 is the NUCLEUS CENTRE and 1 is the droplet wall. The
pipeline's own figures use rho_wall, which is 0 at the nuclear SURFACE and
needs distance_px and inside_nucleus -- neither is exported. Cell 106 of the
pipeline notebook warns about exactly this.

Consequences, stated up front so no figure here is over-read:

  * Radial shells are in centre-normalised rho, NOT microns from the nuclear
    surface. Shell boundaries are not comparable to the v18.1 shell figures.
  * Envelope roses show the rho at which membrane intensity peaks, not an
    envelope radius in microns. The shape is comparable; the units are not.
  * W_Wall_Proximity_Index, Is_Ridge_Point and Cluster_ID are NaN/placeholder
    in the export, so nothing here uses them.

All ANGULAR statistics -- asymmetry score, resultant direction, Rayleigh
tests, alignment-averaged profiles -- are unaffected, because they do not
depend on radial calibration.

To make the radial figures fully reproducible, add Distance_px and
Inside_Nucleus (or R_From_Edge_um) to build_radial_profile_table.

ANGLE CONVENTION
----------------
The sweep casts rays as y = cy + r*sin(theta) and image y increases downward,
so theta increases CLOCKWISE on screen with theta=0 pointing right (+x).
Polar axes here are set to match, as _orient_polar does in the pipeline.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colors

import nsdb

SHELLS = ((0.00, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.00))
SHELL_COLORS = ("#7fd4b0", "#2fa383", "#4c6fbf", "#1a1a2e")


# =============================================================================
# loading
# =============================================================================

def load_nuclei(experiment=None, qc="PASS", intensity=False, path=None) -> pd.DataFrame:
    """Nuclei with condition metadata. Thin wrapper so notebooks import one module."""
    return nsdb.nuclei(experiment=experiment, qc=qc, intensity=intensity, path=path)


def sweep_channel(experiment: str, path=None) -> str:
    """The single channel label present in the radial export.

    The sweep runs on cfg.radial_channel_index only, so Channel is one constant
    string for the whole file. Detected rather than assumed, because which
    channel it was depends on the config at run time.
    """
    ch = nsdb.radial(experiment, columns=["channel"], path=path)["channel"]
    vals = ch.dropna().unique().tolist()
    if len(vals) != 1:
        warnings.warn(f"expected one channel in the radial export, found {vals}")
    return vals[0] if vals else ""


def load_sweep(experiment: str, time_frames=None, nuclei=None, channel=None,
               path=None) -> pd.DataFrame:
    """Radial sweep rows, filtered inside the parquet reader.

    The file is millions of rows; always pass time_frames or nuclei unless you
    genuinely need all of it.
    """
    if channel is None:
        channel = sweep_channel(experiment, path=path)
    filters = [("channel", "==", channel)]
    if time_frames is not None:
        filters.append(("time_frame", "in", list(np.atleast_1d(time_frames))))
    cols = ["nucleus_id", "time_frame", "theta_deg", "rho_normalized",
            "channel", "intensity"]
    df = nsdb.radial(experiment, columns=cols, filters=filters, path=path)
    if nuclei is not None:
        df = df[df.nucleus_id.isin(np.atleast_1d(nuclei))]
    return df


# =============================================================================
# angular statistics
# =============================================================================

def _boxplot(ax, data, labels, **kw):
    """boxplot with the tick-label kwarg matplotlib actually accepts.

    `labels=` was renamed to `tick_labels=` in matplotlib 3.9 and removed
    later, so the name depends on the installed version. Try the new one,
    fall back to the old.
    """
    try:
        return ax.boxplot(data, tick_labels=labels, **kw)
    except TypeError:
        return ax.boxplot(data, labels=labels, **kw)


def _orient_polar(ax) -> None:
    """theta=0 at +x, increasing clockwise -- matches the displayed image."""
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)


def _resultant(theta_rad: np.ndarray, weight: np.ndarray):
    """Weighted circular resultant. Returns (R, phi_rad)."""
    w = np.asarray(weight, dtype=float)
    ok = np.isfinite(w) & np.isfinite(theta_rad) & (w > 0)
    if ok.sum() == 0 or w[ok].sum() <= 0:
        return np.nan, np.nan
    z = np.sum(w[ok] * np.exp(1j * theta_rad[ok])) / w[ok].sum()
    return float(abs(z)), float(np.angle(z))


def angular_profile(df: pd.DataFrame, n_bins: int = 36):
    """Bin one nucleus/frame/shell into an angular intensity profile.

    Returns (theta_centres_rad, mean_intensity, sample_count) with NaN where no
    ray sampled that wedge -- which happens where the droplet wall clips the
    sweep, and is why R_geom below matters.
    """
    edges = np.linspace(0, 360, n_bins + 1)
    idx = pd.cut(df["theta_deg"] % 360, edges, labels=False, include_lowest=True)
    g = df.assign(_b=idx).groupby("_b")["intensity"]
    mean = g.mean().reindex(range(n_bins)).to_numpy(dtype=float)
    count = g.size().reindex(range(n_bins)).fillna(0).to_numpy(dtype=float)
    centres = np.radians(0.5 * (edges[:-1] + edges[1:]))
    return centres, mean, count


def asymmetry(sweep: pd.DataFrame, shells=SHELLS, n_bins: int = 36,
              baseline_pct: float = 10.0) -> pd.DataFrame:
    """Per (nucleus, frame, shell) membrane asymmetry.

    R        intensity-weighted resultant length after subtracting a shell
             baseline -- 0 is perfectly isotropic, 1 is all signal at one angle
    phi      direction of that resultant, radians, image convention
    R_geom   resultant of the SAMPLE COUNTS alone. This is the asymmetry you
             would measure from an isotropic nucleus purely because the droplet
             wall clipped some rays. R must beat R_geom to mean anything.
    p        Rayleigh approximation exp(-n_eff * R^2) using the effective
             weight count, so a profile carried by two bright bins is not
             credited with the full bin count.
    """
    out = []
    for (nid, t), g in sweep.groupby(["nucleus_id", "time_frame"], sort=False):
        for (lo, hi), label in zip(shells, [f"({lo:.2f}, {hi:.2f}]" for lo, hi in shells]):
            sub = g[(g.rho_normalized > lo) & (g.rho_normalized <= hi)]
            if sub.empty:
                continue
            th, mean, count = angular_profile(sub, n_bins)
            filled = np.isfinite(mean)
            if filled.sum() < 3:
                continue
            base = np.nanpercentile(mean[filled], baseline_pct)
            w = np.clip(mean - base, 0, None)
            w[~filled] = 0.0
            R, phi = _resultant(th, w)
            Rg, _ = _resultant(th, count)
            n_eff = (w.sum() ** 2 / np.sum(w ** 2)) if np.sum(w ** 2) > 0 else 0.0
            out.append(dict(nucleus_id=nid, time_frame=t, shell=label,
                            shell_lo=lo, shell_hi=hi, R=R, phi=phi, R_geom=Rg,
                            n_eff=n_eff, p=float(np.exp(-n_eff * R ** 2)) if R == R else np.nan,
                            n_samples=len(sub), mean_intensity=float(np.nanmean(mean))))
    return pd.DataFrame(out)


def ray_peaks(sweep: pd.DataFrame) -> pd.DataFrame:
    """Per (nucleus, frame, ray): the rho at which intensity peaks.

    The pipeline's envelope rose plots this radius in microns. Only
    centre-normalised rho survives the export, so this is the same quantity in
    different units -- comparable in shape, not in scale.
    """
    idx = sweep.groupby(["nucleus_id", "time_frame", "theta_deg"])["intensity"].idxmax()
    peaks = sweep.loc[idx, ["nucleus_id", "time_frame", "theta_deg",
                            "rho_normalized", "intensity"]]
    return peaks.rename(columns={"rho_normalized": "rho_at_peak",
                                 "intensity": "peak_intensity"})


def population_direction(asym: pd.DataFrame) -> pd.DataFrame:
    """Do nuclei agree on a lab-frame direction, per shell per frame?

    Each nucleus contributes its phi with unit weight, so one very bright
    nucleus cannot carry the population. p is the Rayleigh approximation.
    """
    rows = []
    for (shell, t), g in asym.groupby(["shell", "time_frame"]):
        phi = g["phi"].dropna().to_numpy()
        if phi.size == 0:
            continue
        R, mean_phi = _resultant(phi, np.ones_like(phi))
        rows.append(dict(shell=shell, time_frame=t, n=phi.size, R_pop=R,
                         mean_phi=mean_phi,
                         p=float(np.exp(-phi.size * R ** 2))))
    return pd.DataFrame(rows)


# =============================================================================
# figures -- size and N/C
# =============================================================================

def plot_area_timecourse(df: pd.DataFrame, ax=None, by_condition: bool = True):
    """Cross-sectional area against true acquisition time."""
    ax = ax or plt.subplots(figsize=(8, 5))[1]
    keys = nsdb.condition_keys(df) if by_condition else []
    groups = df.groupby(keys, dropna=False, observed=True) if keys else [("all", df)]
    cmap = plt.get_cmap("viridis")
    for i, (name, g) in enumerate(groups):
        c = cmap(i / max(len(groups) - 1, 1)) if keys else "#4c6fbf"
        ax.scatter(g.time_min, g.cross_sectional_area_um2, s=6, alpha=.20, color=c, lw=0)
        med = g.groupby("time_frame").agg(t=("time_min", "median"),
                                          m=("cross_sectional_area_um2", "median"))
        ax.plot(med.t, med.m, "-o", ms=4, color=c,
                label=" / ".join(map(str, np.atleast_1d(name))))
    ax.set_xlabel("true acquisition time (min)")
    ax.set_ylabel("cross-sectional area (µm²)")
    ax.set_title("Nuclear cross-sectional area over time")
    if keys:
        ax.legend(frameon=False, fontsize=8)
    return ax.figure


def plot_area_bins(df: pd.DataFrame, bin_min: float = 6.0, ax=None):
    """Boxplot per time bin with the median trace -- the v18.1 area figure."""
    ax = ax or plt.subplots(figsize=(11, 5))[1]
    edges = np.arange(0, df.time_min.max() + bin_min, bin_min)
    lab = pd.cut(df.time_min, edges, labels=[f"{int(a)}-{int(b)}"
                                             for a, b in zip(edges[:-1], edges[1:])],
                 include_lowest=True)
    d = df.assign(bin=lab).dropna(subset=["bin"])
    order = [c for c in d["bin"].cat.categories if (d["bin"] == c).any()]
    data = [d.loc[d["bin"] == c, "cross_sectional_area_um2"].to_numpy() for c in order]
    bp = _boxplot(ax, data, order, patch_artist=True, showfliers=False,
                  medianprops=dict(color="k"))
    for b in bp["boxes"]:
        b.set(facecolor="#b8c6e8", alpha=.85, edgecolor="#33415c")
    for i, v in enumerate(data, start=1):
        ax.scatter(np.random.normal(i, .07, v.size), v, s=4, color="k", alpha=.25, lw=0)
        ax.annotate(f"n={v.size}", (i, ax.get_ylim()[1]), ha="center", fontsize=8,
                    color="0.35", xytext=(0, 4), textcoords="offset points")
    ax.plot(range(1, len(data) + 1), [np.median(v) for v in data], "-o",
            color="#c1121f", ms=4, label="median")
    ax.set_xlabel(f"acquisition time bin (min, width {int(bin_min)})")
    ax.set_ylabel("cross-sectional area (µm²)")
    ax.set_title("Nuclear cross-sectional area per time interval")
    ax.legend(frameon=False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return ax.figure


def plot_nc_ratio(df: pd.DataFrame, ax=None):
    """N/C ratio against time.

    Note this is computed WITHOUT background subtraction: the background_*
    columns are empty in the v18.1 export, so the absolute level is not
    trustworthy even though the trend may be.
    """
    ax = ax or plt.subplots(figsize=(8, 5))[1]
    ax.scatter(df.time_min, df.nc_ratio, s=6, alpha=.20, color="#2fa383", lw=0)
    q = df.groupby("time_frame").agg(t=("time_min", "median"),
                                     m=("nc_ratio", "median"),
                                     lo=("nc_ratio", lambda s: s.quantile(.25)),
                                     hi=("nc_ratio", lambda s: s.quantile(.75)))
    ax.plot(q.t, q.m, "-o", color="#14746f", ms=4, label="median ± IQR")
    ax.fill_between(q.t, q.lo, q.hi, color="#14746f", alpha=.18, lw=0)
    ax.set_xlabel("true acquisition time (min)")
    ax.set_ylabel("N/C ratio")
    ax.set_title("N/C ratio over time  (no background subtraction — see docstring)")
    ax.legend(frameon=False)
    return ax.figure


def plot_dual_axis(df: pd.DataFrame, ax=None):
    """N/C ratio (left) and cross-sectional area (right) against time."""
    ax = ax or plt.subplots(figsize=(9, 5))[1]
    g = df.groupby("time_frame")
    t = g.time_min.median()
    nc = g.nc_ratio.median()
    nc_lo, nc_hi = g.nc_ratio.quantile(.25), g.nc_ratio.quantile(.75)
    ar = g.cross_sectional_area_um2.median()
    ar_lo = g.cross_sectional_area_um2.quantile(.25)
    ar_hi = g.cross_sectional_area_um2.quantile(.75)

    ax.plot(t, nc, "-o", color="#14746f", ms=5, label="N/C ratio")
    ax.fill_between(t, nc_lo, nc_hi, color="#14746f", alpha=.15, lw=0)
    ax.set_ylabel("N/C ratio", color="#14746f")
    ax.tick_params(axis="y", labelcolor="#14746f")
    ax.set_xlabel("true acquisition time (min)")

    ax2 = ax.twinx()
    ax2.plot(t, ar, "-s", color="#c1121f", ms=5, label="cross-sectional area")
    ax2.fill_between(t, ar_lo, ar_hi, color="#c1121f", alpha=.13, lw=0)
    ax2.set_ylabel("cross-sectional area (µm²)", color="#c1121f")
    ax2.tick_params(axis="y", labelcolor="#c1121f")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="lower right")
    ax.set_title("N/C ratio and nuclear area (medians ± IQR)")
    return ax.figure


# =============================================================================
# figures -- radial sweep
# =============================================================================

def plot_envelope_rose_individual(peaks: pd.DataFrame, nuclei=None, n: int = 4):
    """Peak-intensity radius by angle, one polar panel per nucleus."""
    if nuclei is None:
        nuclei = (peaks.groupby("nucleus_id").size().sort_values(ascending=False)
                  .head(n).index.tolist())
    frames = sorted(peaks.time_frame.unique())
    cmap = plt.get_cmap("viridis")
    cn = {t: cmap(i / max(len(frames) - 1, 1)) for i, t in enumerate(frames)}

    fig, axes = plt.subplots(1, len(nuclei), figsize=(4.2 * len(nuclei), 4.6),
                             subplot_kw={"projection": "polar"})
    for ax, nid in zip(np.atleast_1d(axes), nuclei):
        _orient_polar(ax)
        d = peaks[peaks.nucleus_id == nid]
        for t, g in d.groupby("time_frame"):
            g = g.sort_values("theta_deg")
            th = np.radians(g.theta_deg.to_numpy())
            r = g.rho_at_peak.to_numpy()
            ax.plot(np.append(th, th[0]), np.append(r, r[0]), lw=.8, color=cn[t])
        ax.set_title(str(nid).split("|")[-1], fontsize=9)
        ax.set_ylim(0, 1)
    handles = [plt.Line2D([], [], color=cn[t], label=f"t={t}") for t in frames]
    fig.legend(handles=handles, loc="center right", frameon=False, fontsize=8, title="frame")
    fig.suptitle("Radius of peak membrane intensity by angle "
                 "(ρ, centre→wall) — individual nuclei")
    fig.tight_layout(rect=[0, 0, .92, .95])
    return fig


def plot_envelope_rose_pooled(peaks: pd.DataFrame, show_iqr: bool = True):
    """Pooled median ± IQR peak radius by angle, per frame."""
    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw={"projection": "polar"})
    _orient_polar(ax)
    frames = sorted(peaks.time_frame.unique())
    cmap = plt.get_cmap("viridis")
    for i, t in enumerate(frames):
        c = cmap(i / max(len(frames) - 1, 1))
        g = (peaks[peaks.time_frame == t].groupby("theta_deg")["rho_at_peak"]
             .agg(med="median", q1=lambda s: s.quantile(.25),
                  q3=lambda s: s.quantile(.75)).reset_index().sort_values("theta_deg"))
        th = np.radians(g.theta_deg.to_numpy())
        th_c = np.append(th, th[0])
        ax.plot(th_c, np.append(g["med"], g["med"].iloc[0]), lw=1.1, color=c, label=f"t={t}")
        if show_iqr:
            ax.fill_between(th_c, np.append(g.q1, g.q1.iloc[0]),
                            np.append(g.q3, g.q3.iloc[0]), color=c, alpha=.10, lw=0)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, title="frame", bbox_to_anchor=(1.22, 1.0))
    ax.set_title("Radius of peak membrane intensity by angle\nmedian ± IQR, pooled over nuclei")
    fig.tight_layout()
    return fig


def plot_angle_distance(sweep: pd.DataFrame, nucleus_id: str, rho_bins: int = 60,
                        max_cols: int = 5, cmap: str = "inferno"):
    """Intensity as a function of angle and radius, one panel per frame."""
    d = sweep[sweep.nucleus_id == nucleus_id]
    frames = sorted(d.time_frame.unique())
    ncol = min(max_cols, len(frames))
    nrow = int(np.ceil(len(frames) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 3.2 * nrow), squeeze=False)
    edges = np.linspace(0, 1, rho_bins + 1)
    vals = d.intensity
    vmin, vmax = np.nanpercentile(vals, [2, 98])
    im = None
    for k, t in enumerate(frames):
        ax = axes[k // ncol][k % ncol]
        sub = d[d.time_frame == t].copy()
        sub["rb"] = pd.cut(sub.rho_normalized, edges, labels=False, include_lowest=True)
        grid = sub.pivot_table(index="theta_deg", columns="rb", values="intensity",
                               aggfunc="mean").reindex(columns=range(rho_bins))
        im = ax.imshow(grid.to_numpy(), aspect="auto", origin="lower", cmap=cmap,
                       vmin=vmin, vmax=vmax, extent=[0, 1, 0, 360])
        ax.set_title(f"t={t}", fontsize=9)
        ax.set_xlabel("ρ (centre → droplet wall)", fontsize=8)
        ax.set_ylabel("θ (deg)", fontsize=8)
        ax.set_yticks([0, 90, 180, 270, 360])
    for k in range(len(frames), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=.7, label="membrane intensity (a.u.)")
    fig.suptitle(f"Membrane intensity by angle and radius — {str(nucleus_id).split('|')[-1]}")
    return fig


def plot_direction_windrose(asym: pd.DataFrame, shell=None, n_bins: int = 24,
                            max_cols: int = 5):
    """Distribution of per-nucleus asymmetry directions, one panel per frame.

    The black line is the population resultant. Its length is R_pop; the
    Rayleigh p tests whether the nuclei agree on a lab-frame direction at all.
    """
    if shell is None:
        shell = sorted(asym.shell.unique())[-2 if asym.shell.nunique() > 1 else 0]
    d = asym[asym.shell == shell]
    pop = population_direction(d).set_index("time_frame")
    frames = sorted(d.time_frame.unique())
    ncol = min(max_cols, len(frames))
    nrow = int(np.ceil(len(frames) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.4 * nrow),
                             subplot_kw={"projection": "polar"}, squeeze=False)
    edges = np.linspace(0, 2 * np.pi, n_bins + 1)
    for k, t in enumerate(frames):
        ax = axes[k // ncol][k % ncol]
        _orient_polar(ax)
        phi = d.loc[d.time_frame == t, "phi"].dropna().to_numpy() % (2 * np.pi)
        counts, _ = np.histogram(phi, bins=edges)
        ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge",
               color="#9b8ec4", edgecolor="w", lw=.4)
        if t in pop.index and np.isfinite(pop.loc[t, "R_pop"]):
            r = pop.loc[t, "R_pop"] * counts.max() if counts.max() else 0
            ax.plot([pop.loc[t, "mean_phi"]] * 2, [0, r], color="k", lw=1.8)
            ax.set_title(f"t={t}  n={phi.size}\nR={pop.loc[t,'R_pop']:.2f}, "
                         f"p={pop.loc[t,'p']:.3g}", fontsize=8)
        ax.set_yticklabels([])
    for k in range(len(frames), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(f"Direction of the membrane-bright side (image frame) — shell ρ {shell}")
    fig.tight_layout(rect=[0, 0, 1, .94])
    return fig


def plot_asymmetry_timecourse(asym: pd.DataFrame, nuclei_df: pd.DataFrame = None,
                              shell=None, bin_min: float = 6.0):
    """Asymmetry against time, its distribution per bin, and population agreement."""
    if shell is None:
        shell = sorted(asym.shell.unique())[-2 if asym.shell.nunique() > 1 else 0]
    d = asym[asym.shell == shell].copy()
    if nuclei_df is not None:
        d = d.merge(nuclei_df[["nucleus_id", "time_frame", "time_min"]],
                    on=["nucleus_id", "time_frame"], how="left")
    else:
        d["time_min"] = d["time_frame"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    a, b, c = axes

    for nid, g in d.groupby("nucleus_id"):
        g = g.sort_values("time_min")
        a.plot(g.time_min, g.R, color="0.75", lw=.5, alpha=.6)
    q = d.groupby("time_frame").agg(t=("time_min", "median"), m=("R", "median"),
                                    lo=("R", lambda s: s.quantile(.25)),
                                    hi=("R", lambda s: s.quantile(.75)))
    a.plot(q.t, q.m, color="#2b6cb0", lw=2, label="median ± IQR")
    a.fill_between(q.t, q.lo, q.hi, color="#2b6cb0", alpha=.25, lw=0)
    a.set_xlabel("true acquisition time (min)")
    a.set_ylabel("asymmetry score R")
    a.set_title("(a) Membrane asymmetry vs time")
    a.legend(frameon=False, fontsize=8)

    edges = np.arange(0, d.time_min.max() + bin_min, bin_min)
    lab = pd.cut(d.time_min, edges, labels=[f"{int(x)}-{int(y)}"
                                            for x, y in zip(edges[:-1], edges[1:])],
                 include_lowest=True)
    dd = d.assign(bin=lab).dropna(subset=["bin"])
    order = [x for x in dd["bin"].cat.categories if (dd["bin"] == x).any()]
    bp = _boxplot(b, [dd.loc[dd["bin"] == x, "R"].to_numpy() for x in order], order,
                  patch_artist=True, showfliers=False, medianprops=dict(color="k"))
    for box in bp["boxes"]:
        box.set(facecolor="#f4c6a8", alpha=.9, edgecolor="#8a5a3b")
    b.set_xlabel(f"time bin (min, width {int(bin_min)})")
    b.set_ylabel("asymmetry score R")
    b.set_title("(b) Distribution per time bin")
    plt.setp(b.get_xticklabels(), rotation=45, ha="right")

    pop = population_direction(d)
    c.plot(pop.time_frame, pop.R_pop, "-o", color="#3fa34d", label="population resultant R")
    crit = np.sqrt(-np.log(0.05) / pop.n.clip(lower=1))
    c.plot(pop.time_frame, crit, "k--", lw=1, label="Rayleigh p=0.05")
    c.set_xlabel("time frame")
    c.set_ylabel("directional concentration R")
    c.set_title("(c) Do all nuclei point the same way?")
    c.legend(frameon=False, fontsize=8)

    fig.suptitle(f"Perinuclear membrane asymmetry — shell ρ {shell}")
    fig.tight_layout(rect=[0, 0, 1, .93])
    return fig


def plot_asymmetry_by_shell(asym: pd.DataFrame):
    """Asymmetry by shell, signal against the geometry floor, population agreement."""
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 4.2))
    shells = sorted(asym.shell.unique())
    cols = dict(zip(shells, SHELL_COLORS))

    for s in shells:
        g = asym[asym.shell == s].groupby("time_frame")
        m, lo, hi = g.R.median(), g.R.quantile(.25), g.R.quantile(.75)
        a.plot(m.index, m, "-o", color=cols[s], ms=4, label=f"ρ {s}")
        a.fill_between(m.index, lo, hi, color=cols[s], alpha=.15, lw=0)
        b.plot(m.index, m, "-o", color=cols[s], ms=4, label=f"ρ {s}")
        b.plot(g.R_geom.median().index, g.R_geom.median(), ":^", color=cols[s], ms=4)
    n_typ = max(int(asym.n_eff.median()), 1)
    a.axhline(np.sqrt(-np.log(0.05) / n_typ), ls="--", color="k", lw=1,
              label="single-nucleus noise floor (p=0.05)")
    a.set_xlabel("time frame"); a.set_ylabel("asymmetry score R")
    a.set_title("(a) Asymmetry score by shell"); a.legend(frameon=False, fontsize=7)
    b.set_xlabel("time frame"); b.set_ylabel("R")
    b.set_title("(b) Signal (solid) vs geometry floor (dotted)")

    pop = population_direction(asym)
    for s in shells:
        g = pop[pop.shell == s]
        c.plot(g.time_frame, g.R_pop, "-o", color=cols[s], ms=4, label=f"ρ {s}")
        c.plot(g.time_frame, np.sqrt(-np.log(.05) / g.n.clip(lower=1)), "--",
               color=cols[s], lw=.8)
    c.set_xlabel("time frame"); c.set_ylabel("population resultant of φ")
    c.set_title("(c) Do nuclei agree on a lab direction?")
    c.legend(frameon=False, fontsize=7)
    fig.suptitle("Normalised asymmetry score by radial shell (ρ, centre→wall)")
    fig.tight_layout(rect=[0, 0, 1, .93])
    return fig


def plot_asymmetry_aggregate(asym: pd.DataFrame, sweep: pd.DataFrame, n_bins: int = 36):
    """Score distribution, ECDF against the geometry floor, and the
    alignment-averaged angular profile (each nucleus rotated so φ→0)."""
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 4.2))
    shells = sorted(asym.shell.unique())
    cols = dict(zip(shells, SHELL_COLORS))

    data = [asym.loc[asym.shell == s, "R"].dropna().to_numpy() for s in shells]
    parts = a.violinplot(data, showmedians=True, showextrema=True)
    for pc, s in zip(parts["bodies"], shells):
        pc.set_facecolor(cols[s]); pc.set_alpha(.55)
    for i, s in enumerate(shells, start=1):
        a.plot([i - .35, i + .35], [asym.loc[asym.shell == s, "R_geom"].median()] * 2,
               ":", color="#c1121f", lw=1.6)
    a.set_xticks(range(1, len(shells) + 1)); a.set_xticklabels(shells, fontsize=7)
    a.set_xlabel("shell (ρ, centre→wall)"); a.set_ylabel("asymmetry score R")
    a.set_title("(a) Score distribution — direction discarded")
    a.plot([], [], ":", color="#c1121f", label="median $R_{geom}$"); a.legend(frameon=False, fontsize=8)

    for s in shells:
        g = asym[asym.shell == s]
        for col, ls in (("R", "-"), ("R_geom", ":")):
            v = np.sort(g[col].dropna().to_numpy())
            if v.size:
                b.plot(v, np.arange(1, v.size + 1) / v.size, ls, color=cols[s], lw=1.3,
                       label=f"ρ {s} (n={v.size})" if col == "R" else None)
    b.set_xlabel("asymmetry score R"); b.set_ylabel("cumulative fraction")
    b.set_title("(b) ECDF (dotted = $R_{geom}$)"); b.legend(frameon=False, fontsize=7)

    phi_lookup = asym.set_index(["nucleus_id", "time_frame", "shell"])["phi"]
    centres = np.linspace(-180, 180, n_bins, endpoint=False) + 180 / n_bins
    for s in shells:
        lo, hi = asym.loc[asym.shell == s, ["shell_lo", "shell_hi"]].iloc[0]
        stack = []
        sub_all = sweep[(sweep.rho_normalized > lo) & (sweep.rho_normalized <= hi)]
        for (nid, t), g in sub_all.groupby(["nucleus_id", "time_frame"], sort=False):
            try:
                phi = phi_lookup.loc[(nid, t, s)]
            except KeyError:
                continue
            if not np.isfinite(phi):
                continue
            th, mean, _ = angular_profile(g, n_bins)
            if not np.isfinite(mean).any() or np.nanmean(mean) <= 0:
                continue
            shift = int(round(np.degrees(phi) / (360 / n_bins)))
            stack.append(np.roll(mean / np.nanmean(mean), -shift + n_bins // 2))
        if stack:
            arr = np.vstack(stack)
            med = np.nanmedian(arr, axis=0)
            se = np.nanstd(arr, axis=0) / max(np.sqrt(arr.shape[0]), 1)
            c.plot(centres, med, color=cols[s], lw=1.4, label=f"ρ {s} (n={arr.shape[0]})")
            c.fill_between(centres, med - se, med + se, color=cols[s], alpha=.2, lw=0)
    c.axhline(1.0, color="k", lw=.7)
    c.set_xlabel("angle from each nucleus's own asymmetry axis (°)")
    c.set_ylabel("intensity / shell mean")
    c.set_xticks([-180, -90, 0, 90, 180])
    c.set_title("(c) Alignment-averaged profile"); c.legend(frameon=False, fontsize=7)
    fig.suptitle("Aggregate membrane asymmetry")
    fig.tight_layout(rect=[0, 0, 1, .93])
    return fig


def plot_shell_rose_grid(sweep: pd.DataFrame, nucleus_id: str, shells=SHELLS,
                         n_bins: int = 36, baseline_pct: float = 10.0):
    """Shell x frame grid of angular roses with the resultant arrow.

    Wedge radius is proportional to sqrt(intensity above the shell baseline),
    so wedge AREA is proportional to intensity. Grey wedges are angles with no
    sample -- almost always droplet-wall clipping.
    """
    d = sweep[sweep.nucleus_id == nucleus_id]
    frames = sorted(d.time_frame.unique())
    fig, axes = plt.subplots(len(shells), len(frames),
                             figsize=(2.3 * len(frames), 2.5 * len(shells)),
                             subplot_kw={"projection": "polar"}, squeeze=False)
    edges = np.linspace(0, 2 * np.pi, n_bins + 1)
    for r, ((lo, hi), col) in enumerate(zip(shells, SHELL_COLORS)):
        for k, t in enumerate(frames):
            ax = axes[r][k]; _orient_polar(ax)
            sub = d[(d.time_frame == t) & (d.rho_normalized > lo)
                    & (d.rho_normalized <= hi)]
            if sub.empty:
                ax.set_xticklabels([]); ax.set_yticklabels([]); continue
            th, mean, count = angular_profile(sub, n_bins)
            base = np.nanpercentile(mean[np.isfinite(mean)], baseline_pct)
            w = np.clip(np.nan_to_num(mean - base), 0, None)
            ax.bar(edges[:-1], np.sqrt(w), width=np.diff(edges), align="edge",
                   color=col, edgecolor="w", lw=.3)
            miss = count == 0
            if miss.any():
                ax.bar(edges[:-1][miss], np.full(miss.sum(), np.sqrt(w).max() or 1),
                       width=np.diff(edges)[miss], align="edge", color="0.85", zorder=0)
            R, phi = _resultant(th, w)
            if np.isfinite(R):
                ax.annotate("", xy=(phi, R * (np.sqrt(w).max() or 1)), xytext=(0, 0),
                            arrowprops=dict(color="#c1121f", width=1.2, headwidth=6))
                ax.set_title(f"R={R:.2f}", fontsize=7, color="0.3")
            ax.set_xticklabels([]); ax.set_yticklabels([])
            if k == 0:
                ax.set_ylabel(f"ρ ({lo:.2f}, {hi:.2f}]", fontsize=7)
            if r == 0:
                ax.text(.5, 1.25, f"t = {t}", transform=ax.transAxes, ha="center", fontsize=9)
    fig.suptitle(f"Membrane intensity by angle and shell — {str(nucleus_id).split('|')[-1]}\n"
                 "wedge area ∝ intensity above shell baseline · red arrow = resultant · "
                 "grey = no sample", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, .93])
    return fig
