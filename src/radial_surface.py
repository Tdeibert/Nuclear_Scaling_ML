"""
radial_surface -- theta x rho intensity surfaces from the radial sweep.

The sweep casts `radial_n_angles` rays from the nucleus centre out to the
droplet edge, sampling at integer pixel steps. Each ray therefore has a
different number of samples, and rho_normalized is not on a shared grid --
so the points must be binned in rho before they can be pivoted into the
regular (n_theta x n_rho) array that plot_surface requires.

Two conventions worth knowing:

* theta is periodic. The grid is closed by repeating the first ray at
  theta + 360 so the surface has no seam.

* image y increases downward, and the sweep casts rays as
  y = cy + r*sin(theta), so theta runs CLOCKWISE on screen with theta=0
  pointing right. Keep that in mind when reading angular features off the
  plot -- it matches the pipeline's own polar figures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sweep_grid(df: pd.DataFrame, rho_bins: int = 60, agg: str = "mean",
               rho_max: float = 1.0, close_theta: bool = True):
    """Bin a long-form sweep into a regular theta x rho intensity grid.

    Returns (THETA, RHO, Z) meshgrid-style arrays ready for plot_surface.
    Z is intensity; NaN where no sample fell in a bin.
    """
    need = {"theta_deg", "rho_normalized", "intensity"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"missing columns: {sorted(missing)}")
    d = df[df["rho_normalized"].between(0, rho_max)].copy()
    if d.empty:
        raise ValueError("no samples in the requested rho range")

    edges = np.linspace(0, rho_max, rho_bins + 1)
    d["rho_bin"] = pd.cut(d["rho_normalized"], edges, labels=False,
                          include_lowest=True)

    grid = (d.pivot_table(index="theta_deg", columns="rho_bin",
                          values="intensity", aggfunc=agg)
              .reindex(columns=range(rho_bins)))

    theta = grid.index.to_numpy(dtype=float)
    z = grid.to_numpy(dtype=float)

    if close_theta and theta.size > 1:
        theta = np.append(theta, theta[0] + 360.0)
        z = np.vstack([z, z[0]])

    rho = 0.5 * (edges[:-1] + edges[1:])          # bin centres
    THETA, RHO = np.meshgrid(theta, rho, indexing="ij")
    return THETA, RHO, z


def fill_gaps(z: np.ndarray) -> np.ndarray:
    """Interpolate NaN holes along rho.

    plot_surface renders a NaN as a hole in the mesh. Sparse rho bins at large
    radius are common (rays get shorter near the droplet wall), so small gaps
    are bridged. Bins with no data at any angle stay NaN -- those are real
    absence, not noise, and should look like absence.
    """
    out = z.copy()
    for i in range(out.shape[0]):
        row = out[i]
        ok = ~np.isnan(row)
        if ok.sum() >= 2:
            idx = np.arange(row.size)
            out[i] = np.interp(idx, idx[ok], row[ok], left=np.nan, right=np.nan)
    return out


def polar_xy(THETA: np.ndarray, RHO: np.ndarray):
    """Convert the theta/rho grid to Cartesian, for a disc-shaped surface.

    Use when you want the surface to sit over the physical footprint of the
    nucleus rather than over a rectangular angle-vs-radius domain.
    """
    th = np.radians(THETA)
    return RHO * np.cos(th), RHO * np.sin(th)
