"""Reusable matplotlib plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal, Sequence

import matplotlib.pyplot as plt


PlotKind = Literal["line", "scatter", "bar"]


def plot_xy(
    x: Sequence[float],
    y: Sequence[float],
    *,
    kind: PlotKind = "line",
    yerr: Sequence[float] | None = None,
    label: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    color: str | None = None,
    marker: str | None = None,
    figsize: tuple[float, float] = (7.0, 4.5),
    grid: bool = True,
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
    dpi: int = 300,
    show: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Create a clean x/y matplotlib plot.

    Parameters
    ----------
    x, y:
        Values to plot. Both sequences must have the same length.
    kind:
        Plot type: "line", "scatter", or "bar".
    yerr:
        Optional vertical error values. Supported for line/scatter via
        ``errorbar`` and for bar via ``bar``.
    ax:
        Existing matplotlib axis to draw on. If omitted, a new figure and axis
        are created.
    save_path:
        Optional output path. Parent directories are created automatically.
    show:
        Call ``plt.show()`` before returning. Keep this ``False`` in scripts
        and most notebooks when you want to further customize the axis.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The figure and axis for additional customization or testing.
    """

    if len(x) != len(y):
        raise ValueError(f"x and y must have the same length, got {len(x)} and {len(y)}")
    if yerr is not None and len(yerr) != len(y):
        raise ValueError(f"yerr must match y length, got {len(yerr)} and {len(y)}")
    if kind not in {"line", "scatter", "bar"}:
        raise ValueError(f"Unsupported plot kind: {kind!r}")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if kind == "line":
        if yerr is None:
            ax.plot(x, y, label=label, color=color, marker=marker)
        else:
            ax.errorbar(x, y, yerr=yerr, label=label, color=color, marker=marker, capsize=3)
    elif kind == "scatter":
        if yerr is None:
            ax.scatter(x, y, label=label, color=color, marker=marker)
        else:
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt=marker or "o",
                label=label,
                color=color,
                capsize=3,
                linestyle="none",
            )
    else:
        ax.bar(x, y, yerr=yerr, label=label, color=color, capsize=3 if yerr is not None else 0)

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if grid:
        ax.grid(True, alpha=0.25)
    if label:
        ax.legend(frameon=False)

    fig.tight_layout()

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def plot_histogram(
    data: Sequence[float],
    *,
    bins: int | Sequence[float] | str = 30,
    density: bool = False,
    alpha: float = 0.7,
    label: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    color: str | None = None,
    figsize: tuple[float, float] = (7.0, 4.5),
    grid: bool = True,
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
    dpi: int = 300,
    show: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Create a histogram plot.

    Parameters
    ----------
    data : Sequence[float]
        The data to plot.
    bins : int, sequence, or str, default 30
        Number of bins, bin edges, or binning strategy.
    density : bool, default False
        If True, draw and return a probability density.
    alpha : float, default 0.7
        Transparency level.
    label : str, optional
        Label for the histogram.
    title : str, optional
        Plot title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    color : str, optional
        Histogram color.
    figsize : tuple[float, float], default (7.0, 4.5)
        Figure size.
    grid : bool, default True
        Whether to show grid.
    ax : plt.Axes, optional
        Existing matplotlib axis to draw on.
    save_path : str or Path, optional
        Output path for saving the plot.
    dpi : int, default 300
        DPI for saved figure.
    show : bool, default False
        Whether to call plt.show().

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        The figure and axis.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.hist(data, bins=bins, density=density, alpha=alpha, label=label, color=color)

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if grid:
        ax.grid(True, alpha=0.25)
    if label:
        ax.legend(frameon=False)

    fig.tight_layout()

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def plot_multiple_xy(
    series: Iterable[tuple[Sequence[float], Sequence[float], str]],
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] = (7.0, 4.5),
    grid: bool = True,
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
    dpi: int = 300,
    show: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot multiple labeled x/y line series on one axis."""

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for x, y, label in series:
        if len(x) != len(y):
            raise ValueError(f"Series {label!r} has mismatched x/y lengths")
        ax.plot(x, y, marker="o", label=label)

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if grid:
        ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax
