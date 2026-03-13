import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from typing import Union, Tuple, Optional, Callable, Any

# ────────────────────────────────────────
#  File loading helpers
# ────────────────────────────────────────

def load_table(
    path: str,
    format: str = "auto",
    hdu: int = 1,
    dropna: bool = False,                  # new: default False → safe
    dropna_subset: Optional[list[str]] = None,  # optional: only drop if these columns are NaN
) -> Union[pd.DataFrame, np.recarray]:
    """
    Unified loader for csv / parquet / fits table.
    
    Args:
        dropna: If True, drop rows where **any** column is NaN (after loading as DataFrame).
        dropna_subset: If provided, only drop rows where these specific columns are NaN.
                       Ignored if dropna=False.
    """
    fmt = format.lower()
    if fmt == "auto":
        if path.endswith(".csv"):
            fmt = "csv"
        elif path.endswith((".parquet", ".pq")):
            fmt = "parquet"
        elif path.endswith((".fits", ".fit")):
            fmt = "fits"

    if fmt == "csv":
        df = pd.read_csv(path)
    elif fmt == "parquet":
        df = pd.read_parquet(path)
    elif fmt == "fits":
        with fits.open(path) as hdul:
            data = hdul[hdu].data
            # Convert FITS recarray → DataFrame for consistency & easier NaN handling
            df = pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    # Optional NaN cleaning (only on DataFrame)
    if dropna:
        if dropna_subset:
            df = df.dropna(subset=dropna_subset)
        else:
            df = df.dropna()  # any column NaN → drop row

    # Decide return type
    if fmt == "fits" and not dropna:
        # If no dropna requested, return original recarray for fidelity
        return data
    else:
        return df


def extract_columns(
    table: Union[pd.DataFrame, np.recarray],
    columns: list[str],
    dropna: bool = True,
) -> dict[str, np.ndarray]:
    """Extract multiple columns → dict of arrays"""
    out = {}
    is_df = isinstance(table, pd.DataFrame)

    for col in columns:
        if is_df:
            s = table[col]
            if dropna:
                s = s.dropna()
            out[col] = s.to_numpy()
        else:  # fits recarray / structured array
            arr = table[col]
            if dropna:
                arr = arr[~np.isnan(arr)]
            out[col] = arr

    return out


# ────────────────────────────────────────
#  Histogram & density helpers
# ────────────────────────────────────────

def hist_density(
    data: np.ndarray,
    bins: Union[int, np.ndarray] = 100,
    range: Optional[Tuple[float, float]] = None,
    clip: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute normalized (density=True) histogram + bin centers"""
    if clip is not None:
        data = data[(data >= clip[0]) & (data <= clip[1])]

    hist, edges = np.histogram(
        data,
        bins=bins,
        range=range,
        density=True
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    return hist, centers


def plot_step_hist(
    hists: list[Tuple[np.ndarray, np.ndarray]],     # list of (values, centers)
    labels: list[str],
    colors: Optional[list[str]] = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "Normalized density",
    xlim: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (9, 5.5),
    save: Optional[str] = None,
) -> None:
    """Plot multiple step histograms on same axes"""
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 0.85, len(hists)))

    plt.figure(figsize=figsize)

    for (vals, cents), label, c in zip(hists, labels, colors):
        plt.step(cents, vals, label=label, where="mid", color=c, lw=1.6)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    if xlim:
        plt.xlim(xlim)
    plt.grid(alpha=0.35)
    plt.legend(frameon=True, loc="best")

    if save:
        plt.savefig(save, dpi=160, bbox_inches="tight")

    plt.show()


# ────────────────────────────────────────
#  Very generic plot wrapper (optional)
# ────────────────────────────────────────

def quick_plot(
    func: Callable,
    *args,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    save: Optional[str] = None,
    **kwargs
) -> None:
    """Minimal wrapper if you want even less boilerplate in notebooks"""
    plt.figure(figsize=kwargs.pop("figsize", (9, 5.5)))
    func(*args, **kwargs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.grid(alpha=0.3)
    if save:
        plt.savefig(save, dpi=160, bbox_inches="tight")
    plt.show()


def flux_to_ab_magnitude(
    flux: np.ndarray,
    zp: float = 22.5,
    min_flux: float = 1e-30,
) -> np.ndarray:
    """
    Convert flux → AB magnitude (any band).
    Safe for zero/negative fluxes → returns NaN.
    """
    flux = np.asarray(flux, dtype=float)
    mag = np.full_like(flux, np.nan)
    valid = flux > min_flux
    mag[valid] = -2.5 * np.log10(flux[valid]) + zp
    return mag