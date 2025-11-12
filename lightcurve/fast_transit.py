import sys
from pathlib import Path
from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def _to_numpy(data: Any) -> np.ndarray:
    if hasattr(data, "to_value"):
        return np.asarray(data.to_value(), dtype=np.float64)
    if hasattr(data, "value"):
        return np.asarray(data.value, dtype=np.float64)
    return np.asarray(data, dtype=np.float64)


def lc_to_arrays(lc: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a lightkurve.LightCurve-like object to (time, flux) numpy arrays."""
    try:
        time_arr = _to_numpy(lc.time)
        flux_arr = _to_numpy(lc.flux)
    except AttributeError as exc:
        raise ValueError("Provided lightcurve lacks time/flux attributes") from exc

    mask = np.isfinite(time_arr) & np.isfinite(flux_arr)
    if not np.any(mask):
        raise ValueError("Lightcurve arrays contain no finite samples")

    return time_arr[mask], flux_arr[mask]

# --- Trapezoidal transit model ---
def trapezoid_model(t, t0, depth, duration, ingress, baseline):
    tau = ingress
    T = duration
    d = depth
    t1 = t0 - T/2
    t2 = t0 - tau/2
    t3 = t0 + tau/2
    t4 = t0 + T/2
    y = np.ones_like(t) * baseline
    # Ingress
    mask_ingress = (t >= t1) & (t < t2)
    y[mask_ingress] -= d * (t[mask_ingress] - t1) / (t2 - t1)
    # Flat bottom
    mask_flat = (t >= t2) & (t <= t3)
    y[mask_flat] -= d
    # Egress
    mask_egress = (t > t3) & (t <= t4)
    y[mask_egress] -= d * (1 - (t[mask_egress] - t3) / (t4 - t3))
    return y


def fit_trapezoid_from_arrays(time: np.ndarray, flux: np.ndarray, outdir: Path | None = None):
    """Fit the trapezoidal transit model to arrays of time and flux.

    Returns fitted parameters (t0, depth, duration, ingress, baseline) and the covariance matrix.
    If plot is True, shows a quick matplotlib visual check.
    """
    # --- Initial guesses ---
    t0_guess = time[np.argmin(flux)]              # Midpoint: time of minimum flux
    depth_guess = np.median(np.abs(flux - np.median(flux)))  # Approximate depth
    duration_guess = 0.05                         # Duration (days), adjust as needed
    ingress_guess = 0.01                          # Ingress/egress duration (days)
    baseline_guess = np.median(flux)              # Out-of-transit baseline

    p0 = [t0_guess, depth_guess, duration_guess, ingress_guess, baseline_guess]

    # --- Fit the model ---
    popt, pcov = curve_fit(trapezoid_model, time, flux, p0=p0, maxfev=10000)
    t0_fit, depth_fit, duration_fit, ingress_fit, baseline_fit = popt

    # --- Print results ---
    print(f"Transit midpoint (t0):       {t0_fit:.6f} days")
    print(f"Transit depth (delta):      {depth_fit:.6f} (fractional)")
    print(f"Transit duration (T):       {duration_fit:.6f} days")
    print(f"Ingress/egress (tau):       {ingress_fit:.6f} days")
    print(f"Baseline flux (out-of-transit): {baseline_fit:.6f}")

    fig, ax = plt.subplots()
    ax.plot(time, flux, '.k', label='Data')
    ax.plot(time, trapezoid_model(time, *popt), 'r-', label='Trapezoid fit')
    ax.set_xlabel('Time [days]')
    ax.set_ylabel('Flux')
    ax.legend()
    ax.set_title('Trapezoidal Transit Fit')
    fig.tight_layout()

    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / "04_fit_trapezoid.png", dpi=150)
        plt.close(fig)
    else:
        plt.show()

    return popt, pcov


def read_csv_lightcurve(csv_file: str | Path, time_col: str = 'time', flux_col: str = 'flux') -> Tuple[np.ndarray, np.ndarray]:
    """Read lightcurve data from a CSV file.
    
    Parameters:
    -----------
    csv_file : str or Path
        Path to the CSV file containing lightcurve data
    time_col : str, optional
        Name of the time column in the CSV file (default: 'time')
    flux_col : str, optional
        Name of the flux column in the CSV file (default: 'flux')
        
    Returns:
    --------
    time : np.ndarray
        Array of time values
    flux : np.ndarray
        Array of flux values
    """
    try:
        data = np.genfromtxt(csv_file, delimiter=",", names=True, dtype=np.float64)
    except Exception as exc:
        raise ValueError(f"Error reading CSV file '{csv_file}': {exc}")

    if isinstance(data, np.ndarray) and data.size == 0:
        raise ValueError(f"CSV file '{csv_file}' is empty or improperly formatted")

    available_cols = data.dtype.names or []
    if time_col not in available_cols:
        raise ValueError(f"Time column '{time_col}' not found in CSV. Available columns: {available_cols}")
    if flux_col not in available_cols:
        raise ValueError(f"Flux column '{flux_col}' not found in CSV. Available columns: {available_cols}")

    time = np.atleast_1d(data[time_col])
    flux = np.atleast_1d(data[flux_col])

    mask = np.isfinite(time) & np.isfinite(flux)
    if not np.any(mask):
        raise ValueError(f"No finite values found in CSV columns '{time_col}' and '{flux_col}'")

    time = time[mask]
    flux = flux[mask]

    print(f"Successfully loaded {len(time)} data points from {csv_file}")
    return time.astype(float), flux.astype(float)


def fit_trapezoid_from_csv(csv_file: str | Path, time_col: str = 'time', flux_col: str = 'flux', outdir: Path | None = None):
    """Fit trapezoidal transit model to lightcurve data from a CSV file.
    
    Parameters:
    -----------
    csv_file : str or Path
        Path to the CSV file containing lightcurve data
    time_col : str, optional
        Name of the time column in the CSV file (default: 'time')
    flux_col : str, optional
        Name of the flux column in the CSV file (default: 'flux')
    outdir : Path, optional
        Directory to save the plot (if None, plot is displayed)
        
    Returns:
    --------
    tuple
        Fitted parameters (t0, depth, duration, ingress, baseline)
    """
    time, flux = read_csv_lightcurve(csv_file, time_col, flux_col)
    popt, _ = fit_trapezoid_from_arrays(time, flux, outdir=outdir)
    return tuple(popt)


def fit_trapezoid_from_lightcurve(lc: Any, outdir: Path | None = None):
    """Convenience wrapper: accept a lightkurve.LightCurve (or equivalent) and fit the model."""
    time, flux = lc_to_arrays(lc)
    popt, _ = fit_trapezoid_from_arrays(time, flux, outdir=outdir)
    return tuple(popt)
