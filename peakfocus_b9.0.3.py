#!/usr/bin/env python3
"""
PeakFocus™ b9.0.3 (Cleaned & Improved, Pre-modularization)

Script for periodic peak detection, segmentation, and ranking in timeseries CSV data.
Incorporates pre-modularization priorities: robust I/O, config/CLI validation, separation, refactored exports, and edge handling.
"""

__version__ = "b9.0.3"

import os
import sys
import json
import argparse
import logging
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import savgol_filter, find_peaks, peak_widths, windows
from scipy.ndimage import gaussian_filter1d
from datetime import datetime

# === Constants ===
MIN_SEGMENT_LENGTH = 1e-3
MIN_STRIDE = 1e-3
SAVGOL_WINDOW_DEFAULT = 51
SAVGOL_POLY_DEFAULT = 3
DEFAULT_DESIRED_SNR = 300
ALLOWED_SNRESTIMATOR = {"median", "mean"}
SUMMARY_FILENAME = "run_summary.csv"
ERROR_LOG_FILENAME = "error_report.txt"

RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"

# === 1. Robust Input/Output & 2. Config/CLI Consistency ===

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PeakFocus data processing")
    parser.add_argument('--config', type=str, default='peakfocus_config.json', help='Path to JSON config file')
    parser.add_argument('--no-interactive', action='store_true', help='Run in non-interactive (batch) mode')
    parser.add_argument('--input-dir', type=str, default='.', help='Directory containing input CSV files')
    parser.add_argument('--file-pattern', type=str, default='*.csv', help='Glob pattern for input files')
    parser.add_argument('--quiet', action='store_true', help='Suppress console logging & progress bar')
    return parser.parse_args()

def validate_prefs(prefs: Dict[str, Any]) -> List[str]:
    """Validate config fields, warn on unknowns, enforce types/ranges. Returns list of warnings."""
    warnings = []
    # Known fields
    allowed_keys = {
        "SkipInitial", "ScanWindowSec", "SegmentLength", "Stride", "DesiredSNR",
        "AutoTrim", "Simulate", "OverlayReshape", "ReplaceWithReshape", "AutoDetectCooldown",
        "ExportSegCSV", "ExportSegPNG", "ExportBestCSV", "ExportBestPNG", "ExportPerFilePDF",
        "ExportOverallPDF", "ExportShapeCSV", "BestSegmentsCount", "ExpectedSpacingMs",
        "ExpectedWidthMs", "SNREstimator", "PerFileBestCount", "OverallBestCount",
        "RankingScheme", "RankWeights", "ExportPerFileBestPNG", "ExportPerFileBestCSV"
    }
    for key in list(prefs.keys()):
        if key not in allowed_keys:
            warnings.append(f"Unknown config key: {key}")
    # Type and value checks
    if prefs.get('SegmentLength', 0) < MIN_SEGMENT_LENGTH:
        warnings.append("SegmentLength must be > 0")
    if prefs.get('Stride', 0) < MIN_STRIDE:
        warnings.append("Stride must be > 0")
    if prefs.get('BestSegmentsCount', 0) < 1:
        warnings.append("BestSegmentsCount must be >= 1")
    if prefs.get("SNREstimator", "median") not in ALLOWED_SNRESTIMATOR:
        warnings.append(f"SNREstimator should be one of {ALLOWED_SNRESTIMATOR}")
    # Normalize/force some config
    prefs['PerFileBestCount']  = max(1, int(prefs.get('PerFileBestCount', 1)))
    prefs['OverallBestCount']  = max(1, int(prefs.get('OverallBestCount', 5)))
    if "ExportPerFileBestPNG" not in prefs:
        prefs["ExportPerFileBestPNG"] = True
    if "ExportPerFileBestCSV" not in prefs:
        prefs["ExportPerFileBestCSV"] = True
    return warnings

def load_preferences(path: str) -> Dict[str, Any]:
    DEFAULT_CONFIG = {
        "SkipInitial": 7.0,
        "ScanWindowSec": 10.0,
        "SegmentLength": 0.300,
        "Stride": 0.150,
        "DesiredSNR": DEFAULT_DESIRED_SNR,
        "AutoTrim": True,
        "Simulate": True,
        "OverlayReshape": True,
        "ReplaceWithReshape": False,
        "AutoDetectCooldown": True,
        "ExportSegCSV": False,
        "ExportSegPNG": False,
        "ExportBestCSV": True,
        "ExportBestPNG": True,
        "ExportPerFilePDF": False,
        "ExportOverallPDF": True,
        "ExportShapeCSV": False,
        "BestSegmentsCount": 3,
        "ExpectedSpacingMs": 50,
        "ExpectedWidthMs": 10,
        "SNREstimator": "median"
    }
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                cfg = json.load(f)
            return {**DEFAULT_CONFIG, **cfg}
        except Exception as e:
            print(f"Warning: failed to load config, using defaults. Error: {e}")
    return DEFAULT_CONFIG.copy()

def save_preferences(path: str, prefs: Dict[str, Any]) -> None:
    with open(path, 'w') as f:
        json.dump(prefs, f, indent=2)

def setup_logging(log_path: Path, level: int = logging.INFO) -> None:
    logging.basicConfig(
        filename=str(log_path),
        filemode='w',
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger('').addHandler(console)

# === 3. Separation: CLI, IO, Analysis, Plotting, Export ===

def robust_read_csv(path: str) -> Optional[pd.DataFrame]:
    """Try to robustly read a CSV file, skip bad header rows, ensure at least two numeric columns."""
    skiprows = 0
    try:
        with open(path, 'r') as f:
            for line in f:
                tokens = line.strip().split(',')
                try:
                    float(tokens[0]); float(tokens[1])
                    break
                except Exception:
                    skiprows += 1
        df = pd.read_csv(path, skiprows=skiprows, header=None)
        if df.shape[1] < 2:
            return None
        return df
    except Exception as e:
        logging.warning(f"Could not read {path}: {e}")
        return None

def is_monotonic_and_unique(arr: np.ndarray) -> bool:
    """True if array is strictly increasing and unique."""
    return np.all(np.diff(arr) > 0) and len(np.unique(arr)) == len(arr)

def log_and_print(msg: str, level=logging.INFO):
    print(msg)
    logging.log(level, msg)

def export_csv(data: dict, out_path: Path):
    try:
        pd.DataFrame(data).to_csv(out_path, index=False)
    except Exception as e:
        logging.warning(f"Failed to write CSV {out_path}: {e}")

def export_png(fig, out_path: Path):
    try:
        fig.savefig(out_path)
        plt.close(fig)
    except Exception as e:
        logging.warning(f"Failed to write PNG {out_path}: {e}")

# === 4. Refactor Repeated Exports/Helpers ===

def export_segment_png(ts, raw, proc, ideal, fname, idx_, outdir, prefs):
    title1 = f"{fname} seg{idx_}"
    fig = plot_segment(ts, raw, proc, ideal, title1, prefs)
    export_png(fig, outdir / f"{fname}_seg{idx_}.png")

def export_segment_csv(ts, raw, proc, ideal, snr_raw, snr_proc, snr_base_raw, snr_base_proc, shape_raw, shape_proc, fname, idx_, outdir, prefs):
    const_len = len(ts)
    out = {
        'time': ts,
        'raw': raw,
        'reshaped': proc
    }
    if prefs.get('Simulate', True) and ideal is not None:
        out['ideal'] = ideal
    out.update({
        'snr_baseline_raw'   : [snr_base_raw]  * const_len,
        'snr_baseline_proc'  : [snr_base_proc] * const_len,
        'snr_raw'            : [snr_raw]       * const_len,
        'snr_reshaped'       : [snr_proc]      * const_len,
        'width_raw_ms'       : [shape_raw['avg_width']]        * const_len,
        'spacing_raw_ms'     : [shape_raw['avg_spacing']]      * const_len,
        'width_resh_ms'      : [shape_proc['avg_width']]   * const_len,
        'spacing_resh_ms'    : [shape_proc['avg_spacing']]* const_len,
    })
    export_csv(out, outdir / f"{fname}_seg{idx_}.csv")

# === 4b. Overview Trace Export ===
def export_overview_trace(ts, raw, corr=None, outdir: Path=None, prefs: Dict[str, Any]=None, max_points: int=5000):
    """Save a lightweight overall trace for the Analyse GUI.

    Writes:
      - overview.csv with columns: time, raw, (optional) reshaped, (optional) ideal
      - overview.png quicklook plot

    `ts`, `raw`, `corr` should already reflect SkipInitial & ScanWindowSec.
    Downsamples to <= max_points for snappy plotting in the GUI.
    """
    try:
        if outdir is None:
            return
        outdir.mkdir(exist_ok=True, parents=True)
        import numpy as _np
        import pandas as _pd
        n = len(ts)
        if n == 0:
            return
        if max_points and n > max_points:
            idx = _np.linspace(0, n-1, max_points).astype(int)
        else:
            idx = _np.arange(n)
        data = {
            'time': _np.asarray(ts)[idx],
            'raw': _np.asarray(raw)[idx],
        }
        if corr is not None:
            data['reshaped'] = _np.asarray(corr)[idx]
        # Optional ideal across the full window for context
        if prefs and prefs.get('Simulate', True):
            try:
                ideal = simulate_ideal(_np.asarray(ts), _np.asarray(raw), _np.asarray(corr) if corr is not None else None,
                                       prefs.get('ExpectedSpacingMs', 0), prefs.get('ExpectedWidthMs', 0))
                data['ideal'] = _np.asarray(ideal)[idx]
            except Exception:
                pass
        df = _pd.DataFrame(data)
        df.to_csv(outdir / 'overview.csv', index=False)
        # quick PNG
        try:
            fig = plt.figure(figsize=(8,2.0), dpi=120)
            ax = fig.add_subplot(111)
            ax.plot(df['time'], df['raw'], label='Raw')
            if 'reshaped' in df.columns:
                ax.plot(df['time'], df['reshaped'], label='Reshaped')
            if 'ideal' in df.columns:
                ax.plot(df['time'], df['ideal'], label='Ideal', alpha=0.6)
            ax.set_xlabel('Time (s)'); ax.set_ylabel('CPS')
            ax.legend(loc='upper right', fontsize=8)
            fig.tight_layout()
            fig.savefig(outdir / 'overview.png')
            plt.close(fig)
        except Exception:
            pass
    except Exception as e:
        logging.warning(f"Failed to export overview trace: {e}")

# === 5. Edge Case and NaN Handling ===

def is_valid_segment_metrics(metrics: dict) -> bool:
    """Return True if all metrics are finite and not NaN."""
    for k, v in metrics.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return False
    return True

def drop_invalid_segments(stats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop segments where metrics are NaN/inf or the segment is empty."""
    return [
        s for s in stats
        if is_valid_segment_metrics(s['shape_raw']) and is_valid_segment_metrics(s['shape_reshaped'])
    ]

# === Interactive CLI Prompts, unchanged but now always reload config after ===

def prompt_yes_no(msg: str, default: bool) -> bool:
    d = 'Y' if default else 'N'
    while True:
        ans = input(f"{CYAN}{msg} (y/n) [{d}]: {RESET}").strip().lower()
        if not ans:
            return default
        if ans in ('y', 'yes'):
            return True
        if ans in ('n', 'no'):
            return False
        print("Please enter 'y' or 'n'.")

def prompt_int(msg: str, default: int) -> int:
    ans = input(f"{CYAN}{msg} [{default}]: {RESET}").strip()
    if not ans:
        return default
    try:
        return int(ans)
    except ValueError:
        print("Invalid integer; using default.")
        return default

def prompt_float(msg: str, default: float) -> float:
    ans = input(f"{CYAN}{msg} [{default}]: {RESET}").strip()
    if not ans:
        return default
    try:
        return float(ans)
    except ValueError:
        print("Invalid number; using default.")
        return default

def interactive_prompt(prefs: Dict[str, Any]) -> Dict[str, Any]:
    print("\n=== Data Handling Configuration ===")
    # Baseline & warm-up
    prefs['AutoTrim'] = prompt_yes_no("Auto-trim baseline warm-up?", prefs['AutoTrim'])
    prefs['AutoDetectCooldown'] = prompt_yes_no("Auto-detect cooldown cutoff?", prefs['AutoDetectCooldown'])
    prefs['SkipInitial'] = prompt_float("Skip initial warm-up time (s)", prefs['SkipInitial'])
    prefs['ScanWindowSec'] = prompt_float("Scan window after warm-up (s)", prefs['ScanWindowSec'])

    # Segmentation
    prefs['SegmentLength'] = prompt_float("Segment length (s)", prefs['SegmentLength'])
    prefs['Stride'] = prompt_float("Stride between segments (s)", prefs['Stride'])

    # Peak expectations & reshape
    prefs['ExpectedSpacingMs'] = prompt_int("Expected spacing between peaks (ms)", prefs['ExpectedSpacingMs'])
    prefs['ExpectedWidthMs']   = prompt_int("Expected peak width (ms)", prefs['ExpectedWidthMs'])
    prefs['OverlayReshape']    = prompt_yes_no("Overlay reshaped peaks on plots?", prefs['OverlayReshape'])
    if prefs['OverlayReshape']:
        prefs['ReplaceWithReshape'] = prompt_yes_no("  → Replace raw data with reshaped?", prefs['ReplaceWithReshape'])
    else:
        prefs['ReplaceWithReshape'] = False
    prefs['Simulate'] = prompt_yes_no("Generate ideal-peak underlay?", prefs['Simulate'])

    print("\n=== Data Outputs Configuration ===")
    print("-> Per-file best segments:")
    prefs['PerFileBestCount'] = prompt_int(
        "  Number of top segments per file to save",
        prefs.get('PerFileBestCount', 1)
    )
    prefs['ExportPerFileBestPNG'] = True
    prefs['ExportPerFileBestCSV'] = True

    print("-> Overall-best segments:")
    prefs['OverallBestCount'] = prompt_int(
        "  Number of top segments overall to save",
        prefs.get('OverallBestCount', prefs['BestSegmentsCount'])
    )
    prefs['ExportOverallPDF'] = True
    prefs['ExportBestPNG']     = True
    prefs['ExportBestCSV']     = True

    print("\n=== Ranking Configuration ===")
    print("  1) Balanced")
    print("  2) SNR > Width > Spacing")
    print("  3) Width > SNR > Spacing")
    print("  4) Spacing > Width > SNR")
    print("  5) Custom weights")
    choice = prompt_int("Choose ranking scheme (1–5)", prefs.get('RankingScheme', 1))
    prefs['RankingScheme'] = choice

    if choice == 5:
        print("Enter relative weights (they’ll be normalized to sum=1):")
        w_snr   = prompt_float("  Weight for SNR",   0.33)
        w_width = prompt_float("  Weight for width", 0.33)
        w_sp    = prompt_float("  Weight for spacing",0.33)
        total   = w_snr + w_width + w_sp
        if total == 0:
            w_snr = w_width = w_sp = 1.0 / 3
            total = 1.0
        prefs['RankWeights'] = {
            'snr':   w_snr/total,
            'width': w_width/total,
            'sp':    w_sp/total
        }
    return prefs

# === Plotting Helper (unchanged, but all calls now routed through export helpers) ===

def plot_segment(ts, raw, proc, ideal, title, prefs):
    fig, ax = plt.subplots(figsize=(8,6))
    ax.set_title(title)
    if ideal is not None:
        ax.fill_between(ts, np.clip(ideal, 0, None), 0, facecolor='lightgray', alpha=0.4, zorder=0)
    if prefs['ReplaceWithReshape']:
        ax.plot(ts, np.clip(proc, 0, None), label='Reshaped', zorder=1)
    else:
        ax.plot(ts, np.clip(raw, 0, None), label='Raw', zorder=1)
        if prefs['OverlayReshape']:
            ax.plot(ts, np.clip(proc, 0, None), label='Reshaped', zorder=2)
    label_axes(ax)
    ax.legend()
    return fig

def label_axes(ax: plt.Axes) -> None:
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Intensity (CPS)')

# === SNR Calculation ===

def calculate_snr(ts, signal, spacing_ms, method="median", fixed_noise=None):
    dt = ts[1] - ts[0] if ts.size > 1 else 1.0
    spacing = spacing_ms / 1000.0
    min_d = max(1, int(0.8 * spacing / dt))
    peaks, _ = find_peaks(np.clip(signal, 0, None), distance=min_d)
    if not len(peaks):
        return 0.0
    noise = fixed_noise if fixed_noise is not None else np.std(signal - np.mean(signal)) + 1e-12
    snr_vals = [signal[p] / noise for p in peaks]
    return float(np.median(snr_vals)) if method == "median" else float(np.mean(snr_vals))

# === Core Analysis Functions ===

def reshape_peak(
    ts: np.ndarray,
    raw_seg: np.ndarray,
    corr_seg: np.ndarray,
    exp_sp_ms: float,
    exp_w_ms: float,
    segment_length: float,
    smooth_factor: float = 1
) -> np.ndarray:
    """
    Reshape a segment to retain only the main peaks, zeroing out the rest.
    """
    dt = ts[1] - ts[0] if ts.size > 1 else 1.0
    sigma_pts = smooth_factor * (exp_w_ms / 1000.0) / (2 * dt)
    detector = gaussian_filter1d(corr_seg, sigma=sigma_pts, mode='nearest')
    spacing_s = exp_sp_ms / 1000.0
    N = max(1, math.ceil(segment_length / spacing_s))
    edges = np.linspace(ts[0], ts[0] + segment_length, N + 1)
    top_idxs = []
    for i in range(N):
        lo, hi = edges[i], edges[i+1]
        mask = (ts >= lo) & (ts < hi) if i < N-1 else (ts >= lo) & (ts <= hi)
        if not np.any(mask):
            continue
        idxs = np.where(mask)[0]
        best = idxs[np.argmax(detector[idxs])]
        top_idxs.append(best)
    half_w = int((exp_w_ms / 1000.0) / (2 * dt))
    cleaned = np.zeros_like(raw_seg)
    for p in top_idxs:
        start = max(0, p - half_w)
        end   = min(raw_seg.size, p + half_w + 1)
        cleaned[start:end] = raw_seg[start:end]
    return cleaned

def simulate_ideal(
    ts: np.ndarray,
    raw: np.ndarray,
    proc: np.ndarray,
    exp_sp_ms: float,
    exp_w_ms: float
) -> np.ndarray:
    """
    Generate ideal peaks as triangular pulses of exactly exp_w_ms width,
    spaced by exp_sp_ms, only where each full pulse fits within [t0, tN].
    """
    dt = ts[1] - ts[0] if ts.size > 1 else 1.0
    amp = max(np.max(raw), np.max(proc)) if proc is not None else np.max(raw)
    sample_count = max(1, int(round((exp_w_ms / 1000.0) / dt)))
    tri = windows.triang(sample_count) * amp
    half = sample_count // 2
    spacing_s = exp_sp_ms / 1000.0
    base = proc if proc is not None else raw
    peaks, _ = find_peaks(base, distance=max(1, int(0.8 * spacing_s / dt)))
    start_t = ts[peaks[0]] if peaks.size > 0 else ts[0]
    centers = np.arange(start_t, ts[-1] + spacing_s, spacing_s)
    valid_centers = [
        c for c in centers
        if (c - half*dt) >= ts[0] and (c + half*dt) <= ts[-1]
    ]
    ideal = np.zeros_like(ts)
    for c in valid_centers:
        idx_center = int(np.argmin(np.abs(ts - c)))
        start_idx = idx_center - half
        for j, val in enumerate(tri):
            pos = start_idx + j
            if 0 <= pos < ideal.size:
                ideal[pos] = max(ideal[pos], val)
    return ideal

def analyze_peak_shape(
    time: np.ndarray,
    sig: np.ndarray,
    exp_sp_ms: float,
    exp_w_ms: float
) -> Dict[str, float]:
    """
    Analyze the peak shape for a segment. Returns a dict of peak metrics.
    If no peaks are found, returns metrics with np.nan or 0.
    """
    if time.size < 2:
        return {
            'num_peaks': 0,
            'avg_spacing': float('nan'),
            'spacing_std': float('nan'),
            'avg_width': float('nan'),
            'width_std': float('nan'),
        }
    dt = time[1] - time[0]
    pad_ms = exp_w_ms / 2.0
    pad_n = int(round(pad_ms/1000.0 / dt))
    padded = np.concatenate([
        np.full(pad_n, np.median(sig[:pad_n]) if pad_n > 0 else 0),
        sig,
        np.full(pad_n, np.median(sig[-pad_n:]) if pad_n > 0 else 0)
    ])
    time_pad = np.concatenate([
        time[0] - np.arange(pad_n,0,-1)*dt,
        time,
        time[-1] + np.arange(1,pad_n+1)*dt
    ])
    exp_sp = exp_sp_ms / 1000.0
    min_dist = max(1, int(0.8 * exp_sp / dt))
    peaks, _ = find_peaks(padded, distance=min_dist)
    if peaks.size == 0:
        return {
            'num_peaks': 0,
            'avg_spacing': float('nan'),
            'spacing_std': float('nan'),
            'avg_width': float('nan'),
            'width_std': float('nan'),
        }
    widths_samples = peak_widths(padded, peaks, rel_height=0.5)[0]
    widths_ms = widths_samples * dt * 1000.0
    real_peaks = peaks[(peaks >= pad_n) & (peaks < pad_n + len(sig))] - pad_n
    if real_peaks.size == 0:
        return {
            'num_peaks': 0,
            'avg_spacing': float('nan'),
            'spacing_std': float('nan'),
            'avg_width': float('nan'),
            'width_std': float('nan'),
        }
    spacings_s = np.diff(time[real_peaks])
    spacings_ms = spacings_s * 1000.0
    return {
      'num_peaks': int(len(real_peaks)),
      'avg_spacing': float(np.mean(spacings_ms)) if spacings_ms.size > 0 else float('nan'),
      'spacing_std': float(np.std(spacings_ms)) if spacings_ms.size > 0 else float('nan'),
      'avg_width': float(np.mean(widths_ms)) if widths_ms.size > 0 else float('nan'),
      'width_std': float(np.std(widths_ms)) if widths_ms.size > 0 else float('nan'),
    }

def segment_timeseries(
    time: np.ndarray,
    sig: np.ndarray,
    length: float,
    stride: float
) -> List[Tuple[np.ndarray, np.ndarray]]:
    segs: List[Tuple[np.ndarray, np.ndarray]] = []
    if time.size == 0:
        return segs
    start, stop = time[0], time[-1]
    cur = start
    while cur + length <= stop:
        mask = (time >= cur) & (time < cur + length)
        if np.any(mask):
            segs.append((time[mask], sig[mask]))
        cur += stride
    last_start = stop - length
    if last_start > start:
        already = any(
            abs(segs_start[0] - last_start) < 1e-12
            for segs_start, _ in segs
        )
        if not already:
            mask = (time >= last_start) & (time <= stop)
            if np.any(mask):
                segs.append((time[mask], sig[mask]))
    return segs

# === File Processing ===

def process_file(
    path: str,
    prefs: Dict[str, Any],
    indiv_folder: Path,
    summary_segments: List[Dict[str, Any]],
    summary_shape: List[Dict[str, Any]],
    error_log: List[str]
) -> None:
    df = robust_read_csv(path)
    if df is None or df.shape[1] < 2:
        error_log.append(f"{path}: Could not parse or insufficient columns.")
        return
    tvals = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    svals = pd.to_numeric(df.iloc[:, 1], errors='coerce')
    mask = (~np.isnan(tvals)) & (~np.isnan(svals))
    time_arr, sig_arr = tvals[mask].values, svals[mask].values
    if time_arr.size == 0 or sig_arr.size == 0:
        error_log.append(f"{path}: No valid numeric data after filtering NaNs.")
        return
    if not is_monotonic_and_unique(time_arr):
        error_log.append(f"{path}: Time column not strictly increasing/unique.")
        return
    warm_mask   = time_arr < prefs['SkipInitial']
    warm_signal = sig_arr[warm_mask]
    warm_noise  = (
        np.std(warm_signal - np.mean(warm_signal)) + 1e-12
        if warm_signal.size > 0 else None
    )
    idx    = np.searchsorted(time_arr, prefs['SkipInitial'], 'left')
    t_trim = time_arr[idx:]
    raw    = sig_arr[idx:]
    if t_trim.size == 0 or raw.size == 0:
        error_log.append(f"{path}: No data after warm-up skip.")
        return
    if prefs['AutoTrim'] and raw.size >= SAVGOL_WINDOW_DEFAULT:
        base = savgol_filter(raw, SAVGOL_WINDOW_DEFAULT, SAVGOL_POLY_DEFAULT)
        corr = raw - base
    else:
        corr = raw.copy()
    scan_end   = prefs['SkipInitial'] + prefs['ScanWindowSec']
    mask_scan  = t_trim <= scan_end
    t_scan     = t_trim[mask_scan]
    raw_scan   = raw[mask_scan]
    corr_scan  = corr[mask_scan]
    if t_scan.size == 0 or raw_scan.size == 0:
        error_log.append(f"{path}: No data in scan window.")
        return

    # Export overall overview trace for GUI (already in Skip+Scan window)
    try:
        fname  = Path(path).stem
        outdir = indiv_folder / fname
        outdir.mkdir(exist_ok=True)
        export_overview_trace(t_scan, raw_scan, corr_scan, outdir, prefs, max_points=5000)
    except Exception as _e:
        logging.warning(f"overview export failed for {path}: {_e}")
    desired_end = t_trim[0] + prefs['SegmentLength']
    if t_scan[-1] < desired_end:
        dt = t_scan[1] - t_scan[0] if t_scan.size > 1 else 1.0
        n_extra = int(np.ceil((desired_end - t_scan[-1]) / dt))
        pad_t    = t_scan[-1] + np.arange(1, n_extra+1) * dt
        pad_raw  = np.full_like(pad_t, raw_scan[-1])
        pad_corr = np.full_like(pad_t, corr_scan[-1])
        t_scan   = np.concatenate([t_scan, pad_t])
        raw_scan = np.concatenate([raw_scan, pad_raw])
        corr_scan= np.concatenate([corr_scan, pad_corr])
    raw_segs  = segment_timeseries(t_scan, raw_scan, prefs['SegmentLength'], prefs['Stride'])
    corr_segs = segment_timeseries(t_scan, corr_scan, prefs['SegmentLength'], prefs['Stride'])
    if prefs['AutoDetectCooldown'] and corr.size > 1:
        dt   = t_trim[1] - t_trim[0]
        sp   = prefs['ExpectedSpacingMs'] / 1000.0
        min_d= max(1, int(0.8 * sp / dt))
        peaks, _ = find_peaks(corr, distance=min_d)
        if peaks.size > 0:
            last_time = t_trim[peaks[-1]]
            filtered = [
                ((r_ts, r_vals), (c_ts, c_vals))
                for (r_ts, r_vals), (c_ts, c_vals)
                in zip(raw_segs, corr_segs)
                if r_ts[0] <= last_time
            ]
            if filtered:
                raw_segs, corr_segs = zip(*filtered)
                raw_segs, corr_segs = list(raw_segs), list(corr_segs)
            else:
                raw_segs, corr_segs = [], []
    oshape = analyze_peak_shape(t_trim, corr, prefs['ExpectedSpacingMs'], prefs['ExpectedWidthMs'])
    oshape['file'] = Path(path).stem
    summary_shape.append(oshape)
    fname  = Path(path).stem
    outdir = indiv_folder / fname
    outdir.mkdir(exist_ok=True)
    stats: List[Dict[str, Any]] = []
    for idx_, ((ts, raw_seg), (_, corr_seg)) in enumerate(zip(raw_segs, corr_segs), start=1):
        r_original = raw_seg.copy()
        proc = reshape_peak(
            ts=ts,
            raw_seg=raw_seg,
            corr_seg=corr_seg,
            exp_sp_ms=prefs['ExpectedSpacingMs'],
            exp_w_ms=prefs['ExpectedWidthMs'],
            segment_length=prefs['SegmentLength']
        )
        ideal = simulate_ideal(
            ts, r_original, proc,
            prefs['ExpectedSpacingMs'],
            prefs['ExpectedWidthMs']
        ) if prefs['Simulate'] else None
        snr_raw       = calculate_snr(ts, r_original, prefs['ExpectedSpacingMs'], prefs['SNREstimator'])
        snr_reshaped  = calculate_snr(ts, proc,       prefs['ExpectedSpacingMs'], prefs['SNREstimator'])
        snr_base_raw  = calculate_snr(ts, r_original, prefs['ExpectedSpacingMs'], prefs['SNREstimator'], fixed_noise=warm_noise)
        snr_base_proc = calculate_snr(ts, proc,       prefs['ExpectedSpacingMs'], prefs['SNREstimator'], fixed_noise=warm_noise)
        shape_raw      = analyze_peak_shape(ts, r_original, prefs['ExpectedSpacingMs'], prefs['ExpectedWidthMs'])
        shape_reshaped = analyze_peak_shape(ts, proc,       prefs['ExpectedSpacingMs'], prefs['ExpectedWidthMs'])
        seg_metrics = {
            'file'              : fname,
            'segment'           : idx_,
            't_seg'             : ts,
            'raw'               : r_original,
            'proc'              : proc,
            'ideal'             : ideal,
            'snr_raw'           : snr_raw,
            'snr_reshaped'      : snr_reshaped,
            'snr_baseline'      : snr_base_raw,
            'snr_baseline_proc' : snr_base_proc,
            'shape_raw'         : shape_raw,
            'shape_reshaped'    : shape_reshaped,
        }
        if not is_valid_segment_metrics(shape_raw) or not is_valid_segment_metrics(shape_reshaped):
            error_log.append(f"{path}: Segment {idx_} has invalid metrics; skipped.")
            continue
        stats.append(seg_metrics)
        if prefs.get('ExportSegPNG', False):
            export_segment_png(ts, r_original, proc, ideal, fname, idx_, outdir, prefs)
        if prefs.get('ExportSegCSV', False):
            export_segment_csv(ts, r_original, proc, ideal, snr_raw, snr_reshaped, snr_base_raw, snr_base_proc,
                               shape_raw, shape_reshaped, fname, idx_, outdir, prefs)
    if not stats:
        error_log.append(f"{path}: No valid segments found after filtering.")
    summary_segments.append({'file': fname, 'stats': stats})

# === Composite scoring and reports ===

def export_run_reports(
    run_folder: Path,
    prefs: Dict[str, Any],
    summary_segments: List[Dict[str, Any]],
    summary_shape: List[Dict[str, Any]]
) -> None:
    indiv_folder = run_folder / 'Individual'
    indiv_folder.mkdir(exist_ok=True)

    def score_segment(seg: Dict[str, Any]) -> float:
        snr_norm = min(
            seg['snr_baseline'] / prefs.get('DesiredSNR', seg['snr_baseline']),
            1.0
        ) if prefs.get('DesiredSNR', 0) else 1.0

        w_ms  = seg['shape_raw']['avg_width']
        sp_ms = seg['shape_raw']['avg_spacing']
        w_score  = max(0.0, 1.0 - abs(w_ms  - prefs['ExpectedWidthMs'])   / prefs['ExpectedWidthMs']) if prefs['ExpectedWidthMs'] else 1.0
        sp_score = max(0.0, 1.0 - abs(sp_ms - prefs['ExpectedSpacingMs']) / prefs['ExpectedSpacingMs']) if prefs['ExpectedSpacingMs'] else 1.0

        scheme = prefs.get('RankingScheme', 1)
        if scheme == 1:
            w_snr, w_width, w_sp = 1/3, 1/3, 1/3
        elif scheme == 2:
            w_snr, w_width, w_sp = 0.5, 0.3, 0.2
        elif scheme == 3:
            w_width, w_snr, w_sp = 0.5, 0.3, 0.2
        elif scheme == 4:
            w_sp, w_width, w_snr = 0.5, 0.3, 0.2
        else:
            rw = prefs.get('RankWeights', {'snr':1/3,'width':1/3,'sp':1/3})
            w_snr   = rw.get('snr',   1/3)
            w_width = rw.get('width', 1/3)
            w_sp    = rw.get('sp',    1/3)
        return snr_norm * w_snr + w_score * w_width + sp_score * w_sp

    # 1. Per-file best
    per_file_best = []
    for itm in summary_segments:
        ranked = sorted(itm['stats'], key=score_segment, reverse=True)
        folder = indiv_folder / itm['file']
        folder.mkdir(exist_ok=True)
        metrics_rows = []
        for rank, b in enumerate(ranked[:prefs['PerFileBestCount']], start=1):
            per_file_best.append(b)
            prefix = f"{rank}_{b['file']}_seg{b['segment']}"
            # PNG
            fig = plot_segment(
                b['t_seg'], b['raw'], b['proc'], b.get('ideal'),
                f"{b['file']} seg{b['segment']}", prefs
            )
            fig.savefig(folder / f"{prefix}.png")
            plt.close(fig)
            # CSV
            out = {
                'time':     b['t_seg'],
                'raw':      b['raw'],
                'reshaped': b['proc'],
                'ideal':    b.get('ideal'),
            }
            out.update({
                'snr_baseline_raw'   : [b['snr_baseline']]      * len(b['t_seg']),
                'snr_baseline_proc'  : [b['snr_baseline_proc']] * len(b['t_seg']),
                'snr_raw'            : [b['snr_raw']]           * len(b['t_seg']),
                'snr_reshaped'       : [b['snr_reshaped']]      * len(b['t_seg']),
                'width_raw_ms'       : [b['shape_raw']['avg_width']]       * len(b['t_seg']),
                'spacing_raw_ms'     : [b['shape_raw']['avg_spacing']]     * len(b['t_seg']),
                'width_resh_ms'      : [b['shape_reshaped']['avg_width']]  * len(b['t_seg']),
                'spacing_resh_ms'    : [b['shape_reshaped']['avg_spacing']]* len(b['t_seg']),
            })
            pd.DataFrame(out).to_csv(folder / f"{prefix}.csv", index=False)
            metrics_rows.append({
                'rank'             : rank,
                'segment'          : b['segment'],
                'snr_baseline_raw' : b['snr_baseline'],
                'snr_baseline_proc': b['snr_baseline_proc'],
                'snr_raw'          : b['snr_raw'],
                'snr_reshaped'     : b['snr_reshaped'],
                'width_raw_ms'     : b['shape_raw']['avg_width'],
                'spacing_raw_ms'   : b['shape_raw']['avg_spacing'],
                'width_resh_ms'    : b['shape_reshaped']['avg_width'],
                'spacing_resh_ms'  : b['shape_reshaped']['avg_spacing'],
            })
        pd.DataFrame(metrics_rows).to_csv(folder / 'metrics.csv', index=False)

    # 2. Rank & rename Individual folders
    file_best_map = {
        itm['file']: max(itm['stats'], key=score_segment)
        for itm in summary_segments if itm['stats']
    }
    sorted_files = sorted(
        file_best_map.items(),
        key=lambda kv: score_segment(kv[1]),
        reverse=True
    )
    for rank, (fname, _) in enumerate(sorted_files, start=1):
        old = indiv_folder / fname
        new = indiv_folder / f"{rank:02d}_{fname}"
        if old.exists() and not new.exists():
            old.rename(new)

    # 3. Overall‐best PDF with summary
    overall_best = sorted(per_file_best, key=score_segment, reverse=True)[:max(1, int(prefs.get('OverallBestCount', 5)))]
    with PdfPages(run_folder / 'overall_best_summary.pdf') as pdf:
        # Config (PORTRAIT A4)
        fig = plt.figure(figsize=(8.27, 11.69))
        plt.axis('off')
        plt.text(0.1, 0.5, json.dumps(prefs, indent=2), family='monospace')
        pdf.savefig(fig)
        plt.close(fig)
        # Summary table (LANDSCAPE A4)
        if not overall_best:
            fig = plt.figure(figsize=(11.69, 8.27))
            ax = fig.add_subplot(111); ax.axis('off')
            ax.text(0.1, 0.5, "No segments available for overall-best summary.", fontsize=14)
            pdf.savefig(fig); plt.close(fig)
        else:
            fig = plt.figure(figsize=(11.69, 8.27))
            ax = fig.add_subplot(111); ax.axis('off')
            rows = []
            for rank, b in enumerate(overall_best, start=1):
                seg_label = f"{rank}. {b['file']} seg{b['segment']}"
                rows.append([
                    seg_label,
                    f"{b['snr_baseline']:.1f}",
                    f"{b['snr_raw']:.1f}",
                    f"{b['shape_raw']['avg_width']:.1f}",
                    f"{b['shape_raw']['avg_spacing']:.1f}",
                    f"{b['snr_reshaped']:.1f}",
                    f"{b['shape_reshaped']['avg_width']:.1f}",
                    f"{b['shape_reshaped']['avg_spacing']:.1f}"
                ])
            col_labels = ["Segment","Base SNR","Raw SNR","Raw W (ms)","Raw Sp (ms)","Resh SNR","Resh W (ms)","Resh Sp (ms)"]
            col_widths = [0.50, 0.07, 0.07, 0.08, 0.08, 0.07, 0.08, 0.08]
            tbl = ax.table(cellText=rows, colLabels=col_labels, colWidths=col_widths,
                           loc='center', cellLoc='left')
            tbl.auto_set_font_size(False); tbl.set_fontsize(8)
            for (row, col), cell in tbl.get_celld().items():
                if row == 0:
                    cell.set_text_props(fontweight='bold')
                elif col > 0:
                    cell.get_text().set_fontfamily('monospace')
            pdf.savefig(fig); plt.close(fig)
            # Chart pages
            for rank, b in enumerate(overall_best, start=1):
                fig = plt.figure(figsize=(11.69, 8.27))
                ax = fig.add_axes([0.08, 0.25, 0.90, 0.70])
                ax.set_title(f"{rank}. {b['file']} seg{b['segment']}")
                if prefs.get('ReplaceWithReshape', False):
                    ax.plot(b['t_seg'], np.clip(b['proc'],0,None), label='Reshaped')
                else:
                    ax.plot(b['t_seg'], np.clip(b['raw'],0,None), label='Raw')
                    ax.plot(b['t_seg'], np.clip(b['proc'],0,None), label='Reshaped')
                if b.get('ideal') is not None:
                    ax.fill_between(b['t_seg'], np.clip(b['ideal'],0,None), 0,
                                    facecolor='lightgray', alpha=0.3, zorder=0)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Intensity (CPS)')
                ax.legend(loc='upper left')
                table_data = [
                    ["Raw Metrics", "", "Reshaped Metrics", ""],
                    ["SNR", "W (ms)", "SNR", "W (ms)"],
                ]
                if prefs.get('ReplaceWithReshape', False):
                    table_data.append([
                        "", "",
                        f"{b['snr_reshaped']:.1f}", f"{b['shape_reshaped']['avg_width']:.1f}"
                    ])
                else:
                    table_data.append([
                        f"{b['snr_raw']:.1f}", f"{b['shape_raw']['avg_width']:.1f}",
                        f"{b['snr_reshaped']:.1f}", f"{b['shape_reshaped']['avg_width']:.1f}"
                    ])
                ax_tab = fig.add_axes([0.08, 0.05, 0.90, 0.15], frameon=False)
                ax_tab.axis('off')
                tbl = ax_tab.table(cellText=table_data, loc='center',
                                   colWidths=[0.25,0.25,0.25,0.25], cellLoc='center')
                tbl.auto_set_font_size(False); tbl.set_fontsize(8)
                pdf.savefig(fig); plt.close(fig)
                
    # 4. Overall PNG & CSV
    for rank, b in enumerate(overall_best, start=1):
        prefix = f"{rank}_{b['file']}_best_seg{b['segment']}"
        fig = plot_segment(
            b['t_seg'], b['raw'], b['proc'], b.get('ideal'),
            f"{rank}. {b['file']} seg{b['segment']}", prefs
        )
        fig.savefig(run_folder / f"{prefix}.png"); plt.close(fig)
        out = {
            'time':       b['t_seg'],
            'raw':        b['raw'],
            'reshaped':   b['proc'],
            'ideal':      b.get('ideal'),
            'snr_raw':    [b['snr_raw']]    * len(b['t_seg']),
            'snr_base':   [b['snr_baseline']]* len(b['t_seg']),
            'width_raw':  [b['shape_raw']['avg_width']]*len(b['t_seg']),
            'spacing_raw':[b['shape_raw']['avg_spacing']]*len(b['t_seg']),
            'width_resh': [b['shape_reshaped']['avg_width']]*len(b['t_seg']),
            'spacing_resh':[b['shape_reshaped']['avg_spacing']]*len(b['t_seg'])
        }
        pd.DataFrame(out).to_csv(run_folder / f"{prefix}.csv", index=False)
    # 5. Shape summary CSV
    if prefs.get('ExportShapeCSV', False):
        pd.DataFrame(summary_shape).to_csv(
            run_folder / 'shape_summary.csv', index=False
        )

# === Entry Point ===

def main() -> None:
    args = parse_args()
    prefs = load_preferences(args.config)
    # Interactive prompt and config reload
    if not args.no_interactive:
        prefs = interactive_prompt(prefs)
        save_preferences(args.config, prefs)
        prefs = load_preferences(args.config)  # always reload for consistency
    warnings = validate_prefs(prefs)
    for w in warnings:
        print(f"Config warning: {w}")
        logging.warning(w)
    run_folder = Path(f"run_{pd.Timestamp.now():%Y%m%d_%H%M%S}")
    indiv_folder = run_folder / 'Individual'
    run_folder.mkdir(exist_ok=True)
    indiv_folder.mkdir(exist_ok=True)
    log_level = logging.WARNING if args.quiet else logging.INFO
    setup_logging(run_folder / 'run.log', level=log_level)
    input_path = Path(args.input_dir)
    files = sorted(list(input_path.glob(args.file_pattern)))
    if not files:
        print("No input files found. Exiting.")
        sys.exit(1)
    summary_segments: List[Dict[str, Any]] =[]
    summary_shape:    List[Dict[str, Any]] =[]
    error_log: List[str] = []
    iterator = files if args.quiet else tqdm(files, desc="Processing files")
    for f in iterator:
        if not args.quiet:
            logging.info(f"Processing {f.name}")
        try:
            process_file(str(f), prefs, indiv_folder, summary_segments, summary_shape, error_log)
        except Exception as e:
            msg = f"Failed {f.name}, skipping. Reason: {e}"
            logging.exception(msg)
            error_log.append(msg)
    export_run_reports(run_folder, prefs, summary_segments, summary_shape)
    # Write error log and summary
    if error_log:
        with open(run_folder / ERROR_LOG_FILENAME, "w") as ef:
            for l in error_log:
                ef.write(l + "\n")
    summary_df = pd.DataFrame([
        {
            'file': s['file'],
            'best_snr': max(st['snr_raw'] for st in s['stats']) if s['stats'] else float('nan')
        } for s in summary_segments
    ])
    summary_df.to_csv(run_folder / SUMMARY_FILENAME, index=False)
    # Print summary
    print(f"\nRun complete. Outputs in: {run_folder}")
    print(f"Processed files: {len(files)}")
    print(f"Files with valid segments: {len([s for s in summary_segments if s['stats']])}")
    print(f"Files skipped or with errors: {len(error_log)}")
    if error_log:
        print(f"See {run_folder / ERROR_LOG_FILENAME} for details.")
    if not args.quiet:
        print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()