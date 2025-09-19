#!/usr/bin/env python3
"""
PeakFocus GUI + Focus (v8)

New in this build
-----------------
• **Overview highlight** of the currently selected segment
  - Uses absolute t0/t1 if present in the segment CSV (columns like `t_abs_start`, `t_abs_end`).
  - Otherwise tries to project the segment onto the overview by time range; if that
    fails (e.g., segment time is relative), it performs a **fast correlation‑based
    alignment** to locate the best match and shades that window.
  - Also drops little markers at detected peak centers inside that window.

• **Interactive peak metrics on the segment plot**
  - Calculates per‑peak **height, area, width (FWHM by default)** and **spacing**.
  - Overlays peak markers/labels on the segment plot.
  - Shows a small table of per‑peak metrics (resizable) and lets you export it.
  - Controls to tweak detection live:
      • Threshold in σ (MAD‑based)  
      • Minimum spacing (ms)  
      • Width @ fraction (e.g., 0.5 for FWHM)

Keep-alive from v7
------------------
• Crosshair + wheel zoom (Ctrl for Y‑zoom), right‑click autoscale.
• Auto‑switch to Focus; de‑dupe rows; fast overview loader; Top‑N export, etc.

Run:
  pip install PySide6 matplotlib pandas numpy
  # optional (hover labels on lines)
  pip install mplcursors
  python peakfocus_gui_plus_v8.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QWidget, QFileDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QDoubleSpinBox, QSpinBox, QCheckBox, QPushButton, QTextEdit,
    QLabel, QGroupBox, QComboBox, QMessageBox, QTabWidget, QSplitter,
    QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar
)

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar

try:
    import mplcursors  # optional
    HAVE_MPLC = True
except Exception:
    HAVE_MPLC = False

DEFAULT_CONFIG = {
    "SkipInitial": 7.0,
    "ScanWindowSec": 10.0,
    "SegmentLength": 0.300,
    "Stride": 0.150,
    "DesiredSNR": 300,
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
    "SNREstimator": "median",
    "PerFileBestCount": 1,
    "OverallBestCount": 5,
    "RankingScheme": 1,
}

OUTPUT_DIR_REGEX = re.compile(r"Outputs in:\s*\"?(.+?run_[0-9]{8}_[0-9]{6})\"?\s*$")
RUN_NAME_REGEX   = re.compile(r"run_[0-9]{8}_[0-9]{6}")
SEG_NUM_RE       = re.compile(r"_seg(\d+)\.csv$")
LEADING_NUM_RE   = re.compile(r"^\d+_+")
PROG_PATTERNS    = [
    re.compile(r"(?i)(?:processing|processed|file)\s*(\d+)\s*(?:/|of)\s*(\d+)"),
    re.compile(r"(?i)(\d+)\s*(?:/|of)\s*(\d+)\s*(?:files|segments)\b"),
]

# ──────────────────────────────────────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────────────────────────────────────

def score_segment_row(row: Dict[str, float], prefs: Dict[str, Any]) -> float:
    base_snr = float(row.get('snr_baseline_raw', row.get('snr_baseline', 0.0)))
    desired = float(prefs.get('DesiredSNR', base_snr or 1.0))
    snr_norm = min(max((base_snr / desired) if desired > 0 else 1.0, 0.0), 1.0)

    w_ms  = float(row.get('width_raw_ms', row.get('width_raw', np.nan)))
    sp_ms = float(row.get('spacing_raw_ms', row.get('spacing_raw', np.nan)))
    exp_w  = float(prefs.get('ExpectedWidthMs', 0) or 0)
    exp_sp = float(prefs.get('ExpectedSpacingMs', 0) or 0)

    w_score  = 1.0 if exp_w  == 0 else max(0.0, 1.0 - abs((w_ms  - exp_w) / exp_w))
    sp_score = 1.0 if exp_sp == 0 else max(0.0, 1.0 - abs((sp_ms - exp_sp) / exp_sp))

    scheme = int(prefs.get('RankingScheme', 1))
    if scheme == 1:      w_snr, w_width, w_sp = 1/3, 1/3, 1/3
    elif scheme == 2:    w_snr, w_width, w_sp = 0.5, 0.3, 0.2
    elif scheme == 3:    w_width, w_snr, w_sp = 0.5, 0.3, 0.2
    elif scheme == 4:    w_sp, w_width, w_snr = 0.5, 0.3, 0.2
    else:
        rw = prefs.get('RankWeights', {'snr':1/3,'width':1/3,'sp':1/3})
        w_snr   = float(rw.get('snr',   1/3))
        w_width = float(rw.get('width', 1/3))
        w_sp    = float(rw.get('sp',    1/3))
    return snr_norm * w_snr + w_score * w_width + sp_score * w_sp

# ──────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────────────────────────

def plot_segment(ax, df: pd.DataFrame, prefs: Dict[str, Any], title: str = "") -> None:
    ax.clear()
    if 'ideal' in df.columns and prefs.get('Simulate', True):
        ax.fill_between(df['time'], np.clip(df['ideal'].values,0,None), 0, alpha=0.3)
    replace = bool(prefs.get('ReplaceWithReshape', False))
    overlay = bool(prefs.get('OverlayReshape', True))
    if replace:
        y = df['reshaped'] if 'reshaped' in df.columns else df['raw']
        ax.plot(df['time'], np.clip(y.values,0,None), label='Reshaped')
    else:
        ax.plot(df['time'], np.clip(df['raw'].values,0,None), label='Raw')
        if overlay and 'reshaped' in df.columns:
            ax.plot(df['time'], np.clip(df['reshaped'].values,0,None), label='Reshaped')
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Intensity (CPS)')
    ax.legend(loc='upper left')

# ──────────────────────────────────────────────────────────────────────────────
# Simple peak finder (no SciPy): MAD threshold + FWHM measures
# ──────────────────────────────────────────────────────────────────────────────

def _mad(x: np.ndarray) -> float:
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))

def _interp_crossing(x: np.ndarray, y: np.ndarray, i0: int, i1: int, level: float) -> float:
    # linear interpolation for y-level crossing between i0 and i1
    x0, y0 = x[i0], y[i0]; x1, y1 = x[i1], y[i1]
    if y1 == y0:
        return x0
    frac = (level - y0) / (y1 - y0)
    return x0 + frac * (x1 - x0)

def measure_peaks(t: np.ndarray, y: np.ndarray, thresh_sigma: float, min_space_s: float, width_frac: float = 0.5) -> List[Dict[str, float]]:
    if len(t) < 3:
        return []
    base = np.median(y)
    yb = y - base
    sigma = _mad(y)
    thr = base + thresh_sigma * sigma

    # candidate local maxima above threshold
    peaks = []
    for i in range(1, len(y)-1):
        if y[i] > thr and y[i] >= y[i-1] and y[i] >= y[i+1]:
            peaks.append(i)
    if not peaks:
        return []

    # enforce minimum spacing by greedy selection on height
    peaks_sorted = sorted(peaks, key=lambda i: y[i], reverse=True)
    chosen: List[int] = []
    for i in peaks_sorted:
        if all(abs(t[i] - t[j]) >= min_space_s for j in chosen):
            chosen.append(i)
    chosen.sort()

    out: List[Dict[str, float]] = []
    for k, i in enumerate(chosen):
        # width at fraction of height
        h = y[i] - base
        level = base + width_frac * h
        # left crossing
        li = i
        while li > 0 and y[li] > level:
            li -= 1
        if li < i:
            tL = _interp_crossing(t, y, li, li+1, level)
        else:
            tL = t[i]
        # right crossing
        ri = i
        while ri < len(y)-1 and y[ri] > level:
            ri += 1
        if ri > i:
            tR = _interp_crossing(t, y, ri-1, ri, level)
        else:
            tR = t[i]
        width_s = max(tR - tL, 0.0)
        # area above baseline between crossings
        i0 = max(li, 0); i1 = min(ri, len(y)-1)
        seg_x = np.concatenate(([tL], t[i0+1:i1], [tR])) if i1 > i0+1 else np.array([tL, tR])
        seg_y = np.concatenate(([level], y[i0+1:i1], [level])) if i1 > i0+1 else np.array([level, level])
        area = float(np.trapz(np.clip(seg_y - base, 0, None), seg_x))
        spacing_ms = float((t[chosen[k+1]] - t[i]) * 1000.0) if k+1 < len(chosen) else np.nan
        out.append({
            'index': int(i),
            'time_s': float(t[i]),
            'height': float(h),
            'area': area,
            'width_ms': float(width_s * 1000.0),
            'spacing_ms': spacing_ms,
        })
    return out

# ──────────────────────────────────────────────────────────────────────────────
# GUI
# ──────────────────────────────────────────────────────────────────────────────

class PeakFocusGUI(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PeakFocus GUI + Focus (v8)")
        self.resize(1320, 860)

        self.process: subprocess.Popen | None = None
        self._output_dir_last: Optional[Path] = None
        self._rows: List[Dict[str, Any]] = []
        self._csv_map: Dict[Tuple[str,int], Path] = {}
        self._rescored_rows: List[Dict[str, Any]] = []

        # Full-trace cache
        self._full_cache: Dict[str, pd.DataFrame] = {}

        self.tabs = QTabWidget(self)
        self.tab_run = QWidget(self)
        self.tab_focus = QWidget(self)
        self.tabs.addTab(self.tab_run,  "Run")
        self.tabs.addTab(self.tab_focus, "Focus")

        outer = QVBoxLayout(self)
        outer.addWidget(self.tabs)

        self._build_run_tab()
        self._build_focus_tab()

    # Run tab (unchanged behavior)
    def _build_run_tab(self) -> None:
        self.edt_script = QLineEdit()
        self.btn_browse_script = QPushButton("Browse…")
        self.edt_input = QLineEdit()
        self.btn_browse_input = QPushButton("Browse…")
        self.edt_pattern = QLineEdit("*.csv")
        self.edt_config = QLineEdit("peakfocus_config.json")

        self.progress = QProgressBar(); self.progress.setTextVisible(True)
        self.chk_auto_switch = QCheckBox("Auto-switch to Focus when complete"); self.chk_auto_switch.setChecked(True)

        self.spn_skip = QDoubleSpinBox(); self.spn_skip.setRange(0, 3600); self.spn_skip.setDecimals(3); self.spn_skip.setValue(DEFAULT_CONFIG["SkipInitial"])
        self.spn_scan = QDoubleSpinBox(); self.spn_scan.setRange(0.1, 3600); self.spn_scan.setDecimals(3); self.spn_scan.setValue(DEFAULT_CONFIG["ScanWindowSec"])
        self.spn_seg  = QDoubleSpinBox(); self.spn_seg.setRange(0.001, 600); self.spn_seg.setDecimals(3); self.spn_seg.setValue(DEFAULT_CONFIG["SegmentLength"])
        self.spn_stride= QDoubleSpinBox(); self.spn_stride.setRange(0.001, 600); self.spn_stride.setDecimals(3); self.spn_stride.setValue(DEFAULT_CONFIG["Stride"])
        self.spn_desired_snr = QSpinBox(); self.spn_desired_snr.setRange(0, 100000); self.spn_desired_snr.setValue(DEFAULT_CONFIG["DesiredSNR"])
        self.spn_exp_spacing = QSpinBox(); self.spn_exp_spacing.setRange(1, 100000); self.spn_exp_spacing.setValue(DEFAULT_CONFIG["ExpectedSpacingMs"])
        self.spn_exp_width   = QSpinBox(); self.spn_exp_width.setRange(1, 100000); self.spn_exp_width.setValue(DEFAULT_CONFIG["ExpectedWidthMs"])
        self.spn_pf_best     = QSpinBox(); self.spn_pf_best.setRange(1, 99); self.spn_pf_best.setValue(DEFAULT_CONFIG.get("PerFileBestCount", 1))
        self.spn_ov_best     = QSpinBox(); self.spn_ov_best.setRange(1, 99); self.spn_ov_best.setValue(DEFAULT_CONFIG.get("OverallBestCount", 5))

        self.chk_autotrim = QCheckBox("Auto‑trim baseline warm‑up");        self.chk_autotrim.setChecked(True)
        self.chk_cooldown = QCheckBox("Auto‑detect cooldown cutoff");        self.chk_cooldown.setChecked(True)
        self.chk_simulate = QCheckBox("Generate ideal‑peak underlay");      self.chk_simulate.setChecked(True)
        self.chk_overlay  = QCheckBox("Overlay reshaped on plots");          self.chk_overlay.setChecked(True)
        self.chk_replace  = QCheckBox("Replace raw with reshaped");          self.chk_replace.setChecked(False)
        self.chk_export_pdf   = QCheckBox("Export overall PDF summary");     self.chk_export_pdf.setChecked(True)
        self.chk_export_shape = QCheckBox("Export shape summary CSV");       self.chk_export_shape.setChecked(False)
        self.chk_export_best_png = QCheckBox("Export overall best PNGs");    self.chk_export_best_png.setChecked(True)
        self.chk_export_best_csv = QCheckBox("Export overall best CSVs");    self.chk_export_best_csv.setChecked(True)

        self.cmb_rank_scheme = QComboBox()
        self.cmb_rank_scheme.addItems([
            "1) Balanced","2) SNR > Width > Spacing","3) Width > SNR > Spacing","4) Spacing > Width > SNR","5) Custom weights"
        ])
        self.cmb_rank_scheme.setCurrentIndex(DEFAULT_CONFIG.get("RankingScheme", 1) - 1)

        self.btn_save = QPushButton("Save Config Only")
        self.btn_run  = QPushButton("Run Analysis")
        self.btn_stop = QPushButton("Stop Run"); self.btn_stop.setEnabled(False)
        self.btn_open_out = QPushButton("Open Output Folder…"); self.btn_open_out.setEnabled(False)

        self.txt_log = QTextEdit(); self.txt_log.setReadOnly(True)

        grp_paths = QGroupBox("Paths & Files"); lay_paths = QFormLayout(grp_paths)
        row_script = QHBoxLayout(); row_script.addWidget(self.edt_script, 1); row_script.addWidget(self.btn_browse_script)
        row_input  = QHBoxLayout(); row_input.addWidget(self.edt_input, 1);  row_input.addWidget(self.btn_browse_input)
        lay_paths.addRow("PeakFocus script:", row_script)
        lay_paths.addRow("Input directory:", row_input)
        lay_paths.addRow("File pattern:", self.edt_pattern)
        lay_paths.addRow("Config file name:", self.edt_config)

        grp_prefs = QGroupBox("Preferences (saved to JSON)"); lay_prefs = QFormLayout(grp_prefs)
        for lbl, w in [
            ("Skip initial warm‑up (s):", self.spn_skip),
            ("Scan window after warm‑up (s):", self.spn_scan),
            ("Segment length (s):", self.spn_seg),
            ("Stride (s):", self.spn_stride),
            ("Desired SNR:", self.spn_desired_snr),
            ("Expected spacing (ms):", self.spn_exp_spacing),
            ("Expected peak width (ms):", self.spn_exp_width),
            ("Per‑file best count:", self.spn_pf_best),
            ("Overall best count:", self.spn_ov_best),
            ("Ranking scheme:", self.cmb_rank_scheme),
        ]:
            lay_prefs.addRow(lbl, w)
        flags_col = QVBoxLayout()
        for w in [self.chk_autotrim, self.chk_cooldown, self.chk_simulate, self.chk_overlay, self.chk_replace,
                  self.chk_export_pdf, self.chk_export_shape, self.chk_export_best_png, self.chk_export_best_csv]:
            flags_col.addWidget(w)
        lay_prefs.addRow(QLabel("Outputs & Options:"), QLabel(""))
        lay_prefs.addRow(flags_col)

        row_btns = QHBoxLayout(); row_btns.addWidget(self.btn_save); row_btns.addStretch(1); row_btns.addWidget(self.btn_stop); row_btns.addWidget(self.btn_run)

        lay_run = QVBoxLayout(self.tab_run)
        lay_run.addWidget(grp_paths)
        lay_run.addWidget(grp_prefs)
        lay_run.addLayout(row_btns)
        lay_run.addWidget(self.progress)
        lay_run.addWidget(self.chk_auto_switch)
        lay_run.addWidget(QLabel("Run log:"))
        lay_run.addWidget(self.txt_log, 1)
        lay_run.addWidget(self.btn_open_out)

        self.btn_browse_script.clicked.connect(self._pick_script)
        self.btn_browse_input.clicked.connect(self._pick_input)
        self.btn_save.clicked.connect(self._save_config)
        self.btn_run.clicked.connect(self._run)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_open_out.clicked.connect(self._open_output)
        self.chk_overlay.toggled.connect(self._sync_replace_enabled)

        self._guess_defaults()

    # Focus tab
    def _build_focus_tab(self) -> None:
        lay = QVBoxLayout(self.tab_focus)

        row = QHBoxLayout()
        self.edt_run = QLineEdit(); self.edt_run.setPlaceholderText("Select a run_YYYYMMDD_HHMMSS folder…")
        btn_browse_run = QPushButton("Browse run folder…")
        self.edt_source_root = QLineEdit(); self.edt_source_root.setPlaceholderText("(Optional) Source data root for original CSVs…")
        btn_browse_src = QPushButton("Browse source root…")
        row.addWidget(self.edt_run, 1); row.addWidget(btn_browse_run)
        row.addSpacing(12)
        row.addWidget(self.edt_source_root, 1); row.addWidget(btn_browse_src)

        self.spn_desired_snr2 = QSpinBox(); self.spn_desired_snr2.setRange(0, 100000); self.spn_desired_snr2.setValue(DEFAULT_CONFIG["DesiredSNR"])
        self.spn_exp_spacing2 = QSpinBox(); self.spn_exp_spacing2.setRange(1, 100000); self.spn_exp_spacing2.setValue(DEFAULT_CONFIG["ExpectedSpacingMs"])
        self.spn_exp_width2   = QSpinBox(); self.spn_exp_width2.setRange(1, 100000); self.spn_exp_width2.setValue(DEFAULT_CONFIG["ExpectedWidthMs"])
        self.cmb_scheme2 = QComboBox(); self.cmb_scheme2.addItems([
            "1) Balanced","2) SNR > Width > Spacing","3) Width > SNR > Spacing","4) Spacing > Width > SNR","5) Custom"
        ]); self.cmb_scheme2.setCurrentIndex(DEFAULT_CONFIG.get("RankingScheme", 1) - 1)
        btn_rescore = QPushButton("Re‑score")

        row2 = QHBoxLayout()
        for lbl, w in [("Desired SNR:", self.spn_desired_snr2), ("Expected spacing (ms):", self.spn_exp_spacing2), ("Expected width (ms):", self.spn_exp_width2), ("Scheme:", self.cmb_scheme2)]:
            row2.addWidget(QLabel(lbl)); row2.addWidget(w)
        row2.addStretch(1); row2.addWidget(btn_rescore)

        row3 = QHBoxLayout()
        self.chk_overlay_prev = QCheckBox("Overlay Reshape in preview"); self.chk_overlay_prev.setChecked(True)
        self.chk_replace_prev = QCheckBox("Replace Raw with Reshape in preview"); self.chk_replace_prev.setChecked(False)
        self.chk_use_all = QCheckBox("Use ALL segments (requires ExportSegCSV)"); self.chk_use_all.setChecked(True)
        self.chk_prefer_overview = QCheckBox("Prefer run overview traces for full plot"); self.chk_prefer_overview.setChecked(True)
        self.chk_highlight = QCheckBox("Highlight segment on overview"); self.chk_highlight.setChecked(True)
        self.spn_maxpts = QSpinBox(); self.spn_maxpts.setRange(500, 200000); self.spn_maxpts.setValue(4000)
        self.spn_maxpts.setToolTip("Downsample full trace to at most this many points")
        self.chk_interactive = QCheckBox("Interactive (cursor + wheel zoom)"); self.chk_interactive.setChecked(True)
        row3.addWidget(self.chk_overlay_prev); row3.addWidget(self.chk_replace_prev); row3.addSpacing(16)
        row3.addWidget(self.chk_use_all); row3.addSpacing(16)
        row3.addWidget(self.chk_prefer_overview); row3.addSpacing(16)
        row3.addWidget(self.chk_highlight); row3.addSpacing(16)
        row3.addWidget(QLabel("Max full‑trace points:")); row3.addWidget(self.spn_maxpts)
        row3.addSpacing(16)
        row3.addWidget(self.chk_interactive)
        row3.addStretch(1)

        # Peak detection controls
        row4 = QHBoxLayout()
        self.spn_thr_sigma = QDoubleSpinBox(); self.spn_thr_sigma.setRange(0.0, 20.0); self.spn_thr_sigma.setDecimals(2); self.spn_thr_sigma.setSingleStep(0.25); self.spn_thr_sigma.setValue(3.0)
        self.spn_min_space = QDoubleSpinBox(); self.spn_min_space.setRange(0.0, 100000.0); self.spn_min_space.setDecimals(1); self.spn_min_space.setSingleStep(1.0); self.spn_min_space.setSuffix(" ms"); self.spn_min_space.setValue(DEFAULT_CONFIG["ExpectedSpacingMs"])
        self.spn_width_frac = QDoubleSpinBox(); self.spn_width_frac.setRange(0.05, 0.95); self.spn_width_frac.setDecimals(2); self.spn_width_frac.setSingleStep(0.05); self.spn_width_frac.setValue(0.5)
        btn_remeasure = QPushButton("Re‑measure peaks")
        for lbl, w in [("Thresh (σ):", self.spn_thr_sigma), ("Min spacing:", self.spn_min_space), ("Width @ frac:", self.spn_width_frac)]:
            row4.addWidget(QLabel(lbl)); row4.addWidget(w)
        row4.addStretch(1); row4.addWidget(btn_remeasure)

        splitter = QSplitter(Qt.Horizontal)
        self.tbl = QTableWidget(0, 9)
        self.tbl.setHorizontalHeaderLabels(["Rank","File","Seg","Score","Base SNR","Raw SNR","Raw W (ms)","Raw Sp (ms)","CSV Path"])
        self.tbl.setColumnHidden(8, True)
        self.tbl.setSortingEnabled(False)
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)

        # Right: vertical splitter → (segment plot) / (overview) / (metrics table)
        right_split = QSplitter(Qt.Vertical)
        # Top: segment
        seg_widget = QWidget(); seg_v = QVBoxLayout(seg_widget); seg_v.setContentsMargins(0,0,0,0)
        self.fig_seg = Figure(figsize=(6,4)); self.ax_seg = self.fig_seg.add_subplot(111)
        self.canvas_seg = FigureCanvas(self.fig_seg); self.toolbar_seg = NavToolbar(self.canvas_seg, seg_widget)
        seg_v.addWidget(self.toolbar_seg); seg_v.addWidget(self.canvas_seg, 1)
        # Mid: overview
        full_widget = QWidget(); full_v = QVBoxLayout(full_widget); full_v.setContentsMargins(0,0,0,0)
        self.fig_full = Figure(figsize=(6,2.8)); self.ax_full = self.fig_full.add_subplot(111)
        self.canvas_full = FigureCanvas(self.fig_full)
        full_v.addWidget(QLabel("File overview (original or run overview):"))
        full_v.addWidget(self.canvas_full, 1)
        # Bottom: metrics table
        self.tbl_metrics = QTableWidget(0, 6)
        self.tbl_metrics.setHorizontalHeaderLabels(["#","Time (s)","Height","Area","Width (ms)","Spacing→ (ms)"])
        self.tbl_metrics.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.btn_export_metrics = QPushButton("Export measured peaks…")
        met_widget = QWidget(); met_v = QVBoxLayout(met_widget); met_v.setContentsMargins(0,0,0,0)
        met_v.addWidget(self.tbl_metrics); met_v.addWidget(self.btn_export_metrics)

        right_split.addWidget(seg_widget)
        right_split.addWidget(full_widget)
        right_split.addWidget(met_widget)
        right_split.setSizes([520, 320, 200])

        splitter.addWidget(self.tbl); splitter.addWidget(right_split)
        splitter.setSizes([560, 760])

        lay.addLayout(row)
        lay.addLayout(row2)
        lay.addLayout(row3)
        lay.addLayout(row4)
        lay.addWidget(splitter, 1)

        btn_browse_run.clicked.connect(self._pick_run_folder)
        btn_browse_src.clicked.connect(self._pick_source_root)
        btn_rescore.clicked.connect(self._rescore)
        btn_remeasure.clicked.connect(self._remeasure)
        self.btn_export_metrics.clicked.connect(self._export_metrics)

        self.tbl.itemSelectionChanged.connect(self._show_selected)
        self.tbl.horizontalHeader().sortIndicatorChanged.connect(lambda *_: self._renumber_ranks())
        self.chk_overlay_prev.toggled.connect(lambda *_: self._show_selected())
        self.chk_replace_prev.toggled.connect(lambda *_: self._show_selected())
        self.chk_highlight.toggled.connect(lambda *_: self._show_selected())

        # Interactivity
        self.chk_interactive.toggled.connect(self._refresh_interactivity)
        self._int_seg = None
        self._int_full = None
        self._refresh_interactivity()

    # Helpers (Run) — same as v7
    def _pick_script(self) -> None:
        fn, _ = QFileDialog.getOpenFileName(self, "Select PeakFocus script", str(Path.cwd()), "Python (*.py)")
        if fn: self.edt_script.setText(fn)
    def _pick_input(self) -> None:
        dn = QFileDialog.getExistingDirectory(self, "Select input folder", str(Path.cwd()))
        if dn: self.edt_input.setText(dn)

    def _pick_run_folder(self) -> None:
        dn = QFileDialog.getExistingDirectory(self, "Select run folder", str(Path.cwd()))
        if dn:
            self.edt_run.setText(dn)
            self._load_run_metrics(Path(dn))
    def _pick_source_root(self) -> None:
        dn = QFileDialog.getExistingDirectory(self, "Select source data root", str(Path.cwd()))
        if dn:
            self.edt_source_root.setText(dn)
            self._full_cache.clear(); self._show_selected()

    def _guess_defaults(self) -> None:
        candidates = list(Path.cwd().glob("peakfocus_b9*.py"))
        if candidates:
            self.edt_script.setText(str(candidates[0]))
        self.edt_input.setText(str(Path.cwd()))
        self._sync_replace_enabled(self.chk_overlay.isChecked())

    def _sync_replace_enabled(self, overlay_enabled: bool) -> None:
        self.chk_replace.setEnabled(overlay_enabled)
        if not overlay_enabled:
            self.chk_replace.setChecked(False)

    def _collect_prefs(self) -> dict:
        prefs = DEFAULT_CONFIG.copy()
        prefs.update({
            "SkipInitial": float(self.spn_skip.value()),
            "ScanWindowSec": float(self.spn_scan.value()),
            "SegmentLength": float(self.spn_seg.value()),
            "Stride": float(self.spn_stride.value()),
            "DesiredSNR": int(self.spn_desired_snr.value()),
            "ExpectedSpacingMs": int(self.spn_exp_spacing.value()),
            "ExpectedWidthMs": int(self.spn_exp_width.value()),
            "PerFileBestCount": int(self.spn_pf_best.value()),
            "OverallBestCount": int(self.spn_ov_best.value()),
            "AutoTrim": self.chk_autotrim.isChecked(),
            "AutoDetectCooldown": self.chk_cooldown.isChecked(),
            "Simulate": self.chk_simulate.isChecked(),
            "OverlayReshape": self.chk_overlay.isChecked(),
            "ReplaceWithReshape": self.chk_replace.isChecked(),
            "ExportOverallPDF": self.chk_export_pdf.isChecked(),
            "ExportShapeCSV": self.chk_export_shape.isChecked(),
            "ExportBestPNG": self.chk_export_best_png.isChecked(),
            "ExportBestCSV": self.chk_export_best_csv.isChecked(),
            "RankingScheme": self.cmb_rank_scheme.currentIndex() + 1,
        })
        return prefs

    def _save_config(self) -> None:
        try:
            cfg_path = Path(self.edt_config.text()).resolve()
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(self._collect_prefs(), f, indent=2)
            self._log(f"Saved config → {cfg_path}")
        except Exception as e:
            QMessageBox.critical(self, "PeakFocus GUI", f"Failed to save config: {e}")

    # Run handling with progress (unchanged from v7)
    def _run(self) -> None:
        if self.process is not None:
            QMessageBox.critical(self, "PeakFocus GUI", "A run is already in progress.")
            return
        script = Path(self.edt_script.text()); input_dir = Path(self.edt_input.text())
        if not script.exists():
            QMessageBox.critical(self, "PeakFocus GUI", "PeakFocus script not found.")
            return
        if not input_dir.exists():
            QMessageBox.critical(self, "PeakFocus GUI", "Input directory does not exist.")
            return
        try:
            cfg_path = Path(self.edt_config.text()).resolve()
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(self._collect_prefs(), f, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "PeakFocus GUI", f"Failed to write config JSON: {e}")
            return
        pattern = self.edt_pattern.text().strip() or "*.csv"
        self.txt_log.clear(); self.btn_run.setEnabled(False); self.btn_stop.setEnabled(True); self.btn_open_out.setEnabled(False)
        self._output_dir_last = None
        self.progress.setRange(0,0); self.progress.setFormat("Running… %p%")

        cmd = [sys.executable, str(script), "--config", str(cfg_path), "--no-interactive",
               "--input-dir", str(input_dir), "--file-pattern", pattern]
        self._log("Launching: \n" + " ".join(cmd))

        def worker():
            try:
                self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
                assert self.process.stdout is not None
                for line in self.process.stdout:
                    self._append_log_from_thread(line.rstrip())
                    m = OUTPUT_DIR_REGEX.search(line)
                    if m:
                        raw = m.group(1).strip().strip('"')
                        p = Path(raw)
                        if not p.is_absolute():
                            p = Path.cwd() / p
                        self._output_dir_last = p.resolve()
                    else:
                        m2 = RUN_NAME_REGEX.search(line)
                        if m2 and not self._output_dir_last:
                            nm = m2.group(0)
                            for base in [Path.cwd(), Path(self.edt_input.text() or '.' )]:
                                cand = (base / nm)
                                if cand.exists():
                                    self._output_dir_last = cand.resolve(); break
                    for pp in PROG_PATTERNS:
                        mm = pp.search(line)
                        if mm:
                            try:
                                cur = int(mm.group(1)); total = int(mm.group(2))
                                QTimer.singleShot(0, lambda c=cur,t=total: self._set_progress(c, t))
                            except Exception:
                                pass
                            break
                rc = self.process.wait()
                self._append_log_from_thread(f"\nProcess finished with exit code {rc}")
            except Exception as e:
                self._append_log_from_thread(f"\nERROR: {e}")
            finally:
                QTimer.singleShot(0, self._run_finished)
        threading.Thread(target=worker, daemon=True).start()

    def _set_progress(self, cur: int, total: int) -> None:
        total = max(total, 1)
        if self.progress.maximum() == 0:
            self.progress.setRange(0, total)
        self.progress.setMaximum(total)
        self.progress.setValue(min(cur, total))
        self.progress.setFormat(f"{cur}/{total} (%p%)")

    def _stop(self) -> None:
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate(); self._log("Termination signal sent.")
            except Exception as e:
                self._log(f"Failed to terminate: {e}")
        else:
            self._log("No active process to stop.")

    def _run_finished(self) -> None:
        if self.progress.maximum() == 0:
            self.progress.setRange(0, 100); self.progress.setValue(100)
        else:
            self.progress.setValue(self.progress.maximum())
        self.btn_run.setEnabled(True); self.btn_stop.setEnabled(False)
        if (not self._output_dir_last) or (self._output_dir_last and not self._output_dir_last.exists()):
            candidates: List[Path] = []
            for base in [Path.cwd(), Path(self.edt_input.text() or '.')]:
                if base.exists():
                    candidates += [p for p in base.glob('run_*') if p.is_dir()]
            if candidates:
                candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                self._output_dir_last = candidates[0].resolve()
        if self._output_dir_last and self._output_dir_last.exists():
            self.btn_open_out.setEnabled(True)
            self.edt_run.setText(str(self._output_dir_last))
            self._load_run_metrics(self._output_dir_last)
            if self.chk_auto_switch.isChecked():
                self.tabs.setCurrentWidget(self.tab_focus)
        self.process = None

    def _open_output(self) -> None:
        if not self._output_dir_last:
            QMessageBox.critical(self, "PeakFocus GUI", "No output folder detected yet.")
            return
        out_dir = self._output_dir_last
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(out_dir))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(out_dir)])
            else:
                subprocess.Popen(["xdg-open", str(out_dir)])
        except Exception as e:
            QMessageBox.critical(self, "PeakFocus GUI", f"Failed to open folder: {e}")

    def _log(self, text: str) -> None:
        self.txt_log.append(text)
        self.txt_log.moveCursor(QTextCursor.End)
        self.txt_log.ensureCursorVisible()

    def _append_log_from_thread(self, text_line: str) -> None:
        QTimer.singleShot(0, lambda: self._log(text_line))

    # ==== Focus (load/score/plot) ====
    def _load_run_metrics(self, run_folder: Path) -> None:
        indiv = run_folder / 'Individual'
        if not indiv.exists():
            QMessageBox.warning(self, "Focus", f"No 'Individual' folder found in\n{run_folder}")
            self.tbl.setRowCount(0); self.ax_seg.clear(); self.canvas_seg.draw(); return
        use_all = self.chk_use_all.isChecked()
        row_map: Dict[Tuple[str,int], Dict[str, Any]] = {}
        csv_map: Dict[Tuple[str,int], Path] = {}
        for sub in sorted(p for p in indiv.iterdir() if p.is_dir()):
            fname = sub.name
            mfile = sub / 'metrics.csv'
            if mfile.exists():
                try:
                    mdf = pd.read_csv(mfile)
                    for _, r in mdf.iterrows():
                        seg = int(r.get('segment', 0))
                        row_map[(fname, seg)] = {
                            'file': fname,
                            'segment': seg,
                            'snr_baseline_raw': float(r.get('snr_baseline_raw', r.get('snr_baseline', 0.0))),
                            'snr_raw': float(r.get('snr_raw', 0.0)),
                            'width_raw_ms': float(r.get('width_raw_ms', r.get('width_raw', np.nan))),
                            'spacing_raw_ms': float(r.get('spacing_raw_ms', r.get('spacing_raw', np.nan))),
                        }
                except Exception:
                    pass
            for csv in sub.glob("*_seg*.csv"):
                m = SEG_NUM_RE.search(csv.name)
                seg_idx = int(m.group(1)) if m else 0
                csv_map[(fname, seg_idx)] = csv
                if use_all:
                    try:
                        df = pd.read_csv(csv, nrows=1)
                        row_map[(fname, seg_idx)] = {
                            'file': fname,
                            'segment': seg_idx,
                            'snr_baseline_raw': float(df.get('snr_baseline_raw', pd.Series([0])).iloc[0]),
                            'snr_raw': float(df.get('snr_raw', pd.Series([0])).iloc[0]),
                            'width_raw_ms': float(df.get('width_raw_ms', pd.Series([np.nan])).iloc[0]),
                            'spacing_raw_ms': float(df.get('spacing_raw_ms', pd.Series([np.nan])).iloc[0]),
                        }
                    except Exception:
                        pass
        self._rows = list(row_map.values())
        self._csv_map = csv_map
        self._rescore()

    def _current_focus_prefs(self) -> Dict[str, Any]:
        return {
            "DesiredSNR": int(self.spn_desired_snr2.value()),
            "ExpectedSpacingMs": int(self.spn_exp_spacing2.value()),
            "ExpectedWidthMs": int(self.spn_exp_width2.value()),
            "RankingScheme": self.cmb_scheme2.currentIndex() + 1,
            "OverlayReshape": self.chk_overlay_prev.isChecked(),
            "ReplaceWithReshape": self.chk_replace_prev.isChecked() and self.chk_overlay_prev.isChecked(),
            "Simulate": True,
        }

    def _rescore(self) -> None:
        prefs = self._current_focus_prefs()
        data = []
        for r in self._rows:
            s = score_segment_row(r, prefs)
            data.append((s, r))
        data.sort(key=lambda x: x[0], reverse=True)
        self._rescored_rows = [
            {
                'rank': i+1,
                'file': r['file'],
                'segment': r['segment'],
                'score': float(score),
                'snr_baseline': float(r.get('snr_baseline_raw', 0.0)),
                'snr_raw': float(r.get('snr_raw', 0.0)),
                'width_raw_ms': float(r.get('width_raw_ms', np.nan)),
                'spacing_raw_ms': float(r.get('spacing_raw_ms', np.nan)),
                'csv_path': str(self._csv_map.get((r['file'], r['segment']), Path()))
            }
            for i, (score, r) in enumerate(data)
        ]
        self._populate_table()

    def _populate_table(self) -> None:
        self.tbl.setSortingEnabled(False)
        self.tbl.blockSignals(True)
        try:
            self.tbl.setRowCount(len(self._rescored_rows))
            for i, row in enumerate(self._rescored_rows):
                vals = [row['rank'], row['file'], row['segment'], row['score'], row['snr_baseline'], row['snr_raw'], row['width_raw_ms'], row['spacing_raw_ms'], row['csv_path']]
                for col, val in enumerate(vals):
                    it = QTableWidgetItem()
                    if isinstance(val, (int, float)):
                        it.setData(Qt.DisplayRole, val)
                    else:
                        it.setText(str(val))
                    if col in (0,2): it.setTextAlignment(Qt.AlignCenter)
                    self.tbl.setItem(i, col, it)
            self.tbl.resizeColumnsToContents()
            if self.tbl.rowCount():
                self.tbl.selectRow(0)
        finally:
            self.tbl.blockSignals(False)
        self.tbl.setSortingEnabled(True)
        self.tbl.sortItems(0, Qt.AscendingOrder)
        self._renumber_ranks()
        QTimer.singleShot(0, self._show_selected)

    def _renumber_ranks(self) -> None:
        for i in range(self.tbl.rowCount()):
            it = self.tbl.item(i, 0)
            if it:
                it.setData(Qt.DisplayRole, i+1)

    # ---- full-trace loading & plotting
    def _load_full_trace_for(self, file_disp: str) -> Optional[pd.DataFrame]:
        if self.chk_prefer_overview.isChecked() and self.edt_run.text():
            run_folder = Path(self.edt_run.text())
            sub = run_folder / 'Individual' / file_disp
            if sub.exists():
                for pattern in ["overview.csv", f"{file_disp}_overview.csv", "*overview*.csv"]:
                    for cand in sub.glob(pattern):
                        try:
                            return pd.read_csv(cand)
                        except Exception:
                            pass
        if not self.edt_source_root.text():
            return None
        src_root = Path(self.edt_source_root.text())
        if not src_root.exists():
            return None
        base = LEADING_NUM_RE.sub("", file_disp)
        for ext in (".csv", ".CSV", ".tsv"):
            cand = src_root / f"{base}{ext}"
            if cand.exists():
                key = str(cand)
                if key in self._full_cache:
                    return self._full_cache[key]
                try:
                    df = pd.read_csv(cand)
                    tcol = next((c for c in df.columns if str(c).lower().startswith('time')), df.columns[0])
                    if df.shape[1] >= 2:
                        ycol = df.columns[1] if df.columns[1] != tcol else df.columns[-1]
                    else:
                        return None
                    maxpts = int(self.spn_maxpts.value())
                    n = len(df)
                    if n > maxpts and maxpts > 0:
                        idx = np.linspace(0, n-1, maxpts).astype(int)
                        slim = pd.DataFrame({'time': df[tcol].to_numpy()[idx], 'raw': df[ycol].to_numpy()[idx]})
                    else:
                        slim = pd.DataFrame({'time': df[tcol], 'raw': df[ycol]})
                    self._full_cache[key] = slim
                    return slim
                except Exception:
                    return None
        return None

    # ---- selection handler: draw plots, highlight segment, measure peaks
    def _show_selected(self) -> None:
        if self.tbl.rowCount() == 0:
            for ax, canvas in [(self.ax_seg, self.canvas_seg), (self.ax_full, self.canvas_full)]:
                ax.clear(); ax.text(0.5,0.5,"No rows", ha='center'); canvas.draw()
            self.tbl_metrics.setRowCount(0)
            return
        sel = self.tbl.selectedItems()
        if not sel:
            self.tbl.selectRow(0)
            sel = self.tbl.selectedItems()
        row = sel[0].row()
        def sval(c):
            it = self.tbl.item(row, c); return it.data(Qt.DisplayRole) if it and it.data(Qt.DisplayRole) is not None else (it.text() if it else "")
        file_disp = str(sval(1))
        seg = int(sval(2)) if str(sval(2)).isdigit() else 0
        csvp = Path(self.tbl.item(row, 8).text()) if self.tbl.item(row, 8) else Path()

        # Segment plot + metrics
        df_seg = None
        if csvp.exists():
            try:
                df_seg = pd.read_csv(csvp)
                plot_segment(self.ax_seg, df_seg, self._current_focus_prefs(), f"{file_disp} seg{seg}")
                self.canvas_seg.draw()
                self._remeasure(df_seg)
            except Exception as e:
                self.ax_seg.clear(); self.ax_seg.text(0.5,0.5,f"Plot error: {e}", ha='center'); self.canvas_seg.draw(); self.tbl_metrics.setRowCount(0)
        else:
            self.ax_seg.clear(); self.ax_seg.text(0.5,0.5,"No CSV to plot", ha='center'); self.canvas_seg.draw(); self.tbl_metrics.setRowCount(0)

        # Overview + highlight
        df_full = self._load_full_trace_for(file_disp)
        if df_full is not None and {'time','raw'}.issubset(df_full.columns):
            self.ax_full.clear(); self.ax_full.plot(df_full['time'], df_full['raw'])
            self.ax_full.set_title(f"{file_disp} — overall trace")
            self.ax_full.set_xlabel('Time (s)'); self.ax_full.set_ylabel('Intensity (CPS)')
            if self.chk_highlight.isChecked() and df_seg is not None:
                self._highlight_segment_on_overview(df_full, df_seg)
            self.canvas_full.draw()
        else:
            self.ax_full.clear(); self.ax_full.text(0.5,0.5,"No full trace available (set source root or export overview)", ha='center'); self.canvas_full.draw()

        self._refresh_interactivity()

    # ---- peak measurement plumbing
    def _remeasure(self, df_seg: Optional[pd.DataFrame] = None) -> None:
        if df_seg is None:
            sel = self.tbl.selectedItems()
            if not sel:
                return
            row = sel[0].row()
            csvp = Path(self.tbl.item(row, 8).text()) if self.tbl.item(row, 8) else Path()
            if not csvp.exists():
                self.tbl_metrics.setRowCount(0); return
            try:
                df_seg = pd.read_csv(csvp)
            except Exception:
                self.tbl_metrics.setRowCount(0); return
        t = df_seg.get('time', None)
        y = df_seg.get('reshaped', df_seg.get('raw', None))
        if t is None or y is None:
            self.tbl_metrics.setRowCount(0); return
        t = np.asarray(t); y = np.asarray(y)
        thr_sigma = float(self.spn_thr_sigma.value())
        min_space_s = float(self.spn_min_space.value()) / 1000.0
        width_frac = float(self.spn_width_frac.value())
        peaks = measure_peaks(t, y, thr_sigma, min_space_s, width_frac)
        # overlay markers on segment plot
        if len(peaks):
            self.ax_seg.plot([p['time_s'] for p in peaks], [y[int(p['index'])] for p in peaks], 'o', ms=5, alpha=0.8)
            # label up to 12 peaks to keep it readable
            for j, p in enumerate(peaks[:12]):
                self.ax_seg.text(p['time_s'], y[int(p['index'])], f"{j+1}", fontsize=8, ha='center', va='bottom')
            self.canvas_seg.draw()
        # fill table
        self.tbl_metrics.setRowCount(len(peaks))
        for i, p in enumerate(peaks):
            vals = [i+1, p['time_s'], p['height'], p['area'], p['width_ms'], p['spacing_ms']]
            for c, v in enumerate(vals):
                it = QTableWidgetItem()
                it.setData(Qt.DisplayRole, float(v) if isinstance(v, (int, float)) else v)
                self.tbl_metrics.setItem(i, c, it)
        self.tbl_metrics.resizeColumnsToContents()

    def _export_metrics(self) -> None:
        if self.tbl_metrics.rowCount() == 0:
            QMessageBox.information(self, "Export", "No measured peaks to export.")
            return
        fn, _ = QFileDialog.getSaveFileName(self, "Save measured peaks", str(Path.cwd()/"peakfocus_measured_peaks.csv"), "CSV Files (*.csv)")
        if not fn:
            return
        try:
            rows = []
            for i in range(self.tbl_metrics.rowCount()):
                rows.append({
                    '#': self.tbl_metrics.item(i,0).data(Qt.DisplayRole),
                    'time_s': self.tbl_metrics.item(i,1).data(Qt.DisplayRole),
                    'height': self.tbl_metrics.item(i,2).data(Qt.DisplayRole),
                    'area': self.tbl_metrics.item(i,3).data(Qt.DisplayRole),
                    'width_ms': self.tbl_metrics.item(i,4).data(Qt.DisplayRole),
                    'spacing_ms': self.tbl_metrics.item(i,5).data(Qt.DisplayRole),
                })
            pd.DataFrame(rows).to_csv(fn, index=False)
            QMessageBox.information(self, "Export", f"Saved → {fn}")
        except Exception as e:
            QMessageBox.critical(self, "Export", f"Failed to save: {e}")

    # ---- overview highlight helpers
    def _highlight_segment_on_overview(self, df_full: pd.DataFrame, df_seg: pd.DataFrame) -> None:
        tf = df_full['time'].to_numpy()
        yf = df_full['raw'].to_numpy()
        # Try absolute bounds first
        t0 = float(df_seg.get('t_abs_start', pd.Series([np.nan])).iloc[0]) if 't_abs_start' in df_seg.columns else np.nan
        t1 = float(df_seg.get('t_abs_end', pd.Series([np.nan])).iloc[0]) if 't_abs_end' in df_seg.columns else np.nan
        if not np.isfinite(t0) or not np.isfinite(t1):
            ts = df_seg.get('time', None)
            if ts is not None:
                ts = np.asarray(ts)
                # If segment time is already within overview range, assume absolute
                if ts.min() >= tf.min()-1e-6 and ts.max() <= tf.max()+1e-6:
                    t0, t1 = float(ts.min()), float(ts.max())
        # If still unknown, perform quick correlation-based localization
        if not (np.isfinite(t0) and np.isfinite(t1)):
            ts = np.asarray(df_seg.get('time', []))
            ys = np.asarray(df_seg.get('reshaped', df_seg.get('raw', [])))
            if len(ts) >= 8 and len(tf) >= len(ts):
                dur = ts.max() - ts.min()
                # Normalize to zero-mean unit-variance to be scale-insensitive
                ys_n = (ys - np.median(ys))
                if np.std(ys_n) > 0:
                    ys_n = ys_n / (np.std(ys_n) + 1e-9)
                # resample seg to fixed 256 points
                nS = 256
                gridS = np.linspace(ts.min(), ts.max(), nS)
                yS = np.interp(gridS, ts, ys_n)
                # stride across overview in ~1/200 of its length
                step = max(int(len(tf) / 200), 1)
                best_score = -1e9; best_i0 = 0; nW = nS
                for i0 in range(0, max(0, len(tf)-nW), step):
                    win_t = tf[i0:i0+nW]
                    if len(win_t) < nW: break
                    win_y = yf[i0:i0+nW]
                    wy = win_y - np.median(win_y)
                    if np.std(wy) > 0:
                        wy = wy / (np.std(wy) + 1e-9)
                    # resample window to segment grid
                    win_grid = np.linspace(win_t.min(), win_t.max(), nS)
                    wyS = np.interp(gridS, win_grid, wy)
                    score = float(np.dot(yS, wyS))
                    if score > best_score:
                        best_score = score; best_i0 = i0
                i1 = min(best_i0 + nW, len(tf)-1)
                t0, t1 = float(tf[best_i0]), float(tf[i1])
        # draw highlight if we have bounds
        if np.isfinite(t0) and np.isfinite(t1) and t1 > t0:
            self.ax_full.axvspan(t0, t1, color='orange', alpha=0.18, label='Selected segment')
            # optional: mark measured peak centers that fall within [t0,t1]
            if 'time' in df_seg.columns:
                tt = np.asarray(df_seg['time'])
                yy = np.asarray(df_seg.get('reshaped', df_seg.get('raw', df_seg.iloc[:,1])))
                thr_sigma = float(self.spn_thr_sigma.value())
                min_space_s = float(self.spn_min_space.value()) / 1000.0
                width_frac = float(self.spn_width_frac.value())
                peaks = measure_peaks(tt, yy, thr_sigma, min_space_s, width_frac)
                pk_times = [p['time_s'] for p in peaks if p['time_s'] >= t0 and p['time_s'] <= t1]
                pk_vals = []
                if len(pk_times):
                    # map their values from overview (interpolate)
                    pk_vals = list(np.interp(pk_times, tf, yf))
                    self.ax_full.plot(pk_times, pk_vals, 'o', ms=4, alpha=0.9)
            self.ax_full.legend(loc='upper right')

    # ---- interactive plot helpers (same as v7)
    def _refresh_interactivity(self) -> None:
        enable = self.chk_interactive.isChecked()
        for handle in (getattr(self, "_int_seg", None), getattr(self, "_int_full", None)):
            if handle:
                self._detach_interactions(handle)
        if enable:
            self._int_seg  = self._attach_interactions(self.canvas_seg, self.ax_seg)
            self._int_full = self._attach_interactions(self.canvas_full, self.ax_full)
        else:
            self._int_seg = self._int_full = None

    def _attach_interactions(self, canvas: FigureCanvas, ax) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        vline = ax.axvline(np.nan, color='k', alpha=0.18)
        hline = ax.axhline(np.nan, color='k', alpha=0.18)
        canvas.draw_idle()
        data['vline'] = vline; data['hline'] = hline

        def on_move(evt):
            if evt.inaxes != ax or evt.xdata is None or evt.ydata is None:
                return
            vline.set_xdata(evt.xdata); hline.set_ydata(evt.ydata)
            canvas.draw_idle()
        def on_scroll(evt):
            if evt.inaxes != ax or evt.xdata is None or evt.ydata is None:
                return
            base = 1.2
            scale = (1/base) if evt.button == 'up' else base
            x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
            if evt.key == 'control':
                new_h = (y1-y0) * scale; yc = evt.ydata; ylow = yc - (yc - y0) * scale; ax.set_ylim(ylow, ylow + new_h)
            else:
                new_w = (x1-x0) * scale; xc = evt.xdata; xlow = xc - (xc - x0) * scale; ax.set_xlim(xlow, xlow + new_w)
            canvas.draw_idle()
        def on_click(evt):
            if evt.inaxes == ax and evt.button == 3:
                ax.relim(); ax.autoscale(); canvas.draw_idle()
        cid_move   = canvas.mpl_connect('motion_notify_event', on_move)
        cid_scroll = canvas.mpl_connect('scroll_event', on_scroll)
        cid_click  = canvas.mpl_connect('button_press_event', on_click)
        data['cids'] = [cid_move, cid_scroll, cid_click]
        if HAVE_MPLC:
            try:
                cur = mplcursors.cursor(ax, hover=True)
                @cur.connect("add")
                def _on_add(sel):
                    try:
                        x, y = sel.target
                    except Exception:
                        x = getattr(sel.annotation, 'x', None) or 0.0
                        y = getattr(sel.annotation, 'y', None) or 0.0
                    sel.annotation.set_text(f"{x:.6g}, {y:.6g}")
                data['cursor'] = cur
            except Exception:
                pass
        return data

    def _detach_interactions(self, handle: Dict[str, Any]) -> None:
        try:
            canvas = self.canvas_seg if handle is self._int_seg else self.canvas_full
            for cid in handle.get('cids', []):
                canvas.mpl_disconnect(cid)
            for k in ('vline','hline'):
                obj = handle.get(k)
                if obj: obj.remove()
            canvas.draw_idle()
        except Exception:
            pass

    # ---- Top-N export (unchanged)
    def _export_topn(self) -> None:
        if self.tbl.rowCount() == 0:
            QMessageBox.information(self, "Export", "Nothing to export yet. Load a run and Re‑score.")
            return
        fn, _ = QFileDialog.getSaveFileName(self, "Save Top‑N as CSV", str(Path.cwd()/"peakfocus_topN.csv"), "CSV Files (*.csv)")
        if not fn:
            return
        try:
            rows = []
            topn = min(int(self.tbl.rowCount()),  max(1, int(self.tbl.rowCount())))
            for i in range(topn):
                rows.append({
                    'rank': self.tbl.item(i,0).data(Qt.DisplayRole),
                    'file': self.tbl.item(i,1).text(),
                    'segment': self.tbl.item(i,2).data(Qt.DisplayRole),
                    'score': self.tbl.item(i,3).data(Qt.DisplayRole),
                    'snr_baseline': self.tbl.item(i,4).data(Qt.DisplayRole),
                    'snr_raw': self.tbl.item(i,5).data(Qt.DisplayRole),
                    'width_raw_ms': self.tbl.item(i,6).data(Qt.DisplayRole),
                    'spacing_raw_ms': self.tbl.item(i,7).data(Qt.DisplayRole),
                    'csv_path': self.tbl.item(i,8).text(),
                })
            pd.DataFrame(rows).to_csv(fn, index=False)
            QMessageBox.information(self, "Export", f"Saved → {fn}")
        except Exception as e:
            QMessageBox.critical(self, "Export", f"Failed to save: {e}")

# Entrypoint

def main() -> int:
    app = QApplication(sys.argv)
    win = PeakFocusGUI()
    win.show()
    act_save = QAction("Save Config", win); act_save.setShortcut("Ctrl+S"); act_save.triggered.connect(win._save_config)
    win.addAction(act_save)
    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())
