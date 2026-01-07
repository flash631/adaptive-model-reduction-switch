from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal as sp_signal  # type: ignore

from config import load_yaml
from foam_ascii import n_cells_from_polymesh
from confinement import compute_confinement, compute_span, write_metrics
from forces import ForceCoeffsSeries, read_force_coeffs_dat, read_forces_dat
from geometry import cylinder_geometry, read_inlet_u_mag
from metrics import mean_after_discard, rms_fluct_after_discard
from plots import (
    save_energy_plot,
    save_error_plot_two_rel_abs,
    save_fft_plot,
    save_force_coeffs_plot,
    save_forces_plot,
    save_singular_values_plot,
)


@dataclass(frozen=True)
class CaseSummary:
    name: str
    n_cells: int
    deltaT: float
    endTime: float
    writeInterval: float
    nWrites: int | None
    nSnapshotsUsed: int
    snapshotStride: int


@dataclass(frozen=True)
class DominantFreqEstimate:
    f_peak_hz: float | None
    freq_hz: np.ndarray
    psd: np.ndarray
    duration_kept_s: float
    n_cycles_kept: float


def _coeffs_from_forces(forces_path: Path, *, u_inf: float, a_ref: float, d_ref: float | None = None) -> ForceCoeffsSeries:
    """
    Recompute Cd/Cl from OpenFOAM `forces` output without rerunning any solver.

    Cd = Fx / (0.5*rho*Uinf^2*Aref), Cl = Fy / (0.5*rho*Uinf^2*Aref) with rho=1.
    """
    rho = 1.0
    u_inf = float(u_inf)
    a_ref = float(a_ref)
    if not (np.isfinite(u_inf) and u_inf > 0 and np.isfinite(a_ref) and a_ref > 0):
        raise ValueError("Invalid U_inf or Aref for coefficient recomputation")
    s = read_forces_dat(forces_path)
    q = 0.5 * rho * u_inf * u_inf * a_ref
    return ForceCoeffsSeries(
        time=np.asarray(s.time, dtype=float),
        Cd=np.asarray(s.Fx, dtype=float) / float(q),
        Cl=np.asarray(s.Fy, dtype=float) / float(q),
        Cm=None,
        magUInf=float(u_inf),
        lRef=float(d_ref) if (d_ref is not None and np.isfinite(d_ref)) else None,
        Aref=float(a_ref),
    )


def _read_control_dict(case_dir: Path) -> dict[str, str]:
    cd = case_dir / "system" / "controlDict"
    txt = cd.read_text(encoding="utf-8", errors="ignore")

    def get_key(k: str) -> str:
        m = re.search(rf"^\s*{re.escape(k)}\s+([^;]+)\s*;", txt, flags=re.M)
        return m.group(1).strip() if m else ""

    return {k: get_key(k) for k in ["deltaT", "endTime", "writeInterval", "writeControl"]}


def _warn(msg: str) -> None:
    print(f"[report_assets] WARNING: {msg}", file=sys.stderr)


def _read_scalar_dict_value(path: Path, key: str) -> str:
    if not path.exists():
        return ""
    txt = path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(rf"^\s*{re.escape(key)}\s+([^;]+)\s*;", txt, flags=re.M)
    return m.group(1).strip() if m else ""


def _read_solver_name(case_dir: Path) -> str:
    cd = case_dir / "system" / "controlDict"
    if not cd.exists():
        return ""
    txt = cd.read_text(encoding="utf-8", errors="ignore")
    for k in ["application", "solver"]:
        m = re.search(rf"^\s*{re.escape(k)}\s+([^;]+)\s*;", txt, flags=re.M)
        if m:
            return m.group(1).strip()
    return ""


def _read_inlet_u(case_dir: Path) -> str:
    u0 = case_dir / "0" / "U"
    if not u0.exists():
        return ""
    txt = u0.read_text(encoding="utf-8", errors="ignore")
    pat = re.compile(
        r"(?s)\bboundaryField\s*\{.*?\}\s*//\s*\*+\s*$",
        flags=re.M,
    )
    m = pat.search(txt)
    body = m.group(0) if m else txt
    patches: dict[str, str] = {}
    for pm in re.finditer(
        r"(?s)^\s*([A-Za-z0-9_]+)\s*\{.*?\btype\s+fixedValue\s*;.*?\bvalue\s+uniform\s*\(\s*([^)]+?)\s*\)\s*;.*?\}",
        body,
        flags=re.M,
    ):
        patches[pm.group(1)] = "(" + " ".join(pm.group(2).split()) + ")"

    for preferred in ["inlet", "left", "inflow"]:
        if preferred in patches:
            return patches[preferred]
    # Fallback: first fixedValue patch.
    if patches:
        return patches[sorted(patches.keys())[0]]
    return ""


def _read_domain_bounds(case_dir: Path) -> str:
    pts = case_dir / "constant" / "polyMesh" / "points"
    if not pts.exists():
        return ""
    try:
        xmin = ymin = zmin = float("inf")
        xmax = ymax = zmax = float("-inf")
        vec_re = re.compile(r"^\s*\(\s*([Ee0-9+\-\.]+)\s+([Ee0-9+\-\.]+)\s+([Ee0-9+\-\.]+)\s*\)\s*$")
        in_list = False
        for line in pts.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not in_list and line.strip() == "(":
                in_list = True
                continue
            if in_list and line.strip() == ")":
                break
            if not in_list:
                continue
            m = vec_re.match(line)
            if not m:
                continue
            x, y, z = float(m.group(1)), float(m.group(2)), float(m.group(3))
            xmin, xmax = min(xmin, x), max(xmax, x)
            ymin, ymax = min(ymin, y), max(ymax, y)
            zmin, zmax = min(zmin, z), max(zmax, z)
        if not np.isfinite([xmin, xmax, ymin, ymax, zmin, zmax]).all():
            return ""
        return f"({xmin:g}, {ymin:g}, {zmin:g})--({xmax:g}, {ymax:g}, {zmax:g})"
    except Exception:
        return ""


def _detect_cylinder_patch_name(case_dir: Path) -> str | None:
    boundary = case_dir / "constant" / "polyMesh" / "boundary"
    if not boundary.exists():
        return None
    lines = boundary.read_text(encoding="utf-8", errors="ignore").splitlines()
    names: list[str] = []
    i = 0
    while i < len(lines) - 1:
        line = re.sub(r"//.*$", "", lines[i]).strip()
        if not line or line.startswith(("/*", "*", ")")):
            i += 1
            continue
        nxt = re.sub(r"//.*$", "", lines[i + 1]).strip()
        if re.fullmatch(r"[A-Za-z0-9_]+", line) and nxt.startswith("{"):
            names.append(line)
        i += 1
    for cand in names:
        if "cylinder" in cand.lower():
            return cand
    for cand in names:
        if "obstacle" in cand.lower():
            return cand
    return None


def _is_float_dirname(name: str) -> bool:
    try:
        float(name)
        return True
    except Exception:
        return False


def _count_time_dirs(root: Path, *, min_time: float) -> int | None:
    if not root.exists():
        return None
    n = 0
    for p in root.iterdir():
        if not p.is_dir() or not _is_float_dirname(p.name):
            continue
        try:
            if float(p.name) >= float(min_time):
                n += 1
        except Exception:
            continue
    return n


def _snapshot_selection_info(cfg: dict[str, Any], results_dir: Path) -> tuple[int | None, int, int]:
    meta_path = results_dir / "snapshots" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    n_used = int(len(meta.get("times", [])))

    export = cfg.get("export", {}) if isinstance(cfg, dict) else {}
    stride = int(export.get("time_stride", 1))
    min_time = float(export.get("min_time", 0.0))

    source = str(meta.get("source", "")).strip().lower()
    n_writes: int | None = None
    if source == "openfoam_ascii":
        case_dir = Path(str(meta.get("case_dir", ""))).expanduser()
        if case_dir:
            n_writes = _count_time_dirs(case_dir, min_time=min_time)
    elif source == "vtk":
        vtk_root = Path(str(meta.get("vtk_root", ""))).expanduser()
        if vtk_root:
            n_writes = _count_time_dirs(vtk_root, min_time=min_time)
    return n_writes, n_used, stride


def _case_summary(name: str, cfg: dict[str, Any], case_dir: Path, results_dir: Path) -> CaseSummary:
    n_cells = n_cells_from_polymesh(case_dir)
    n_writes, n_used, stride = _snapshot_selection_info(cfg, results_dir)

    of = cfg.get("openfoam", {}) if isinstance(cfg, dict) else {}
    try:
        delta_t = float(of.get("deltaT"))
        end_time = float(of.get("endTime"))
        write_interval = float(of.get("writeInterval"))
    except Exception:
        cd = _read_control_dict(case_dir)
        delta_t = float(cd["deltaT"])
        end_time = float(cd["endTime"])
        write_interval = float(cd["writeInterval"])

    return CaseSummary(
        name=name,
        n_cells=int(n_cells),
        deltaT=float(delta_t),
        endTime=float(end_time),
        writeInterval=float(write_interval),
        nWrites=n_writes,
        nSnapshotsUsed=int(n_used),
        snapshotStride=int(stride),
    )


def _write_table_tex(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def esc(s: str) -> str:
        # Minimal escaping for LaTeX tables.
        return (
            s.replace("\\", r"\textbackslash{}")
            .replace("_", r"\_")
            .replace("%", r"\%")
            .replace("&", r"\&")
        )
    colspec = "l" * len(headers)
    lines = []
    lines.append(r"\begin{tabular}{" + colspec + "}")
    lines.append(r"\hline")
    lines.append(" & ".join(esc(h) for h in headers) + r" \\")
    lines.append(r"\hline")
    for r in rows:
        lines.append(" & ".join(esc(c) for c in r) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_table_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    txt = ",".join(headers) + "\n"
    for r in rows:
        txt += ",".join(r) + "\n"
    path.write_text(txt, encoding="utf-8")


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _find_latest_postproc_dat(case_dir: Path, obj_name: str, filename: str) -> Path | None:
    root = case_dir / "postProcessing" / obj_name
    if not root.exists():
        return None

    def is_float_dirname(name: str) -> bool:
        try:
            float(name)
            return True
        except Exception:
            return False

    candidates: list[tuple[float, Path]] = []
    for p in root.iterdir():
        if p.is_dir() and is_float_dirname(p.name):
            fp = p / filename
            if fp.exists():
                candidates.append((float(p.name), fp))
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[-1][1]

    for fp in root.rglob(filename):
        if fp.is_file():
            return fp
    return None


def _read_force_coeffs_series(case_dir: Path, obj_name: str) -> ForceCoeffsSeries | None:
    root = case_dir / "postProcessing" / obj_name
    if not root.exists():
        return None

    # Read all segments and merge by time (adaptive runs may restart and create multiple segment dirs).
    seg_files: list[Path] = []
    for fn in ["forceCoeffs.dat", "coefficient.dat"]:
        for fp in root.rglob(fn):
            if fp.is_file():
                seg_files.append(fp)
    if not seg_files:
        return None

    t_all: list[np.ndarray] = []
    cd_all: list[np.ndarray] = []
    cl_all: list[np.ndarray] = []
    mag_u: float | None = None
    l_ref: float | None = None
    a_ref: float | None = None

    for fp in sorted(seg_files, key=lambda p: p.as_posix()):
        try:
            s = read_force_coeffs_dat(fp)
        except Exception:
            continue
        t_all.append(np.asarray(s.time, dtype=float).reshape(-1))
        cd_all.append(np.asarray(s.Cd, dtype=float).reshape(-1))
        cl_all.append(np.asarray(s.Cl, dtype=float).reshape(-1))
        if mag_u is None and s.magUInf is not None:
            mag_u = float(s.magUInf)
        if l_ref is None and s.lRef is not None:
            l_ref = float(s.lRef)
        if a_ref is None and s.Aref is not None:
            a_ref = float(s.Aref)

    if not t_all:
        return None

    t = np.concatenate(t_all)
    cd = np.concatenate(cd_all)
    cl = np.concatenate(cl_all)
    order = np.argsort(t)
    t = t[order]
    cd = cd[order]
    cl = cl[order]

    # Deduplicate times (keep last occurrence) via reverse pass.
    seen: set[float] = set()
    keep_rev: list[int] = []
    for i in range(t.size - 1, -1, -1):
        ti = float(t[i])
        if ti in seen:
            continue
        seen.add(ti)
        keep_rev.append(i)
    keep = np.array(sorted(keep_rev), dtype=int)
    t = t[keep]
    cd = cd[keep]
    cl = cl[keep]

    # Drop non-finite samples and gross outliers (typically restart transients).
    # These can occur at the first step after restarting from a ROM-written state.
    cd_abs_max = 200.0
    cl_abs_max = 200.0
    m = np.isfinite(t) & np.isfinite(cd) & np.isfinite(cl)
    m &= (np.abs(cd) <= cd_abs_max) & (np.abs(cl) <= cl_abs_max)
    t = t[m]
    cd = cd[m]
    cl = cl[m]

    return ForceCoeffsSeries(time=t, Cd=cd, Cl=cl, Cm=None, magUInf=mag_u, lRef=l_ref, Aref=a_ref)


def _svd_energy_at_rank(X: np.ndarray, r: int) -> float:
    # X: (n_snap, n_dof)
    if X.ndim != 2:
        raise ValueError("Expected 2D matrix for SVD energy")
    Xc = X - X.mean(axis=0, keepdims=True)
    # Economy SVD: Xc = U S V^T with S sorted decreasing
    _U, s, _Vt = np.linalg.svd(Xc, full_matrices=False)
    e = s**2
    if e.sum() <= 0:
        return float("nan")
    r_eff = max(1, min(int(r), e.size))
    return float(e[:r_eff].sum() / e.sum())


def _svd_svals_cum_energy(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # X: (n_snap, n_dof)
    if X.ndim != 2:
        raise ValueError("Expected 2D matrix for SVD")
    Xc = X - X.mean(axis=0, keepdims=True)
    _U, s, _Vt = np.linalg.svd(Xc, full_matrices=False)
    e = s**2
    if float(e.sum()) <= 0.0:
        cum = np.full((s.size,), float("nan"), dtype=float)
    else:
        cum = np.cumsum(e) / float(e.sum())
    return s.astype(float), cum.astype(float)


def _read_validation_stats(adaptive_dir: Path) -> dict[str, float]:
    path = adaptive_dir / "validation.csv"
    if not path.exists():
        return {"mean_err_U": float("nan"), "max_err_U": float("nan"), "mean_err_p": float("nan"), "max_err_p": float("nan")}
    import csv

    e_u = []
    e_p = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                eu = float(row["err_U"])
                ep = float(row["err_p"])
                if np.isfinite(eu) and np.isfinite(ep):
                    e_u.append(eu)
                    e_p.append(ep)
            except Exception:
                continue
    if not e_u:
        return {"mean_err_U": float("nan"), "max_err_U": float("nan"), "mean_err_p": float("nan"), "max_err_p": float("nan")}
    return {
        "mean_err_U": float(np.mean(e_u)),
        "max_err_U": float(np.max(e_u)),
        "mean_err_p": float(np.mean(e_p)),
        "max_err_p": float(np.max(e_p)),
    }


def _rom_time_fraction(adaptive_dir: Path, t0: float, t1: float) -> float:
    path = adaptive_dir / "timeline.csv"
    if not path.exists():
        return float("nan")
    import csv

    times = []
    models = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                times.append(float(row["time"]))
                models.append(str(row["model"]).strip())
            except Exception:
                continue
    if not times:
        return float("nan")
    prev = float(t0)
    rom_dur = 0.0
    total = max(1e-12, float(t1 - t0))
    for t, m in zip(times, models, strict=False):
        dur = max(0.0, min(t, t1) - prev)
        if m.upper() == "ROM":
            rom_dur += dur
        prev = float(t)
    # If last row doesn't reach t1, assume it continues with last model.
    if prev < t1 and models:
        if models[-1].upper() == "ROM":
            rom_dur += (t1 - prev)
    return float(rom_dur / total)


def _timeline_model_at_times(adaptive_dir: Path, times: np.ndarray) -> np.ndarray | None:
    """
    Map sample times -> active model label ("FOM"/"ROM") using adaptive/timeline.csv.

    Returns an array of strings aligned with `times`, or None if timeline.csv is missing.
    """
    path = adaptive_dir / "timeline.csv"
    if not path.exists():
        return None
    import csv

    t_marks: list[float] = []
    m_marks: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                t_marks.append(float(row["time"]))
                m_marks.append(str(row["model"]).strip().upper())
            except Exception:
                continue
    if not t_marks:
        return None
    t_marks_arr = np.asarray(t_marks, dtype=float)
    m_marks_arr = np.asarray(m_marks, dtype=object)

    times = np.asarray(times, dtype=float).reshape(-1)
    idx = np.searchsorted(t_marks_arr, times, side="right") - 1
    idx = np.clip(idx, 0, t_marks_arr.size - 1)
    out = m_marks_arr[idx]
    return out.astype(str)


def _mask_series_by_model(series: ForceCoeffsSeries, adaptive_dir: Path, model: str) -> ForceCoeffsSeries | None:
    """
    Keep only samples that fall in the requested model interval ("FOM" or "ROM"),
    based on adaptive/timeline.csv.
    """
    model = str(model).strip().upper()
    labels = _timeline_model_at_times(adaptive_dir, series.time)
    if labels is None:
        return None
    keep = (labels == model)
    if int(np.sum(keep)) < 2:
        return None
    return ForceCoeffsSeries(
        time=series.time[keep],
        Cd=series.Cd[keep],
        Cl=series.Cl[keep],
        Cm=None,
        magUInf=series.magUInf,
        lRef=series.lRef,
        Aref=series.Aref,
    )

def _write_summary_tex(path: Path, name: str, cfg: dict[str, Any], case_dir: Path, results_dir: Path) -> None:
    snap_meta = json.loads((results_dir / "snapshots" / "meta.json").read_text(encoding="utf-8"))
    n_writes, n_snap_used, stride = _snapshot_selection_info(cfg, results_dir)
    export = cfg.get("export", {}) if isinstance(cfg, dict) else {}
    max_snaps = int(export.get("max_snapshots", 200))
    min_time = float(export.get("min_time", 0.0))
    snap_source = str(snap_meta.get("source", "Not recorded")).strip().replace("_", r"\_")

    r_u_req = int(cfg["rom"]["r_u"])
    r_p_req = int(cfg["rom"]["r_p"])
    energy_thresh = float(cfg.get("rom", {}).get("energy_threshold", 0.999))

    # Prefer the ROM build's recorded effective ranks (after any POD clamping),
    # otherwise fall back to config defaults.
    r_u = None
    r_p = None
    rom_meta_path = results_dir / "rom" / "rom_meta.json"
    if rom_meta_path.exists():
        try:
            rom_meta = json.loads(rom_meta_path.read_text(encoding="utf-8"))
            r_u = int(rom_meta.get("r_u"))
            r_p = int(rom_meta.get("r_p"))
        except Exception:
            r_u = None
            r_p = None
    if r_u is None:
        r_u = int(r_u_req)
    if r_p is None:
        r_p = int(r_p_req)

    # Energy at chosen ranks from snapshot matrices.
    U = np.load(results_dir / "snapshots" / "U.npy")
    p = np.load(results_dir / "snapshots" / "p.npy")
    n_cells_u = int(U.shape[1])
    n_cells_p = int(p.shape[1]) if p.ndim == 2 else -1
    if n_cells_p != n_cells_u:
        _warn(f"{name}: snapshot nCells mismatch between U ({n_cells_u}) and p ({n_cells_p}); using U nCells")
    U_flat = U.reshape(U.shape[0], n_cells_u * 3)
    p_flat = p.reshape(p.shape[0], n_cells_u)
    e_u = _svd_energy_at_rank(U_flat, r=r_u)
    e_p = _svd_energy_at_rank(p_flat, r=r_p)

    adaptive_dir = results_dir / "adaptive"
    stats = _read_validation_stats(adaptive_dir)
    t0 = float(cfg["adaptive"]["start_time"])
    t1 = float(cfg["adaptive"]["end_time"])
    frac_rom = _rom_time_fraction(adaptive_dir, t0=t0, t1=t1)

    speed_path = adaptive_dir / "speed.json"
    wall = json.loads(speed_path.read_text(encoding="utf-8")) if speed_path.exists() else {}
    fom_s = float(wall.get("fom_s", float("nan")))
    rom_s = float(wall.get("rom_s", float("nan")))
    io_s = float(wall.get("io_s", float("nan")))
    total_s = float(np.nansum([fom_s, rom_s, io_s]))

    # Measured baseline (FOM-only), when available.
    fom_only_path = results_dir / "fom_only" / "speed.json"
    fom_only = json.loads(fom_only_path.read_text(encoding="utf-8")) if fom_only_path.exists() else {}
    fom_only_wall_s = float(fom_only.get("wall_s", float("nan")))
    fom_only_clock_s = float(fom_only.get("clock_s", float("nan")))
    # Prefer OpenFOAM-reported ClockTime (parsed from logs) when present.
    # Speed baseline: use measured wall time (Python stopwatch) as the primary quantity.
    fom_only_t = fom_only_wall_s
    speed_ratio = (fom_only_t / total_s) if np.isfinite(fom_only_t) and total_s > 0 else float("nan")
    slowdown_factor = (total_s / fom_only_t) if np.isfinite(fom_only_t) and fom_only_t > 0 and np.isfinite(total_s) else float("nan")
    speed_label = "Not recorded"
    if np.isfinite(speed_ratio):
        speed_label = "speedup" if speed_ratio > 1.0 else "slowdown"

    def fmt(x: float) -> str:
        return "Not recorded" if not np.isfinite(x) else f"{x:.3g}"

    def fmt_energy(x: float) -> str:
        return "Not recorded" if not np.isfinite(x) else f"{x:.8f}"

    tex = []
    tex.append(r"\paragraph{Summary (%s).}" % name)
    tex.append(r"\begin{itemize}")
    rule = r"source=%s, stride=%d, max=%d, minTime=%s" % (snap_source, int(stride), int(max_snaps), f"{min_time:g}")
    if n_writes is None:
        tex.append(r"\item Snapshots: nSnapshotsUsed=%d (%s)." % (int(n_snap_used), rule))
    else:
        tex.append(r"\item Snapshots: nWrites=%d available, nSnapshotsUsed=%d (%s)." % (int(n_writes), int(n_snap_used), rule))
    tex.append(
        r"\item POD ranks: requested $(r_U,r_p)=(%d,%d)$, effective $(r_U,r_p)=(%d,%d)$; captured energy: $E_U(r_U)=%s$, $E_p(r_p)=%s$."
        % (r_u_req, r_p_req, r_u, r_p, fmt_energy(e_u), fmt_energy(e_p))
    )
    tex.append(
        r"\item Rank selection rule: $r_{\mathrm{eff}}=\min(r_{\mathrm{req}}, \min(n_{\mathrm{snap}}, n_{\mathrm{dof}}))$; energy threshold=%s (used only when $r$ is not specified)."
        % (f"{energy_thresh:.6g}")
    )
    tex.append(
        r"\item Validation errors: mean/max $(U)=%s/%s$, mean/max $(p)=%s/%s$."
        % (fmt(stats["mean_err_U"]), fmt(stats["max_err_U"]), fmt(stats["mean_err_p"]), fmt(stats["max_err_p"]))
    )
    tex.append(r"\item ROM usage: %s of simulated time in ROM." % (fmt(frac_rom)))
    tex.append(
        r"\item Wall time (adaptive): FOM=%s s, ROM=%s s, I/O=%s s, total=%s s."
        % (fmt(fom_s), fmt(rom_s), fmt(io_s), fmt(total_s))
    )
    if np.isfinite(speed_ratio) and speed_ratio > 1.0:
        tex.append(
            r"\item Baseline (FOM-only): wall=%s s (ClockTime=%s s); speedup=%s (baseline vs adaptive)."
            % (fmt(fom_only_wall_s), fmt(fom_only_clock_s), fmt(speed_ratio))
        )
    elif np.isfinite(speed_ratio) and speed_ratio > 0.0:
        tex.append(
            r"\item Baseline (FOM-only): wall=%s s (ClockTime=%s s); slowdown=%s (adaptive vs baseline)."
            % (fmt(fom_only_wall_s), fmt(fom_only_clock_s), fmt(slowdown_factor))
        )
    else:
        tex.append(
            r"\item Baseline (FOM-only): wall=%s s (ClockTime=%s s); speed comparison: Not recorded."
            % (fmt(fom_only_wall_s), fmt(fom_only_clock_s))
        )
    tex.append(r"\end{itemize}")
    path.write_text("\n".join(tex) + "\n", encoding="utf-8")


def _write_runtime_summary(results_dir: Path, name: str) -> None:
    """
    Writes results/*/metrics/runtime_summary.csv for a single case.
    """
    metrics_dir = results_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    adaptive_speed = results_dir / "adaptive" / "speed.json"
    wall = json.loads(adaptive_speed.read_text(encoding="utf-8")) if adaptive_speed.exists() else {}
    a_fom = float(wall.get("fom_s", float("nan")))
    a_rom = float(wall.get("rom_s", float("nan")))
    a_io = float(wall.get("io_s", float("nan")))
    a_total = float(np.nansum([a_fom, a_rom, a_io]))

    fom_only_speed = results_dir / "fom_only" / "speed.json"
    fom_only = json.loads(fom_only_speed.read_text(encoding="utf-8")) if fom_only_speed.exists() else {}
    f_wall = float(fom_only.get("wall_s", float("nan")))
    f_clock = float(fom_only.get("clock_s", float("nan")))
    f_time = f_wall
    ratio = (f_time / a_total) if np.isfinite(f_time) and a_total > 0 else float("nan")
    label = ""
    if np.isfinite(ratio):
        label = "speedup" if ratio > 1.0 else "slowdown"

    out = metrics_dir / "runtime_summary.csv"
    out.write_text(
        "case,adaptive_fom_s,adaptive_rom_s,adaptive_io_s,adaptive_total_s,fom_only_wall_s,fom_only_clock_s,fom_only_time_s,ratio_fomonly_over_adaptive,label\n"
        f"{name},{a_fom:.16g},{a_rom:.16g},{a_io:.16g},{a_total:.16g},{f_wall:.16g},{f_clock:.16g},{f_time:.16g},{ratio:.16g},{label}\n",
        encoding="utf-8",
    )


def _dominant_freq_welch(
    time: np.ndarray,
    signal: np.ndarray,
    *,
    discard_frac: float = 0.30,
    f_max_hz: float = 10.0,
    min_duration_s: float = 1.0,
) -> DominantFreqEstimate:
    """
    Returns (f_peak, freq, psd) from a Welch PSD estimate.
    If the series is too short, returns (None, empty, empty).

    Per Milestone 3:
      - concatenate/sort/dedup is handled upstream
      - resample to uniform dt with interpolation
      - discard first 30% of samples (transient removal)
      - scipy.signal.welch with a Hann window
    """
    t = np.asarray(time, dtype=float).reshape(-1)
    x = np.asarray(signal, dtype=float).reshape(-1)
    if t.size < 8 or t.size != x.size:
        return DominantFreqEstimate(None, np.array([]), np.array([]), 0.0, 0.0)

    # Sort and drop non-finite/duplicate timestamps.
    m = np.isfinite(t) & np.isfinite(x)
    t = t[m]
    x = x[m]
    if t.size < 8:
        return DominantFreqEstimate(None, np.array([]), np.array([]), 0.0, 0.0)
    order = np.argsort(t)
    t = t[order]
    x = x[order]
    t_unique, unique_idx = np.unique(t, return_index=True)
    t = t_unique
    x = x[unique_idx]
    if t.size < 8:
        return DominantFreqEstimate(None, np.array([]), np.array([]), 0.0, 0.0)

    # Welch expects uniform sampling; resample with dt = median(diff(t)).
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size < 8:
        return DominantFreqEstimate(None, np.array([]), np.array([]), 0.0, 0.0)
    dt_med = float(np.median(dt))
    if not np.isfinite(dt_med) or dt_med <= 0:
        return DominantFreqEstimate(None, np.array([]), np.array([]), 0.0, 0.0)
    dt_max = float(np.max(dt)) if dt.size else float("nan")
    # If the series has large gaps (e.g., masked adaptive intervals), PSD estimates become unreliable.
    if np.isfinite(dt_max) and dt_max > 10.0 * dt_med:
        return DominantFreqEstimate(None, np.array([]), np.array([]), 0.0, 0.0)
    fs = 1.0 / dt_med

    t_uniform = np.arange(float(t[0]), float(t[-1]) + 0.5 * dt_med, dt_med, dtype=float)
    if t_uniform.size < 16:
        return DominantFreqEstimate(None, np.array([]), np.array([]), 0.0, 0.0)
    x_uniform = np.interp(t_uniform, t, x).astype(float, copy=False)

    # Ignore initial transient (default: discard first 30% of samples).
    discard_frac = float(discard_frac)
    discard_frac = min(max(discard_frac, 0.0), 0.95)
    k0 = int(np.floor(discard_frac * t_uniform.size))
    x_uniform = x_uniform[k0:]
    if x_uniform.size < 16:
        return DominantFreqEstimate(None, np.array([]), np.array([]), 0.0, 0.0)

    min_duration_s = float(min_duration_s)
    if not np.isfinite(min_duration_s):
        min_duration_s = 0.0
    if min_duration_s > 0.0:
        duration = float(dt_med) * float(max(0, x_uniform.size - 1))
        if duration < min_duration_s:
            return DominantFreqEstimate(None, np.array([]), np.array([]), duration, 0.0)

    nperseg = min(1024, x_uniform.size)
    if nperseg < 16:
        duration = float(dt_med) * float(max(0, x_uniform.size - 1))
        return DominantFreqEstimate(None, np.array([]), np.array([]), duration, 0.0)
    freq, psd = sp_signal.welch(
        x_uniform,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
    )
    if freq.size < 3:
        duration = float(dt_med) * float(max(0, x_uniform.size - 1))
        return DominantFreqEstimate(None, freq, psd, duration, 0.0)

    f_max_hz = max(0.0, float(f_max_hz))
    mask = (freq > 0.0) & (freq <= f_max_hz)
    if int(np.sum(mask)) < 3:
        duration = float(dt_med) * float(max(0, x_uniform.size - 1))
        return DominantFreqEstimate(None, freq, psd, duration, 0.0)

    idx = np.where(mask)[0]
    i = int(idx[np.argmax(psd[idx])])
    f_peak = float(freq[i])
    duration = float(dt_med) * float(max(0, x_uniform.size - 1))
    n_cycles = float(f_peak * duration) if (np.isfinite(f_peak) and duration > 0) else 0.0
    return DominantFreqEstimate(f_peak, freq, psd, duration, n_cycles)


def _write_force_stats_table(out_tab: Path, low_case: Path, high_case: Path, low_res: Path, high_res: Path) -> None:
    headers = [
        "case",
        "source",
        "Cd_mean",
        "Cd_rms",
        "Cl_mean",
        "Cl_rms",
        "f_peak_Hz",
        "St",
        "D",
        "span",
        "Aref",
        "beta",
        "U_inf",
        "Re",
    ]

    @dataclass(frozen=True)
    class _Row:
        case: str
        source: str
        Cd_mean: float | None
        Cd_rms: float | None
        Cl_mean: float | None
        Cl_rms: float | None
        f_peak_Hz: float | None
        St: float | None
        D: float | None
        span: float | None
        Aref: float | None
        beta: float | None
        U_inf: float | None
        Re: float | None
        st_is_indicative: bool

    rows: list[_Row] = []
    any_indicative = False
    indicative_note = (
        r"\footnotesize St is marked as ``indicative'' when fewer than 8 dominant cycles are present "
        r"after discarding the first 30\% transient."
    )

    def _detect_cylinder_patch_name(case_dir: Path) -> str:
        boundary = case_dir / "constant" / "polyMesh" / "boundary"
        if not boundary.exists():
            return "cylinder"
        txt = boundary.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"^\\s*cylinder\\s*\\{", txt, flags=re.M):
            return "cylinder"
        for m in re.finditer(r"^\\s*([A-Za-z0-9_]+)\\s*\\{", txt, flags=re.M):
            name = m.group(1)
            if "cylinder" in name.lower():
                return name
        return "cylinder"

    def _write_geometry_json(case_dir: Path, res_dir: Path) -> tuple[float | None, float | None, float | None]:
        metrics_dir = res_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        out_path = metrics_dir / "geometry.json"
        patch = _detect_cylinder_patch_name(case_dir)
        try:
            geom = cylinder_geometry(case_dir, patch_name=patch)
        except Exception:
            return None, None, None
        out_path.write_text(json.dumps(geom, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        d = float(geom.get("D", float("nan")))
        span_ref = float(geom.get("span", geom.get("span_ref", float("nan"))))
        a_ref = float(geom.get("Aref", float("nan")))
        if not np.isfinite(d):
            d = None
        if not np.isfinite(span_ref):
            span_ref = None
        if not np.isfinite(a_ref):
            a_ref = None
        return d, span_ref, a_ref

    for name, case_dir, res_dir in [("low", low_case, low_res), ("high", high_case, high_res)]:
        base_case_dir = res_dir / "fom_only" / "fom_only_case"
        adapt_case_dir = res_dir / "adaptive" / "adaptive_case"
        adaptive_dir = res_dir / "adaptive"

        # For geometry and "recorded" physical parameters, prefer the actual run case folder.
        geom_case_dir = base_case_dir if (base_case_dir / "system" / "controlDict").exists() else adapt_case_dir if (adapt_case_dir / "system" / "controlDict").exists() else case_dir
        d_geom, span_geom, a_ref_geom = _write_geometry_json(geom_case_dir, res_dir)

        # Confinement/span JSON audit files (explicitly requested by the report).
        beta_case: float | None = None
        try:
            write_metrics(geom_case_dir, res_dir / "metrics")
            conf = compute_confinement(geom_case_dir)
            beta_case = float(conf.get("beta", float("nan")))
            if not np.isfinite(beta_case):
                beta_case = None
        except Exception:
            beta_case = None

        series_list: list[tuple[str, ForceCoeffsSeries]] = []
        base = _read_force_coeffs_series(base_case_dir, "forceCoeffsIncompressible")
        if base is None:
            base = _read_force_coeffs_series(base_case_dir, "forceCoeffs1")
        if base is not None:
            series_list.append(("FOM-only", base))

        # Prefer a single postprocessed coefficient timeline (computed from the fields),
        # then split into FOM/ROM intervals using adaptive/timeline.csv.
        full = _read_force_coeffs_series(adapt_case_dir, "forceCoeffsIncompressible")
        if full is None:
            full = _read_force_coeffs_series(adapt_case_dir, "forceCoeffsROM")
        if full is not None:
            fom_seg = _mask_series_by_model(full, adaptive_dir, "FOM")
            rom_seg = _mask_series_by_model(full, adaptive_dir, "ROM")
            if fom_seg is not None:
                series_list.append(("adaptive (FOM segments)", fom_seg))
            if rom_seg is not None:
                series_list.append(("ROM intervals (postProcess)", rom_seg))
        else:
            # Fallback: use solver-written forceCoeffs when postprocess is unavailable.
            adapt = _read_force_coeffs_series(adapt_case_dir, "forceCoeffs1")
            if adapt is not None:
                series_list.append(("adaptive (FOM segments)", adapt))

        if not series_list:
            rows.append(
                _Row(
                    case=name,
                    source="Not recorded",
                    Cd_mean=None,
                    Cd_rms=None,
                    Cl_mean=None,
                    Cl_rms=None,
                    f_peak_Hz=None,
                    St=None,
                    D=None,
                    span=None,
                    Aref=None,
                    beta=beta_case,
                    U_inf=None,
                    Re=None,
                    st_is_indicative=False,
                )
            )
            continue

        def _as_finite_or_none(x: float | None) -> float | None:
            if x is None:
                return None
            x = float(x)
            return x if np.isfinite(x) else None

        def fmt(x: float | None) -> str:
            if x is None or not np.isfinite(x):
                return "Not recorded"
            return f"{x:.6g}"

        # Use consistent geometry/Uinf/Strouhal across rows within a case to avoid "Not recorded"
        # for discontinuous adaptive segments.
        inlet_u_case = _as_finite_or_none(read_inlet_u_mag(geom_case_dir))
        d_case = _as_finite_or_none(d_geom)
        span_case = _as_finite_or_none(span_geom)
        a_ref_case = _as_finite_or_none(a_ref_geom)
        if d_case is None:
            for _src, s in series_list:
                d_case = _as_finite_or_none(s.lRef)
                if d_case is not None:
                    break

        if inlet_u_case is None:
            for _src, s in series_list:
                inlet_u_case = _as_finite_or_none(s.magUInf)
                if inlet_u_case is not None:
                    break

        if a_ref_case is None:
            for _src, s in series_list:
                a_ref_case = _as_finite_or_none(s.Aref)
                if a_ref_case is not None:
                    break

        # Audit coefficient normalization metadata against mesh-derived values.
        if inlet_u_case is not None or a_ref_case is not None:
            for src, s in series_list:
                if inlet_u_case is not None and s.magUInf is not None and np.isfinite(s.magUInf):
                    rel = abs(float(s.magUInf) - float(inlet_u_case)) / max(abs(float(inlet_u_case)), 1e-12)
                    if rel > 1e-6:
                        _warn(f"{name}: {src} magUInf={float(s.magUInf):.6g} differs from inlet |U|={float(inlet_u_case):.6g}")
                if a_ref_case is not None and s.Aref is not None and np.isfinite(s.Aref):
                    rel = abs(float(s.Aref) - float(a_ref_case)) / max(abs(float(a_ref_case)), 1e-12)
                    if rel > 1e-6:
                        _warn(f"{name}: {src} Aref={float(s.Aref):.6g} differs from mesh-derived Aref={float(a_ref_case):.6g}")

        # Explicit coefficient normalization audit/recompute:
        # if the case is not strictly 2D (span != 1), recompute solver-produced coefficients from forces.
        must_recompute = (span_case is not None) and np.isfinite(span_case) and (abs(float(span_case) - 1.0) > 1e-12)
        if must_recompute:
            if inlet_u_case is None or d_case is None or a_ref_case is None:
                _warn(f"{name}: span != 1 but missing U_inf/D/Aref; cannot recompute coefficients from forces")
            else:
                # Baseline (FOM-only) replacement.
                forces_base = base_case_dir / "postProcessing" / "forces1" / "0" / "forces.dat"
                if forces_base.exists():
                    try:
                        base_from_forces = _coeffs_from_forces(forces_base, u_inf=float(inlet_u_case), a_ref=float(a_ref_case), d_ref=float(d_case))
                        series_list = [("FOM-only", base_from_forces)] + [(src, s) for (src, s) in series_list if src != "FOM-only"]
                    except Exception:
                        _warn(f"{name}: failed to recompute baseline Cd/Cl from {forces_base}")
                # Adaptive solver segments (FOM bursts) replacement (forces exist only when solver runs).
                forces_adapt = adapt_case_dir / "postProcessing" / "forces1" / "0" / "forces.dat"
                if forces_adapt.exists():
                    try:
                        adapt_from_forces = _coeffs_from_forces(forces_adapt, u_inf=float(inlet_u_case), a_ref=float(a_ref_case), d_ref=float(d_case))
                        series_list = [(src, s) for (src, s) in series_list if src != "adaptive (FOM segments)"]
                        series_list.append(("adaptive (FOM segments)", adapt_from_forces))
                    except Exception:
                        _warn(f"{name}: failed to recompute adaptive Cd/Cl from {forces_adapt}")

        # nu: kinematic viscosity from case dictionaries (fallback: transportProperties).
        nu_case: float | None = None
        for cand in ["physicalProperties", "transportProperties"]:
            nu_s = _read_scalar_dict_value(geom_case_dir / "constant" / cand, "nu")
            if nu_s:
                try:
                    nu_case = float(nu_s)
                except Exception:
                    nu_case = None
                break

        re_case: float | None = None
        if inlet_u_case is not None and d_case is not None and nu_case is not None and nu_case > 0:
            re_case = float(inlet_u_case * d_case / nu_case)

        f_peak_case: float | None = None
        n_cycles_case: float | None = None
        # Prefer the continuous baseline for frequency; fallback to full postprocess series.
        if base is not None:
            est = _dominant_freq_welch(base.time, base.Cl)
            f_peak_case = est.f_peak_hz
            n_cycles_case = est.n_cycles_kept
        if f_peak_case is None and full is not None:
            est = _dominant_freq_welch(full.time, full.Cl)
            f_peak_case = est.f_peak_hz
            n_cycles_case = est.n_cycles_kept
        if f_peak_case is None:
            for _src, s in series_list:
                est = _dominant_freq_welch(s.time, s.Cl)
                f_peak_case = est.f_peak_hz
                n_cycles_case = est.n_cycles_kept
                if f_peak_case is not None:
                    break

        st_case: float | None = None
        if f_peak_case is not None and d_case is not None and inlet_u_case is not None and inlet_u_case > 0:
            st_case = float(f_peak_case * d_case / inlet_u_case)

        st_is_indicative = (n_cycles_case is not None) and np.isfinite(n_cycles_case) and (float(n_cycles_case) < 8.0)
        if st_case is not None and st_is_indicative:
            any_indicative = True

        for source, series in series_list:
            cd_mean = mean_after_discard(series.Cd, discard_frac=0.30)
            cd_rms = rms_fluct_after_discard(series.Cd, discard_frac=0.30)
            cl_mean = mean_after_discard(series.Cl, discard_frac=0.30)
            cl_rms = rms_fluct_after_discard(series.Cl, discard_frac=0.30)

            rows.append(
                _Row(
                    case=name,
                    source=source,
                    Cd_mean=float(cd_mean) if np.isfinite(cd_mean) else None,
                    Cd_rms=float(cd_rms) if np.isfinite(cd_rms) else None,
                    Cl_mean=float(cl_mean) if np.isfinite(cl_mean) else None,
                    Cl_rms=float(cl_rms) if np.isfinite(cl_rms) else None,
                    f_peak_Hz=_as_finite_or_none(f_peak_case),
                    St=_as_finite_or_none(st_case),
                    D=_as_finite_or_none(d_case),
                    span=_as_finite_or_none(span_case),
                    Aref=_as_finite_or_none(a_ref_case),
                    beta=_as_finite_or_none(beta_case),
                    U_inf=_as_finite_or_none(inlet_u_case),
                    Re=_as_finite_or_none(re_case),
                    st_is_indicative=bool(st_is_indicative),
                )
            )

    def _fmt_csv(x: float | None) -> str:
        if x is None or not np.isfinite(x):
            return "N/A"
        return f"{float(x):.16g}"

    def _fmt_tex(x: float | None) -> str:
        if x is None or not np.isfinite(x):
            return "Not recorded"
        return f"{float(x):.6g}"

    out_csv = out_tab / "force_stats.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text(
        ",".join(headers)
        + "\n"
        + "".join(
            ",".join(
                [
                    r.case,
                    r.source,
                    _fmt_csv(r.Cd_mean),
                    _fmt_csv(r.Cd_rms),
                    _fmt_csv(r.Cl_mean),
                    _fmt_csv(r.Cl_rms),
                    _fmt_csv(r.f_peak_Hz),
                    _fmt_csv(r.St),
                    _fmt_csv(r.D),
                    _fmt_csv(r.span),
                    _fmt_csv(r.Aref),
                    _fmt_csv(r.beta),
                    _fmt_csv(r.U_inf),
                    _fmt_csv(r.Re),
                ]
            )
            + "\n"
            for r in rows
        ),
        encoding="utf-8",
    )

    tex_path = out_tab / "force_stats.tex"
    tex_path.parent.mkdir(parents=True, exist_ok=True)

    def esc(s: str) -> str:
        return (
            s.replace("\\", r"\textbackslash{}")
            .replace("_", r"\_")
            .replace("%", r"\%")
            .replace("&", r"\&")
        )

    colspec = "llrrrrrlrrrrrr"
    lines: list[str] = []
    lines.append(r"\begin{tabular}{" + colspec + "}")
    lines.append(r"\hline")
    lines.append(" & ".join(esc(h) for h in headers) + r" \\")
    lines.append(r"\hline")
    for r in rows:
        st_tex = _fmt_tex(r.St)
        if r.St is not None and r.st_is_indicative:
            st_tex = f"{float(r.St):.6g} (indicative)"
        lines.append(
            " & ".join(
                [
                    esc(r.case),
                    esc(r.source),
                    esc(_fmt_tex(r.Cd_mean)),
                    esc(_fmt_tex(r.Cd_rms)),
                    esc(_fmt_tex(r.Cl_mean)),
                    esc(_fmt_tex(r.Cl_rms)),
                    esc(_fmt_tex(r.f_peak_Hz)),
                    esc(st_tex),
                    esc(_fmt_tex(r.D)),
                    esc(_fmt_tex(r.span)),
                    esc(_fmt_tex(r.Aref)),
                    esc(_fmt_tex(r.beta)),
                    esc(_fmt_tex(r.U_inf)),
                    esc(_fmt_tex(r.Re)),
                ]
            )
            + r" \\"
        )
    if any_indicative:
        lines.append(r"\hline")
        lines.append(r"\multicolumn{" + str(len(headers)) + r"}{l}{" + indicative_note + r"} \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_validation_plot(adaptive_dir: Path, out: Path, title: str) -> None:
    """
    Regenerate the validation plot from adaptive/validation.csv.

    We do this in the report pipeline (instead of only copying solver-generated PNGs) because
    many runs record "rom_precheck" rows with NaN errors, which can produce blank plots if
    not filtered.
    """
    path = adaptive_dir / "validation.csv"
    if not path.exists():
        return
    import csv

    t_list: list[float] = []
    err_u: list[float] = []
    err_p: list[float] = []
    abs_u: list[float] = []
    abs_p: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                t_list.append(float(row["time"]))
                err_u.append(float(row["err_U"]))
                err_p.append(float(row["err_p"]))
                abs_u.append(float(row.get("abs_U", "nan")))
                abs_p.append(float(row.get("abs_p", "nan")))
            except Exception:
                continue
    if not t_list:
        return
    save_error_plot_two_rel_abs(
        np.asarray(t_list, dtype=float),
        np.asarray(err_u, dtype=float),
        np.asarray(err_p, dtype=float),
        np.asarray(abs_u, dtype=float),
        np.asarray(abs_p, dtype=float),
        out,
        title,
        rel_ylabel="L2 relative error",
        abs_ylabel="L2 absolute error",
    )


def _pick_case_dir_for_recorded_params(*, base_case_dir: Path, results_dir: Path) -> Path:
    """
    Pick the OpenFOAM case directory whose dictionaries correspond to the run that produced results.

    Preference order:
      1) results/*/fom_only/fom_only_case (uses config writeInterval/endTime)
      2) results/*/adaptive/adaptive_case (may have writeInterval=1 for frequent writes)
      3) cases/* (template/work_dir)
    """
    candidates = [
        results_dir / "fom_only" / "fom_only_case",
        results_dir / "adaptive" / "adaptive_case",
        base_case_dir,
    ]
    for c in candidates:
        if (c / "system" / "controlDict").exists():
            return c
    return base_case_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--low-config", default="rom/configs/low.yaml")
    ap.add_argument("--high-config", default="rom/configs/high.yaml")
    ap.add_argument("--out-fig", default="report/figures")
    ap.add_argument("--out-tab", default="report/tables")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    low_cfg = load_yaml(root / args.low_config)
    high_cfg = load_yaml(root / args.high_config)

    out_fig = (root / args.out_fig).resolve()
    out_tab = (root / args.out_tab).resolve()
    out_fig.mkdir(parents=True, exist_ok=True)
    out_tab.mkdir(parents=True, exist_ok=True)

    low_case = (root / low_cfg["case"]["work_dir"]).resolve()
    low_res = (root / low_cfg["case"]["results_dir"]).resolve()
    high_case = (root / high_cfg["case"]["work_dir"]).resolve()
    high_res = (root / high_cfg["case"]["results_dir"]).resolve()

    # Tables: case settings and ROM settings
    low_sum = _case_summary("low", low_cfg, low_case, low_res)
    high_sum = _case_summary("high", high_cfg, high_case, high_res)

    headers = ["case", "nCells", "deltaT", "endTime", "writeInterval", "nWrites", "nSnapshotsUsed", "snapshotStride"]
    def _fmt_int(x: int | None) -> str:
        return "Not recorded" if x is None else str(int(x))
    rows = [
        [
            low_sum.name,
            str(low_sum.n_cells),
            f"{low_sum.deltaT:g}",
            f"{low_sum.endTime:g}",
            f"{low_sum.writeInterval:g}",
            _fmt_int(low_sum.nWrites),
            str(low_sum.nSnapshotsUsed),
            str(low_sum.snapshotStride),
        ],
        [
            high_sum.name,
            str(high_sum.n_cells),
            f"{high_sum.deltaT:g}",
            f"{high_sum.endTime:g}",
            f"{high_sum.writeInterval:g}",
            _fmt_int(high_sum.nWrites),
            str(high_sum.nSnapshotsUsed),
            str(high_sum.snapshotStride),
        ],
    ]
    _write_table_csv(out_tab / "case_settings.csv", headers, rows)
    _write_table_tex(out_tab / "case_settings.tex", headers, rows)

    def rom_row(cfg: dict[str, Any], name: str, n_cells: int) -> list[str]:
        return [
            name,
            str(cfg["rom"]["r_u"]),
            str(cfg["rom"]["r_p"]),
            str(cfg["rom"]["ridge"]),
            str(cfg["adaptive"]["tol"]),
            str(cfg["adaptive"]["train_window"]),
            str(cfg["adaptive"]["validate_every"]),
            str(cfg["adaptive"]["rom_horizon"]),
            str(n_cells),
        ]

    headers2 = ["case", "r_U", "r_p", "ridge", "tol", "train_window", "validate_every", "rom_horizon", "nCells"]
    rows2 = [rom_row(low_cfg, "low", low_sum.n_cells), rom_row(high_cfg, "high", high_sum.n_cells)]
    _write_table_csv(out_tab / "rom_settings.csv", headers2, rows2)
    _write_table_tex(out_tab / "rom_settings.tex", headers2, rows2)

    # POD energy at the chosen ranks (8 decimals).
    pod_headers = ["case", "r_U", "E_U(r_U)", "r_p", "E_p(r_p)"]
    pod_rows: list[list[str]] = []
    for name, cfg, case_dir, res_dir in [
        ("low", low_cfg, low_case, low_res),
        ("high", high_cfg, high_case, high_res),
    ]:
        try:
            U = np.load(res_dir / "snapshots" / "U.npy")
            p = np.load(res_dir / "snapshots" / "p.npy")
            n_cells_u = int(U.shape[1])
            n_cells_p = int(p.shape[1]) if p.ndim == 2 else -1
            if n_cells_p != n_cells_u:
                _warn(f"{name}: snapshot nCells mismatch between U ({n_cells_u}) and p ({n_cells_p}); using U nCells")
            U_flat = U.reshape(U.shape[0], n_cells_u * 3)
            p_flat = p.reshape(p.shape[0], n_cells_u)

            # Prefer recorded effective ranks (after any POD clamping), otherwise config ranks.
            r_u = int(cfg["rom"]["r_u"])
            r_p = int(cfg["rom"]["r_p"])
            rom_meta_path = res_dir / "rom" / "rom_meta.json"
            if rom_meta_path.exists():
                try:
                    meta = json.loads(rom_meta_path.read_text(encoding="utf-8"))
                    r_u = int(meta.get("r_u", r_u))
                    r_p = int(meta.get("r_p", r_p))
                except Exception:
                    pass

            e_u = _svd_energy_at_rank(U_flat, r=r_u)
            e_p = _svd_energy_at_rank(p_flat, r=r_p)
            e_u_s = "Not recorded" if not np.isfinite(e_u) else f"{e_u:.8f}"
            e_p_s = "Not recorded" if not np.isfinite(e_p) else f"{e_p:.8f}"
        except Exception:
            r_u = int(cfg["rom"]["r_u"])
            r_p = int(cfg["rom"]["r_p"])
            e_u_s = "Not recorded"
            e_p_s = "Not recorded"
        pod_rows.append([name, str(r_u), e_u_s, str(r_p), e_p_s])
    _write_table_csv(out_tab / "pod_energy.csv", pod_headers, pod_rows)
    _write_table_tex(out_tab / "pod_energy.tex", pod_headers, pod_rows)

    # Case setup extracted from OpenFOAM dictionaries.
    setup_headers = ["case", "solver", "deltaT", "endTime", "writeInterval", "nu", "inlet_U", "domain_bounds", "beta"]
    setup_rows = []
    for name, base_case_dir, res_dir in [("low", low_case, low_res), ("high", high_case, high_res)]:
        case_dir = _pick_case_dir_for_recorded_params(base_case_dir=base_case_dir, results_dir=res_dir)
        cd = _read_control_dict(case_dir)
        solver = _read_solver_name(case_dir) or "Not recorded"
        if solver == "Not recorded":
            _warn(f"{name}: could not read solver name from system/controlDict")
        deltaT = cd.get("deltaT") or "Not recorded"
        endTime = cd.get("endTime") or "Not recorded"
        writeInterval = cd.get("writeInterval") or "Not recorded"
        nu = _read_scalar_dict_value(case_dir / "constant" / "physicalProperties", "nu") or "Not recorded"
        if nu == "Not recorded":
            _warn(f"{name}: could not read nu from constant/physicalProperties")
        inlet_u = _read_inlet_u(case_dir) or "Not recorded"
        if inlet_u == "Not recorded":
            _warn(f"{name}: could not read inlet velocity from 0/U fixedValue")
        bounds = _read_domain_bounds(case_dir) or "Not recorded"
        if bounds == "Not recorded":
            _warn(f"{name}: could not compute domain bounds from constant/polyMesh/points")
        # Confinement/span metrics are derived from mesh data; write them to results/*/metrics for auditability.
        beta_s = "Not recorded"
        try:
            metrics_dir = res_dir / "metrics"
            write_metrics(case_dir, metrics_dir)
            conf = compute_confinement(case_dir)
            try:
                d_chk = float(conf.get("D", float("nan")))
                if np.isfinite(d_chk) and abs(d_chk - 2.0) > 1e-6:
                    _warn(f"{name}: cylinder diameter D={d_chk:g} (expected ~2)")
            except Exception:
                pass
            beta = float(conf.get("beta", float("nan")))
            if np.isfinite(beta):
                beta_s = f"{beta:.6g}"
        except Exception:
            _warn(f"{name}: could not compute confinement/span metrics (beta/span/Aref)")
        setup_rows.append([name, solver, deltaT, endTime, writeInterval, nu, inlet_u, bounds, beta_s])
    _write_table_tex(out_tab / "case_setup.tex", setup_headers, setup_rows)

    # Results summaries for Sections 7/8.
    _write_summary_tex(out_tab / "summary_low.tex", "low", low_cfg, low_case, low_res)
    _write_summary_tex(out_tab / "summary_high.tex", "high", high_cfg, high_case, high_res)
    _write_runtime_summary(low_res, "low")
    _write_runtime_summary(high_res, "high")
    _write_force_stats_table(out_tab, low_case=low_case, high_case=high_case, low_res=low_res, high_res=high_res)

    # Copy existing plots into report/figures with stable names.
    for name, res in [("low", low_res), ("high", high_res)]:
        _copy_if_exists(res / "rom" / "svd_U.png", out_fig / f"{name}_svd_U.png")
        _copy_if_exists(res / "rom" / "svd_p.png", out_fig / f"{name}_svd_p.png")
        _copy_if_exists(res / "rom" / "energy_U.png", out_fig / f"{name}_energy_U.png")
        _copy_if_exists(res / "rom" / "energy_p.png", out_fig / f"{name}_energy_p.png")
        _copy_if_exists(res / "rom" / "err_U.png", out_fig / f"{name}_err_U.png")
        _copy_if_exists(res / "rom" / "err_p.png", out_fig / f"{name}_err_p.png")
        _copy_if_exists(res / "rom" / "err_rom_only.png", out_fig / f"{name}_err_rom_only.png")

        _copy_if_exists(res / "adaptive" / "errU_validate.png", out_fig / f"{name}_errU_validate.png")
        _copy_if_exists(res / "adaptive" / "errp_validate.png", out_fig / f"{name}_errp_validate.png")
        _write_validation_plot(res / "adaptive", out_fig / f"{name}_err_validate.png", "Validation error (U and p)")
        _copy_if_exists(res / "adaptive" / "timeline.png", out_fig / f"{name}_timeline.png")
        _copy_if_exists(res / "adaptive" / "speed.png", out_fig / f"{name}_speed.png")

        # If ROM-only POD plots aren't available (e.g., high config), generate them from snapshots.
        if not (out_fig / f"{name}_svd_U.png").exists() or not (out_fig / f"{name}_svd_p.png").exists() or not (out_fig / f"{name}_energy_U.png").exists() or not (out_fig / f"{name}_energy_p.png").exists():
            try:
                U = np.load(res / "snapshots" / "U.npy")
                p = np.load(res / "snapshots" / "p.npy")
                n_cells = int(U.shape[1])
                U_flat = U.reshape(U.shape[0], n_cells * 3)
                p_flat = p.reshape(p.shape[0], n_cells)
                s_u, e_u = _svd_svals_cum_energy(U_flat)
                s_p, e_p = _svd_svals_cum_energy(p_flat)
                if not (out_fig / f"{name}_svd_U.png").exists():
                    save_singular_values_plot(s_u, out_fig / f"{name}_svd_U.png", f"POD singular values (U, {name})")
                if not (out_fig / f"{name}_energy_U.png").exists():
                    save_energy_plot(e_u, out_fig / f"{name}_energy_U.png", f"Cumulative POD energy (U, {name})")
                if not (out_fig / f"{name}_svd_p.png").exists():
                    save_singular_values_plot(s_p, out_fig / f"{name}_svd_p.png", f"POD singular values (p, {name})")
                if not (out_fig / f"{name}_energy_p.png").exists():
                    save_energy_plot(e_p, out_fig / f"{name}_energy_p.png", f"Cumulative POD energy (p, {name})")
            except Exception:
                pass

    # Forces plots (FOM-only baseline, if forces exist). Prefer the run results so the plotted
    # time window matches the reported run duration; fall back to template cases when needed.
    forces_low = _find_latest_postproc_dat(low_res / "fom_only" / "fom_only_case", "forces1", "forces.dat")
    if forces_low is None:
        forces_low = _find_latest_postproc_dat(root / "cases/low", "forces1", "forces.dat")
    if forces_low is not None and forces_low.exists():
        s = read_forces_dat(forces_low)
        save_forces_plot(s.time, s.Fx, s.Fy, out_fig / "low_forces.png", "Cylinder forces (low FOM)")

    forces_high = _find_latest_postproc_dat(high_res / "fom_only" / "fom_only_case", "forces1", "forces.dat")
    if forces_high is None:
        forces_high = _find_latest_postproc_dat(root / "cases/high", "forces1", "forces.dat")
    if forces_high is not None and forces_high.exists():
        s = read_forces_dat(forces_high)
        save_forces_plot(s.time, s.Fx, s.Fy, out_fig / "high_forces.png", "Cylinder forces (high FOM)")

    # Force coefficients plots (baseline vs adaptive, using postprocessed coefficients when available).
    psd_captions: dict[str, str] = {}
    for name, _case_dir, res_dir in [("low", low_case, low_res), ("high", high_case, high_res)]:
        base_case_dir = res_dir / "fom_only" / "fom_only_case"
        adapt_case_dir = res_dir / "adaptive" / "adaptive_case"
        adaptive_dir = res_dir / "adaptive"

        geom_case_dir = (
            base_case_dir
            if (base_case_dir / "system" / "controlDict").exists()
            else adapt_case_dir
            if (adapt_case_dir / "system" / "controlDict").exists()
            else _case_dir
        )
        u_inf_case = read_inlet_u_mag(geom_case_dir)
        patch = _detect_cylinder_patch_name(geom_case_dir) or "cylinder"
        d_ref: float | None = None
        a_ref_case: float | None = None
        span_case: float | None = None
        try:
            geom = cylinder_geometry(geom_case_dir, patch_name=patch)
            d_ref = float(geom.get("D", float("nan")))
        except Exception:
            d_ref = None
        try:
            span_meta = compute_span(geom_case_dir, patch_name=patch)
            span_case = float(span_meta.get("span", float("nan")))
            a_ref_case = float(span_meta.get("Aref", float("nan")))
        except Exception:
            span_case = None
            a_ref_case = None

        base = _read_force_coeffs_series(base_case_dir, "forceCoeffsIncompressible")
        if base is None:
            base = _read_force_coeffs_series(base_case_dir, "forceCoeffs1")
        # If the case is not strictly 2D (span != 1), recompute baseline coefficients from `forces`.
        must_recompute = (
            span_case is not None
            and np.isfinite(span_case)
            and (abs(float(span_case) - 1.0) > 1e-12)
            and (u_inf_case is not None)
            and (a_ref_case is not None)
        )
        if must_recompute and base is not None:
            forces_base = base_case_dir / "postProcessing" / "forces1" / "0" / "forces.dat"
            if forces_base.exists():
                try:
                    base = _coeffs_from_forces(forces_base, u_inf=float(u_inf_case), a_ref=float(a_ref_case), d_ref=d_ref)
                except Exception:
                    _warn(f"{name}: failed to recompute baseline coefficients from forces; using forceCoeffs output")

        # Prefer a single postprocessed series over the adaptive case fields, then split by model.
        full = _read_force_coeffs_series(adapt_case_dir, "forceCoeffsIncompressible")
        if full is None:
            full = _read_force_coeffs_series(adapt_case_dir, "forceCoeffsROM")
        fom_seg = _mask_series_by_model(full, adaptive_dir, "FOM") if full is not None else None
        rom_seg = _mask_series_by_model(full, adaptive_dir, "ROM") if full is not None else None

        # Fallback: solver-written coefficients during FOM bursts.
        adapt_solver = _read_force_coeffs_series(adapt_case_dir, "forceCoeffs1")

        if base is not None:
            t2 = cd2 = cl2 = None
            t3 = cd3 = cl3 = None
            label2 = "Adaptive FOM"
            label3 = "ROM post"

            # If span != 1, prefer coefficients recomputed from adaptive `forces` for the FOM segments.
            if must_recompute:
                forces_adapt = adapt_case_dir / "postProcessing" / "forces1" / "0" / "forces.dat"
                if forces_adapt.exists():
                    try:
                        adapt_f = _coeffs_from_forces(forces_adapt, u_inf=float(u_inf_case), a_ref=float(a_ref_case), d_ref=d_ref)
                        t2, cd2, cl2 = adapt_f.time, adapt_f.Cd, adapt_f.Cl
                        label2 = "Adaptive FOM (forces)"
                    except Exception:
                        pass

            if rom_seg is not None and fom_seg is not None:
                if t2 is None:
                    t2, cd2, cl2 = fom_seg.time, fom_seg.Cd, fom_seg.Cl
                t3, cd3, cl3 = rom_seg.time, rom_seg.Cd, rom_seg.Cl
            elif full is not None:
                if t2 is None:
                    t2, cd2, cl2 = full.time, full.Cd, full.Cl
                label2 = "Adaptive post"
                label3 = ""
            elif adapt_solver is not None:
                if t2 is None:
                    t2, cd2, cl2 = adapt_solver.time, adapt_solver.Cd, adapt_solver.Cl
                label2 = "Adaptive FOM"
                label3 = ""

            save_force_coeffs_plot(
                base.time,
                base.Cd,
                base.Cl,
                out_fig / f"{name}_coeffs.png",
                f"Force coefficients ({name})",
                t2=t2,
                cd2=cd2,
                cl2=cl2,
                t3=t3,
                cd3=cd3,
                cl3=cl3,
                label1="FOM-only",
                label2=label2,
                label3=label3,
            )

            est = _dominant_freq_welch(base.time, base.Cl)
            if est.freq_hz.size and est.psd.size:
                indicative = (est.f_peak_hz is not None) and np.isfinite(est.n_cycles_kept) and (float(est.n_cycles_kept) < 8.0)
                title = f"$C_l$ Welch PSD ({name})"
                if est.f_peak_hz is not None:
                    title = f"$C_l$ Welch PSD ({name}), peak={est.f_peak_hz:.3g} Hz"
                if indicative:
                    title = f"Indicative: {title} (insufficient cycles)"
                save_fft_plot(
                    est.freq_hz,
                    est.psd,
                    out_fig / f"{name}_cl_fft.png",
                    title,
                    ylabel="PSD",
                    xlim=None,
                )
                if indicative:
                    psd_captions[name] = (
                        r"Lift-coefficient Welch PSD (%s): Indicative: insufficient cycles in the post-transient window "
                        r"($n_{\mathrm{cycles}}=%.3g$, $T_{\mathrm{window}}=%.3g\,\mathrm{s}$)."
                        % (name, float(est.n_cycles_kept), float(est.duration_kept_s))
                    )
                else:
                    psd_captions[name] = r"Lift-coefficient Welch PSD (%s): dominant frequency estimate from $C_l(t)$." % name
            else:
                psd_captions[name] = r"Lift-coefficient Welch PSD (%s): not generated (insufficient data after transient removal)." % name
        else:
            series = full or adapt_solver
            if series is None:
                continue
            save_force_coeffs_plot(
                series.time,
                series.Cd,
                series.Cl,
                out_fig / f"{name}_coeffs.png",
                f"Force coefficients ({name}, adaptive)",
            )
            est = _dominant_freq_welch(series.time, series.Cl)
            if est.freq_hz.size and est.psd.size:
                indicative = (est.f_peak_hz is not None) and np.isfinite(est.n_cycles_kept) and (float(est.n_cycles_kept) < 8.0)
                title = f"$C_l$ Welch PSD ({name})"
                if est.f_peak_hz is not None:
                    title = f"$C_l$ Welch PSD ({name}), peak={est.f_peak_hz:.3g} Hz"
                if indicative:
                    title = f"Indicative: {title} (insufficient cycles)"
                save_fft_plot(
                    est.freq_hz,
                    est.psd,
                    out_fig / f"{name}_cl_fft.png",
                    title,
                    ylabel="PSD",
                    xlim=None,
                )
                if indicative:
                    psd_captions[name] = (
                        r"Lift-coefficient Welch PSD (%s): Indicative: insufficient cycles in the post-transient window "
                        r"($n_{\mathrm{cycles}}=%.3g$, $T_{\mathrm{window}}=%.3g\,\mathrm{s}$)."
                        % (name, float(est.n_cycles_kept), float(est.duration_kept_s))
                    )
                else:
                    psd_captions[name] = r"Lift-coefficient Welch PSD (%s): dominant frequency estimate from $C_l(t)$." % name
            else:
                psd_captions[name] = r"Lift-coefficient Welch PSD (%s): not generated (insufficient data after transient removal)." % name

    captions_path = out_tab / "captions.tex"
    captions_lines: list[str] = []
    for key in ["low", "high"]:
        cap = psd_captions.get(key)
        if not cap:
            continue
        captions_lines.append(r"\renewcommand{\caption" + key + r"clfft}{" + cap + r"}")
    if captions_lines:
        captions_path.write_text("\n".join(captions_lines) + "\n", encoding="utf-8")

    print(f"[report_assets] Wrote tables to: {out_tab}")
    print(f"[report_assets] Wrote/copied figures to: {out_fig}")


if __name__ == "__main__":
    main()
