#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ForceStats:
    case: str
    source: str
    Cd_mean: float
    Cl_rms: float
    f_peak_hz: float | None
    st: float | None
    st_note: str
    beta: float | None


def _is_float_dirname(name: str) -> bool:
    try:
        float(name)
        return True
    except Exception:
        return False


def _strip_foam_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//.*?$", "", text, flags=re.M)
    return text


def _read_inlet_u_mag(case_dir: Path) -> float | None:
    u0 = case_dir / "0" / "U"
    if not u0.exists():
        return None
    txt = _strip_foam_comments(u0.read_text(encoding="utf-8", errors="ignore"))
    for m in re.finditer(
        r"^\s*([A-Za-z0-9_]+)\s*\{.*?\btype\s+fixedValue\s*;.*?\bvalue\s+uniform\s*\(\s*([^)]+?)\s*\)\s*;.*?\}",
        txt,
        flags=re.M | re.S,
    ):
        nums = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", m.group(2))
        if len(nums) >= 3:
            try:
                ux, uy, uz = float(nums[0]), float(nums[1]), float(nums[2])
                return float((ux * ux + uy * uy + uz * uz) ** 0.5)
            except Exception:
                continue
    return None


def _is_case_2d(case_dir: Path) -> bool:
    boundary = case_dir / "constant" / "polyMesh" / "boundary"
    if not boundary.exists():
        return False
    txt = boundary.read_text(encoding="utf-8", errors="ignore")
    return bool(re.search(r"(?m)^\s*type\s+empty\s*;", txt))


def _detect_cylinder_patch_name(case_dir: Path) -> str:
    boundary = case_dir / "constant" / "polyMesh" / "boundary"
    if not boundary.exists():
        return "cylinder"
    txt = boundary.read_text(encoding="utf-8", errors="ignore")
    for cand in ("cylinder", "Cylinder", "obstacle", "Obstacle"):
        if re.search(rf"^\s*{re.escape(cand)}\s*$", txt, flags=re.M):
            return cand
    for m in re.finditer(r"^\s*([A-Za-z0-9_]+)\s*\{", txt, flags=re.M):
        name = m.group(1)
        if "cylinder" in name.lower():
            return name
    return "cylinder"


def _domain_bounds_from_points(case_dir: Path) -> dict[str, float] | None:
    pts_path = case_dir / "constant" / "polyMesh" / "points"
    if not pts_path.exists():
        return None
    txt = pts_path.read_text(encoding="utf-8", errors="ignore")
    nums = re.findall(r"\(\s*([-+0-9Ee\.]+)\s+([-+0-9Ee\.]+)\s+([-+0-9Ee\.]+)\s*\)", txt)
    if not nums:
        return None
    xs = [float(a) for a, _, _ in nums]
    ys = [float(b) for _, b, _ in nums]
    zs = [float(c) for _, _, c in nums]
    return {
        "xmin": float(min(xs)),
        "xmax": float(max(xs)),
        "ymin": float(min(ys)),
        "ymax": float(max(ys)),
        "zmin": float(min(zs)),
        "zmax": float(max(zs)),
    }


def _cylinder_diameter_from_patch_vertices(case_dir: Path, patch_name: str) -> float | None:
    # Best-effort: parse patch vertices from polyMesh files (OpenFOAM ASCII).
    poly = case_dir / "constant" / "polyMesh"
    boundary = poly / "boundary"
    points_path = poly / "points"
    faces_path = poly / "faces"
    if not (boundary.exists() and points_path.exists() and faces_path.exists()):
        return None

    def clean(line: str) -> str:
        return re.sub(r"//.*$", "", line).strip()

    lines = boundary.read_text(encoding="utf-8", errors="ignore").splitlines()
    in_list = False
    i = 0
    start_face = None
    n_faces = None
    while i < len(lines):
        s = clean(lines[i])
        if not in_list:
            if s == "(":
                in_list = True
            i += 1
            continue
        if s == ")":
            break
        if not s:
            i += 1
            continue
        m = re.fullmatch(r"([A-Za-z0-9_]+)(?:\s*\{)?", s)
        if not m:
            i += 1
            continue
        name = m.group(1)
        if "{" not in s:
            j = i + 1
            while j < len(lines) and not clean(lines[j]):
                j += 1
            if j >= len(lines) or "{" not in clean(lines[j]):
                i += 1
                continue
            i = j
        depth = 0
        block_lines: list[str] = []
        while i < len(lines):
            s2 = clean(lines[i])
            if s2:
                depth += s2.count("{")
                depth -= s2.count("}")
                block_lines.append(s2)
            i += 1
            if depth <= 0 and block_lines:
                break
        if name != patch_name:
            continue
        block = "\n".join(block_lines)
        sm = re.search(r"\bstartFace\s+(\d+)\s*;", block)
        nm = re.search(r"\bnFaces\s+(\d+)\s*;", block)
        if sm and nm:
            start_face = int(sm.group(1))
            n_faces = int(nm.group(1))
            break

    if start_face is None or n_faces is None or n_faces < 1:
        return None

    # Read points
    pts_txt = points_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    pts: list[tuple[float, float, float]] = []
    in_pts = False
    vec_re = re.compile(r"^\s*\(\s*([Ee0-9+\-\.]+)\s+([Ee0-9+\-\.]+)\s+([Ee0-9+\-\.]+)\s*\)\s*$")
    for line in pts_txt:
        s = clean(line)
        if not in_pts:
            if s == "(":
                in_pts = True
            continue
        if s == ")":
            break
        m = vec_re.match(s)
        if not m:
            continue
        pts.append((float(m.group(1)), float(m.group(2)), float(m.group(3))))
    if not pts:
        return None
    P = np.asarray(pts, dtype=float)

    # Read faces (only need patch faces range)
    faces_txt = faces_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    faces: list[list[int]] = []
    in_faces = False
    for line in faces_txt:
        s = clean(line)
        if not in_faces:
            if s == "(":
                in_faces = True
            continue
        if s == ")":
            break
        m = re.match(r"^(\d+)\s*\(([^)]*)\)\s*$", s)
        if not m:
            continue
        idx = [int(x) for x in re.findall(r"\d+", m.group(2))]
        if idx:
            faces.append(idx)
    if not faces:
        return None
    if start_face + n_faces > len(faces):
        return None

    v_idx: set[int] = set()
    for fi in range(start_face, start_face + n_faces):
        for v in faces[fi]:
            v_idx.add(int(v))
    if not v_idx:
        return None
    Pv = P[sorted(v_idx), :]
    y0 = float(np.min(Pv[:, 1]))
    y1 = float(np.max(Pv[:, 1]))
    return float(y1 - y0)


def _read_forcecoeffs_dat(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Returns (t, Cd, Cl, header_meta)
    """
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    header: dict[str, Any] = {}
    cols: list[str] | None = None
    data: list[list[float]] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#"):
            # Parse known header scalars of form: "# key : value"
            m = re.match(r"^#\s*([A-Za-z0-9_]+)\s*:\s*(.*)$", s)
            if m:
                k = m.group(1)
                v = m.group(2).strip()
                header[k] = v
            # Detect column header
            if "Time" in s and ("Cd" in s or "Cl" in s):
                cols = [c for c in re.split(r"\s+", s.lstrip("#").strip()) if c]
            continue
        # Data line
        nums = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", s)
        if not nums:
            continue
        data.append([float(x) for x in nums])
    if cols is None:
        # Fallback: assume OpenFOAM default ordering: Time, Cm, Cd, Cl, ...
        cols = ["Time", "Cm", "Cd", "Cl"]

    if not data:
        raise ValueError(f"No data parsed from {path}")
    A = np.asarray(data, dtype=float)
    # Map columns
    def idx(name: str) -> int:
        if name in cols:
            return cols.index(name)
        raise ValueError(f"Missing column {name} in header of {path}")

    t = A[:, idx("Time")]
    cd = A[:, idx("Cd")]
    cl = A[:, idx("Cl")]
    return t, cd, cl, header


def _find_forcecoeffs_files(case_dir: Path, obj_name: str) -> list[Path]:
    root = case_dir / "postProcessing" / obj_name
    if not root.exists():
        return []
    # Prefer numeric time directories (sorted).
    candidates: list[tuple[float, Path]] = []
    for p in root.iterdir():
        if p.is_dir() and _is_float_dirname(p.name):
            for fn in ("forceCoeffs.dat", "coefficient.dat"):
                fp = p / fn
                if fp.exists():
                    candidates.append((float(p.name), fp))
                    break
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return [p for _t, p in candidates]
    # Fallback: any matching file under the tree.
    out: list[Path] = []
    for fn in ("forceCoeffs.dat", "coefficient.dat"):
        out.extend(sorted(root.rglob(fn)))
    return out


def _sort_and_dedup_time(t: np.ndarray, *ys: np.ndarray) -> tuple[np.ndarray, ...]:
    t = np.asarray(t, dtype=float).reshape(-1)
    ys = tuple(np.asarray(y, dtype=float).reshape(-1) for y in ys)
    if any(y.size != t.size for y in ys):
        raise ValueError("t/y length mismatch")
    if t.size == 0:
        return (t,) + ys
    m = np.isfinite(t)
    if not np.any(m):
        return (np.asarray([], dtype=float),) + tuple(np.asarray([], dtype=float) for _ in ys)
    t = t[m]
    ys = tuple(y[m] for y in ys)
    order = np.argsort(t, kind="mergesort")
    t = t[order]
    ys = tuple(y[order] for y in ys)
    # Drop duplicate times (keep last occurrence).
    t_rev = t[::-1]
    _, rev_idx = np.unique(t_rev, return_index=True)
    keep = (t.size - 1 - rev_idx).astype(int)
    keep.sort()
    t = t[keep]
    ys = tuple(y[keep] for y in ys)
    return (t,) + ys


def _read_forcecoeffs_series(
    case_dir: Path, obj_name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]] | None:
    files = _find_forcecoeffs_files(case_dir, obj_name)
    if not files:
        return None
    t_all: list[np.ndarray] = []
    cd_all: list[np.ndarray] = []
    cl_all: list[np.ndarray] = []
    hdr: dict[str, Any] = {}
    for fp in files:
        t, cd, cl, h = _read_forcecoeffs_dat(fp)
        t_all.append(t)
        cd_all.append(cd)
        cl_all.append(cl)
        if h:
            hdr = h
    t_cat = np.concatenate(t_all) if t_all else np.asarray([], dtype=float)
    cd_cat = np.concatenate(cd_all) if cd_all else np.asarray([], dtype=float)
    cl_cat = np.concatenate(cl_all) if cl_all else np.asarray([], dtype=float)
    t_cat, cd_cat, cl_cat = _sort_and_dedup_time(t_cat, cd_cat, cl_cat)  # type: ignore[misc]
    # Match rom/python/report_assets.py: drop non-finite samples and gross outliers (typically restart transients).
    cd_abs_max = 200.0
    cl_abs_max = 200.0
    m = np.isfinite(t_cat) & np.isfinite(cd_cat) & np.isfinite(cl_cat)
    m &= (np.abs(cd_cat) <= cd_abs_max) & (np.abs(cl_cat) <= cl_abs_max)
    t_cat = t_cat[m]
    cd_cat = cd_cat[m]
    cl_cat = cl_cat[m]
    if t_cat.size < 2:
        return None
    return t_cat, cd_cat, cl_cat, hdr


def _trim_transient(t: np.ndarray, y: np.ndarray, discard_frac: float) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if t.size != y.size:
        raise ValueError("t/y length mismatch")
    if t.size < 4:
        return t, y
    # Match rom/python/metrics.py (round, not floor) for consistency with Table 5.
    k0 = int(max(0, round(float(discard_frac) * t.size)))
    k0 = max(0, min(k0, t.size - 2))
    return t[k0:], y[k0:]


def _welch_peak_hz(t: np.ndarray, y: np.ndarray) -> tuple[float | None, float]:
    """
    Dominant-frequency estimate from a Welch PSD on uniformly resampled data.

    Returns (f_peak_hz, n_cycles_kept). If the series is not suitable for PSD (too short,
    ill-conditioned dt, or large gaps), returns (None, 0.0).
    """
    t = np.asarray(t, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if t.size < 16 or t.size != y.size:
        return None, 0.0

    t, y = _sort_and_dedup_time(t, y)  # type: ignore[misc]
    if t.size < 16:
        return None, 0.0

    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size < 8:
        return None, 0.0
    dt_med = float(np.median(dt))
    if not np.isfinite(dt_med) or dt_med <= 0:
        return None, 0.0
    dt_max = float(np.max(dt)) if dt.size else float("nan")
    # Large gaps -> unreliable PSD after interpolation (typical for discontinuous masked series).
    if np.isfinite(dt_max) and dt_max > 10.0 * dt_med:
        return None, 0.0

    fs = 1.0 / dt_med
    t_uniform = np.arange(float(t[0]), float(t[-1]) + 0.5 * dt_med, dt_med, dtype=float)
    if t_uniform.size < 16:
        return None, 0.0
    y_uniform = np.interp(t_uniform, t, y).astype(float, copy=False)

    f_max_hz = 10.0
    try:
        from scipy import signal as sp_signal  # type: ignore

        nperseg = min(1024, y_uniform.size)
        if nperseg < 16:
            return None, 0.0
        freq, psd = sp_signal.welch(
            y_uniform,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=nperseg // 2,
            detrend="constant",
            scaling="density",
        )
        if freq.size < 4:
            return None, 0.0
        mask = (freq > 0.0) & (freq <= f_max_hz)
        if int(np.sum(mask)) < 3:
            return None, 0.0
        idx = np.where(mask)[0]
        i = int(idx[np.argmax(psd[idx])])
        f_peak = float(freq[i])
    except Exception:
        # Minimal fallback: FFT peak (less robust than Welch).
        n = int(2 ** int(np.ceil(np.log2(y_uniform.size))))
        yf = np.fft.rfft(y_uniform - float(np.mean(y_uniform)), n=n)
        freq = np.fft.rfftfreq(n, d=dt_med)
        psd = np.abs(yf) ** 2
        if freq.size < 4:
            return None, 0.0
        mask = (freq > 0.0) & (freq <= f_max_hz)
        if int(np.sum(mask)) < 3:
            return None, 0.0
        idx = np.where(mask)[0]
        i = int(idx[np.argmax(psd[idx])])
        f_peak = float(freq[i])

    duration = float(dt_med) * float(max(0, y_uniform.size - 1))
    n_cycles = float(f_peak * duration) if (np.isfinite(f_peak) and duration > 0) else 0.0
    return f_peak, n_cycles


def _mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else float("nan")


def _rms_fluct(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    mu = float(np.mean(x))
    return float(np.sqrt(np.mean((x - mu) ** 2)))


def _stats_for_series(
    *,
    case: str,
    source: str,
    t: np.ndarray,
    cd: np.ndarray,
    cl: np.ndarray,
    discard_frac: float,
    beta: float | None,
    f_peak_case: float | None,
    st_case: float | None,
    st_note: str,
) -> ForceStats:
    t, cd, cl = _sort_and_dedup_time(t, cd, cl)  # type: ignore[misc]
    t2, cd2 = _trim_transient(t, cd, discard_frac)
    _, cl2 = _trim_transient(t, cl, discard_frac)
    cd_mean = _mean(cd2)
    cl_rms = _rms_fluct(cl2)
    return ForceStats(
        case=case,
        source=source,
        Cd_mean=cd_mean,
        Cl_rms=cl_rms,
        f_peak_hz=f_peak_case,
        st=st_case,
        st_note=st_note,
        beta=beta,
    )


def _read_nu(case_dir: Path) -> float | None:
    phys = case_dir / "constant" / "physicalProperties"
    if not phys.exists():
        return None
    txt = _strip_foam_comments(phys.read_text(encoding="utf-8", errors="ignore"))
    m = re.search(r"(?m)^\s*nu\s+([^;]+);", txt)
    if not m:
        return None
    try:
        return float(m.group(1).strip())
    except Exception:
        return None


def _case_geometry(case_dir: Path) -> tuple[float | None, float | None, float | None, float | None]:
    patch = _detect_cylinder_patch_name(case_dir)
    D = _cylinder_diameter_from_patch_vertices(case_dir, patch)
    is2d = _is_case_2d(case_dir)
    bounds = _domain_bounds_from_points(case_dir)
    beta = None
    if bounds is not None and D is not None:
        H = float(bounds["ymax"] - bounds["ymin"])
        beta = float(D / H) if H > 0 else float("nan")
    span = 1.0 if is2d else None
    Aref = (float(D) * float(span)) if (D is not None and span is not None) else None
    return D, span, Aref, beta


def _write_csv(path: Path, rows: list[ForceStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "source",
                "Cd_mean",
                "Cl_rms",
                "f_peak_Hz",
                "St",
                "beta",
            ],
        )
        w.writeheader()
        for r in rows:
            f_str = "N/A" if r.f_peak_hz is None else f"{r.f_peak_hz:.8g}"
            st_str = "N/A" if r.st is None else f"{r.st:.8g}"
            if r.st is not None and r.st_note:
                st_str = f"{st_str} ({r.st_note})"
            w.writerow(
                {
                    "case": r.case,
                    "source": r.source,
                    "Cd_mean": "N/A" if not np.isfinite(r.Cd_mean) else f"{r.Cd_mean:.8g}",
                    "Cl_rms": "N/A" if not np.isfinite(r.Cl_rms) else f"{r.Cl_rms:.8g}",
                    "f_peak_Hz": f_str,
                    "St": st_str,
                    "beta": "N/A" if (r.beta is None or not np.isfinite(r.beta)) else f"{r.beta:.8g}",
                }
            )


def _write_tex_table(path: Path, rows: list[ForceStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["case", "source", "Cd_mean", "Cl_rms", "f_peak_Hz", "St", "beta"]
    lines: list[str] = []
    lines.append("\\begin{tabular}{lllllll}")
    lines.append("\\hline")
    lines.append(" & ".join(h.replace("_", "\\_") for h in headers) + " \\\\")
    lines.append("\\hline")
    for r in rows:
        f_str = "N/A" if r.f_peak_hz is None else f"{r.f_peak_hz:.6g}"
        st_str = "N/A" if r.st is None else f"{r.st:.6g}"
        if r.st is not None and r.st_note:
            st_str = f"{st_str} ({r.st_note})"
        b_str = "N/A" if (r.beta is None or not np.isfinite(r.beta)) else f"{r.beta:.6g}"
        lines.append(
            " & ".join(
                [
                    r.case,
                    r.source.replace("_", "\\_"),
                    "N/A" if not np.isfinite(r.Cd_mean) else f"{r.Cd_mean:.6g}",
                    "N/A" if not np.isfinite(r.Cl_rms) else f"{r.Cl_rms:.6g}",
                    f_str,
                    st_str,
                    b_str,
                ]
            )
            + " \\\\"
        )
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _timeline_model_at_times(adaptive_dir: Path, times: np.ndarray) -> np.ndarray | None:
    """
    Map sample times -> active model label ("FOM"/"ROM") using adaptive/timeline.csv.
    """
    path = adaptive_dir / "timeline.csv"
    if not path.exists():
        return None
    import csv as _csv

    t_marks: list[float] = []
    m_marks: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = _csv.DictReader(f)
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


def _mask_series_by_model(
    t: np.ndarray, cd: np.ndarray, cl: np.ndarray, adaptive_dir: Path, model: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Keep only samples that fall in the requested model interval ("FOM" or "ROM"),
    based on adaptive/timeline.csv.
    """
    labels = _timeline_model_at_times(adaptive_dir, t)
    if labels is None:
        return None
    model = str(model).strip().upper()
    keep = labels == model
    if int(np.sum(keep)) < 2:
        return None
    return t[keep], cd[keep], cl[keep]


def _compute_for_case(case: str, root: Path, discard_frac: float) -> list[ForceStats]:
    results_dir = (root / "results" / case).resolve()
    base = results_dir / "fom_only" / "fom_only_case"
    adapt = results_dir / "adaptive" / "adaptive_case"
    if not base.exists():
        raise FileNotFoundError(base)
    if not adapt.exists():
        raise FileNotFoundError(adapt)

    D, _span, _Aref, beta = _case_geometry(base)
    U_inf = _read_inlet_u_mag(base) or _read_inlet_u_mag(adapt)

    rows: list[ForceStats] = []

    # Prefer postprocessed coefficients (from fields) when present, else solver forceCoeffs.
    base_series = _read_forcecoeffs_series(base, "forceCoeffsIncompressible") or _read_forcecoeffs_series(base, "forceCoeffs1")
    if base_series is None:
        raise FileNotFoundError(f"No forceCoeffs output under {base}/postProcessing")
    t_base, cd_base, cl_base, _hdr = base_series

    # Case-level dominant frequency is computed from the best continuous baseline available,
    # then reused for all sources (adaptive segments can be discontinuous).
    t_trim, cl_trim = _trim_transient(t_base, cl_base, discard_frac)
    f_peak_case, n_cycles = _welch_peak_hz(t_trim, cl_trim)
    st_case: float | None = None
    st_note = ""
    if f_peak_case is not None and D is not None and U_inf is not None and U_inf > 0 and D > 0:
        st_case = float(f_peak_case * D / U_inf)
        if not np.isfinite(n_cycles) or float(n_cycles) < 8.0:
            st_note = "indicative"

    rows.append(
        _stats_for_series(
            case=case,
            source="FOM-only",
            t=t_base,
            cd=cd_base,
            cl=cl_base,
            discard_frac=discard_frac,
            beta=beta,
            f_peak_case=f_peak_case,
            st_case=st_case,
            st_note=st_note,
        )
    )

    # Prefer a single postprocessed coefficient timeline (computed from fields),
    # then split into FOM/ROM intervals using adaptive/timeline.csv (as in rom/python/report_assets.py).
    full_series = _read_forcecoeffs_series(adapt, "forceCoeffsIncompressible") or _read_forcecoeffs_series(adapt, "forceCoeffsROM")
    adaptive_dir = results_dir / "adaptive"
    if full_series is not None:
        t_full, cd_full, cl_full, _ = full_series
        fom_seg = _mask_series_by_model(t_full, cd_full, cl_full, adaptive_dir, "FOM")
        rom_seg = _mask_series_by_model(t_full, cd_full, cl_full, adaptive_dir, "ROM")
        if fom_seg is not None:
            t2, cd2, cl2 = fom_seg
            rows.append(
                _stats_for_series(
                    case=case,
                    source="adaptive (FOM segments)",
                    t=t2,
                    cd=cd2,
                    cl=cl2,
                    discard_frac=discard_frac,
                    beta=beta,
                    f_peak_case=f_peak_case,
                    st_case=st_case,
                    st_note=st_note,
                )
            )
        if rom_seg is not None:
            t3, cd3, cl3 = rom_seg
            rows.append(
                _stats_for_series(
                    case=case,
                    source="ROM intervals (postProcess)",
                    t=t3,
                    cd=cd3,
                    cl=cl3,
                    discard_frac=discard_frac,
                    beta=beta,
                    f_peak_case=f_peak_case,
                    st_case=st_case,
                    st_note=st_note,
                )
            )
    else:
        # Fallback: use solver-written forceCoeffs when postprocess is unavailable.
        adapt_series = _read_forcecoeffs_series(adapt, "forceCoeffs1")
        if adapt_series is not None:
            t2, cd2, cl2, _ = adapt_series
            rows.append(
                _stats_for_series(
                    case=case,
                    source="adaptive (FOM segments)",
                    t=t2,
                    cd=cd2,
                    cl=cl2,
                    discard_frac=discard_frac,
                    beta=beta,
                    f_peak_case=f_peak_case,
                    st_case=st_case,
                    st_note=st_note,
                )
            )

    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute mean(Cd), rms(Cl), and dominant f from forceCoeffs output.")
    ap.add_argument("--case", choices=["low", "high"], action="append", default=[], help="Which case(s) to process")
    ap.add_argument("--discard-frac", type=float, default=0.30, help="Initial transient fraction to discard")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    cases = args.case or ["low", "high"]
    discard_frac = float(args.discard_frac)
    if not (0.0 <= discard_frac < 0.95):
        raise SystemExit("--discard-frac must be in [0, 0.95)")

    # Unconfined reference values requested by the report (used as a sanity anchor only).
    cd_ref_unconf = 1.316
    st_ref_unconf = 0.1958
    print(f"[compute_force_stats] unconfined Re=200 reference: Cd≈{cd_ref_unconf:g}, St≈{st_ref_unconf:g} (sanity anchor)")

    for c in cases:
        rows = _compute_for_case(c, root=root, discard_frac=discard_frac)
        out_csv = (root / "results" / c / "force_stats.csv").resolve()
        out_csv_compat = (root / "results" / f"force_stats_{c}.csv").resolve()
        out_tex = (root / "report" / "tables" / f"force_stats_checks_{c}.tex").resolve()
        _write_csv(out_csv, rows)
        _write_csv(out_csv_compat, rows)
        _write_tex_table(out_tex, rows)
        print(f"[compute_force_stats] Wrote: {out_csv}")
        print(f"[compute_force_stats] Wrote: {out_csv_compat}")
        print(f"[compute_force_stats] Wrote: {out_tex}")
        # Reminder line requested by task.
        beta = next((r.beta for r in rows if r.beta is not None), None)
        if beta is not None and np.isfinite(beta) and float(beta) >= 0.2:
            print("[compute_force_stats] NOTE: Large deviation is expected under strong confinement, but only after configuration checks pass.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[compute_force_stats] FAIL: {exc}", file=sys.stderr)
        raise
