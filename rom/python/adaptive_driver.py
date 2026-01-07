from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from config import get, load_yaml
from foam_ascii import n_cells_from_polymesh, read_internal_field, save_meta
from foam_control import detect_runner, set_control_dict_entries
from foam_io import prepare_rom_case, write_time_fields
from metrics import l2_abs, l2_abs_volweighted, l2_rel, l2_rel_volweighted
from opinf import OpInfModel, fit_quadratic
from pod import PODModel, fit_pod
from plots import save_error_plot
from plots import save_error_plot_two, save_error_plot_two_rel_abs, save_model_timeline_plot, save_speed_bar


def _is_float_dirname(name: str) -> bool:
    try:
        float(name)
        return True
    except Exception:
        return False


def _extract_brace_block(text: str, start_idx: int) -> tuple[str, int]:
    if start_idx < 0 or start_idx >= len(text) or text[start_idx] != "{":
        raise ValueError("start_idx must point to '{'")
    depth = 0
    i = start_idx
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start_idx : i + 1], i + 1
        i += 1
    raise ValueError("Unbalanced braces while parsing dictionary")


def _list_time_dirs(case_dir: Path) -> list[tuple[float, Path]]:
    out: list[tuple[float, Path]] = []
    for p in case_dir.iterdir():
        if p.is_dir() and _is_float_dirname(p.name):
            out.append((float(p.name), p))
    out.sort(key=lambda x: x[0])
    return out


def _fmt_time(t: float) -> str:
    return f"{t:g}"

def _resolve_time_dir(case_dir: Path, t: float, *, tol: float) -> tuple[float, Path]:
    """
    Resolve a requested time to an existing OpenFOAM time directory.

    Returns (t_actual, path) where t_actual is float(path.name).
    """
    td = case_dir / _fmt_time(t)
    if td.exists():
        try:
            return float(td.name), td
        except Exception:
            return float(t), td

    tdirs = _list_time_dirs(case_dir)
    if not tdirs:
        raise FileNotFoundError(f"No time directories found under {case_dir}")
    nearest_t, nearest_p = min(tdirs, key=lambda x: abs(x[0] - t))
    if abs(nearest_t - t) > float(tol):
        raise FileNotFoundError(f"Time directory {_fmt_time(t)} not found (nearest: {nearest_t:g}, tol={tol:g})")
    return float(nearest_t), nearest_p


def _interp_coeffs(t: np.ndarray, a: np.ndarray, t_query: float) -> np.ndarray:
    """
    Linear interpolation of reduced coefficients a(t) at a single time.
    """
    tq = float(t_query)
    t = np.asarray(t, dtype=float).reshape(-1)
    a = np.asarray(a, dtype=float)
    if t.size < 2:
        raise ValueError("Need at least 2 time points for interpolation")
    if a.ndim != 2 or a.shape[0] != t.size:
        raise ValueError("a must be (n_time, r) aligned with t")
    if tq < float(t[0]) - 1e-12 or tq > float(t[-1]) + 1e-12:
        raise ValueError("t_query out of bounds for interpolation")
    if abs(tq - float(t[0])) <= 1e-12:
        return a[0].copy()
    if abs(tq - float(t[-1])) <= 1e-12:
        return a[-1].copy()
    r = int(a.shape[1])
    out = np.empty((r,), dtype=float)
    for j in range(r):
        out[j] = float(np.interp(tq, t, a[:, j]))
    return out


def _make_time_grid(t0: float, t1: float, dt: float) -> np.ndarray:
    if dt <= 0:
        raise ValueError("dt must be positive")
    n = int(round((float(t1) - float(t0)) / float(dt)))
    if n < 1:
        raise ValueError("time window too small for dt")
    t = float(t0) + float(dt) * np.arange(n + 1, dtype=float)
    t[0] = float(t0)
    t[-1] = float(t1)
    if np.any(np.diff(t) <= 0):
        raise RuntimeError("generated non-increasing time grid")
    return t


def _run(cmd: list[str], *, cwd: Path, log_path: Path) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n[cmd] (cwd={cwd}) {' '.join(cmd)}\n")
        f.flush()
        p = subprocess.Popen(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT)
        rc = p.wait()
        if rc != 0:
            raise RuntimeError(f"Command failed (rc={rc}): {' '.join(cmd)} (see {log_path})")
    return time.perf_counter() - t0


def _read_fields_at_time(case_dir: Path, t: float, n_cells: int, *, tol: float = 1e-9) -> tuple[float, np.ndarray, np.ndarray]:
    t_act, td = _resolve_time_dir(case_dir, t, tol=tol)
    U = read_internal_field(td / "U", n_cells=n_cells)
    p = read_internal_field(td / "p", n_cells=n_cells)
    if U.shape != (n_cells, 3) or p.shape != (n_cells,):
        raise ValueError("Field shape mismatch")
    return t_act, U, p


def _remove_pressure_gauge(p: np.ndarray, *, volumes: np.ndarray | None) -> np.ndarray:
    """
    Remove the arbitrary pressure gauge (constant offset) from incompressible kinematic pressure.

    For ROM/FOM comparisons, this makes the error metric gauge-invariant.
    """
    p = np.asarray(p, dtype=float).reshape(-1)
    if volumes is not None:
        v = np.asarray(volumes, dtype=float).reshape(-1)
        if v.shape != p.shape:
            raise ValueError("volumes shape mismatch vs p")
        denom = float(np.sum(v))
        mu = float(np.sum(v * p) / denom) if denom > 0 else float(np.mean(p))
    else:
        mu = float(np.mean(p))
    return p - mu


def _collect_window(case_dir: Path, t_start: float, t_end: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tdirs = _list_time_dirs(case_dir)
    times = np.array([t for t, _ in tdirs if t_start - 1e-12 <= t <= t_end + 1e-12], dtype=float)
    if times.size < 3:
        raise RuntimeError(f"Not enough snapshots in [{t_start}, {t_end}] (found {times.size})")
    n_cells = n_cells_from_polymesh(case_dir)
    U_list = []
    p_list = []
    for t in times:
        _t_act, U, p = _read_fields_at_time(case_dir, t, n_cells=n_cells)
        U_list.append(U)
        p_list.append(p)
    U_snap = np.stack(U_list, axis=0)
    p_snap = np.stack(p_list, axis=0)
    return times, U_snap, p_snap


@dataclass
class ROMBundle:
    pod_u: PODModel
    pod_p: PODModel
    opinf: OpInfModel
    r_u: int
    r_p: int
    z_bound_train: np.ndarray  # (r_u+r_p,) max |z| observed in training, z=(a-mu)/sigma


def _train_rom(
    times: np.ndarray,
    U_snap: np.ndarray,
    p_snap: np.ndarray,
    cfg: dict[str, Any],
    *,
    volumes: np.ndarray | None,
) -> ROMBundle:
    times = np.asarray(times, dtype=float).reshape(-1)
    n_snap, n_cells = U_snap.shape[0], U_snap.shape[1]
    if p_snap.shape != (n_snap, n_cells):
        raise ValueError("p_snap shape mismatch")

    # Discard an initial transient fraction before POD/OpInf. Early-time snapshots often contain
    # initialization artefacts (e.g., large transients) that destabilize learned dynamics.
    train_discard_frac = float(get(cfg, "rom.train_discard_frac", 0.0))
    if train_discard_frac > 0:
        k0 = int(np.floor(train_discard_frac * float(n_snap)))
        k0 = max(0, min(k0, n_snap - 3))
        if k0 > 0:
            times = times[k0:]
            U_snap = U_snap[k0:]
            p_snap = p_snap[k0:]
            n_snap = int(times.size)

    # Drop extreme pressure outliers (usually an initialization artefact) before POD/OpInf.
    # This prevents a single bad early-time snapshot from dominating the ROM and causing blow-ups.
    p_train_max_abs = float(get(cfg, "rom.train_p_max_abs", 200.0))

    U_flat = U_snap.reshape(n_snap, n_cells * 3)

    # Pressure in incompressible settings is gauge-free (defined up to a constant).
    # Remove the per-snapshot gauge (mean) before building the POD basis and training dynamics.
    p0 = p_snap.reshape(n_snap, n_cells)
    if volumes is not None and volumes.shape == (n_cells,):
        v = volumes.reshape(1, n_cells)
        denom = float(np.sum(v))
        mu = (p0 * v).sum(axis=1, keepdims=True) / denom if denom > 0 else p0.mean(axis=1, keepdims=True)
        p_flat = p0 - mu
    else:
        p_flat = p0 - p0.mean(axis=1, keepdims=True)

    p_maxabs = np.max(np.abs(p_flat), axis=1)
    keep = (p_maxabs <= p_train_max_abs) & np.isfinite(p_maxabs)
    if int(np.sum(keep)) < 3:
        raise RuntimeError(
            f"Too few snapshots after pressure outlier filtering (kept {int(np.sum(keep))}/{n_snap}); "
            f"consider increasing rom.train_p_max_abs (current {p_train_max_abs:g})."
        )
    if not np.all(keep):
        times = times[keep]
        U_flat = U_flat[keep]
        p_flat = p_flat[keep]
        n_snap = int(times.size)

    rom_cfg = cfg.get("rom", {})
    r_u = int(rom_cfg.get("r_u", 8))
    r_p = int(rom_cfg.get("r_p", 6))
    energy = float(rom_cfg.get("energy_threshold", 0.999))
    ridge = float(rom_cfg.get("ridge", 1e-6))

    pod_u = fit_pod(U_flat, r=r_u, energy_threshold=energy, center=True)
    pod_p = fit_pod(p_flat, r=r_p, energy_threshold=energy, center=True)

    # Effective ranks after any clamping inside POD.
    r_u_eff = int(pod_u.basis.shape[1])
    r_p_eff = int(pod_p.basis.shape[1])

    a_u = pod_u.project(U_flat)
    a_p = pod_p.project(p_flat)
    a = np.hstack([a_u, a_p])

    opinf = fit_quadratic(times, a, ridge=ridge, normalize=True)
    z = (a - opinf.mu[None, :]) / opinf.sigma[None, :]
    z_bound_train = np.nanmax(np.abs(z), axis=0)
    z_bound_train = np.where(np.isfinite(z_bound_train), z_bound_train, 0.0).astype(float)
    return ROMBundle(pod_u=pod_u, pod_p=pod_p, opinf=opinf, r_u=r_u_eff, r_p=r_p_eff, z_bound_train=z_bound_train)


def _integrate_rom(
    times: np.ndarray,
    a0: np.ndarray,
    opinf: OpInfModel,
    cfg: dict[str, Any],
    *,
    z_bound_train: np.ndarray | None = None,
) -> np.ndarray:
    from rom_integrate import integrate_fixed_step, integrate_ivp

    dt = float(times[1] - times[0])
    method = str(get(cfg, "rom.integrator", "BDF"))
    # Soft bounds in normalized coordinates, used both for detection and fallback clipping.
    z_train_bound = float(get(cfg, "rom.clip_factor", 1.25))
    z_min_abs = float(get(cfg, "rom.clip_min_abs", 1.0))
    z0 = (a0 - opinf.mu) / opinf.sigma
    if z_bound_train is not None:
        z_bound_train = np.asarray(z_bound_train, dtype=float).reshape(-1)
        if z_bound_train.shape != z0.shape:
            raise ValueError("z_bound_train shape mismatch")
        z_bound = z_train_bound * np.maximum(z_bound_train, z_min_abs)
    else:
        z_bound = np.maximum(np.abs(z0) * z_train_bound, z_min_abs)

    res = integrate_ivp(opinf.rhs, times, a0=a0, method=method, max_step=dt)
    if res.success:
        # Even when solve_ivp "succeeds", unstable ROMs can drift far outside the local training
        # neighborhood and produce nonphysical fields that make the subsequent FOM burst crash.
        # Detect this early and fall back to clipped fixed-step integration.
        z = (np.asarray(res.a, dtype=float) - opinf.mu[None, :]) / opinf.sigma[None, :]
        if np.all(np.isfinite(z)) and np.all(np.abs(z) <= z_bound[None, :]):
            return res.a

    def rhs_z(_t: float, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float).reshape(-1)
        return opinf.c + opinf.A @ z + opinf.H @ np.kron(z, z)

    res2 = integrate_fixed_step(rhs_z, times, z0, clip=(-z_bound, z_bound))
    if not res2.success:
        raise RuntimeError(f"ROM integration failed (solve_ivp: {res.message}; RK4+clip: {res2.message})")
    return opinf.mu[None, :] + opinf.sigma[None, :] * res2.a


def _write_rom_segment(case_dir: Path, times: np.ndarray, U_rec: np.ndarray, p_rec: np.ndarray) -> None:
    # Writes each time directory by copying the previous directory and replacing internalField (U,p).
    for k in range(1, times.size):
        t_prev = _fmt_time(times[k - 1])
        t_cur = _fmt_time(times[k])
        write_time_fields(case_dir, t_cur, U=U_rec[k], p=p_rec[k], template_time=t_prev)


def _have_postprocess() -> bool:
    return shutil.which("postProcess") is not None


def _find_latest_pp_file(case_dir: Path, obj_name: str, filenames: tuple[str, ...]) -> Path | None:
    root = case_dir / "postProcessing" / obj_name
    if not root.exists():
        return None
    candidates: list[tuple[float, Path]] = []
    for p in root.iterdir():
        if p.is_dir() and _is_float_dirname(p.name):
            for fn in filenames:
                fp = p / fn
                if fp.exists():
                    candidates.append((float(p.name), fp))
                    break
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[-1][1]
    # Fallback: any match under tree.
    for fn in filenames:
        for fp in root.rglob(fn):
            if fp.is_file():
                return fp
    return None


def _have_foam_postprocess() -> bool:
    return shutil.which("foamPostProcess") is not None


def _write_forces_incompressible_dict(path: Path, *, patch: str, cof_r: tuple[float, float, float]) -> None:
    cx, cy, cz = cof_r
    txt = f"""/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  13
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/

patches     ({patch});

rho         rhoInf;
rhoInf      1;

// Moment calculation parameters
CofR        ({cx:g} {cy:g} {cz:g});

#includeEtc "caseDicts/functions/forces/forcesIncompressible.cfg"

// ************************************************************************* //
"""
    path.write_text(txt, encoding="utf-8")


def _read_forcecoeffs_min(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Minimal parser for OpenFOAM forceCoeffs.dat/coefficient.dat.
    Returns (t, Cd, Cl).
    """
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    cols: list[str] | None = None
    data: list[list[float]] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#"):
            if "Time" in s and ("Cd" in s or "Cl" in s):
                cols = [c for c in re.split(r"\s+", s.lstrip("#").strip()) if c]
            continue
        nums = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", s)
        if nums:
            data.append([float(x) for x in nums])
    if not data:
        raise ValueError(f"No data parsed from {path}")
    A = np.asarray(data, dtype=float)
    if cols is None:
        cols = ["Time", "Cm", "Cd", "Cl"]

    def idx(name: str) -> int:
        if name in cols:
            return cols.index(name)
        raise ValueError(f"Missing column {name} in {path}")

    t = A[:, idx("Time")]
    cd = A[:, idx("Cd")]
    cl = A[:, idx("Cl")]
    return t, cd, cl


def _robust_baseline_refs(t: np.ndarray, cd: np.ndarray, cl: np.ndarray, *, discard_frac: float) -> tuple[float, float]:
    t = np.asarray(t, dtype=float).reshape(-1)
    cd = np.asarray(cd, dtype=float).reshape(-1)
    cl = np.asarray(cl, dtype=float).reshape(-1)
    if t.size < 8:
        return float(np.nanmedian(cd)), float(np.nanstd(cl))
    k0 = int(np.floor(float(discard_frac) * t.size))
    k0 = max(0, min(k0, t.size - 3))
    cd2 = cd[k0:]
    cl2 = cl[k0:]
    cd2 = cd2[np.isfinite(cd2)]
    cl2 = cl2[np.isfinite(cl2)]
    if cd2.size:
        lo, hi = np.percentile(cd2, [5.0, 95.0])
        cd2 = cd2[(cd2 >= lo) & (cd2 <= hi)]
    cd_ref = float(np.median(cd2)) if cd2.size else float("nan")
    if cl2.size:
        cl_mu = float(np.mean(cl2))
        cl_rms = float(np.sqrt(np.mean((cl2 - cl_mu) ** 2)))
    else:
        cl_rms = float("nan")
    return cd_ref, cl_rms


def _read_control_dict_norm(case_dir: Path) -> tuple[float | None, float | None, float | None]:
    """
    Best-effort read of magUInf/lRef/Aref from the forceCoeffs1 functionObject in system/controlDict.
    """
    cd = case_dir / "system" / "controlDict"
    if not cd.exists():
        return None, None, None
    txt = cd.read_text(encoding="utf-8", errors="ignore")
    txt = re.sub(r"/\*.*?\*/", "", txt, flags=re.S)
    txt = re.sub(r"//.*?$", "", txt, flags=re.M)
    m = re.search(r"(?m)^[ \t]*forceCoeffs1[ \t]*$", txt)
    if not m:
        return None, None, None
    brace_idx = txt.find("{", m.end())
    if brace_idx < 0:
        return None, None, None
    blk, _ = _extract_brace_block(txt, brace_idx)

    def getf(key: str) -> float | None:
        mm = re.search(rf"(?m)^\s*{re.escape(key)}\s+([^;]+)\s*;", blk)
        if not mm:
            return None
        try:
            return float(mm.group(1).strip())
        except Exception:
            return None

    return getf("magUInf"), getf("lRef"), getf("Aref")


def _read_last_forces_total(path: Path) -> tuple[float, float] | None:
    """
    Read the total (pressure + viscous) Fx,Fy from the last data line of forces.dat.
    """
    if not path.exists():
        return None
    last = None
    for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        last = s
    if last is None:
        return None
    nums = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", last)
    # Format: time, pFx pFy pFz, vFx vFy vFz, ... (moments) ...
    if len(nums) < 7:
        return None
    try:
        pfx, pfy = float(nums[1]), float(nums[2])
        vfx, vfy = float(nums[4]), float(nums[5])
        return float(pfx + vfx), float(pfy + vfy)
    except Exception:
        return None


def _force_gate_check(
    *,
    case_dir: Path,
    time_name: str,
    runner_log: Path,
    u_inf: float,
    a_ref: float,
    cd_ref: float,
    cl_rms_ref: float,
    cfg: dict[str, Any],
) -> tuple[bool, dict[str, float]]:
    """
    Compute (Cd,Cl) at `time_name` from forces1 and decide whether to allow starting FOM from this ROM state.
    """
    fg = get(cfg, "adaptive.force_gate", {}) or {}
    cd_factor_max = float(fg.get("cd_factor_max", 2.0))
    cd_abs_max = float(fg.get("cd_abs_max", 50.0))
    cl_rms_factor_max = float(fg.get("cl_rms_factor_max", 4.0))
    cl_abs_max = float(fg.get("cl_abs_max", 50.0))
    if not (_have_foam_postprocess() and u_inf > 0 and a_ref > 0):
        return True, {}

    # Compute forces via foamPostProcess (avoids ambiguity between functionObject instance names vs function types).
    patch = str(get(cfg, "openfoam.force_patch", "cylinder"))
    cof_r = (0.0, 0.0, 0.0)
    func_name = "forcesGate"
    func_path = case_dir / "system" / func_name
    _write_forces_incompressible_dict(func_path, patch=patch, cof_r=cof_r)
    _run(
        [
            "foamPostProcess",
            "-case",
            str(case_dir),
            "-solver",
            "incompressibleFluid",
            "-func",
            func_name,
            "-fields",
            "(U p)",
            "-time",
            str(time_name),
        ],
        cwd=case_dir,
        log_path=runner_log,
    )
    fp = case_dir / "postProcessing" / func_name / str(time_name) / "forces.dat"
    fxy = _read_last_forces_total(fp) if fp.exists() else None
    if fxy is None:
        return True, {}
    fx, fy = fxy
    q = 0.5 * float(u_inf) * float(u_inf) * float(a_ref)
    cd = float(fx / q)
    cl = float(fy / q)
    meta = {"Cd": cd, "Cl": cl}

    if not (np.isfinite(cd) and np.isfinite(cl)):
        return False, meta
    if cd < 0.0 or cd > cd_abs_max:
        return False, meta
    if abs(cl) > cl_abs_max:
        return False, meta
    if np.isfinite(cd_ref) and cd_ref > 0 and cd > cd_factor_max * float(cd_ref):
        return False, meta
    if np.isfinite(cl_rms_ref) and cl_rms_ref > 0 and abs(cl) > cl_rms_factor_max * float(cl_rms_ref):
        return False, meta
    return True, meta


def _delete_time_dirs_after(case_dir: Path, t_keep: float) -> None:
    """
    Delete all numeric time directories strictly after t_keep.

    Used to discard ROM-written times before running a fallback FOM segment.
    """
    for t_dir, p in reversed(_list_time_dirs(case_dir)):
        if float(t_dir) > float(t_keep) + 1e-12:
            shutil.rmtree(p, ignore_errors=True)


def _ensure_cell_volumes(case_dir: Path, time_name: str, n_cells: int, log_path: Path) -> np.ndarray | None:
    """
    Create/read cell volume weights for volume-weighted errors.
    Uses `postProcess -func writeCellVolumes` and reads volScalarField 'V' (or 'cellVolumes').
    """
    td = case_dir / time_name
    if not td.exists():
        return None
    # OpenFOAM's writeCellVolumes commonly writes 'Vc' (cell volume), but older setups may use 'V'.
    for fname in ["Vc", "V", "cellVolumes"]:
        fp = td / fname
        if fp.exists():
            try:
                v = read_internal_field(fp, n_cells=n_cells)
                if v.shape == (n_cells,):
                    return v
            except Exception:
                pass

    if not _have_postprocess():
        return None
    # Run postProcess and try again.
    _run(["postProcess", "-func", "writeCellVolumes", "-time", time_name], cwd=case_dir, log_path=log_path)
    for fname in ["Vc", "V", "cellVolumes"]:
        fp = td / fname
        if fp.exists():
            try:
                v = read_internal_field(fp, n_cells=n_cells)
                if v.shape == (n_cells,):
                    return v
            except Exception:
                continue
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true", help="Overwrite existing adaptive outputs")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    base_dir = Path(get(cfg, "case.work_dir", "cases/low")).resolve()
    results_dir = Path(get(cfg, "case.results_dir", "results/low")).resolve()
    out_dir = results_dir / "adaptive"
    case_dir = out_dir / "adaptive_case"
    out_dir.mkdir(parents=True, exist_ok=True)

    if case_dir.exists():
        if not args.force:
            raise RuntimeError(f"{case_dir} exists; re-run with --force to overwrite")
        shutil.rmtree(case_dir)

    # Start from the low case (already configured).
    prepare_rom_case(base_dir, case_dir)

    # Ensure time settings match config.
    dt = float(get(cfg, "openfoam.deltaT", 0.005))
    set_control_dict_entries(
        case_dir / "system" / "controlDict",
        {
            "deltaT": str(dt),
            "writeControl": "timeStep",
            "writeInterval": "1",
            "writeFormat": "ascii",
            "purgeWrite": "0",
            "runTimeModifiable": "true",
        },
    )

    # Mesh
    log_path = results_dir / "logs" / "adaptive_driver.log"
    if not (case_dir / "constant" / "polyMesh").exists():
        _run(["blockMesh"], cwd=case_dir, log_path=log_path)

    runner = detect_runner(case_dir)

    # Adaptive parameters
    t_start = float(get(cfg, "adaptive.start_time", 0.0))
    t_final = float(get(cfg, "adaptive.end_time", float(get(cfg, "openfoam.endTime", 1.0))))
    train_window = float(get(cfg, "adaptive.train_window", 0.2))
    validate_every = int(get(cfg, "adaptive.validate_every", 20))
    validate_steps = int(get(cfg, "adaptive.validate_steps", 1))
    tol = float(get(cfg, "adaptive.tol", 0.05))
    retrain_steps = int(get(cfg, "adaptive.retrain_steps", 20))
    max_cycles = int(get(cfg, "adaptive.max_cycles", 50))
    rom_horizon = float(get(cfg, "adaptive.rom_horizon", t_final))

    n_cells = n_cells_from_polymesh(case_dir)
    volumes = _ensure_cell_volumes(case_dir, time_name=_fmt_time(t_start), n_cells=n_cells, log_path=log_path)

    timeline_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    wall = {"fom_s": 0.0, "rom_s": 0.0, "io_s": 0.0}

    def record_timepoint(t: float, model: str, err_u: float | None = None, err_p: float | None = None) -> None:
        timeline_rows.append(
            {
                "time": float(t),
                "model": model,
                "err_U": "" if err_u is None else float(err_u),
                "err_p": "" if err_p is None else float(err_p),
            }
        )

    # 1) Training FOM segment
    t_train_end = min(t_final, t_start + train_window)
    set_control_dict_entries(
        case_dir / "system" / "controlDict",
        {"startFrom": "startTime", "startTime": str(t_start), "endTime": str(t_train_end)},
    )
    wall["fom_s"] += _run(runner, cwd=case_dir, log_path=log_path)

    times_train, U_train, p_train = _collect_window(case_dir, t_start, t_train_end)
    if U_train.shape[1] != n_cells or p_train.shape[1] != n_cells:
        raise RuntimeError(f"nCells mismatch: polyMesh has {n_cells}, snapshots have U:{U_train.shape[1]} p:{p_train.shape[1]}")

    # Persist training snapshots for reproducibility.
    snap_dir = out_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    np.save(snap_dir / "U.npy", U_train)
    np.save(snap_dir / "p.npy", p_train)
    save_meta(snap_dir / "meta.json", {"times": times_train.tolist(), "n_cells": int(n_cells), "source": "adaptive_fom"})

    rom = _train_rom(times_train, U_train, p_train, cfg=cfg, volumes=volumes)

    # Optional coefficient-based gate: reject ROM states that imply absurd Cd/Cl before starting a FOM burst.
    fg = get(cfg, "adaptive.force_gate", {}) or {}
    force_gate_enabled = bool(fg.get("enabled", False))
    force_gate_discard = float(fg.get("discard_frac", 0.30))
    u_inf_norm, _l_ref_norm, a_ref_norm = _read_control_dict_norm(case_dir)
    if u_inf_norm is None:
        u_inf_norm = float(get(cfg, "openfoam.U_inlet_mag", 1.0))
    if a_ref_norm is None:
        # Default for this project: 2D coefficients per unit span.
        a_ref_norm = 2.0

    cd_ref = float("nan")
    cl_rms_ref = float("nan")
    if force_gate_enabled:
        # Prefer a precomputed FOM-only baseline if available, else use the training segment.
        base_case = (results_dir / "fom_only" / "fom_only_case").resolve()
        base_case = base_case if base_case.exists() else case_dir
        base_fc = _find_latest_pp_file(base_case, "forceCoeffsIncompressible", ("forceCoeffs.dat", "coefficient.dat"))
        if base_fc is None:
            base_fc = _find_latest_pp_file(base_case, "forceCoeffs1", ("forceCoeffs.dat", "coefficient.dat"))
        if base_fc is not None:
            try:
                t_b, cd_b, cl_b = _read_forcecoeffs_min(base_fc)
                cd_ref, cl_rms_ref = _robust_baseline_refs(t_b, cd_b, cl_b, discard_frac=force_gate_discard)
            except Exception:
                pass

    t = float(times_train[-1])
    record_timepoint(t, "FOM")

    # 2) Adaptive loop
    cycle = 0
    while t + 1e-12 < t_final and cycle < max_cycles:
        cycle += 1

        dt_tol = max(1e-9, 0.51 * dt)
        t_act, _ = _resolve_time_dir(case_dir, t, tol=dt_tol)
        t = float(t_act)

        # Chunk end: validate_every steps ahead, capped by rom_horizon window.
        t_horizon_end = min(t_final, t + rom_horizon)
        t_chunk_end = min(t_horizon_end, t + validate_every * dt)
        t_val_end = min(t_final, t_chunk_end + validate_steps * dt)
        if t_val_end <= t:
            break

        # Build ROM initial condition from current (FOM) fields.
        _t0_act, U0, p0 = _read_fields_at_time(case_dir, t, n_cells=n_cells, tol=dt_tol)
        a_u0 = rom.pod_u.project(U0.reshape(1, -1))[0]
        p0_g = _remove_pressure_gauge(p0, volumes=volumes if (volumes is not None and volumes.shape == (n_cells,)) else None)
        a_p0 = rom.pod_p.project(p0_g.reshape(1, -1))[0]
        a0 = np.hstack([a_u0, a_p0])

        # Integrate ROM from t to t_val_end (includes validation horizon).
        t_eval = _make_time_grid(t, t_val_end, dt)

        t_rom0 = time.perf_counter()
        a_hat = _integrate_rom(t_eval, a0=a0, opinf=rom.opinf, cfg=cfg, z_bound_train=rom.z_bound_train)
        wall["rom_s"] += time.perf_counter() - t_rom0

        a_u_hat = a_hat[:, : rom.r_u]
        a_p_hat = a_hat[:, rom.r_u :]
        U_rec = rom.pod_u.reconstruct(a_u_hat).reshape(t_eval.size, n_cells, 3)
        p_rec = rom.pod_p.reconstruct(a_p_hat).reshape(t_eval.size, n_cells)

        # Safety check: if the ROM state at the start of the validation burst is clearly nonphysical,
        # don't start the FOM from it (it can crash the solver). Fall back to a FOM retrain segment.
        max_u_mag = float(get(cfg, "rom.max_u_mag", 10.0))
        max_p_abs_raw = get(cfg, "rom.max_p_abs", 50.0)
        max_p_abs = None if max_p_abs_raw is None else float(max_p_abs_raw)
        k_chunk = int(np.argmin(np.abs(t_eval - t_chunk_end)))
        u_max = float(np.max(np.linalg.norm(U_rec[k_chunk], axis=1)))
        p_chunk = p_rec[k_chunk]
        p_chunk_g = _remove_pressure_gauge(
            p_chunk,
            volumes=volumes if (volumes is not None and volumes.shape == (n_cells,)) else None,
        )
        p_abs = float(np.max(np.abs(p_chunk_g)))
        precheck_fail = (u_max > max_u_mag) or (max_p_abs is not None and p_abs > max_p_abs)
        if precheck_fail:
            try:
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(
                        "[rom_precheck] "
                        f"t={t_eval[k_chunk]:g} u_max={u_max:.6g} (limit {max_u_mag:.6g}) "
                        f"p_abs_gaugeRemoved={p_abs:.6g} (limit {('off' if max_p_abs is None else f'{max_p_abs:.6g}')})\n"
                    )
            except Exception:
                pass
            validation_rows.append(
                {
                    "time": float(t_eval[k_chunk]),
                    "err_U": float("nan"),
                    "err_p": float("nan"),
                    "abs_U": float("nan"),
                    "abs_p": float("nan"),
                    "accepted": 0.0,
                    "metric": "rom_precheck",
                }
            )

            _delete_time_dirs_after(case_dir, t)
            t_retrain_end = min(t_final, float(t) + retrain_steps * dt)
            set_control_dict_entries(
                case_dir / "system" / "controlDict",
                {"startFrom": "startTime", "startTime": _fmt_time(t), "endTime": _fmt_time(t_retrain_end)},
            )
            wall["fom_s"] += _run(runner, cwd=case_dir, log_path=log_path)

            t_retrain_act, _ = _resolve_time_dir(case_dir, t_retrain_end, tol=dt_tol)
            times_new, U_new, p_new = _collect_window(case_dir, float(t), float(t_retrain_act))
            times_train = np.concatenate([times_train, times_new[1:]])
            U_train = np.concatenate([U_train, U_new[1:]], axis=0)
            p_train = np.concatenate([p_train, p_new[1:]], axis=0)

            np.save(snap_dir / "U.npy", U_train)
            np.save(snap_dir / "p.npy", p_train)
            save_meta(
                snap_dir / "meta.json",
                {"times": times_train.tolist(), "n_cells": int(n_cells), "source": "adaptive_fom_append"},
            )

            rom = _train_rom(times_train, U_train, p_train, cfg=cfg, volumes=volumes)
            record_timepoint(float(t_retrain_act), "FOM")
            t = float(t_retrain_act)
            continue

        t_io0 = time.perf_counter()
        _write_rom_segment(case_dir, t_eval, U_rec=U_rec, p_rec=p_rec)
        wall["io_s"] += time.perf_counter() - t_io0

        if force_gate_enabled and _have_postprocess():
            # Gate at the start of the validation burst (t_chunk_end): if the ROM-written state yields
            # absurd coefficients, don't attempt a FOM restart from it.
            t_chunk_act, _ = _resolve_time_dir(case_dir, t_chunk_end, tol=dt_tol)
            ok, meta = _force_gate_check(
                case_dir=case_dir,
                time_name=_fmt_time(t_chunk_act),
                runner_log=log_path,
                u_inf=float(u_inf_norm),
                a_ref=float(a_ref_norm),
                cd_ref=float(cd_ref),
                cl_rms_ref=float(cl_rms_ref),
                cfg=cfg,
            )
            if not ok:
                try:
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(
                            "[rom_force_gate] "
                            f"t={t_chunk_act:g} Cd={meta.get('Cd', float('nan')):.6g} Cl={meta.get('Cl', float('nan')):.6g} "
                            f"Cd_ref={cd_ref:.6g} Cl_rms_ref={cl_rms_ref:.6g}\n"
                        )
                except Exception:
                    pass
                validation_rows.append(
                    {
                        "time": float(t_chunk_act),
                        "err_U": float("nan"),
                        "err_p": float("nan"),
                        "abs_U": float("nan"),
                        "abs_p": float("nan"),
                        "accepted": 0.0,
                        "metric": "rom_force_gate",
                    }
                )
                _delete_time_dirs_after(case_dir, t)
                t_retrain_end = min(t_final, float(t) + retrain_steps * dt)
                set_control_dict_entries(
                    case_dir / "system" / "controlDict",
                    {"startFrom": "startTime", "startTime": _fmt_time(t), "endTime": _fmt_time(t_retrain_end)},
                )
                wall["fom_s"] += _run(runner, cwd=case_dir, log_path=log_path)
                t_retrain_act, _ = _resolve_time_dir(case_dir, t_retrain_end, tol=dt_tol)
                times_new, U_new, p_new = _collect_window(case_dir, float(t), float(t_retrain_act))
                times_train = np.concatenate([times_train, times_new[1:]])
                U_train = np.concatenate([U_train, U_new[1:]], axis=0)
                p_train = np.concatenate([p_train, p_new[1:]], axis=0)
                np.save(snap_dir / "U.npy", U_train)
                np.save(snap_dir / "p.npy", p_train)
                save_meta(
                    snap_dir / "meta.json",
                    {"times": times_train.tolist(), "n_cells": int(n_cells), "source": "adaptive_fom_append"},
                )
                rom = _train_rom(times_train, U_train, p_train, cfg=cfg, volumes=volumes)
                record_timepoint(float(t_retrain_act), "FOM")
                t = float(t_retrain_act)
                continue

        # Validation burst: run FOM from t_chunk_end -> t_val_end
        if t_val_end <= t_chunk_end + 1e-12:
            # No room for a validation step (end of simulation). Keep ROM state.
            record_timepoint(t_val_end, "ROM")
            t = t_val_end
            continue

        # Align the validation burst start time to an existing directory (ROM-wrote times).
        t_chunk_act, _ = _resolve_time_dir(case_dir, t_chunk_end, tol=dt_tol)
        set_control_dict_entries(
            case_dir / "system" / "controlDict",
            {"startFrom": "startTime", "startTime": _fmt_time(t_chunk_act), "endTime": _fmt_time(t_val_end)},
        )
        wall["fom_s"] += _run(runner, cwd=case_dir, log_path=log_path)

        # Align comparison to the actual output time directory produced by the FOM burst.
        t_val_act, U_fom, p_fom = _read_fields_at_time(case_dir, t_val_end, n_cells=n_cells, tol=dt_tol)
        a_val = _interp_coeffs(t_eval, a_hat, t_val_act)
        a_u_val = a_val[: rom.r_u][None, :]
        a_p_val = a_val[rom.r_u :][None, :]
        U_rom = rom.pod_u.reconstruct(a_u_val).reshape(n_cells, 3)
        p_rom = rom.pod_p.reconstruct(a_p_val).reshape(n_cells,)

        # Gauge-invariant pressure comparison (incompressible kinematic pressure p is defined up to a constant).
        p_fom_g = _remove_pressure_gauge(p_fom, volumes=volumes if (volumes is not None and volumes.shape == (n_cells,)) else None)
        p_rom_g = _remove_pressure_gauge(p_rom, volumes=volumes if (volumes is not None and volumes.shape == (n_cells,)) else None)
        if volumes is not None and volumes.shape == (n_cells,):
            err_u = l2_rel_volweighted(U_rom, U_fom, volumes=volumes)
            err_p = l2_rel_volweighted(p_rom_g, p_fom_g, volumes=volumes)
            abs_u = l2_abs_volweighted(U_rom, U_fom, volumes=volumes)
            abs_p = l2_abs_volweighted(p_rom_g, p_fom_g, volumes=volumes)
            metric_name = "vol_weighted"
        else:
            err_u = l2_rel(U_rom, U_fom)
            err_p = l2_rel(p_rom_g, p_fom_g)
            abs_u = l2_abs(U_rom, U_fom)
            abs_p = l2_abs(p_rom_g, p_fom_g)
            metric_name = "unweighted"
        validation_rows.append(
            {
                "time": float(t_val_act),
                "err_U": float(err_u),
                "err_p": float(err_p),
                "abs_U": float(abs_u),
                "abs_p": float(abs_p),
                "accepted": float(max(err_u, err_p) <= tol),
                "metric": metric_name,
            }
        )

        accepted = (max(err_u, err_p) <= tol)
        if accepted:
            # Accept ROM segment but "correct" the state to the validated FOM field at t_val_end.
            # This makes subsequent ROM rollouts start from a FOM state (time-consistent and reduces drift).
            record_timepoint(float(t_val_act), "ROM", err_u, err_p)
            t = float(t_val_act)
            continue

        # Reject: switch to FOM for retrain_steps, append snapshots, retrain ROM.
        t_retrain_end = min(t_final, float(t_val_act) + retrain_steps * dt)
        set_control_dict_entries(
            case_dir / "system" / "controlDict",
            {"startFrom": "startTime", "startTime": _fmt_time(t_val_act), "endTime": _fmt_time(t_retrain_end)},
        )
        wall["fom_s"] += _run(runner, cwd=case_dir, log_path=log_path)

        t_retrain_act, _ = _resolve_time_dir(case_dir, t_retrain_end, tol=dt_tol)
        times_new, U_new, p_new = _collect_window(case_dir, float(t_val_act), float(t_retrain_act))
        # Append, dropping duplicate first entry at t_val_end.
        times_train = np.concatenate([times_train, times_new[1:]])
        U_train = np.concatenate([U_train, U_new[1:]], axis=0)
        p_train = np.concatenate([p_train, p_new[1:]], axis=0)

        np.save(snap_dir / "U.npy", U_train)
        np.save(snap_dir / "p.npy", p_train)
        save_meta(snap_dir / "meta.json", {"times": times_train.tolist(), "n_cells": int(n_cells), "source": "adaptive_fom_append"})

        rom = _train_rom(times_train, U_train, p_train, cfg=cfg, volumes=volumes)
        record_timepoint(float(t_retrain_act), "FOM", err_u, err_p)
        t = float(t_retrain_act)

    # Restore global time bounds in controlDict so post-processing tools can see the full run.
    # The adaptive loop repeatedly restarts from intermediate times, leaving startTime/endTime
    # pointing at the last validation/retrain segment.
    try:
        set_control_dict_entries(
            case_dir / "system" / "controlDict",
            {"startFrom": "startTime", "startTime": str(t_start), "endTime": str(t_final)},
        )
    except Exception:
        pass

    # Write outputs
    with (out_dir / "timeline.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["time", "model", "err_U", "err_p"])
        w.writeheader()
        for r in timeline_rows:
            w.writerow(r)

    with (out_dir / "validation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["time", "err_U", "err_p", "abs_U", "abs_p", "accepted", "metric"])
        w.writeheader()
        for r in validation_rows:
            w.writerow(r)

    # Write a unified error time series for the pipeline/report.
    metrics_dir = results_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    err_ts = metrics_dir / "error_timeseries.csv"
    with err_ts.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["time", "err_U", "err_p", "abs_U", "abs_p", "accepted", "metric", "source"])
        w.writeheader()
        for r in validation_rows:
            w.writerow(
                {
                    "time": r["time"],
                    "err_U": r["err_U"],
                    "err_p": r["err_p"],
                    "abs_U": r.get("abs_U", ""),
                    "abs_p": r.get("abs_p", ""),
                    "accepted": r["accepted"],
                    "metric": r.get("metric", ""),
                    "source": "adaptive_validation",
                }
            )

    (out_dir / "speed.json").write_text(json.dumps(wall, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (out_dir / "speed.csv").open("w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=["component", "seconds"])
        wcsv.writeheader()
        for k in ["fom_s", "rom_s", "io_s"]:
            wcsv.writerow({"component": k, "seconds": wall[k]})

    if validation_rows:
        t_v = np.array([r["time"] for r in validation_rows], dtype=float)
        e_u = np.array([r["err_U"] for r in validation_rows], dtype=float)
        e_p = np.array([r["err_p"] for r in validation_rows], dtype=float)
        a_u = np.array([r.get("abs_U", float("nan")) for r in validation_rows], dtype=float)
        a_p = np.array([r.get("abs_p", float("nan")) for r in validation_rows], dtype=float)
        save_error_plot(t_v, e_u, out_dir / "errU_validate.png", "Validation error (U)")
        save_error_plot(t_v, e_p, out_dir / "errp_validate.png", "Validation error (p)")
        # Combined figure with both relative (top) and absolute (bottom) curves.
        save_error_plot_two_rel_abs(
            t_v,
            e_u,
            e_p,
            a_u,
            a_p,
            out_dir / "err_validate.png",
            "Validation error (U and p)",
            rel_ylabel="L2 relative error",
            abs_ylabel="L2 absolute error",
        )

    if timeline_rows:
        t_t = np.array([r["time"] for r in timeline_rows], dtype=float)
        m_t = [str(r["model"]) for r in timeline_rows]
        save_model_timeline_plot(t_t, m_t, out_dir / "timeline.png")
        save_speed_bar(wall, out_dir / "speed.png")

    # Print a concise debug summary (used for diagnosing switching behavior).
    if validation_rows:
        e_u = np.array([r["err_U"] for r in validation_rows], dtype=float)
        e_p = np.array([r["err_p"] for r in validation_rows], dtype=float)
        accepted_n = int(sum(1 for r in validation_rows if float(r.get("accepted", 0.0)) >= 0.5))
        print(
            "[adaptive_driver] debug:",
            f"tol={tol:g}",
            f"train_window={train_window:g}",
            f"validate_every={validate_every}",
            f"validate_steps={validate_steps}",
            f"dt={dt:g}",
            f"metric={validation_rows[0].get('metric','')}",
            f"validation_n={len(validation_rows)}",
            f"accepted_n={accepted_n}",
            f"errU_mean={float(np.mean(e_u)):.3g}",
            f"errU_max={float(np.max(e_u)):.3g}",
            f"errp_mean={float(np.mean(e_p)):.3g}",
            f"errp_max={float(np.max(e_p)):.3g}",
        )
    else:
        print("[adaptive_driver] debug:", f"tol={tol:g}", "validation_n=0")

    print(f"[adaptive_driver] Wrote: {out_dir}/timeline.csv {out_dir}/validation.csv {out_dir}/speed.json")
    print(f"[adaptive_driver] Case: {case_dir}")


if __name__ == "__main__":
    main()
