from __future__ import annotations

import argparse
import json
import re
import subprocess
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from config import load_yaml
from foam_io import prepare_rom_case, write_time_fields
from foam_ascii import n_cells_from_polymesh, read_internal_field
from metrics import (
    l2_abs_time_series,
    l2_abs_time_series_volweighted,
    l2_rel_time_series,
    l2_rel_time_series_volweighted,
)
from opinf import fit_quadratic
from pod import fit_pod
from plots import save_energy_plot, save_error_plot, save_error_plot_two_rel_abs, save_singular_values_plot
from rom_integrate import integrate_fixed_step, integrate_ivp


def _load_snapshots(results_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    snap_dir = results_dir / "snapshots"
    U = np.load(snap_dir / "U.npy")
    p = np.load(snap_dir / "p.npy")
    meta = json.loads((snap_dir / "meta.json").read_text(encoding="utf-8"))
    if U.ndim != 3 or U.shape[2] != 3:
        raise ValueError(f"Unexpected U shape: {U.shape}")
    if p.ndim != 2:
        raise ValueError(f"Unexpected p shape: {p.shape}")
    if U.shape[0] != p.shape[0]:
        raise ValueError("Snapshot count mismatch U vs p")
    if U.shape[1] != p.shape[1]:
        raise ValueError("nCells mismatch U vs p")
    if not np.all(np.isfinite(U)) or not np.all(np.isfinite(p)):
        raise ValueError("NaN/Inf in snapshots")
    return U.astype(float), p.astype(float), meta


def _have_postprocess() -> bool:
    return shutil.which("postProcess") is not None


def _mean_subtract_pressure_series(p: np.ndarray, volumes: np.ndarray | None) -> np.ndarray:
    """
    Remove the (volume-)mean pressure at each time:
        p'(t) = p(t) - <p(t)>.

    This avoids gauge/offset effects in relative pressure error metrics.
    """
    P = np.asarray(p, dtype=float)
    if P.ndim != 2:
        raise ValueError("Expected p as (nSnap, nCells)")
    if volumes is None:
        mean = np.mean(P, axis=1)
    else:
        v = np.asarray(volumes, dtype=float).reshape(-1)
        if v.shape[0] != P.shape[1]:
            raise ValueError("volumes length mismatch for pressure series")
        w = v / float(np.sum(v))
        mean = np.sum(P * w[None, :], axis=1)
    return P - mean[:, None]


def _read_label_list(path: Path) -> np.ndarray:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    # Strip block and line comments.
    txt = re.sub(r"/\*.*?\*/", "", txt, flags=re.S)
    txt = re.sub(r"//.*?$", "", txt, flags=re.M)
    m = re.search(r"\n\s*(\d+)\s*\(\s*(.*?)\s*\)\s*$", txt, flags=re.S)
    if not m:
        raise ValueError(f"Could not parse labelList from {path}")
    body = m.group(2)
    return np.fromstring(body, sep=" ", dtype=np.int64)


def _pressure_jump_series(case_dir: Path, p_series: np.ndarray) -> np.ndarray | None:
    """
    Proxy for grad(p): pressure jump across internal faces:
        dp_f = p(neighbour) - p(owner)  for each internal face.
    Returns (nSnap, nInternalFaces) or None if polyMesh connectivity unavailable.
    """
    poly = case_dir / "constant" / "polyMesh"
    owner_path = poly / "owner"
    neigh_path = poly / "neighbour"
    if not owner_path.exists() or not neigh_path.exists():
        return None
    try:
        owner = _read_label_list(owner_path)
        neigh = _read_label_list(neigh_path)
    except Exception:
        return None

    n_internal = int(neigh.size)
    if n_internal < 1 or owner.size < n_internal:
        return None
    owner_int = owner[:n_internal].astype(np.int64, copy=False)
    neigh = neigh.astype(np.int64, copy=False)

    P = np.asarray(p_series, dtype=float)
    if P.ndim != 2:
        return None
    # Bounds safety.
    if int(np.max(owner_int)) >= P.shape[1] or int(np.max(neigh)) >= P.shape[1]:
        return None
    return P[:, neigh] - P[:, owner_int]


def _ensure_cell_volumes(case_dir: Path, time_name: str = "0") -> np.ndarray | None:
    """
    Try to obtain cell volumes as an OpenFOAM volScalarField (typically named 'V')
    by running `postProcess -func writeCellVolumes`. Returns None if unavailable.
    """
    try:
        n_cells = int(n_cells_from_polymesh(case_dir))
    except Exception:
        return None

    td = case_dir / time_name
    if not td.exists():
        return None

    candidates = ["Vc", "V", "cellVolumes"]
    for c in candidates:
        fp = td / c
        if fp.exists():
            try:
                v = read_internal_field(fp, n_cells=n_cells)
                if v.shape == (n_cells,):
                    return v
            except Exception:
                pass

    if not _have_postprocess():
        return None

    # Generate volumes and retry.
    subprocess.run(["postProcess", "-func", "writeCellVolumes", "-time", time_name], cwd=str(case_dir), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for c in candidates:
        fp = td / c
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
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    case_dir = Path(cfg["case"]["work_dir"]).resolve()
    results_dir = Path(cfg["case"]["results_dir"]).resolve()
    out_dir = results_dir / "rom"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = results_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    U_snap, p_snap, meta = _load_snapshots(results_dir)
    times = np.asarray(meta["times"], dtype=float)
    n_snap, n_cells = U_snap.shape[0], U_snap.shape[1]

    U_flat = U_snap.reshape(n_snap, n_cells * 3)
    p_flat = p_snap.reshape(n_snap, n_cells)

    rom_cfg = cfg.get("rom", {})
    r_u = int(rom_cfg.get("r_u", rom_cfg.get("r", 8)))
    r_p = int(rom_cfg.get("r_p", rom_cfg.get("r", 6)))
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

    model = fit_quadratic(times, a, ridge=ridge, normalize=True)

    # ROM-only forecast over the same time grid.
    dt = float(times[1] - times[0])
    methods = [str(rom_cfg.get("integrator", "BDF")), "Radau", "RK45"]
    res = None
    for meth in methods:
        res_try = integrate_ivp(model.rhs, times, a0=a[0], rtol=1e-6, atol=1e-9, method=meth, max_step=dt)
        if res_try.success:
            res = res_try
            break
    if res is None:
        res = res_try
    if not res.success:
        # Fallback: fixed-step integration with clipping in normalized coordinates.
        z_train = (a - model.mu[None, :]) / model.sigma[None, :]
        z_max = np.max(np.abs(z_train), axis=0)
        clip_factor = float(rom_cfg.get("clip_factor", 1.25))
        z_bound = np.where(z_max > 0, clip_factor * z_max, 5.0)

        def rhs_z(_t: float, z: np.ndarray) -> np.ndarray:
            z = np.asarray(z, dtype=float).reshape(-1)
            return model.c + model.A @ z + model.H @ np.kron(z, z)

        z0 = (a[0] - model.mu) / model.sigma
        res_z = integrate_fixed_step(rhs_z, times, z0, clip=(-z_bound, z_bound))
        if not res_z.success:
            raise RuntimeError(f"ROM integration failed (solve_ivp: {res.message}; RK4+clip: {res_z.message})")
        a_hat = model.mu[None, :] + model.sigma[None, :] * res_z.a
        res = res_z
    else:
        a_hat = res.a
    a_u_hat = a_hat[:, :r_u_eff]
    a_p_hat = a_hat[:, r_u_eff:]

    U_rec = pod_u.reconstruct(a_u_hat).reshape(n_snap, n_cells, 3)
    p_rec = pod_p.reconstruct(a_p_hat).reshape(n_snap, n_cells)

    # Error metrics vs FOM snapshots.
    volumes = _ensure_cell_volumes(case_dir, time_name=str(meta.get("template_time", "0")))
    # Pressure: compute errors on mean-subtracted pressure to avoid gauge/offset artifacts.
    p_rec_err = _mean_subtract_pressure_series(p_rec, volumes=volumes)
    p_snap_err = _mean_subtract_pressure_series(p_snap, volumes=volumes)
    if volumes is not None and volumes.shape == (n_cells,):
        np.save(metrics_dir / "cell_volumes.npy", volumes)
        err_U = l2_rel_time_series_volweighted(U_rec, U_snap, volumes=volumes)
        err_p = l2_rel_time_series_volweighted(p_rec_err, p_snap_err, volumes=volumes)
        abs_U = l2_abs_time_series_volweighted(U_rec, U_snap, volumes=volumes)
        abs_p = l2_abs_time_series_volweighted(p_rec_err, p_snap_err, volumes=volumes)
        metric_name = "vol_weighted"
    else:
        err_U = l2_rel_time_series(U_rec, U_snap)
        err_p = l2_rel_time_series(p_rec_err, p_snap_err)
        abs_U = l2_abs_time_series(U_rec, U_snap)
        abs_p = l2_abs_time_series(p_rec_err, p_snap_err)
        metric_name = "unweighted"

    # Optional alternate pressure metric: internal-face pressure jumps (proxy for grad(p)).
    dp_fom = _pressure_jump_series(case_dir, p_series=p_snap)
    dp_rom = _pressure_jump_series(case_dir, p_series=p_rec)
    err_dp = abs_dp = None
    if dp_fom is not None and dp_rom is not None and dp_fom.shape == dp_rom.shape and dp_fom.shape[0] == n_snap:
        err_dp = l2_rel_time_series(dp_rom, dp_fom)
        abs_dp = l2_abs_time_series(dp_rom, dp_fom)
        np.save(out_dir / "err_p_jump.npy", err_dp)
        np.save(out_dir / "abs_p_jump.npy", abs_dp)
        save_error_plot(times, err_dp, out_dir / "err_p_jump.png", "ROM vs FOM error (pressure jump, proxy for grad(p))")
        save_error_plot(times, abs_dp, out_dir / "abs_p_jump.png", "ROM vs FOM absolute error (pressure jump)", ylabel="L2 absolute error")

    # Store a reusable error time series CSV for the report/pipeline.
    csv_path = metrics_dir / "error_timeseries_rom_only.csv"
    csv_path.write_text(
        "time,err_U,err_p,abs_U,abs_p,metric,source\n"
        + "\n".join(
            f"{t:g},{eu:.16g},{ep:.16g},{au:.16g},{ap:.16g},{metric_name},rom_only"
            for t, eu, ep, au, ap in zip(times, err_U, err_p, abs_U, abs_p, strict=False)
        )
        + "\n",
        encoding="utf-8",
    )

    np.save(out_dir / "a_fom.npy", a)
    np.save(out_dir / "a_rom.npy", a_hat)
    np.save(out_dir / "err_U.npy", err_U)
    np.save(out_dir / "err_p.npy", err_p)
    np.save(out_dir / "abs_U.npy", abs_U)
    np.save(out_dir / "abs_p.npy", abs_p)
    (out_dir / "rom_meta.json").write_text(
        json.dumps(
            {
                "r_u": r_u_eff,
                "r_p": r_p_eff,
                "ridge": ridge,
                "times": meta["times"],
                "n_cells": int(n_cells),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    # Plots.
    save_singular_values_plot(pod_u.svals, out_dir / "svd_U.png", "POD singular values (U)")
    save_singular_values_plot(pod_p.svals, out_dir / "svd_p.png", "POD singular values (p)")
    save_energy_plot(pod_u.energy, out_dir / "energy_U.png", "Cumulative POD energy (U)")
    save_energy_plot(pod_p.energy, out_dir / "energy_p.png", "Cumulative POD energy (p)")
    save_error_plot(times, err_U, out_dir / "err_U.png", "ROM vs FOM error (U)")
    save_error_plot(times, err_p, out_dir / "err_p.png", "ROM vs FOM error (p)")
    save_error_plot(times, abs_U, out_dir / "abs_U.png", "ROM vs FOM absolute error (U)", ylabel="L2 absolute error")
    save_error_plot(times, abs_p, out_dir / "abs_p.png", "ROM vs FOM absolute error (p)", ylabel="L2 absolute error")

    # Avoid the t=0 point for the combined relative/absolute plot (often trivial/ill-conditioned).
    t_plot = times[1:] if times.size > 1 else times
    err_U_plot = err_U[1:] if err_U.size > 1 else err_U
    err_p_plot = err_p[1:] if err_p.size > 1 else err_p
    abs_U_plot = abs_U[1:] if abs_U.size > 1 else abs_U
    abs_p_plot = abs_p[1:] if abs_p.size > 1 else abs_p
    save_error_plot_two_rel_abs(
        t_plot,
        err_U_plot,
        err_p_plot,
        abs_U_plot,
        abs_p_plot,
        out_dir / "err_rom_only.png",
        "ROM-only forecast error (U and p)",
        rel_ylabel="L2 relative error",
        abs_ylabel="L2 absolute error",
    )

    # Write a ROM OpenFOAM case with reconstructed fields.
    rom_case_dir = results_dir / "rom_case"
    prepare_rom_case(case_dir, rom_case_dir)

    write_stride = int(rom_cfg.get("write_stride", 10))
    template_time = str(rom_cfg.get("template_time", "0"))
    for i in range(0, n_snap, max(1, write_stride)):
        tname = f"{times[i]:g}"
        if tname == template_time:
            # Keep the template directory intact.
            continue
        write_time_fields(rom_case_dir, tname, U=U_rec[i], p=p_rec[i], template_time=template_time)

    print(f"[build_rom_low] Wrote ROM outputs to: {out_dir}")
    print(f"[build_rom_low] Wrote ROM OpenFOAM case to: {rom_case_dir}")


if __name__ == "__main__":
    main()
