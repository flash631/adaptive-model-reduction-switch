from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from config import load_yaml
from foam_ascii import n_cells_from_polymesh, read_internal_field, save_meta


def _is_float_dirname(name: str) -> bool:
    try:
        float(name)
        return True
    except Exception:
        return False


def _list_time_dirs(case_dir: Path) -> list[Path]:
    times = []
    for p in case_dir.iterdir():
        if p.is_dir() and _is_float_dirname(p.name):
            times.append(p)
    return sorted(times, key=lambda p: float(p.name))


def _detect_fields_in_dataset(ds: Any) -> tuple[str, str]:
    # Heuristics for foamToVTK outputs.
    field_names = set()
    try:
        field_names |= set(getattr(ds, "cell_data", {}).keys())
    except Exception:
        pass
    try:
        field_names |= set(getattr(ds, "point_data", {}).keys())
    except Exception:
        pass

    u_name = "U" if "U" in field_names else next((n for n in field_names if n.lower() == "u"), "")
    p_name = "p" if "p" in field_names else next((n for n in field_names if n.lower() in {"p", "p_rgh"}), "")
    if not u_name or not p_name:
        raise ValueError(f"Could not detect U/p fields in dataset. Available: {sorted(field_names)}")
    return u_name, p_name


def _read_from_vtk(vtk_root: Path, fields: list[str], stride: int, max_snapshots: int) -> tuple[np.ndarray, np.ndarray, dict]:
    try:
        import pyvista as pv  # type: ignore
    except Exception as exc:
        raise RuntimeError("pyvista not installed") from exc

    if not vtk_root.exists():
        raise FileNotFoundError(f"VTK directory not found: {vtk_root}")

    # Common foamToVTK pattern: VTK/<time>/*.{vtk,vtu,vtm}
    time_dirs = [p for p in vtk_root.iterdir() if p.is_dir() and _is_float_dirname(p.name)]
    time_dirs = sorted(time_dirs, key=lambda p: float(p.name))

    datasets: list[tuple[float, Any]] = []
    for td in time_dirs[:: max(1, stride)]:
        # Prefer internalMesh if present
        candidates = []
        for ext in ("vtu", "vtk", "vtm"):
            candidates += list(td.glob(f"*internalMesh*.{ext}"))
        if not candidates:
            for ext in ("vtu", "vtk", "vtm"):
                candidates += list(td.glob(f"*.{ext}"))
        if not candidates:
            continue

        mesh = pv.read(candidates[0])
        if hasattr(mesh, "combine"):
            try:
                mesh = mesh.combine()
            except Exception:
                pass
        datasets.append((float(td.name), mesh))
        if len(datasets) >= max_snapshots:
            break

    if not datasets:
        raise RuntimeError(f"No readable VTK datasets found under {vtk_root}")

    U_list: list[np.ndarray] = []
    p_list: list[np.ndarray] = []
    times: list[float] = []

    for t, ds in datasets:
        u_name, p_name = _detect_fields_in_dataset(ds)
        # Prefer cell data for ROM snapshots.
        if u_name in getattr(ds, "cell_data", {}):
            U = np.asarray(ds.cell_data[u_name], dtype=float)
        else:
            U = np.asarray(ds.point_data[u_name], dtype=float)
        if p_name in getattr(ds, "cell_data", {}):
            p = np.asarray(ds.cell_data[p_name], dtype=float)
        else:
            p = np.asarray(ds.point_data[p_name], dtype=float)

        if U.ndim != 2 or U.shape[1] != 3:
            raise ValueError(f"Unexpected U shape at t={t}: {U.shape}")
        if p.ndim != 1:
            p = p.reshape(-1)

        if not np.all(np.isfinite(U)) or not np.all(np.isfinite(p)):
            raise ValueError(f"Found NaN/Inf at t={t}")

        U_list.append(U)
        p_list.append(p)
        times.append(t)

    n_cells = U_list[0].shape[0]
    if any(U.shape[0] != n_cells for U in U_list):
        raise ValueError("Inconsistent number of cells across VTK snapshots")
    if any(p.shape[0] != n_cells for p in p_list):
        raise ValueError("Inconsistent p size across VTK snapshots")

    U_mat = np.stack(U_list, axis=0)  # (n_snap, n_cells, 3)
    p_mat = np.stack(p_list, axis=0)  # (n_snap, n_cells)

    meta = {
        "source": "vtk",
        "vtk_root": str(vtk_root),
        "times": times,
        "n_cells": n_cells,
    }
    return U_mat, p_mat, meta


def _read_from_openfoam(case_dir: Path, fields: list[str], stride: int, max_snapshots: int, min_time: float) -> tuple[np.ndarray, np.ndarray, dict]:
    n_cells = n_cells_from_polymesh(case_dir)
    time_dirs = [td for td in _list_time_dirs(case_dir) if float(td.name) >= min_time]
    time_dirs = time_dirs[:: max(1, stride)]
    if max_snapshots > 0:
        time_dirs = time_dirs[:max_snapshots]
    if not time_dirs:
        raise RuntimeError(f"No time directories found in {case_dir} at/after min_time={min_time}")

    U_list: list[np.ndarray] = []
    p_list: list[np.ndarray] = []
    times: list[float] = []
    for td in time_dirs:
        U = read_internal_field(td / "U", n_cells=n_cells)
        p = read_internal_field(td / "p", n_cells=n_cells)
        if U.shape != (n_cells, 3):
            raise ValueError(f"{td}/U: expected {(n_cells,3)}, got {U.shape}")
        if p.shape != (n_cells,):
            raise ValueError(f"{td}/p: expected {(n_cells,)}, got {p.shape}")
        if not np.all(np.isfinite(U)) or not np.all(np.isfinite(p)):
            raise ValueError(f"Found NaN/Inf at time {td.name}")
        U_list.append(U)
        p_list.append(p)
        times.append(float(td.name))

    U_mat = np.stack(U_list, axis=0)
    p_mat = np.stack(p_list, axis=0)
    meta = {
        "source": "openfoam_ascii",
        "case_dir": str(case_dir),
        "times": times,
        "n_cells": n_cells,
    }
    return U_mat, p_mat, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config (e.g. rom/configs/low.yaml)")
    ap.add_argument("--case", default=None, help="Override case directory (default from config)")
    ap.add_argument("--out", default=None, help="Output dir (default from config)")
    ap.add_argument("--prefer-vtk", action="store_true", help="Prefer VTK (pyvista) if available")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    case_dir = Path(args.case or cfg["case"]["work_dir"]).resolve()
    out_dir = Path(args.out or cfg["case"]["results_dir"]).resolve() / "snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)

    fields = list(cfg.get("export", {}).get("snapshot_fields", ["U", "p"]))
    stride = int(cfg.get("export", {}).get("time_stride", 1))
    max_snaps = int(cfg.get("export", {}).get("max_snapshots", 200))
    min_time = float(cfg.get("export", {}).get("min_time", 0.0))

    vtk_root = case_dir / "VTK"
    used_vtk = False

    U_mat: np.ndarray
    p_mat: np.ndarray
    meta: dict

    if args.prefer_vtk:
        try:
            U_mat, p_mat, meta = _read_from_vtk(vtk_root, fields=fields, stride=stride, max_snapshots=max_snaps)
            used_vtk = True
        except Exception as exc:
            print(f"[snapshot_export] VTK read unavailable ({exc}); falling back to OpenFOAM ASCII parsing.")

    if not used_vtk:
        U_mat, p_mat, meta = _read_from_openfoam(
            case_dir, fields=fields, stride=stride, max_snapshots=max_snaps, min_time=min_time
        )

    # Save as snapshots along time axis (n_snap, ...)
    np.save(out_dir / "U.npy", U_mat)
    np.save(out_dir / "p.npy", p_mat)
    save_meta(out_dir / "meta.json", meta | {"fields": fields})

    # Lightweight sanity checks for later stages.
    if U_mat.shape[0] != p_mat.shape[0]:
        raise ValueError("Snapshot count mismatch between U and p")
    if U_mat.shape[1] != meta["n_cells"]:
        raise ValueError("nCells mismatch vs stored meta")
    if not np.all(np.isfinite(U_mat)) or not np.all(np.isfinite(p_mat)):
        raise ValueError("NaN/Inf in saved snapshots")

    print(f"[snapshot_export] Wrote: {out_dir}/U.npy {U_mat.shape}, {out_dir}/p.npy {p_mat.shape}")
    print(f"[snapshot_export] Wrote: {out_dir}/meta.json (source={meta['source']})")


if __name__ == "__main__":
    main()

