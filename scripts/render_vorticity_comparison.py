#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np


def _run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed (rc={p.returncode}): {' '.join(cmd)}\n{p.stdout}")


def _ensure_fields(case_dir: Path, time_name: str) -> None:
    # Ensure vorticity and cell centres exist (best-effort; will overwrite if present).
    _run(["postProcess", "-case", str(case_dir), "-func", "vorticity", "-time", str(time_name)])
    _run(["postProcess", "-case", str(case_dir), "-func", "writeCellCentres", "-time", str(time_name)])


def _nearest_time_dir(case_dir: Path, t_req: float) -> str:
    times: list[float] = []
    for p in case_dir.iterdir():
        if not p.is_dir():
            continue
        try:
            times.append(float(p.name))
        except Exception:
            continue
    if not times:
        raise RuntimeError(f"No time directories found under {case_dir}")
    t = float(times[int(np.argmin(np.abs(np.asarray(times) - float(t_req))))])
    for p in case_dir.iterdir():
        if p.is_dir():
            try:
                if abs(float(p.name) - t) < 1e-12:
                    return p.name
            except Exception:
                continue
    return f"{t:g}"


def _robust_absmax(a: np.ndarray, pct: float = 99.0) -> float:
    x = np.asarray(a, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 8:
        return float(np.nanmax(np.abs(x))) if x.size else 1.0
    v = float(np.nanpercentile(np.abs(x), pct))
    if not np.isfinite(v) or v <= 0:
        v = float(np.nanmax(np.abs(x))) if x.size else 1.0
    return float(v) if v > 0 else 1.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a FOM vs ROM spanwise vorticity comparison (no pvpython).")
    ap.add_argument("--fom-case", default="cases/low", help="FOM case directory")
    ap.add_argument("--rom-case", default="results/low/rom_case", help="ROM case directory")
    ap.add_argument("--time", type=float, default=0.5, help="Requested time (nearest available time dir is used)")
    ap.add_argument("--out", default="report/figures/fom_vs_rom_vortZ_mid_low.png", help="Output PNG path")
    ap.add_argument("--patch", default="cylinder", help="Cylinder patch name (for hole masking)")
    ap.add_argument("--left-title", default="FOM", help="Left panel title")
    ap.add_argument("--right-title", default="ROM", help="Right panel title")
    ap.add_argument("--cmap", default="coolwarm", help="Matplotlib colormap name")
    ap.add_argument("--levels", type=int, default=160, help="Filled-contour levels (higher looks smoother)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    fom_case = (root / args.fom_case).resolve()
    rom_case = (root / args.rom_case).resolve()
    time_fom = _nearest_time_dir(fom_case, float(args.time))
    time_rom = _nearest_time_dir(rom_case, float(args.time))
    out = (root / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    # Import small OpenFOAM ASCII readers and geometry helpers from the project.
    sys.path.insert(0, str((root / "rom" / "python").resolve()))
    from foam_ascii import n_cells_from_polymesh, read_internal_field  # type: ignore
    from geometry import cylinder_geometry, estimate_cylinder_diameter  # type: ignore

    if subprocess.run(["bash", "-lc", "command -v postProcess"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
        raise RuntimeError("OpenFOAM postProcess not available (source OpenFOAM-13 before running).")

    # Ensure required derived fields exist for both cases.
    _ensure_fields(fom_case, time_name=time_fom)
    _ensure_fields(rom_case, time_name=time_rom)

    def load(case_dir: Path, time_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = n_cells_from_polymesh(case_dir)
        C = read_internal_field(case_dir / time_name / "C", n_cells=n)
        w = read_internal_field(case_dir / time_name / "vorticity", n_cells=n)
        if C.shape != (n, 3) or w.shape != (n, 3):
            raise ValueError(f"Unexpected field shapes under {case_dir}/{time_name}: C{C.shape} vorticity{w.shape}")
        x = C[:, 0].astype(float, copy=False)
        y = C[:, 1].astype(float, copy=False)
        z = C[:, 2].astype(float, copy=False)
        wz = w[:, 2].astype(float, copy=False)  # spanwise vorticity
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(wz)
        x, y, z, wz = x[m], y[m], z[m], wz[m]
        z0 = float(np.median(z))
        z_span = float(np.max(z) - np.min(z)) if z.size else 0.0
        tol = max(1e-9, 0.05 * max(z_span, 1e-6))
        ms = np.abs(z - z0) <= tol
        return x[ms], y[ms], z[ms], wz[ms]

    xF, yF, _zF, wzF = load(fom_case, time_fom)
    xR, yR, _zR, wzR = load(rom_case, time_rom)

    # Robust shared color scale (avoid single-cell spikes).
    absmax = max(_robust_absmax(wzF, pct=99.0), _robust_absmax(wzR, pct=99.0))
    absmax = float(absmax) if absmax > 0 else 1.0

    # Shared view limits (union bounds).
    xmin = float(min(np.min(xF), np.min(xR)))
    xmax = float(max(np.max(xF), np.max(xR)))
    ymin = float(min(np.min(yF), np.min(yR)))
    ymax = float(max(np.max(yF), np.max(yR)))

    # Cylinder hole mask (triangles whose centroid is inside the cylinder).
    circle = estimate_cylinder_diameter(fom_case, patch_name=str(args.patch))
    if circle is not None and math.isfinite(circle.radius) and circle.radius > 0:
        cx, cy = circle.center_xy
        r0 = float(circle.radius)
    else:
        geom = cylinder_geometry(fom_case, patch_name=str(args.patch))
        cx, cy = 0.0, 0.0
        r0 = 0.5 * float(geom.get("D", 2.0))
    r_mask = 1.03 * float(r0)

    def tri_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        import matplotlib.tri as mtri  # type: ignore

        tri = mtri.Triangulation(x, y)
        t = tri.triangles
        xc = (x[t[:, 0]] + x[t[:, 1]] + x[t[:, 2]]) / 3.0
        yc = (y[t[:, 0]] + y[t[:, 1]] + y[t[:, 2]]) / 3.0
        inside = (xc - cx) ** 2 + (yc - cy) ** 2 < (r_mask**2)
        tri.set_mask(inside)
        return tri

    triF = tri_mask(xF, yF)
    triR = tri_mask(xR, yR)

    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.colors import Normalize  # type: ignore
    from matplotlib.ticker import MaxNLocator  # type: ignore
    from mpl_toolkits.axes_grid1 import make_axes_locatable  # type: ignore

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), dpi=200, sharex=True, sharey=True)
    nlevels = max(16, int(args.levels))
    levels = np.linspace(-absmax, absmax, nlevels)
    cmap = str(args.cmap)
    norm = Normalize(vmin=-absmax, vmax=absmax)

    im0 = axes[0].tricontourf(triF, wzF, levels=levels, cmap=cmap, norm=norm)
    axes[0].set_title(str(args.left_title), fontsize=11)
    im1 = axes[1].tricontourf(triR, wzR, levels=levels, cmap=cmap, norm=norm)
    axes[1].set_title(str(args.right_title), fontsize=11)

    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax, im in zip(axes, (im0, im1), strict=True):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3.5%", pad=0.02)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=9)
        cbar.locator = MaxNLocator(nbins=7)
        cbar.update_ticks()
        cbar.set_label(r"$\omega_z$", rotation=90, fontsize=10)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.06, wspace=0.08)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[render_vorticity_comparison] Wrote: {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[render_vorticity_comparison] FAIL: {exc}", file=sys.stderr)
        raise
