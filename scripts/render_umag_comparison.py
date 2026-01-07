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
    # Preserve OpenFOAM string format for directory names when possible.
    for p in case_dir.iterdir():
        if p.is_dir():
            try:
                if abs(float(p.name) - t) < 1e-12:
                    return p.name
            except Exception:
                continue
    return f"{t:g}"


def _ensure_cell_centres(case_dir: Path, time_name: str) -> None:
    # Ensure cell centres exist (best-effort).
    _run(["postProcess", "-case", str(case_dir), "-func", "writeCellCentres", "-time", str(time_name)])


def _robust_range(a: np.ndarray, pct: float = 99.0) -> tuple[float, float]:
    x = np.asarray(a, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 8:
        lo = float(np.nanmin(x)) if x.size else 0.0
        hi = float(np.nanmax(x)) if x.size else 1.0
        return lo, (hi if hi > lo else lo + 1.0)
    lo = float(np.nanpercentile(x, 100.0 - pct))
    hi = float(np.nanpercentile(x, pct))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(x))
        hi = float(np.nanmax(x))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a FOM vs ROM |U| comparison (no pvpython).")
    ap.add_argument("--left-case", required=True, help="Left case directory (e.g., cases/low or results/high/fom_only/fom_only_case)")
    ap.add_argument("--right-case", required=True, help="Right case directory (e.g., results/low/rom_case or results/high/adaptive/adaptive_case)")
    ap.add_argument("--time", type=float, default=0.5, help="Requested time (nearest available time dir is used)")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--patch", default="cylinder", help="Cylinder patch name (for hole masking)")
    ap.add_argument("--left-title", default="Left", help="Left panel title")
    ap.add_argument("--right-title", default="Right", help="Right panel title")
    ap.add_argument("--cmap", default="coolwarm", help="Matplotlib colormap name")
    ap.add_argument("--levels", type=int, default=160, help="Filled-contour levels (higher looks smoother)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    left_case = (root / args.left_case).resolve()
    right_case = (root / args.right_case).resolve()
    out = (root / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if subprocess.run(["bash", "-lc", "command -v postProcess"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
        raise RuntimeError("OpenFOAM postProcess not available (source OpenFOAM-13 before running).")

    sys.path.insert(0, str((root / "rom" / "python").resolve()))
    from foam_ascii import n_cells_from_polymesh, read_internal_field  # type: ignore
    from geometry import cylinder_geometry, estimate_cylinder_diameter  # type: ignore

    t_left = _nearest_time_dir(left_case, float(args.time))
    t_right = _nearest_time_dir(right_case, float(args.time))
    _ensure_cell_centres(left_case, t_left)
    _ensure_cell_centres(right_case, t_right)

    def load(case_dir: Path, time_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = n_cells_from_polymesh(case_dir)
        C = read_internal_field(case_dir / time_name / "C", n_cells=n)
        U = read_internal_field(case_dir / time_name / "U", n_cells=n)
        if C.shape != (n, 3) or U.shape != (n, 3):
            raise ValueError(f"Unexpected field shapes under {case_dir}/{time_name}: C{C.shape} U{U.shape}")
        x, y, z = C[:, 0].astype(float), C[:, 1].astype(float), C[:, 2].astype(float)
        umag = np.linalg.norm(U.astype(float), axis=1)
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(umag)
        x, y, z, umag = x[m], y[m], z[m], umag[m]
        # Slice at mid-plane in z to avoid triangulation issues if the mesh has multiple z layers.
        z0 = float(np.median(z))
        z_span = float(np.max(z) - np.min(z)) if z.size else 0.0
        tol = max(1e-9, 0.05 * max(z_span, 1e-6))
        ms = np.abs(z - z0) <= tol
        return x[ms], y[ms], z[ms], umag[ms]

    xL, yL, _zL, uL = load(left_case, t_left)
    xR, yR, _zR, uR = load(right_case, t_right)

    # Shared color scale (robust to spikes).
    loL, hiL = _robust_range(uL, pct=99.0)
    loR, hiR = _robust_range(uR, pct=99.0)
    vmin = float(min(loL, loR))
    vmax = float(max(hiL, hiR))
    if vmax <= vmin:
        vmax = vmin + 1.0
    # For speed magnitude, include 0 in the color scale when possible.
    if vmin > 0.0:
        vmin = 0.0

    xmin = float(min(np.min(xL), np.min(xR)))
    xmax = float(max(np.max(xL), np.max(xR)))
    ymin = float(min(np.min(yL), np.min(yR)))
    ymax = float(max(np.max(yL), np.max(yR)))

    # Cylinder hole mask (triangles whose centroid is inside the cylinder).
    circle = estimate_cylinder_diameter(left_case, patch_name=str(args.patch))
    if circle is not None and math.isfinite(circle.radius) and circle.radius > 0:
        cx, cy = circle.center_xy
        r0 = float(circle.radius)
    else:
        geom = cylinder_geometry(left_case, patch_name=str(args.patch))
        cx, cy = 0.0, 0.0
        r0 = 0.5 * float(geom.get("D", 2.0))
    r_mask = 1.03 * float(r0)

    def tri_mask(x: np.ndarray, y: np.ndarray) -> "object":
        import matplotlib.tri as mtri  # type: ignore

        tri = mtri.Triangulation(x, y)
        t = tri.triangles
        xc = (x[t[:, 0]] + x[t[:, 1]] + x[t[:, 2]]) / 3.0
        yc = (y[t[:, 0]] + y[t[:, 1]] + y[t[:, 2]]) / 3.0
        inside = (xc - cx) ** 2 + (yc - cy) ** 2 < (r_mask**2)
        tri.set_mask(inside)
        return tri

    triL = tri_mask(xL, yL)
    triR = tri_mask(xR, yR)

    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.colors import Normalize  # type: ignore
    from matplotlib.ticker import MaxNLocator  # type: ignore
    from mpl_toolkits.axes_grid1 import make_axes_locatable  # type: ignore

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), dpi=200, sharex=True, sharey=True)
    norm = Normalize(vmin=vmin, vmax=vmax)
    nlevels = max(16, int(args.levels))
    levels = np.linspace(vmin, vmax, nlevels)
    cmap = str(args.cmap)

    im0 = axes[0].tricontourf(triL, uL, levels=levels, cmap=cmap, norm=norm)
    axes[0].set_title(str(args.left_title), fontsize=11)
    im1 = axes[1].tricontourf(triR, uR, levels=levels, cmap=cmap, norm=norm)
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
        cbar.locator = MaxNLocator(nbins=6)
        cbar.update_ticks()
        cbar.set_label(r"$|U|$", rotation=90, fontsize=10)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.06, wspace=0.08)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[render_umag_comparison] Wrote: {out}")


if __name__ == "__main__":
    main()
