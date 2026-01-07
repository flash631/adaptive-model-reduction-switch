#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from dataclasses import dataclass
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
    for p in case_dir.iterdir():
        if p.is_dir():
            try:
                if abs(float(p.name) - t) < 1e-12:
                    return p.name
            except Exception:
                continue
    return f"{t:g}"


def _ensure_cell_centres(case_dir: Path, time_name: str) -> None:
    _run(["postProcess", "-case", str(case_dir), "-func", "writeCellCentres", "-time", str(time_name)])


def _ensure_vorticity(case_dir: Path, time_name: str) -> None:
    _run(["postProcess", "-case", str(case_dir), "-func", "vorticity", "-time", str(time_name)])


def _robust_range(a: np.ndarray, pct: float) -> tuple[float, float]:
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


@dataclass(frozen=True)
class Spec:
    t_req: float
    out: Path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render one or more OpenFOAM field snapshots with a shared color scale and per-image right-side colorbars (no pvpython)."
    )
    ap.add_argument("--case", required=True, help="Case directory (e.g., cases/low or results/low/modes_case)")
    ap.add_argument("--field", choices=["U", "p", "vorticity"], required=True)
    ap.add_argument("--component", choices=["mag", "x", "y", "z"], default="mag", help="Vector component for U/vorticity")
    ap.add_argument("--patch", default="cylinder", help="Cylinder patch name (for hole masking)")
    ap.add_argument("--spec", action="append", default=[], help="Repeatable TIME:OUT spec, e.g. 0.5:report/figures/U_low_tmid.png")
    ap.add_argument("--cmap", default="coolwarm", help="Matplotlib colormap")
    ap.add_argument("--levels", type=int, default=160, help="Filled-contour levels")
    ap.add_argument("--pct", type=float, default=99.0, help="Robust percentile for vmin/vmax estimation")
    ap.add_argument("--vmin", type=float, default=float("nan"), help="Override vmin")
    ap.add_argument("--vmax", type=float, default=float("nan"), help="Override vmax")
    args = ap.parse_args()

    if not args.spec:
        raise SystemExit("ERROR: at least one --spec TIME:OUT must be provided")

    root = Path(__file__).resolve().parents[1]
    case_dir = (root / args.case).resolve()

    if subprocess.run(["bash", "-lc", "command -v postProcess"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
        raise RuntimeError("OpenFOAM postProcess not available (source OpenFOAM-13 before running).")

    sys.path.insert(0, str((root / "rom" / "python").resolve()))
    from foam_ascii import n_cells_from_polymesh, read_internal_field  # type: ignore
    from geometry import cylinder_geometry, estimate_cylinder_diameter  # type: ignore

    specs: list[Spec] = []
    for raw in args.spec:
        if ":" not in raw:
            raise SystemExit(f"ERROR: invalid --spec {raw!r} (expected TIME:OUT)")
        t_s, out_s = raw.split(":", 1)
        try:
            t_req = float(t_s)
        except Exception:
            raise SystemExit(f"ERROR: invalid TIME in --spec {raw!r}")
        out = (root / out_s).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        specs.append(Spec(t_req=t_req, out=out))

    def _vec_component(v: np.ndarray, comp: str) -> np.ndarray:
        if comp == "mag":
            return np.linalg.norm(v.astype(float), axis=1)
        idx = {"x": 0, "y": 1, "z": 2}[comp]
        return v[:, idx].astype(float)

    def load_at(time_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = n_cells_from_polymesh(case_dir)
        C = read_internal_field(case_dir / time_name / "C", n_cells=n)
        if C.shape != (n, 3):
            raise ValueError(f"Unexpected C shape under {case_dir}/{time_name}: {C.shape}")
        x = C[:, 0].astype(float, copy=False)
        y = C[:, 1].astype(float, copy=False)
        z = C[:, 2].astype(float, copy=False)

        if args.field == "U":
            U = read_internal_field(case_dir / time_name / "U", n_cells=n)
            if U.shape != (n, 3):
                raise ValueError(f"Unexpected U shape under {case_dir}/{time_name}: {U.shape}")
            s = _vec_component(U, str(args.component))
            label = r"$|U|$" if args.component == "mag" else r"$U_{%s}$" % str(args.component)
        elif args.field == "vorticity":
            w = read_internal_field(case_dir / time_name / "vorticity", n_cells=n)
            if w.shape != (n, 3):
                raise ValueError(f"Unexpected vorticity shape under {case_dir}/{time_name}: {w.shape}")
            s = _vec_component(w, str(args.component))
            label = r"$\omega_z$" if args.component == "z" else (r"$|\omega|$" if args.component == "mag" else r"$\omega_{%s}$" % str(args.component))
        else:
            p = read_internal_field(case_dir / time_name / "p", n_cells=n)
            p = np.asarray(p, dtype=float).reshape(-1)
            if p.size != n:
                raise ValueError(f"Unexpected p shape under {case_dir}/{time_name}: {p.shape}")
            s = p
            label = r"$p$"

        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(s)
        x, y, z, s = x[m], y[m], z[m], s[m]

        z0 = float(np.median(z))
        z_span = float(np.max(z) - np.min(z)) if z.size else 0.0
        tol = max(1e-9, 0.05 * max(z_span, 1e-6))
        ms = np.abs(z - z0) <= tol
        return x[ms], y[ms], s[ms], label

    # Determine time directories; ensure derived fields.
    time_dirs: list[str] = []
    for s in specs:
        tdir = _nearest_time_dir(case_dir, s.t_req)
        time_dirs.append(tdir)
        _ensure_cell_centres(case_dir, tdir)
        if args.field == "vorticity":
            _ensure_vorticity(case_dir, tdir)

    # Preload to compute shared bounds.
    loaded: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []
    for tdir in time_dirs:
        loaded.append(load_at(tdir))

    xs = np.concatenate([a[0] for a in loaded]) if loaded else np.zeros((0,))
    ys = np.concatenate([a[1] for a in loaded]) if loaded else np.zeros((0,))
    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    ymin, ymax = float(np.min(ys)), float(np.max(ys))

    scalar_all = np.concatenate([a[2] for a in loaded]) if loaded else np.zeros((0,))
    if np.isfinite(args.vmin) and np.isfinite(args.vmax):
        vmin, vmax = float(args.vmin), float(args.vmax)
    else:
        lo, hi = _robust_range(scalar_all, pct=float(args.pct))
        vmin, vmax = float(lo), float(hi)

        if args.field == "U" and args.component == "mag":
            vmin = 0.0
        if args.field in {"p", "vorticity"}:
            if vmin < 0.0 < vmax:
                absmax = max(abs(vmin), abs(vmax))
                vmin, vmax = -absmax, absmax
        if vmax <= vmin:
            vmax = vmin + 1.0
        if args.field == "U" and args.component == "mag" and vmin > 0.0:
            vmin = 0.0

    levels_n = max(16, int(args.levels))
    levels = np.linspace(vmin, vmax, levels_n)

    # Cylinder hole mask (triangles whose centroid is inside the cylinder).
    circle = estimate_cylinder_diameter(case_dir, patch_name=str(args.patch))
    if circle is not None and math.isfinite(circle.radius) and circle.radius > 0:
        cx, cy = circle.center_xy
        r0 = float(circle.radius)
    else:
        geom = cylinder_geometry(case_dir, patch_name=str(args.patch))
        cx, cy = 0.0, 0.0
        r0 = 0.5 * float(geom.get("D", 2.0))
    r_mask = 1.03 * float(r0)

    import matplotlib.pyplot as plt  # type: ignore
    import matplotlib.tri as mtri  # type: ignore
    from matplotlib.colors import Normalize  # type: ignore
    from matplotlib.ticker import MaxNLocator  # type: ignore
    from mpl_toolkits.axes_grid1 import make_axes_locatable  # type: ignore

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = str(args.cmap)

    for spec, tdir, (x, y, s, label) in zip(specs, time_dirs, loaded, strict=True):
        tri = mtri.Triangulation(x, y)
        t = tri.triangles
        xc = (x[t[:, 0]] + x[t[:, 1]] + x[t[:, 2]]) / 3.0
        yc = (y[t[:, 0]] + y[t[:, 1]] + y[t[:, 2]]) / 3.0
        inside = (xc - cx) ** 2 + (yc - cy) ** 2 < (r_mask**2)
        tri.set_mask(inside)

        # Match the visual style used by the side-by-side comparison figures:
        # use a per-panel aspect ratio close to the domain, so the plot fills the canvas
        # without large vertical whitespace when included as a subfigure in the report.
        fig = plt.figure(figsize=(5.25, 3.6), dpi=200)
        ax = fig.add_subplot(1, 1, 1)
        im = ax.tricontourf(tri, s, levels=levels, cmap=cmap, norm=norm)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks([])
        ax.set_yticks([])

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3.5%", pad=0.02)
        cbar = fig.colorbar(im, cax=cax)
        cbar.locator = MaxNLocator(nbins=7)
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=9)
        cbar.set_label(label, rotation=90, fontsize=10)

        fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
        fig.savefig(spec.out, bbox_inches="tight")
        plt.close(fig)
        print(f"[render_snapshots_shared] time={tdir} -> {spec.out}")


if __name__ == "__main__":
    main()
