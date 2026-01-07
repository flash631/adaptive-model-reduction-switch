from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from config import get, load_yaml
from geometry import cylinder_geometry, estimate_cylinder_diameter, read_inlet_u_mag


def _is_float_dirname(name: str) -> bool:
    try:
        float(name)
        return True
    except Exception:
        return False


def _detect_cylinder_patch_name(case_dir: Path) -> str:
    boundary = case_dir / "constant" / "polyMesh" / "boundary"
    if not boundary.exists():
        return "cylinder"
    txt = boundary.read_text(encoding="utf-8", errors="ignore")
    # Prefer obvious names.
    for cand in ["cylinder", "Cylinder", "obstacle", "Obstacle"]:
        if cand in txt:
            return "cylinder" if "cylinder" in cand.lower() else "obstacle"
    # Fallback: try to parse patch names (best-effort).
    lines = txt.splitlines()
    names: list[str] = []
    for i in range(len(lines) - 1):
        a = lines[i].strip()
        b = lines[i + 1].strip()
        if a and a[0].isalpha() and a.replace("_", "").isalnum() and b.startswith("{"):
            names.append(a)
    for n in names:
        if "cylinder" in n.lower():
            return n
    return "cylinder"


def _write_force_coeffs_dict(
    path: Path,
    *,
    patch: str,
    mag_u_inf: float,
    d_ref: float,
    a_ref: float,
    cof_r: tuple[float, float, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Configuration file for `foamPostProcess -func forceCoeffsIncompressible`.
    # This mirrors OpenFOAM tutorial usage: system/forceCoeffsIncompressible + include cfg.
    cx, cy, cz = cof_r
    txt = f"""/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  13
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/

patches     ({patch});

magUInf     {mag_u_inf:g};
Aref        {a_ref:g};
dragDir     (1 0 0);
liftDir     (0 1 0);

rho         rhoInf;
rhoInf      1;

// Moment calculation parameters
lRef        {d_ref:g};
CofR        ({cx:g} {cy:g} {cz:g});
pitchAxis   (0 0 1);

#includeEtc "caseDicts/functions/forces/forceCoeffsIncompressible.cfg"

// ************************************************************************* //
"""
    path.write_text(txt, encoding="utf-8")


def _find_force_coeffs_dat(case_dir: Path, obj_name: str) -> Path | None:
    root = case_dir / "postProcessing" / obj_name
    if not root.exists():
        return None
    # Prefer the latest numeric time directory.
    candidates: list[tuple[float, Path]] = []
    for p in root.iterdir():
        if p.is_dir() and _is_float_dirname(p.name):
            for fn in ["forceCoeffs.dat", "coefficient.dat"]:
                fp = p / fn
                if fp.exists():
                    candidates.append((float(p.name), fp))
                    break
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[-1][1]
    # Fallback: any expected filename under this tree.
    for fn in ["forceCoeffs.dat", "coefficient.dat"]:
        for fp in root.rglob(fn):
            if fp.is_file():
                return fp
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--case", choices=["adaptive", "fom_only", "rom_case"], default="adaptive")
    ap.add_argument(
        "--func-name",
        default="forceCoeffsIncompressible",
        help="foamPostProcess template name for incompressible force coefficients",
    )
    ap.add_argument("--force", action="store_true", help="Overwrite existing postProcessing output for func-name")
    args = ap.parse_args()

    if shutil.which("foamPostProcess") is None and shutil.which("postProcess") is None:
        raise RuntimeError("OpenFOAM post-processing not available (foamPostProcess/postProcess not found in PATH)")

    cfg: dict[str, Any] = load_yaml(args.config)
    results_dir = Path(get(cfg, "case.results_dir", "results/low")).resolve()

    if args.case == "adaptive":
        case_dir = results_dir / "adaptive" / "adaptive_case"
    elif args.case == "fom_only":
        case_dir = results_dir / "fom_only" / "fom_only_case"
    else:
        case_dir = results_dir / "rom_case"

    if not case_dir.exists():
        raise FileNotFoundError(case_dir)

    t0 = float(get(cfg, "adaptive.start_time", 0.0))
    # Use an appropriate time horizon:
    # - For baseline FOM-only, post-process the full baseline horizon (openfoam.endTime).
    # - For adaptive/rom_case, default to the adaptive horizon (adaptive.end_time).
    if args.case == "fom_only":
        t1 = float(get(cfg, "openfoam.endTime", float(get(cfg, "adaptive.end_time", 1.0))))
    else:
        t1 = float(get(cfg, "adaptive.end_time", float(get(cfg, "openfoam.endTime", 1.0))))
    patch = _detect_cylinder_patch_name(case_dir)
    inlet_u = read_inlet_u_mag(case_dir) or 1.0
    try:
        geom = cylinder_geometry(case_dir, patch_name=patch)
    except Exception:
        geom = {"D": 1.0, "span_ref": 1.0}
    d_ref = float(geom.get("D", 1.0)) if float(geom.get("D", 1.0)) > 0 else 1.0
    span_ref = float(geom.get("span_ref", 1.0)) if float(geom.get("span_ref", 1.0)) > 0 else 1.0
    circle = estimate_cylinder_diameter(case_dir, patch_name=patch)
    cx, cy = circle.center_xy if circle is not None else (0.0, 0.0)
    a_ref = float(d_ref * span_ref)

    metrics_dir = results_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    sys_functions = case_dir / "system" / "functions"
    if sys_functions.exists():
        # foamPostProcess prefers system/functions over system/controlDict/functions; remove to avoid conflicts.
        if not args.force:
            raise RuntimeError(f"{sys_functions} exists; re-run with --force to remove it for postProcess")
        if sys_functions.is_dir():
            shutil.rmtree(sys_functions)
        else:
            sys_functions.unlink()

    func_path = case_dir / "system" / args.func_name
    _write_force_coeffs_dict(
        func_path,
        patch=patch,
        mag_u_inf=float(inlet_u),
        d_ref=float(d_ref),
        a_ref=float(a_ref),
        cof_r=(float(cx), float(cy), 0.0),
    )

    out_pp = case_dir / "postProcessing" / args.func_name
    if out_pp.exists() and not args.force:
        raise RuntimeError(f"{out_pp} exists; re-run with --force to overwrite")
    if out_pp.exists() and args.force:
        shutil.rmtree(out_pp)

    cmd = [
        "foamPostProcess",
        "-case",
        str(case_dir),
        "-solver",
        "incompressibleFluid",
        "-func",
        args.func_name,
        "-fields",
        "(U p)",
        "-time",
        f"{t0:g}:{t1:g}",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"foamPostProcess failed (rc={p.returncode}).\nstdout:\n{p.stdout}\nstderr:\n{p.stderr}")

    dat_path = _find_force_coeffs_dat(case_dir, args.func_name)
    if dat_path is None:
        raise RuntimeError(f"Could not find forceCoeffs.dat under {case_dir}/postProcessing/{args.func_name}")

    meta = {
        "case": args.case,
        "case_dir": str(case_dir),
        "func_name": args.func_name,
        "func_path": str(func_path),
        "patch": patch,
        "time_range": [t0, t1],
        "force_coeffs_dat": str(dat_path),
    }
    (metrics_dir / f"{args.func_name}_{args.case}_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[postprocess_force_coeffs] Wrote: {dat_path}")
    print(f"[postprocess_force_coeffs] Meta: {metrics_dir}/{args.func_name}_{args.case}_meta.json")


if __name__ == "__main__":
    main()
