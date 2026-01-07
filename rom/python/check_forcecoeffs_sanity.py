from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from forces import read_force_coeffs_dat


def _is_float_dirname(name: str) -> bool:
    try:
        float(name)
        return True
    except Exception:
        return False


def _find_latest_coeffs_file(root: Path) -> Path | None:
    if not root.exists():
        return None
    candidates: list[tuple[float, Path]] = []
    for p in root.iterdir():
        if not (p.is_dir() and _is_float_dirname(p.name)):
            continue
        for fn in ("forceCoeffs.dat", "coefficient.dat"):
            fp = p / fn
            if fp.exists():
                candidates.append((float(p.name), fp))
                break
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[-1][1]
    for fn in ("forceCoeffs.dat", "coefficient.dat"):
        for fp in root.rglob(fn):
            if fp.is_file():
                return fp
    return None


def _resolve_input(target: Path, obj: str | None) -> Path:
    if target.is_file():
        return target
    if not target.is_dir():
        raise FileNotFoundError(target)

    pp = target / "postProcessing"
    if not pp.exists():
        raise FileNotFoundError(pp)

    preferred = []
    if obj:
        preferred = [obj]
    else:
        for name in ["forceCoeffsIncompressible", "forceCoeffs1", "forceCoeffsROM"]:
            if (pp / name).is_dir():
                preferred.append(name)
        preferred.extend(sorted([p.name for p in pp.iterdir() if p.is_dir() and p.name.startswith("forceCoeffs")]))

    seen: set[str] = set()
    for name in preferred:
        if name in seen:
            continue
        seen.add(name)
        fp = _find_latest_coeffs_file(pp / name)
        if fp is not None:
            return fp

    raise FileNotFoundError(f"No forceCoeffs output found under {pp}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="Path to forceCoeffs.dat/coefficient.dat OR an OpenFOAM case directory.")
    ap.add_argument(
        "--obj",
        default=None,
        help="Specific postProcessing object name (e.g. forceCoeffs1, forceCoeffsIncompressible).",
    )
    ap.add_argument("--discard-frac", type=float, default=0.2, help="Fraction of initial samples to discard.")
    args = ap.parse_args()

    fp = _resolve_input(Path(args.target), args.obj)
    s = read_force_coeffs_dat(fp)

    k0 = int(max(0, round(float(args.discard_frac) * s.time.size)))
    cd = np.asarray(s.Cd, dtype=float).reshape(-1)[k0:]
    cl = np.asarray(s.Cl, dtype=float).reshape(-1)[k0:]
    if cd.size < 2 or cl.size < 2:
        raise RuntimeError(f"Too few samples after discard (n={cd.size}) from {fp}")

    cd_mean = float(np.mean(cd))
    cl_rms = float(np.sqrt(np.mean((cl - float(np.mean(cl))) ** 2)))

    print(f"[check_forcecoeffs_sanity] file: {fp}")
    print(f"[check_forcecoeffs_sanity] n_total={s.time.size} n_used={cd.size} discard_frac={float(args.discard_frac):g}")
    if s.magUInf is not None or s.lRef is not None or s.Aref is not None:
        print(f"[check_forcecoeffs_sanity] header: magUInf={s.magUInf} lRef={s.lRef} Aref={s.Aref}")
    print(f"[check_forcecoeffs_sanity] stats: Cd_mean={cd_mean:.6g} Cl_rms={cl_rms:.6g}")

    warn = []
    if abs(cd_mean) > 10:
        warn.append("|Cd_mean| > 10")
    if cl_rms > 10:
        warn.append("Cl_rms > 10")

    if warn:
        print(f"[check_forcecoeffs_sanity] WARN: {', '.join(warn)}")
        print("[check_forcecoeffs_sanity] Likely causes to check:")
        print("- wrong patch list (should be ONLY the cylinder wall patch)")
        print("- wrong rho/rhoInf handling for incompressible kinematic pressure (p is p/rho)")
        print("- wrong Aref/lRef or magUInf (Aref=D*span, lRef=D, magUInf=|U_inlet|)")
    else:
        print("[check_forcecoeffs_sanity] OK: coefficients look within expected O(1..10) range.")


if __name__ == "__main__":
    main()

