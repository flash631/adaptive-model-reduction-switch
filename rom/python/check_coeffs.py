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


def _find_latest_dat(case_dir: Path, obj_name: str) -> Path | None:
    root = case_dir / "postProcessing" / obj_name
    if not root.exists():
        return None

    # Prefer a numeric time folder with a known filename.
    candidates: list[tuple[float, Path]] = []
    for p in root.iterdir():
        if not (p.is_dir() and _is_float_dirname(p.name)):
            continue
        for fn in ["coefficient.dat", "forceCoeffs.dat"]:
            fp = p / fn
            if fp.exists():
                candidates.append((float(p.name), fp))
                break
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[-1][1]

    # Fallback: any supported file under the tree.
    for fn in ["coefficient.dat", "forceCoeffs.dat"]:
        for fp in root.rglob(fn):
            if fp.is_file():
                return fp
    return None


def _stats(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float).reshape(-1)
    return float(np.mean(x)), float(np.sqrt(np.mean((x - float(np.mean(x))) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "case_dirs",
        nargs="+",
        help="Case directories containing postProcessing/forceCoeffs*",
    )
    ap.add_argument(
        "--obj",
        action="append",
        default=[],
        help="Specific forceCoeffs object name(s) (repeatable). If omitted, scans postProcessing for dirs starting with 'forceCoeffs'.",
    )
    ap.add_argument(
        "--discard-frac",
        type=float,
        default=0.2,
        help="Discard this fraction of initial samples before computing stats (default: 0.2).",
    )
    args = ap.parse_args()

    discard_frac = float(args.discard_frac)
    if not (0.0 <= discard_frac < 1.0):
        raise SystemExit("--discard-frac must be in [0,1)")

    for case_dir_s in args.case_dirs:
        case_dir = Path(case_dir_s).resolve()
        if not case_dir.exists():
            print(f"[check_coeffs] MISSING: {case_dir}")
            continue

        pp = case_dir / "postProcessing"
        if not pp.exists():
            print(f"[check_coeffs] No postProcessing in: {case_dir}")
            continue

        if args.obj:
            obj_names = list(args.obj)
        else:
            obj_names = sorted([p.name for p in pp.iterdir() if p.is_dir() and p.name.startswith("forceCoeffs")])

        if not obj_names:
            print(f"[check_coeffs] No forceCoeffs* dirs in: {pp}")
            continue

        print(f"[check_coeffs] case: {case_dir}")
        for obj in obj_names:
            fp = _find_latest_dat(case_dir, obj_name=obj)
            if fp is None:
                print(f"  - {obj}: missing coefficient.dat/forceCoeffs.dat")
                continue

            try:
                s = read_force_coeffs_dat(fp)
            except Exception as e:
                print(f"  - {obj}: failed to parse {fp}: {e}")
                continue

            k0 = int(max(0, round(discard_frac * s.time.size)))
            cd = np.asarray(s.Cd, dtype=float).reshape(-1)[k0:]
            cl = np.asarray(s.Cl, dtype=float).reshape(-1)[k0:]
            if cd.size < 2 or cl.size < 2:
                print(f"  - {obj}: too few samples after discard (n={cd.size}) in {fp}")
                continue

            cd_mean, cd_rms = _stats(cd)
            cl_mean, cl_rms = _stats(cl)

            warn = []
            if abs(float(cd_mean)) > 10:
                warn.append("|Cd_mean|>10")
            if float(cl_rms) > 10:
                warn.append("Cl_rms>10")
            warn_s = f"  WARN({', '.join(warn)})" if warn else ""

            print(
                f"  - {obj}: Cd_mean={cd_mean:.6g} Cd_rms={cd_rms:.6g} Cl_mean={cl_mean:.6g} Cl_rms={cl_rms:.6g}{warn_s}"
            )


if __name__ == "__main__":
    main()
