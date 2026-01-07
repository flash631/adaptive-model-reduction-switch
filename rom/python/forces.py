from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from forcecoeffs_io import read_force_coeffs, read_forces


@dataclass(frozen=True)
class ForcesSeries:
    time: np.ndarray  # (n,)
    Fx: np.ndarray  # total force x
    Fy: np.ndarray  # total force y
    Fz: np.ndarray  # total force z


@dataclass(frozen=True)
class ForceCoeffsSeries:
    time: np.ndarray  # (n,)
    Cd: np.ndarray
    Cl: np.ndarray
    Cm: np.ndarray | None = None
    magUInf: float | None = None
    lRef: float | None = None
    Aref: float | None = None


def read_forces_dat(path: Path) -> ForcesSeries:
    df = read_forces(path)
    return ForcesSeries(
        time=np.asarray(df["time"], dtype=float),
        Fx=np.asarray(df["Fx"], dtype=float),
        Fy=np.asarray(df["Fy"], dtype=float),
        Fz=np.asarray(df["Fz"], dtype=float),
    )


def read_force_coeffs_dat(path: Path) -> ForceCoeffsSeries:
    """
    Parse OpenFOAM forceCoeffs output (typically postProcessing/forceCoeffs*/0/forceCoeffs.dat).
    Detects column ordering from the "# Time ..." header when present.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    magUInf: float | None = None
    lRef: float | None = None
    Aref: float | None = None

    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line.startswith("#"):
            continue
        m = re.match(r"^#\s*magUInf\s*:\s*([Ee0-9+\-\.]+)\s*$", line)
        if m:
            try:
                magUInf = float(m.group(1))
            except Exception:
                pass
        m = re.match(r"^#\s*lRef\s*:\s*([Ee0-9+\-\.]+)\s*$", line)
        if m:
            try:
                lRef = float(m.group(1))
            except Exception:
                pass
        m = re.match(r"^#\s*Aref\s*:\s*([Ee0-9+\-\.]+)\s*$", line)
        if m:
            try:
                Aref = float(m.group(1))
            except Exception:
                pass

    df = read_force_coeffs(path)
    cm_arr = np.asarray(df["Cm"], dtype=float) if "Cm" in df.columns else None
    return ForceCoeffsSeries(
        time=np.asarray(df["time"], dtype=float),
        Cd=np.asarray(df["Cd"], dtype=float),
        Cl=np.asarray(df["Cl"], dtype=float),
        Cm=cm_arr,
        magUInf=magUInf,
        lRef=lRef,
        Aref=Aref,
    )
