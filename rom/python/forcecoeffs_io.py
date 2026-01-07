from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

_FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def _iter_noncomment_lines(lines: Iterable[str]) -> Iterable[str]:
    for raw in lines:
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        yield s


def _normalize_token(tok: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", tok).lower()


def _find_force_coeffs_header_tokens(lines: Iterable[str]) -> list[str] | None:
    hdr: list[str] | None = None
    for raw in lines:
        s = raw.strip()
        if not s.startswith("#"):
            continue
        body = s.lstrip("#").strip()
        if not body:
            continue
        if "Time" in body and ("Cd" in body or "Cl" in body or "Cm" in body):
            hdr = [t for t in re.split(r"\s+", body) if t]
    return hdr


def _dedupe_sort_time(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_values("time", kind="mergesort")
    df = df.drop_duplicates(subset=["time"], keep="last")
    df = df.reset_index(drop=True)
    return df


def read_force_coeffs(path: str | Path) -> pd.DataFrame:
    """
    Read OpenFOAM forceCoeffs-style output into a DataFrame.

    Returns columns: time, Cd, Cl, and Cm if present in the file.
    Column indices are detected from the "# Time ..." header when present.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    hdr = _find_force_coeffs_header_tokens(text)
    if hdr is None:
        raise ValueError(f"Could not find a '# Time ...' header in {path}")

    idx: dict[str, int] = {}
    for i, tok in enumerate(hdr):
        idx[_normalize_token(tok)] = i

    def require(name: str) -> int:
        k = _normalize_token(name)
        if k not in idx:
            raise ValueError(f"Missing column '{name}' in header of {path}: {hdr}")
        return int(idx[k])

    t_i = require("Time")
    cd_i = require("Cd")
    cl_i = require("Cl")
    cm_i = idx.get(_normalize_token("Cm"))

    t_list: list[float] = []
    cd_list: list[float] = []
    cl_list: list[float] = []
    cm_list: list[float] = []

    for line in _iter_noncomment_lines(text):
        nums = [float(x) for x in _FLOAT_RE.findall(line)]
        if not nums:
            continue
        if max(t_i, cd_i, cl_i) >= len(nums):
            continue
        t_list.append(nums[t_i])
        cd_list.append(nums[cd_i])
        cl_list.append(nums[cl_i])
        if cm_i is not None and cm_i < len(nums):
            cm_list.append(nums[cm_i])

    if len(t_list) < 2:
        raise ValueError(f"Could not parse forceCoeffs data from {path} (parsed {len(t_list)} lines)")

    out: dict[str, np.ndarray] = {
        "time": np.asarray(t_list, dtype=float),
        "Cd": np.asarray(cd_list, dtype=float),
        "Cl": np.asarray(cl_list, dtype=float),
    }
    if cm_i is not None and len(cm_list) == len(t_list):
        out["Cm"] = np.asarray(cm_list, dtype=float)

    return _dedupe_sort_time(pd.DataFrame(out))


def read_forces(path: str | Path) -> pd.DataFrame:
    """
    Read OpenFOAM forces output into a DataFrame.

    Returns columns: time, Fx, Fy, Fz where F* = pressure + viscous.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    t_list: list[float] = []
    fx_list: list[float] = []
    fy_list: list[float] = []
    fz_list: list[float] = []

    for line in _iter_noncomment_lines(path.read_text(encoding="utf-8", errors="ignore").splitlines()):
        nums = [float(x) for x in _FLOAT_RE.findall(line)]
        if len(nums) < 7:
            continue
        t = nums[0]
        fpx, fpy, fpz, fvx, fvy, fvz = nums[1:7]
        t_list.append(t)
        fx_list.append(fpx + fvx)
        fy_list.append(fpy + fvy)
        fz_list.append(fpz + fvz)

    if len(t_list) < 2:
        raise ValueError(f"Could not parse forces data from {path} (parsed {len(t_list)} lines)")

    df = pd.DataFrame(
        {
            "time": np.asarray(t_list, dtype=float),
            "Fx": np.asarray(fx_list, dtype=float),
            "Fy": np.asarray(fy_list, dtype=float),
            "Fz": np.asarray(fz_list, dtype=float),
        }
    )
    return _dedupe_sort_time(df)

