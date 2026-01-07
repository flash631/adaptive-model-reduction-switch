from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from geometry import cylinder_geometry


def _detect_cylinder_patch_name(case_dir: Path) -> str:
    boundary = case_dir / "constant" / "polyMesh" / "boundary"
    if not boundary.exists():
        return "cylinder"
    lines = boundary.read_text(encoding="utf-8", errors="ignore").splitlines()

    def clean(line: str) -> str:
        return re.sub(r"//.*$", "", line).strip()

    # Parse patch names inside the top-level boundary list.
    in_list = False
    names: list[str] = []
    i = 0
    while i < len(lines) - 1:
        s = clean(lines[i])
        if not in_list:
            if s == "(":
                in_list = True
            i += 1
            continue
        if s == ")":
            break
        if not s:
            i += 1
            continue
        nxt = clean(lines[i + 1])
        if re.fullmatch(r"[A-Za-z0-9_]+", s) and nxt.startswith("{"):
            names.append(s)
        i += 1

    for cand in names:
        if "cylinder" in cand.lower():
            return cand
    for cand in names:
        if "obstacle" in cand.lower():
            return cand
    return "cylinder"


def _domain_bounds_from_points(case_dir: Path) -> dict[str, float]:
    """
    Read domain bounds from constant/polyMesh/points.
    """
    pts_path = case_dir / "constant" / "polyMesh" / "points"
    if not pts_path.exists():
        raise FileNotFoundError(pts_path)
    txt = pts_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    pts: list[list[float]] = []
    in_list = False
    vec_re = re.compile(r"^\s*\(\s*([Ee0-9+\-\.]+)\s+([Ee0-9+\-\.]+)\s+([Ee0-9+\-\.]+)\s*\)\s*$")
    for line in txt:
        s = re.sub(r"//.*$", "", line).strip()
        if not in_list:
            if s == "(":
                in_list = True
            continue
        if s == ")":
            break
        m = vec_re.match(s)
        if not m:
            continue
        pts.append([float(m.group(1)), float(m.group(2)), float(m.group(3))])
    if not pts:
        raise ValueError(f"Failed to parse any points from {pts_path}")

    P = np.asarray(pts, dtype=float)
    mn = np.min(P, axis=0)
    mx = np.max(P, axis=0)
    return {
        "xmin": float(mn[0]),
        "xmax": float(mx[0]),
        "ymin": float(mn[1]),
        "ymax": float(mx[1]),
        "zmin": float(mn[2]),
        "zmax": float(mx[2]),
    }


def _empty_patch_names(boundary_path: Path) -> list[str]:
    if not boundary_path.exists():
        return []
    lines = boundary_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    def clean(line: str) -> str:
        return re.sub(r"//.*$", "", line).strip()

    in_list = False
    i = 0
    empties: list[str] = []
    while i < len(lines):
        s = clean(lines[i])
        if not in_list:
            if s == "(":
                in_list = True
            i += 1
            continue
        if s == ")":
            break
        if not s:
            i += 1
            continue
        m = re.fullmatch(r"([A-Za-z0-9_]+)(?:\s*\{)?", s)
        if not m:
            i += 1
            continue
        name = m.group(1)
        if "{" not in s:
            j = i + 1
            while j < len(lines) and not clean(lines[j]):
                j += 1
            if j >= len(lines) or "{" not in clean(lines[j]):
                i += 1
                continue
            i = j
        depth = 0
        block_lines: list[str] = []
        while i < len(lines):
            s2 = clean(lines[i])
            if s2:
                depth += s2.count("{")
                depth -= s2.count("}")
                block_lines.append(s2)
            i += 1
            if depth <= 0 and block_lines:
                break
        if re.search(r"(?m)^\s*type\s+empty\s*;", "\n".join(block_lines)):
            empties.append(name)
    return empties


def compute_confinement(case_dir: Path, *, patch_name: str | None = None) -> dict[str, Any]:
    """
    Compute channel height H from domain y-bounds and blockage ratio beta = D/H.
    """
    patch = patch_name or _detect_cylinder_patch_name(case_dir)
    geom = cylinder_geometry(case_dir, patch_name=patch)
    D = float(geom["D"])

    b = _domain_bounds_from_points(case_dir)
    H = float(b["ymax"] - b["ymin"])
    beta = float(D / H) if H > 0 else float("nan")
    return {
        "D": D,
        "H": H,
        "beta": beta,
        "domain_y_bounds": [float(b["ymin"]), float(b["ymax"])],
        "domain_bounds": b,
        "cylinder_patch": patch,
    }


def compute_span(case_dir: Path, *, patch_name: str | None = None) -> dict[str, Any]:
    """
    Determine span for coefficient normalization:
      - If any front/back-like patches are marked `empty`, span=1.0 (2D convention).
      - Otherwise span is the z-extent of the cylinder patch bounding box.
    """
    poly = case_dir / "constant" / "polyMesh"
    empties = _empty_patch_names(poly / "boundary")
    patch = patch_name or _detect_cylinder_patch_name(case_dir)
    geom = cylinder_geometry(case_dir, patch_name=patch)
    D = float(geom["D"])
    span_geom = float(geom["span_geom"])
    span = 1.0 if empties else span_geom
    return {
        "span": float(span),
        "Aref": float(D * span),
        "is_2d": bool(bool(empties)),
        "empty_patches": empties,
        "span_geom": span_geom,
        "cylinder_patch": patch,
    }


def write_metrics(case_dir: Path, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    conf = compute_confinement(case_dir)
    span = compute_span(case_dir)
    p_conf = out_dir / "confinement.json"
    p_span = out_dir / "span.json"
    p_conf.write_text(json.dumps(conf, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    p_span.write_text(json.dumps(span, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p_conf, p_span


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="OpenFOAM case directory (contains constant/polyMesh)")
    ap.add_argument("--out-dir", required=True, help="Output metrics directory (e.g., results/low/metrics)")
    args = ap.parse_args()
    case_dir = Path(args.case).resolve()
    out_dir = Path(args.out_dir).resolve()
    p_conf, p_span = write_metrics(case_dir, out_dir)
    print(f"[confinement] Wrote: {p_conf}")
    print(f"[confinement] Wrote: {p_span}")


if __name__ == "__main__":
    main()
