from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def _read_poly_points(points_path: Path) -> np.ndarray:
    txt = points_path.read_text(encoding="utf-8", errors="ignore").splitlines()
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
        raise ValueError(f"Failed to parse points from {points_path}")
    return np.asarray(pts, dtype=float)


def _read_poly_faces(faces_path: Path) -> list[list[int]]:
    txt = faces_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    faces: list[list[int]] = []
    in_list = False
    for line in txt:
        s = re.sub(r"//.*$", "", line).strip()
        if not in_list:
            if s == "(":
                in_list = True
            continue
        if s == ")":
            break
        if not s:
            continue
        # Examples:
        # 4(0 1 2 3)
        # 3(12 13 14)
        m = re.match(r"^(\d+)\s*\(([^)]*)\)\s*$", s)
        if not m:
            continue
        idx = [int(x) for x in re.findall(r"\d+", m.group(2))]
        if idx:
            faces.append(idx)
    if not faces:
        raise ValueError(f"Failed to parse faces from {faces_path}")
    return faces


def _read_boundary_patch_info(boundary_path: Path, patch_name: str) -> tuple[int, int] | None:
    """
    Returns (startFace, nFaces) for patch_name, or None if not found.
    """
    txt_lines = boundary_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    def clean(line: str) -> str:
        return re.sub(r"//.*$", "", line).strip()

    # The boundary file contains an integer patch count and a top-level list: ( ... )
    in_list = False
    i = 0
    while i < len(txt_lines):
        s = clean(txt_lines[i])
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

        # Patch blocks are commonly formatted as:
        #   patchName
        #   {
        #       ...
        #   }
        m = re.fullmatch(r"([A-Za-z0-9_]+)(?:\s*\{)?", s)
        if not m:
            i += 1
            continue
        name = m.group(1)

        # Advance to the opening brace if it's on the next non-empty line.
        if "{" not in s:
            j = i + 1
            while j < len(txt_lines) and not clean(txt_lines[j]):
                j += 1
            if j >= len(txt_lines) or "{" not in clean(txt_lines[j]):
                i += 1
                continue
            i = j

        # Collect block lines until braces balance.
        depth = 0
        block_lines: list[str] = []
        while i < len(txt_lines):
            s2 = clean(txt_lines[i])
            if s2:
                depth += s2.count("{")
                depth -= s2.count("}")
                block_lines.append(s2)
            i += 1
            if depth <= 0 and block_lines:
                break

        if name != patch_name:
            continue

        block = "\n".join(block_lines)
        sm = re.search(r"\bstartFace\s+(\d+)\s*;", block)
        nm = re.search(r"\bnFaces\s+(\d+)\s*;", block)
        if not sm or not nm:
            return None
        return int(sm.group(1)), int(nm.group(1))

    return None


def _has_empty_patch(boundary_path: Path) -> bool:
    if not boundary_path.exists():
        return False
    txt = boundary_path.read_text(encoding="utf-8", errors="ignore")
    return bool(re.search(r"^\s*type\s+empty\s*;", txt, flags=re.M))


def _empty_patch_names(boundary_path: Path) -> list[str]:
    """
    Return names of patches whose boundary type is `empty`.

    OpenFOAM 2D cases typically mark the front/back patches as `empty`. Some tutorials
    name this patch `frontAndBack`, while others may use `defaultFaces`.
    """
    if not boundary_path.exists():
        return []
    txt_lines = boundary_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    def clean(line: str) -> str:
        return re.sub(r"//.*$", "", line).strip()

    in_list = False
    i = 0
    names: list[str] = []
    while i < len(txt_lines):
        s = clean(txt_lines[i])
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
            while j < len(txt_lines) and not clean(txt_lines[j]):
                j += 1
            if j >= len(txt_lines) or "{" not in clean(txt_lines[j]):
                i += 1
                continue
            i = j
        depth = 0
        block_lines: list[str] = []
        while i < len(txt_lines):
            s2 = clean(txt_lines[i])
            if s2:
                depth += s2.count("{")
                depth -= s2.count("}")
                block_lines.append(s2)
            i += 1
            if depth <= 0 and block_lines:
                break
        block = "\n".join(block_lines)
        if re.search(r"(?m)^\s*type\s+empty\s*;", block):
            names.append(name)
    return names


def is_case_2d(case_dir: Path) -> bool:
    poly = case_dir / "constant" / "polyMesh"
    boundary_path = poly / "boundary"
    empties = _empty_patch_names(boundary_path)
    if not empties:
        return False
    # Prefer "front/back" style names, but accept any empty patch as a 2D indicator.
    for n in empties:
        nl = n.lower()
        if "front" in nl or "back" in nl:
            return True
    return True


def patch_vertices(case_dir: Path, patch_name: str) -> np.ndarray:
    poly = case_dir / "constant" / "polyMesh"
    boundary_path = poly / "boundary"
    points_path = poly / "points"
    faces_path = poly / "faces"
    if not (boundary_path.exists() and points_path.exists() and faces_path.exists()):
        raise FileNotFoundError("Missing polyMesh files under constant/polyMesh")

    info = _read_boundary_patch_info(boundary_path, patch_name=patch_name)
    if info is None:
        raise ValueError(f"Patch '{patch_name}' not found in {boundary_path}")
    start_face, n_faces = info
    if n_faces < 1:
        raise ValueError(f"Patch '{patch_name}' has no faces in {boundary_path}")

    pts = _read_poly_points(points_path)
    faces = _read_poly_faces(faces_path)
    if start_face + n_faces > len(faces):
        raise ValueError(f"Patch '{patch_name}' face range out of bounds for {faces_path}")

    vidx: set[int] = set()
    for fi in range(start_face, start_face + n_faces):
        for v in faces[fi]:
            vidx.add(int(v))
    if not vidx:
        raise ValueError(f"Patch '{patch_name}' has no vertices in {faces_path}")

    return pts[sorted(vidx), :].astype(float, copy=False)


def cylinder_geometry(case_dir: Path, patch_name: str = "cylinder") -> dict[str, Any]:
    """
    Returns geometry derived from the cylinder wall patch:
      - D: estimated diameter (max of x-extent and y-extent of patch vertices)
      - D_x: x-extent of patch vertices
      - D_y: y-extent of patch vertices
      - span_geom: z-extent of patch vertices
      - span_ref: reference span used for force normalization (1 for strictly-2D/empty cases, else span_geom)
      - Aref: D * span_ref
    """
    poly = case_dir / "constant" / "polyMesh"
    boundary_path = poly / "boundary"
    pts = patch_vertices(case_dir, patch_name=patch_name)
    x0, y0, z0 = np.min(pts, axis=0)
    x1, y1, z1 = np.max(pts, axis=0)
    d_x = float(x1 - x0)
    d_y = float(y1 - y0)
    # Per reporting convention in this project, use the y-extent as the primary D and
    # retain x-extent as a cross-check (for a circular cylinder these should match).
    d = float(d_y)
    span_geom = float(z1 - z0)
    empty_patches = _empty_patch_names(boundary_path)
    is_2d = bool(empty_patches)
    # For strictly 2D cases (with an `empty` patch), use a unit reference span for reporting and
    # coefficient normalization (consistent with common 2D cylinder conventions).
    span_ref = 1.0 if is_2d else span_geom
    a_ref = float(d * span_ref)
    d_ratio = float(d_x / d_y) if (np.isfinite(d_x) and np.isfinite(d_y) and abs(d_y) > 0) else float("nan")
    return {
        "patch": patch_name,
        "D": d,
        "D_x": d_x,
        "D_y": d_y,
        "span_geom": span_geom,
        "span": float(span_ref),
        "span_ref": float(span_ref),
        "Aref": a_ref,
        "is_2d": bool(is_2d),
        "empty_patches": empty_patches,
        "D_x_over_D_y": d_ratio,
        "bounds": {"x": [float(x0), float(x1)], "y": [float(y0), float(y1)], "z": [float(z0), float(z1)]},
    }


@dataclass(frozen=True)
class CircleFit:
    center_xy: tuple[float, float]
    radius: float

    @property
    def diameter(self) -> float:
        return 2.0 * float(self.radius)


def fit_circle_xy(points_xy: np.ndarray) -> CircleFit:
    """
    Algebraic circle fit (Kåsa): solves x^2 + y^2 = a x + b y + c.
    """
    P = np.asarray(points_xy, dtype=float)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError("points_xy must be (n,2)")
    if P.shape[0] < 8:
        raise ValueError("Need at least 8 points for a stable circle fit")
    x = P[:, 0]
    y = P[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x**2 + y**2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, bb, c = sol
    cx = 0.5 * float(a)
    cy = 0.5 * float(bb)
    r2 = float(c) + cx * cx + cy * cy
    r = float(np.sqrt(max(r2, 0.0)))
    return CircleFit(center_xy=(cx, cy), radius=r)


def estimate_cylinder_diameter(case_dir: Path, patch_name: str) -> CircleFit | None:
    """
    Estimate cylinder diameter by fitting a circle to boundary patch vertices.
    Returns None if files/patch not available.
    """
    poly = case_dir / "constant" / "polyMesh"
    boundary_path = poly / "boundary"
    points_path = poly / "points"
    faces_path = poly / "faces"
    if not (boundary_path.exists() and points_path.exists() and faces_path.exists()):
        return None

    info = _read_boundary_patch_info(boundary_path, patch_name=patch_name)
    if info is None:
        return None
    start_face, n_faces = info
    if n_faces < 1:
        return None

    try:
        pts = _read_poly_points(points_path)
        faces = _read_poly_faces(faces_path)
    except Exception:
        return None
    if start_face + n_faces > len(faces):
        return None

    vidx: set[int] = set()
    for fi in range(start_face, start_face + n_faces):
        for v in faces[fi]:
            vidx.add(int(v))
    if not vidx:
        return None

    P = pts[sorted(vidx), :]
    Pxy = P[:, :2]
    try:
        return fit_circle_xy(Pxy)
    except Exception:
        return None


def read_inlet_u_mag(case_dir: Path) -> float | None:
    u0 = case_dir / "0" / "U"
    if not u0.exists():
        return None
    txt = u0.read_text(encoding="utf-8", errors="ignore")

    # Find fixedValue patch values of form: value uniform (Ux Uy Uz);
    patches: list[tuple[str, tuple[float, float, float]]] = []
    for pm in re.finditer(
        r"^\s*([A-Za-z0-9_]+)\s*\{.*?\btype\s+fixedValue\s*;.*?\bvalue\s+uniform\s*\(\s*([^)]+?)\s*\)\s*;.*?\}",
        txt,
        flags=re.M | re.S,
    ):
        nums = [float(x) for x in _FLOAT_RE.findall(pm.group(2))]
        if len(nums) >= 3:
            patches.append((pm.group(1), (nums[0], nums[1], nums[2])))

    if not patches:
        return None
    for preferred in ["inlet", "left", "inflow"]:
        for name, v in patches:
            if name.lower() == preferred:
                return float(np.linalg.norm(np.asarray(v, dtype=float)))
    # Fallback: first fixedValue.
    return float(np.linalg.norm(np.asarray(patches[0][1], dtype=float)))
