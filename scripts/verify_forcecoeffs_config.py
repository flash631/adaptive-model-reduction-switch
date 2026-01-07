#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _strip_foam_comments(text: str) -> str:
    # Remove /* ... */ first (can span lines), then // comments.
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//.*?$", "", text, flags=re.M)
    return text


def _extract_brace_block(text: str, start_idx: int) -> tuple[str, int]:
    if start_idx < 0 or start_idx >= len(text) or text[start_idx] != "{":
        raise ValueError("start_idx must point to '{'")
    depth = 0
    i = start_idx
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start_idx : i + 1], i + 1
        i += 1
    raise ValueError("Unbalanced braces while parsing dictionary")


def _find_named_block(text: str, name: str) -> str | None:
    # Match: <name> { ... }
    m = re.search(r"(^|\s)" + re.escape(name) + r"\s*\{", text, flags=re.M)
    if not m:
        return None
    brace_idx = text.find("{", m.end() - 1)
    if brace_idx < 0:
        return None
    blk, _ = _extract_brace_block(text, brace_idx)
    return blk


def _parse_keyvals_one_level(block_text: str) -> dict[str, str]:
    """
    Parse key/value pairs of the form: key  value;
    Best-effort and intentionally minimal (enough for forceCoeffs audit).
    """
    txt = block_text.strip()
    if txt.startswith("{") and txt.endswith("}"):
        txt = txt[1:-1]
    out: dict[str, str] = {}
    for line in txt.splitlines():
        s = line.strip()
        if not s or s in ("{", "}"):
            continue
        if s.startswith("#include"):
            continue
        m = re.match(r"^([A-Za-z0-9_]+)\s+(.*?)\s*;\s*$", s)
        if not m:
            continue
        out[m.group(1)] = m.group(2).strip()
    return out


def _parse_paren_list(value: str) -> list[str]:
    v = value.strip()
    if not (v.startswith("(") and v.endswith(")")):
        return [v]
    body = v[1:-1].strip()
    if not body:
        return []
    return [t for t in re.split(r"\s+", body) if t]


def _parse_vec3(value: str) -> tuple[float, float, float] | None:
    v = value.strip()
    if not (v.startswith("(") and v.endswith(")")):
        return None
    nums = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", v)
    if len(nums) < 3:
        return None
    try:
        return float(nums[0]), float(nums[1]), float(nums[2])
    except Exception:
        return None


def _parse_float(value: str) -> float | None:
    try:
        return float(value.strip())
    except Exception:
        return None


def _detect_cylinder_patch_name(case_dir: Path) -> str:
    boundary = case_dir / "constant" / "polyMesh" / "boundary"
    if not boundary.exists():
        return "cylinder"
    txt = boundary.read_text(encoding="utf-8", errors="ignore")
    for cand in ("cylinder", "Cylinder", "obstacle", "Obstacle"):
        if re.search(rf"^\s*{re.escape(cand)}\s*$", txt, flags=re.M):
            return cand
    # Fallback: take first patch containing 'cylinder'.
    for m in re.finditer(r"^\s*([A-Za-z0-9_]+)\s*\{", txt, flags=re.M):
        name = m.group(1)
        if "cylinder" in name.lower():
            return name
    return "cylinder"


def _is_case_2d(case_dir: Path) -> bool:
    boundary = case_dir / "constant" / "polyMesh" / "boundary"
    if not boundary.exists():
        return False
    txt = boundary.read_text(encoding="utf-8", errors="ignore")
    return bool(re.search(r"(?m)^\s*type\s+empty\s*;", txt))


def _domain_bounds_from_points(case_dir: Path) -> dict[str, float] | None:
    pts_path = case_dir / "constant" / "polyMesh" / "points"
    if not pts_path.exists():
        return None
    txt = pts_path.read_text(encoding="utf-8", errors="ignore")
    nums = re.findall(r"\(\s*([-+0-9Ee\.]+)\s+([-+0-9Ee\.]+)\s+([-+0-9Ee\.]+)\s*\)", txt)
    if not nums:
        return None
    xs = [float(a) for a, _, _ in nums]
    ys = [float(b) for _, b, _ in nums]
    zs = [float(c) for _, _, c in nums]
    return {
        "xmin": float(min(xs)),
        "xmax": float(max(xs)),
        "ymin": float(min(ys)),
        "ymax": float(max(ys)),
        "zmin": float(min(zs)),
        "zmax": float(max(zs)),
    }


def _read_inlet_u_mag(case_dir: Path) -> float | None:
    u0 = case_dir / "0" / "U"
    if not u0.exists():
        return None
    txt = _strip_foam_comments(u0.read_text(encoding="utf-8", errors="ignore"))
    # Find fixedValue patch values of form: value uniform (Ux Uy Uz);
    for m in re.finditer(
        r"^\s*([A-Za-z0-9_]+)\s*\{.*?\btype\s+fixedValue\s*;.*?\bvalue\s+uniform\s*\(\s*([^)]+?)\s*\)\s*;.*?\}",
        txt,
        flags=re.M | re.S,
    ):
        nums = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", m.group(2))
        if len(nums) >= 3:
            try:
                ux, uy, uz = float(nums[0]), float(nums[1]), float(nums[2])
                return float((ux * ux + uy * uy + uz * uz) ** 0.5)
            except Exception:
                continue
    return None


def _parse_forcecoeffs_from_control_dict(control_dict: Path) -> dict[str, dict[str, str]]:
    txt = _strip_foam_comments(control_dict.read_text(encoding="utf-8", errors="ignore"))
    # OpenFOAM commonly formats this as:
    #   functions
    #   {
    # so the opening brace may be on the next line.
    m = re.search(r"(?m)^[ \t]*functions[ \t]*$", txt)
    if not m:
        return {}
    brace_idx = txt.find("{", m.end())
    if brace_idx < 0:
        return {}
    functions_block, _ = _extract_brace_block(txt, brace_idx)
    body = functions_block[1:-1]

    out: dict[str, dict[str, str]] = {}
    i = 0
    while i < len(body):
        # Skip whitespace.
        while i < len(body) and body[i].isspace():
            i += 1
        if i >= len(body):
            break
        # Read identifier.
        mname = re.match(r"([A-Za-z0-9_]+)", body[i:])
        if not mname:
            i += 1
            continue
        name = mname.group(1)
        i += len(name)
        while i < len(body) and body[i].isspace():
            i += 1
        if i >= len(body) or body[i] != "{":
            continue
        blk, j = _extract_brace_block(body, i)
        i = j
        kv = _parse_keyvals_one_level(blk)
        if kv.get("type", "").strip() == "forceCoeffs":
            out[name] = kv
    return out


def _parse_forcecoeffs_simple_dict(path: Path) -> dict[str, str]:
    txt = _strip_foam_comments(path.read_text(encoding="utf-8", errors="ignore"))
    return _parse_keyvals_one_level(txt)


@dataclass(frozen=True)
class ForceCoeffsConfig:
    name: str
    source: str
    patches: list[str]
    dragDir: tuple[float, float, float] | None
    liftDir: tuple[float, float, float] | None
    magUInf: float | None
    lRef: float | None
    Aref: float | None
    rho: str | None
    rhoInf: float | None
    CofR: tuple[float, float, float] | None
    pitchAxis: tuple[float, float, float] | None


def _as_forcecoeffs_config(name: str, source: str, kv: dict[str, str]) -> ForceCoeffsConfig:
    patches = _parse_paren_list(kv.get("patches", ""))
    return ForceCoeffsConfig(
        name=name,
        source=source,
        patches=patches,
        dragDir=_parse_vec3(kv.get("dragDir", "")),
        liftDir=_parse_vec3(kv.get("liftDir", "")),
        magUInf=_parse_float(kv.get("magUInf", "")),
        lRef=_parse_float(kv.get("lRef", "")),
        Aref=_parse_float(kv.get("Aref", "")),
        rho=kv.get("rho"),
        rhoInf=_parse_float(kv.get("rhoInf", "")),
        CofR=_parse_vec3(kv.get("CofR", "")),
        pitchAxis=_parse_vec3(kv.get("pitchAxis", "")),
    )


def _main_one(case_dir: Path, case_name: str, *, out_dir: Path) -> Path:
    case_dir = case_dir.resolve()
    if not case_dir.exists():
        raise FileNotFoundError(case_dir)

    cyl_patch = _detect_cylinder_patch_name(case_dir)
    is_2d = _is_case_2d(case_dir)
    inlet_u = _read_inlet_u_mag(case_dir)
    bounds = _domain_bounds_from_points(case_dir)
    beta = None
    if bounds is not None:
        H = float(bounds["ymax"] - bounds["ymin"])
        # Prefer lRef from forceCoeffs if present (reference length == D).
        # We'll fill this after we parse configs.
        beta = float("nan") if H <= 0 else None

    configs: list[ForceCoeffsConfig] = []

    # controlDict forceCoeffs
    cd = case_dir / "system" / "controlDict"
    if cd.exists():
        fcs = _parse_forcecoeffs_from_control_dict(cd)
        for name, kv in sorted(fcs.items()):
            configs.append(_as_forcecoeffs_config(name, "system/controlDict:functions", kv))

    # standalone function dicts used by foamPostProcess (if present)
    for fn in ("forceCoeffsIncompressible", "forceCoeffsROM", "forceCoeffs"):
        p = case_dir / "system" / fn
        if p.exists() and p.is_file():
            kv = _parse_forcecoeffs_simple_dict(p)
            # Only treat it as a forceCoeffs dict if it declares patches + Aref.
            if "patches" in kv or "Aref" in kv:
                configs.append(_as_forcecoeffs_config(fn, f"system/{fn}", kv))

    if not configs:
        raise RuntimeError(f"No forceCoeffs configuration found under {case_dir}/system")

    # Validation
    errors: list[str] = []
    for c in configs:
        if [p for p in c.patches if p] != [cyl_patch]:
            errors.append(
                f"{case_name}:{c.source}:{c.name}: patches must be ({cyl_patch}) only; found ({' '.join(c.patches)})"
            )
        if c.Aref is None or not (c.Aref > 0):
            errors.append(f"{case_name}:{c.source}:{c.name}: Aref must be > 0 (found {c.Aref})")
        if c.lRef is None or not (c.lRef > 0):
            errors.append(f"{case_name}:{c.source}:{c.name}: lRef must be > 0 (found {c.lRef})")
        if c.magUInf is None or not (c.magUInf > 0):
            errors.append(f"{case_name}:{c.source}:{c.name}: magUInf must be > 0 (found {c.magUInf})")
        if c.dragDir is None or c.liftDir is None:
            errors.append(f"{case_name}:{c.source}:{c.name}: dragDir/liftDir must be set")
        if c.rho == "rhoInf" and (c.rhoInf is None or not (c.rhoInf > 0)):
            errors.append(f"{case_name}:{c.source}:{c.name}: rho=rhoInf but rhoInf missing/invalid ({c.rhoInf})")

    if errors:
        msg = "\n".join(f"ERROR: {e}" for e in errors)
        raise RuntimeError(msg)

    # Fill beta using lRef as D (reference diameter) when available.
    if bounds is not None:
        H = float(bounds["ymax"] - bounds["ymin"])
        d_ref = next((c.lRef for c in configs if c.lRef and c.lRef > 0), None)
        if d_ref is not None and H > 0:
            beta = float(d_ref / H)

    # Emit JSON summary
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"forcecoeffs_config_{case_name}.json"

    def cfg_to_dict(c: ForceCoeffsConfig) -> dict[str, Any]:
        return {
            "name": c.name,
            "source": c.source,
            "patches": c.patches,
            "dragDir": c.dragDir,
            "liftDir": c.liftDir,
            "magUInf": c.magUInf,
            "lRef": c.lRef,
            "Aref": c.Aref,
            "rho": c.rho,
            "rhoInf": c.rhoInf,
            "CofR": c.CofR,
            "pitchAxis": c.pitchAxis,
        }

    summary = {
        "case": case_name,
        "case_dir": str(case_dir),
        "cylinder_patch": cyl_patch,
        "is_2d": bool(is_2d),
        "inlet_magU": inlet_u,
        "domain_bounds": bounds,
        "blockage_beta": beta,
        "forceCoeffs": [cfg_to_dict(c) for c in configs],
        "notes": [
            "Checks enforced: forceCoeffs patches must be cylinder-only; Aref/lRef/magUInf must be positive.",
            "Convention: for 2D cases, coefficients are interpreted per unit span.",
        ],
    }
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Console output (report-friendly)
    print(f"[verify_forcecoeffs_config] case={case_name} case_dir={case_dir}")
    print(f"[verify_forcecoeffs_config] cylinder_patch={cyl_patch} is_2d={is_2d} inlet_magU={inlet_u}")
    if beta is not None:
        print(f"[verify_forcecoeffs_config] blockage_beta=D/H={beta:.6g}")
    for c in configs:
        print(
            f"[verify_forcecoeffs_config] {c.source}:{c.name}: patches={c.patches} "
            f"magUInf={c.magUInf} lRef={c.lRef} Aref={c.Aref} "
            f"dragDir={c.dragDir} liftDir={c.liftDir} rho={c.rho} rhoInf={c.rhoInf}"
        )
    print(f"[verify_forcecoeffs_config] Wrote: {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit OpenFOAM forceCoeffs patch selection + normalization.")
    ap.add_argument(
        "--case",
        action="append",
        default=[],
        help="Case spec as name=path (repeatable), e.g. low=cases/low",
    )
    ap.add_argument("--out-dir", default="results", help="Output directory for JSON summaries")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = (root / args.out_dir).resolve()

    specs = list(args.case)
    if not specs:
        specs = ["low=cases/low", "high=cases/high"]

    wrote: list[Path] = []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--case must be name=path, got: {spec}")
        name, rel = spec.split("=", 1)
        wrote.append(_main_one((root / rel).resolve(), name.strip(), out_dir=out_dir))

    if not wrote:
        raise SystemExit(2)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[verify_forcecoeffs_config] FAIL: {exc}", file=sys.stderr)
        raise
