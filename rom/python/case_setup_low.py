from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from config import load_yaml
from geometry import cylinder_geometry, estimate_cylinder_diameter, read_inlet_u_mag


def _scale_block_cells(block_mesh_dict_text: str, factor: float, min_cells: int) -> str:
    def repl(match: re.Match[str]) -> str:
        prefix = match.group(1)
        nx, ny, nz = (int(match.group(i)) for i in range(2, 5))
        nx2 = max(min_cells, int(round(nx * factor)))
        ny2 = max(min_cells, int(round(ny * factor)))
        nz2 = max(min_cells, int(round(nz * factor)))
        return f"{prefix}({nx2} {ny2} {nz2})"

    # Only scale the (nx ny nz) tuple that follows a `hex (v0..v7)` block declaration.
    pat = re.compile(
        r"(\bhex\s*\(\s*[^)]*?\)\s*)\(\s*(\d+)\s+(\d+)\s+(\d+)\s*\)",
        flags=re.S,
    )
    return pat.sub(repl, block_mesh_dict_text)


def _find_matching_brace(text: str, open_index: int) -> int:
    depth = 0
    for i in range(open_index, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("Unbalanced braces while parsing controlDict functions block")


def _functions_block_indices(txt: str) -> tuple[int, int] | None:
    m = re.search(r"(^\s*functions\s*\n\s*\{)", txt, flags=re.M)
    if not m:
        return None
    open_brace = txt.find("{", m.end() - 1)
    close_brace = _find_matching_brace(txt, open_brace)
    return open_brace, close_brace


def _ensure_functions_block(txt: str) -> str:
    if re.search(r"(^\s*functions\s*\n\s*\{)", txt, flags=re.M):
        return txt
    return txt + "\n\nfunctions\n{\n}\n"


def _ensure_function_object(txt: str, name: str, block: str) -> str:
    txt = _ensure_functions_block(txt)
    idx = _functions_block_indices(txt)
    if idx is None:
        raise RuntimeError("Failed to create/locate functions block")
    open_brace, close_brace = idx
    inside = txt[open_brace + 1 : close_brace]
    if re.search(rf"(^\s*{re.escape(name)}\s*\{{)", inside, flags=re.M):
        return txt
    new_inside = inside.rstrip() + "\n" + block.rstrip() + "\n"
    return txt[: open_brace + 1] + new_inside + txt[close_brace:]


def _boundary_patch_names(case_dir: Path) -> list[str]:
    boundary = case_dir / "constant" / "polyMesh" / "boundary"
    if not boundary.exists():
        return []
    lines = boundary.read_text(encoding="utf-8", errors="ignore").splitlines()
    names: list[str] = []
    i = 0
    while i < len(lines) - 1:
        line = re.sub(r"//.*$", "", lines[i]).strip()
        if not line or line.startswith(("/*", "*", ")")):
            i += 1
            continue
        nxt = re.sub(r"//.*$", "", lines[i + 1]).strip()
        if re.fullmatch(r"[A-Za-z0-9_]+", line) and nxt.startswith("{"):
            names.append(line)
        i += 1
    return names


def _ensure_control_dict_settings(case_dir: Path, cfg: dict[str, Any]) -> None:
    control_dict = case_dir / "system" / "controlDict"
    if not control_dict.exists():
        raise FileNotFoundError(f"Missing controlDict: {control_dict}")

    txt = control_dict.read_text(encoding="utf-8", errors="ignore")

    # Avoid replacing keys inside the `functions{}` block when editing top-level entries.
    functions_match = re.search(r"(^\s*functions\s*\n\s*\{)", txt, flags=re.M)
    if functions_match:
        header_txt = txt[: functions_match.start()]
        functions_and_rest = txt[functions_match.start() :]
    else:
        header_txt = txt
        functions_and_rest = ""

    def set_entry(name: str, value: str) -> None:
        nonlocal header_txt
        # Replace "name   ...;" or append if missing.
        pat = re.compile(rf"(^\s*{re.escape(name)}\s+)([^;]+)(;)", flags=re.M)
        if pat.search(header_txt):
            # Avoid regex backreference ambiguity when value begins with digits (e.g. "1.0" -> "\11" group).
            header_txt = pat.sub(lambda m: f"{m.group(1)}{value}{m.group(3)}", header_txt)
        else:
            header_txt += f"\n{name}    {value};\n"

    of = cfg["openfoam"]
    set_entry("endTime", str(of["endTime"]))
    set_entry("deltaT", str(of["deltaT"]))
    set_entry("writeControl", str(of.get("writeControl", "timeStep")))
    set_entry("writeInterval", str(of.get("writeInterval", 1)))
    set_entry("writeFormat", str(of.get("writeFormat", "ascii")))
    set_entry("purgeWrite", str(of.get("purgeWrite", 0)))

    txt = header_txt + functions_and_rest

    # OpenFOAM-13 does not provide the historical `fieldMinMax` functionObject.
    # Use `volFieldValue` equivalents instead (min/max monitoring).
    txt = _ensure_function_object(
        txt,
        "UminMag",
        """
    UminMag
    {
        type            volFieldValue;
        libs            ("libfieldFunctionObjects.so");
        log             yes;
        writeFields     false;
        writeControl    timeStep;
        writeInterval   10;
        cellZone        all;
        operation       minMag;
        fields          (U);
    }
""",
    )
    txt = _ensure_function_object(
        txt,
        "UmaxMag",
        """
    UmaxMag
    {
        type            volFieldValue;
        libs            ("libfieldFunctionObjects.so");
        log             yes;
        writeFields     false;
        writeControl    timeStep;
        writeInterval   10;
        cellZone        all;
        operation       maxMag;
        fields          (U);
    }
""",
    )
    txt = _ensure_function_object(
        txt,
        "pMin",
        """
    pMin
    {
        type            volFieldValue;
        libs            ("libfieldFunctionObjects.so");
        log             yes;
        writeFields     false;
        writeControl    timeStep;
        writeInterval   10;
        cellZone        all;
        operation       min;
        fields          (p);
    }
""",
    )
    txt = _ensure_function_object(
        txt,
        "pMax",
        """
    pMax
    {
        type            volFieldValue;
        libs            ("libfieldFunctionObjects.so");
        log             yes;
        writeFields     false;
        writeControl    timeStep;
        writeInterval   10;
        cellZone        all;
        operation       max;
        fields          (p);
    }
""",
    )

    control_dict.write_text(txt, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--post-mesh",
        action="store_true",
        help="After mesh exists, add cylinder forces/forceCoeffs functionObjects if a cylinder-like patch is detected.",
    )
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    case_dir = Path(cfg["case"]["work_dir"]).resolve()

    block_mesh_dict = case_dir / "system" / "blockMeshDict"
    if block_mesh_dict.exists():
        txt = block_mesh_dict.read_text(encoding="utf-8", errors="ignore")
        factor = float(cfg.get("mesh", {}).get("coarsen_factor", 1.0))
        min_cells = int(cfg.get("mesh", {}).get("min_cells_per_dir", 1))
        if factor != 1.0:
            txt2 = _scale_block_cells(txt, factor=factor, min_cells=min_cells)
            block_mesh_dict.write_text(txt2, encoding="utf-8")

    _ensure_control_dict_settings(case_dir, cfg=cfg)

    if args.post_mesh:
        patches = _boundary_patch_names(case_dir)
        patches_lc = [p.lower() for p in patches]
        cyl_patch = ""
        for cand in patches:
            if "cylinder" in cand.lower():
                cyl_patch = cand
                break
        if not cyl_patch:
            for cand in patches:
                if "obstacle" in cand.lower():
                    cyl_patch = cand
                    break

        if cyl_patch:
            control_dict = case_dir / "system" / "controlDict"
            txt = control_dict.read_text(encoding="utf-8", errors="ignore")
            inlet_u = read_inlet_u_mag(case_dir) or 1.0
            try:
                geom = cylinder_geometry(case_dir, patch_name=cyl_patch)
            except Exception:
                geom = {"D": 1.0, "span_ref": 1.0}
            d_ref = float(geom.get("D", 1.0)) if float(geom.get("D", 1.0)) > 0 else 1.0
            span_ref = float(geom.get("span_ref", 1.0)) if float(geom.get("span_ref", 1.0)) > 0 else 1.0
            a_ref = float(d_ref * span_ref)
            circle = estimate_cylinder_diameter(case_dir, patch_name=cyl_patch)
            cx, cy = circle.center_xy if circle is not None else (0.0, 0.0)
            forces_entry = f"""
    forces1
    {{
        type            forces;
        libs            ("libforces.so");
        writeControl    timeStep;
        writeInterval   1;
        patches         ({cyl_patch});
        rho             rhoInf;
        rhoInf          1;
        CofR            ({cx:g} {cy:g} 0);
        pitchAxis       (0 0 1);
    }}
"""
            force_coeffs_entry = f"""
    forceCoeffs1
    {{
        type            forceCoeffs;
        libs            ("libforces.so");
        writeControl    timeStep;
        writeInterval   1;
        patches         ({cyl_patch});
        rho             rhoInf;
        rhoInf          1;
        CofR            ({cx:g} {cy:g} 0);
        liftDir         (0 1 0);
        dragDir         (1 0 0);
	        pitchAxis       (0 0 1);
	        magUInf         {inlet_u:g};
	        lRef            {d_ref:g};
	        Aref            {a_ref:g};
	    }}
"""
            txt2 = _ensure_function_object(txt, "forces1", forces_entry)
            txt2 = _ensure_function_object(txt2, "forceCoeffs1", force_coeffs_entry)
            if txt2 != txt:
                control_dict.write_text(txt2, encoding="utf-8")
                print(f"[case_setup_low] Added forces/forceCoeffs functionObjects for patch: {cyl_patch}")
        else:
            if patches:
                print(f"[case_setup_low] No cylinder-like patch detected. Patches: {patches}")
            else:
                print("[case_setup_low] No boundary file found yet; skipping forces functionObject.")

    print(f"[case_setup_low] Updated OpenFOAM dictionaries in {case_dir}")


if __name__ == "__main__":
    main()
