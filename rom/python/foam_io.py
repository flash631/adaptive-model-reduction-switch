from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np

from foam_ascii import _strip_foam_comments, n_cells_from_polymesh


def prepare_rom_case(template_case_dir: Path, out_case_dir: Path) -> None:
    if out_case_dir.exists():
        shutil.rmtree(out_case_dir)
    shutil.copytree(template_case_dir, out_case_dir)

    # Remove time dirs except 0/ (keep initial conditions and boundary types).
    for p in out_case_dir.iterdir():
        if p.is_dir():
            name = p.name
            if name == "0" or name == "0.org":
                continue
            try:
                float(name)
            except Exception:
                continue
            shutil.rmtree(p)


def _format_nonuniform_scalar(values: np.ndarray) -> str:
    values = values.reshape(-1)
    n = values.shape[0]
    lines = "\n".join(f"{v:.12g}" for v in values)
    return f"internalField   nonuniform List<scalar>\n{n}\n(\n{lines}\n)\n;"


def _format_nonuniform_vector(values: np.ndarray) -> str:
    values = values.reshape(-1, 3)
    n = values.shape[0]
    lines = "\n".join(f"({v[0]:.12g} {v[1]:.12g} {v[2]:.12g})" for v in values)
    return f"internalField   nonuniform List<vector>\n{n}\n(\n{lines}\n)\n;"


def _replace_internal_field(text: str, replacement_block: str) -> str:
    # Replace uniform or nonuniform internalField statements.
    txt = _strip_foam_comments(text)
    # Find 'internalField ...;' and replace in original text using span indices on original.
    m = re.search(r"internalField\s+[^;]+;", text, flags=re.S)
    if m and "nonuniform" not in m.group(0):
        return text[: m.start()] + replacement_block + text[m.end() :]

    # Nonuniform list uses 'internalField nonuniform ... ( ... ) ;' or 'internalField nonuniform ...;'
    # Replace from 'internalField' up to the terminating ';' after the matching ')'.
    m2 = re.search(r"internalField\s+nonuniform\s+List<[^>]+>\s+\d+\s*\(", text, flags=re.S)
    if not m2:
        # Fallback: just replace first internalField ...; match if present.
        if m:
            return text[: m.start()] + replacement_block + text[m.end() :]
        raise ValueError("Could not locate internalField block to replace")

    start = m2.start()
    # Find matching ')' for the list, then the following ';'
    i0 = text.find("(", m2.end() - 1)
    depth = 0
    end_paren = None
    for i in range(i0, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                end_paren = i
                break
    if end_paren is None:
        raise ValueError("Unbalanced parentheses in internalField list")
    semi = text.find(";", end_paren)
    if semi == -1:
        raise ValueError("Could not find ';' terminating internalField list")
    return text[:start] + replacement_block + text[semi + 1 :]


def write_time_fields(
    case_dir: Path,
    time_name: str,
    *,
    U: np.ndarray,
    p: np.ndarray,
    template_time: str = "0",
) -> None:
    time_dir = case_dir / time_name
    template_dir = case_dir / template_time
    if not template_dir.exists():
        raise FileNotFoundError(f"Template time directory not found: {template_dir}")

    if time_dir.exists():
        shutil.rmtree(time_dir)

    # IMPORTANT: Do not copy the entire previous time directory.
    # Restarting a solver from ROM-written U/p with a copied (inconsistent) phi, time-state,
    # or other auxiliary fields can cause large transients and solver failures.
    # Copy only the required fields we overwrite (U, p) and let OpenFOAM regenerate others.
    time_dir.mkdir(parents=True, exist_ok=True)
    for field_name in ["U", "p"]:
        src = template_dir / field_name
        if not src.exists():
            raise FileNotFoundError(f"Missing field {field_name} in template time: {src}")
        shutil.copy2(src, time_dir / field_name)

    n_cells = n_cells_from_polymesh(case_dir)
    U = np.asarray(U, dtype=float)
    p = np.asarray(p, dtype=float)
    if U.shape != (n_cells, 3):
        raise ValueError(f"U shape mismatch: expected {(n_cells,3)}, got {U.shape}")
    if p.shape != (n_cells,):
        raise ValueError(f"p shape mismatch: expected {(n_cells,)}, got {p.shape}")
    if not np.all(np.isfinite(U)) or not np.all(np.isfinite(p)):
        raise ValueError("NaN/Inf in reconstructed fields")

    U_path = time_dir / "U"
    p_path = time_dir / "p"
    U_text = U_path.read_text(encoding="utf-8", errors="ignore")
    p_text = p_path.read_text(encoding="utf-8", errors="ignore")

    U_new = _replace_internal_field(U_text, _format_nonuniform_vector(U))
    p_new = _replace_internal_field(p_text, _format_nonuniform_scalar(p))

    U_path.write_text(U_new, encoding="utf-8")
    p_path.write_text(p_new, encoding="utf-8")
