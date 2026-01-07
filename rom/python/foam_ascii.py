from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np


def _strip_foam_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//.*?$", "", text, flags=re.M)
    return text


def n_cells_from_polymesh(case_dir: Path) -> int:
    owner = case_dir / "constant" / "polyMesh" / "owner"
    neighbour = case_dir / "constant" / "polyMesh" / "neighbour"
    if not owner.exists():
        raise FileNotFoundError(f"Missing polyMesh owner file: {owner}")

    def _read_int_list(path: Path) -> np.ndarray:
        txt = _strip_foam_comments(path.read_text(encoding="utf-8", errors="ignore"))
        # The owner file structure is: header ... <nEntries> ( ...ints... )
        m = re.search(r"\n\s*(\d+)\s*\(\s*(.*?)\s*\)\s*$", txt, flags=re.S)
        if not m:
            raise ValueError(f"Could not parse integer list from {path}")
        body = m.group(2)
        vals = np.fromstring(body, sep=" ", dtype=np.int64)
        return vals

    owner_vals = _read_int_list(owner)
    max_val = int(owner_vals.max()) if owner_vals.size else 0
    if neighbour.exists():
        neighbour_vals = _read_int_list(neighbour)
        if neighbour_vals.size:
            max_val = max(max_val, int(neighbour_vals.max()))
    return max_val + 1


def read_internal_field(path: Path, n_cells: int) -> np.ndarray:
    txt = _strip_foam_comments(path.read_text(encoding="utf-8", errors="ignore"))
    m = re.search(r"internalField\s+([^;]+);", txt)
    if m and m.group(1).strip().startswith("uniform"):
        payload = m.group(1).strip()
        if "(" in payload:
            vec = re.search(r"uniform\s*\(\s*([^\)]+)\)", payload)
            if not vec:
                raise ValueError(f"Could not parse uniform vector in {path}")
            vals = np.fromstring(vec.group(1), sep=" ", dtype=float)
            if vals.size != 3:
                raise ValueError(f"Expected 3 components for vector in {path}")
            return np.tile(vals[None, :], (n_cells, 1))
        scalar = re.search(r"uniform\s*([^\s]+)", payload)
        if not scalar:
            raise ValueError(f"Could not parse uniform scalar in {path}")
        val = float(scalar.group(1))
        return np.full((n_cells,), val, dtype=float)

    # nonuniform (robust to nested parentheses in vector lists)
    m = re.search(r"internalField\s+nonuniform\s+List<([^>]+)>", txt)
    if not m:
        raise ValueError(f"Could not locate internalField in {path}")
    kind = m.group(1).strip()

    after = txt[m.end() :]
    # Find the entry count (first integer after the type)
    m_n = re.search(r"\b(\d+)\b", after)
    if not m_n:
        raise ValueError(f"Could not parse nonuniform list length in {path}")
    n = int(m_n.group(1))
    after_n = after[m_n.end() :]

    # Find the opening '(' for the list
    open_idx_rel = after_n.find("(")
    if open_idx_rel == -1:
        raise ValueError(f"Could not find opening '(' for internalField list in {path}")
    start = m.end() + m_n.end() + open_idx_rel + (0)  # absolute-ish anchor is not used further
    list_text = after_n[open_idx_rel:]

    # Scan forward to match the outer list parentheses.
    depth = 0
    end_rel = None
    for i, ch in enumerate(list_text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                end_rel = i
                break
    if end_rel is None:
        raise ValueError(f"Could not find matching ')' for internalField list in {path}")
    body = list_text[1:end_rel]  # inside the outer parentheses

    if kind in {"scalar", "double", "float"}:
        arr = np.fromstring(body, sep=" ", dtype=float)
        if arr.size != n:
            raise ValueError(f"{path}: expected {n} scalars, got {arr.size}")
        return arr
    if kind in {"vector"}:
        triples = re.findall(r"\(\s*([^\)]+)\s*\)", body, flags=re.S)
        if len(triples) != n:
            raise ValueError(f"{path}: expected {n} vectors, got {len(triples)}")
        data = np.vstack([np.fromstring(t, sep=" ", dtype=float) for t in triples])
        if data.shape != (n, 3):
            raise ValueError(f"{path}: expected {(n,3)}, got {data.shape}")
        return data

    raise ValueError(f"Unsupported internalField kind {kind!r} in {path}")


def save_meta(path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
