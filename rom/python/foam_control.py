from __future__ import annotations

import re
from pathlib import Path


def _find_functions_start(text: str) -> int | None:
    m = re.search(r"(^\s*functions\s*(?:\n|\r\n)\s*\{)", text, flags=re.M)
    return m.start() if m else None


def _split_header_functions(text: str) -> tuple[str, str]:
    idx = _find_functions_start(text)
    if idx is None:
        return text, ""
    return text[:idx], text[idx:]


def set_control_dict_entries(control_dict: Path, entries: dict[str, str]) -> None:
    if not control_dict.exists():
        raise FileNotFoundError(f"Missing controlDict: {control_dict}")
    txt = control_dict.read_text(encoding="utf-8", errors="ignore")
    header, rest = _split_header_functions(txt)

    def set_one(name: str, value: str) -> None:
        nonlocal header
        pat = re.compile(rf"(^\s*{re.escape(name)}\s+)([^;]+)(;)", flags=re.M)
        if pat.search(header):
            header = pat.sub(lambda m: f"{m.group(1)}{value}{m.group(3)}", header)
        else:
            header += f"\n{name}    {value};\n"

    for k, v in entries.items():
        set_one(k, v)

    control_dict.write_text(header + rest, encoding="utf-8")


def detect_runner(case_dir: Path) -> list[str]:
    control_dict = case_dir / "system" / "controlDict"
    txt = control_dict.read_text(encoding="utf-8", errors="ignore")
    solver = ""
    app = ""
    m1 = re.search(r"^\s*solver\s+([^\s;]+)\s*;", txt, flags=re.M)
    if m1:
        solver = m1.group(1)
    m2 = re.search(r"^\s*application\s+([^\s;]+)\s*;", txt, flags=re.M)
    if m2:
        app = m2.group(1)
    if solver:
        return ["foamRun", "-solver", solver]
    if app:
        return [app]
    return ["icoFoam"]
