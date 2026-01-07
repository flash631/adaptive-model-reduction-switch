from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from config import get, load_yaml
from foam_control import detect_runner, set_control_dict_entries
from foam_io import prepare_rom_case


def _parse_execution_clock_time(log_text: str) -> tuple[float | None, float | None]:
    # Matches: "ExecutionTime = 0.560974 s  ClockTime = 1 s"
    pat = re.compile(r"ExecutionTime\s*=\s*([0-9eE+\-\.]+)\s*s\s*ClockTime\s*=\s*([0-9eE+\-\.]+)\s*s")
    matches = list(pat.finditer(log_text))
    if not matches:
        return None, None
    m = matches[-1]
    try:
        return float(m.group(1)), float(m.group(2))
    except Exception:
        return None, None


def _run(cmd: list[str], *, cwd: Path, log_path: Path) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n[cmd] (cwd={cwd}) {' '.join(cmd)}\n")
        f.flush()
        p = subprocess.Popen(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT)
        rc = p.wait()
        if rc != 0:
            raise RuntimeError(f"Command failed (rc={rc}): {' '.join(cmd)} (see {log_path})")
    return time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    base_case = Path(get(cfg, "case.work_dir", "cases/low")).resolve()
    results_dir = Path(get(cfg, "case.results_dir", "results/low")).resolve()

    out_dir = results_dir / "fom_only"
    case_dir = out_dir / "fom_only_case"
    out_dir.mkdir(parents=True, exist_ok=True)

    if case_dir.exists():
        if not args.force:
            raise RuntimeError(f"{case_dir} exists; rerun with --force to overwrite")
        # prepare_rom_case deletes time dirs; easiest is to nuke and recreate.
        for p in case_dir.iterdir():
            if p.is_dir() or p.is_file():
                pass
        # overwrite via prepare_rom_case below

    prepare_rom_case(base_case, case_dir)

    dt = float(get(cfg, "openfoam.deltaT", 0.005))
    t0 = float(get(cfg, "adaptive.start_time", 0.0))
    # Baseline run horizon is controlled by the OpenFOAM settings (case-level endTime),
    # not the adaptive driver horizon.
    t1 = float(get(cfg, "openfoam.endTime", float(get(cfg, "adaptive.end_time", 1.0))))

    # Use config writing cadence (keep disk use manageable for long horizons).
    set_control_dict_entries(
        case_dir / "system" / "controlDict",
        {
            "startFrom": "startTime",
            "startTime": str(t0),
            "endTime": str(t1),
            "deltaT": str(dt),
            "writeControl": str(get(cfg, "openfoam.writeControl", "timeStep")),
            "writeInterval": str(get(cfg, "openfoam.writeInterval", 10)),
            "writeFormat": str(get(cfg, "openfoam.writeFormat", "ascii")),
            "purgeWrite": str(get(cfg, "openfoam.purgeWrite", 0)),
            "runTimeModifiable": "true",
        },
    )

    runner = detect_runner(case_dir)
    log_path = results_dir / "logs" / "fom_only.log"
    if not (case_dir / "constant" / "polyMesh").exists():
        _run(["blockMesh"], cwd=case_dir, log_path=log_path)
    wall_s = _run(runner, cwd=case_dir, log_path=log_path)
    txt = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
    exec_s, clock_s = _parse_execution_clock_time(txt)

    meta: dict[str, Any] = {
        "case_dir": str(case_dir),
        "log_path": str(log_path),
        "t0": t0,
        "t1": t1,
        "deltaT": dt,
        "wall_s": float(wall_s),
        "execution_s": exec_s,
        "clock_s": clock_s,
    }
    (out_dir / "speed.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[fom_only_run] Wrote: {out_dir}/speed.json")
    print(f"[fom_only_run] Case: {case_dir}")


if __name__ == "__main__":
    main()
