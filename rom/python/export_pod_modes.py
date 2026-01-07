from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from config import load_yaml
from foam_io import prepare_rom_case, write_time_fields
from pod import fit_pod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config (low/high)")
    ap.add_argument("--out-case", required=True, help="Output OpenFOAM case directory for modes")
    ap.add_argument("--nmodes", type=int, default=3)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    root = Path(__file__).resolve().parents[2]
    case_dir = (root / cfg["case"]["work_dir"]).resolve()
    results_dir = (root / cfg["case"]["results_dir"]).resolve()

    snap_dir = results_dir / "snapshots"
    U = np.load(snap_dir / "U.npy")  # (n_snap, n_cells, 3)
    meta = json.loads((snap_dir / "meta.json").read_text(encoding="utf-8"))
    n_snap, n_cells = U.shape[0], U.shape[1]

    U_flat = U.reshape(n_snap, n_cells * 3)
    r_u = int(cfg.get("rom", {}).get("r_u", 8))
    pod_u = fit_pod(U_flat, r=max(r_u, args.nmodes), center=True)

    out_case = Path(args.out_case).resolve()
    prepare_rom_case(case_dir, out_case)

    modes_meta = {"n_cells": int(n_cells), "nmodes": int(args.nmodes), "scales": {}}
    zeros_p = np.zeros((n_cells,), dtype=float)
    for k in range(args.nmodes):
        mode = pod_u.basis[:, k].reshape(n_cells, 3)
        # Scale mode for visualization: max |mode| ~= 1
        mag = np.linalg.norm(mode, axis=1)
        s = float(np.max(mag)) if np.max(mag) > 0 else 1.0
        mode_vis = mode / s
        time_name = str(1001 + k)
        write_time_fields(out_case, time_name, U=mode_vis, p=zeros_p, template_time="0")
        modes_meta["scales"][time_name] = s

    (out_case / "modes_meta.json").write_text(json.dumps(modes_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[export_pod_modes] Wrote modes case: {out_case}")


if __name__ == "__main__":
    main()

