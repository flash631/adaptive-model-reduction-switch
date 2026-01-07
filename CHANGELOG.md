# Changelog

## Unreleased

- Added `scripts/verify_forcecoeffs_config.py` to audit `forceCoeffs` patch selection and normalization and emit `results/forcecoeffs_config_{low,high}.json`.
- Added `scripts/compute_force_stats.py` to compute mean $C_d$, RMS $C_l$, and a dominant shedding frequency (Welch/FFT fallback) and emit `results/force_stats_{low,high}.csv` plus report-ready LaTeX tables.
- Added `scripts/render_vorticity_comparison.py` to generate a side-by-side spanwise-vorticity snapshot (FOM vs ROM) without ParaView, written to `report/figures/fom_vs_rom_vortZ_mid_low.png`.
- Added `scripts/render_umag_comparison.py` to generate side-by-side $|U|$ snapshots (no ParaView), written to `report/figures/fom_vs_rom_Umag_mid_{low,high}_clean.png`.
- Updated the no-ParaView render scripts to use consistent `coolwarm` colormaps and per-panel colorbars aligned tight to each subplot (fixes report figure colorbar placement).
- Added `build_report.sh` to run verification + stats + `scripts/07_make_report.sh` and copy `report/build/report.pdf` to `report.pdf` at repo root.
- Tightened high-fidelity robustness by adding an optional coefficient-based ROM gate in `rom/python/adaptive_driver.py` (configurable in `rom/configs/{low,high}.yaml`) that rejects ROM-written states with absurd implied $C_d/C_l$ before starting a FOM burst.
- Cleaned generated BibTeX author formatting for `Benner2020` (Pontes Duff spacing) via `rom/python/references_bib.py`.
- Updated validation error plots to show discrete validation points (markers) in addition to connecting lines.
- Added `scripts/render_snapshots_shared.py` and wired it into `scripts/07_make_report.sh` to regenerate Figures 23/24/27/29 snapshots (U-mag, p, vorticity, POD modes) with consistent colormaps and tight right-side colorbars (no ParaView required).
