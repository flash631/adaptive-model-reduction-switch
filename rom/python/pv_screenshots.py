from __future__ import annotations

import argparse
import subprocess
import sys
import shutil
from pathlib import Path


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed (rc={p.returncode}): {' '.join(cmd)}\n{p.stdout}")


def _have_pvpython() -> bool:
    return shutil.which("pvpython") is not None


def _have_foamToVTK() -> bool:
    return shutil.which("foamToVTK") is not None


def _vtk_has_field(vtk_dir: Path, pattern: str, field: str) -> bool:
    p = next(iter(vtk_dir.glob(pattern)), None)
    if p is None or not p.exists():
        return False
    try:
        return field.encode("utf-8") in p.read_bytes()
    except Exception:
        return False


def _ensure_vtk_for_case(case_dir: Path, times: list[str], expected_globs: list[str], fields: str = "(U p)") -> None:
    if not _have_foamToVTK():
        return
    vtk_dir = case_dir / "VTK"
    vtk_dir.mkdir(parents=True, exist_ok=True)
    if all(any(vtk_dir.glob(pat)) for pat in expected_globs):
        return
    time_sel = ",".join(times)
    _run(["foamToVTK", "-useTimeName", "-time", time_sel, "-fields", fields], cwd=case_dir)


def _have_postProcess() -> bool:
    return subprocess.run(["bash", "-lc", "command -v postProcess"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def _ensure_vorticity_field(case_dir: Path, time_name: str) -> None:
    if not _have_postProcess():
        return
    field_path = case_dir / time_name / "vorticity"
    if field_path.exists():
        return
    _run(["postProcess", "-func", "vorticity", "-time", time_name], cwd=case_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["wrap", "pv"], default="wrap")
    ap.add_argument("--figdir", default="report/figures")
    ap.add_argument("--low-vtk-dir", default="cases/low/VTK")
    ap.add_argument("--rom-case", default="results/low/rom_case")
    ap.add_argument("--modes-case", default="results/low/modes_case")
    ap.add_argument("--high-case", default="cases/high")
    ap.add_argument("--high-fom-case", default="results/high/fom_only/fom_only_case")
    ap.add_argument("--high-adaptive-case", default="results/high/adaptive/adaptive_case")
    ap.add_argument("--dt-low", type=float, default=0.005)
    ap.add_argument("--dt-high", type=float, default=0.0025)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    figdir = (root / args.figdir).resolve()
    figdir.mkdir(parents=True, exist_ok=True)

    if args.mode == "wrap":
        if not _have_pvpython():
            print("[pv_screenshots] pvpython not found. Manual screenshot checklist:", file=sys.stderr)
            print("- Open `cases/low/low.foam` (create empty `low.foam` if needed)", file=sys.stderr)
            print("- Save mesh whole + zoom near cylinder", file=sys.stderr)
            print("- Save U-mag and p at t=0,0.5,1.0", file=sys.stderr)
            print("- Save vorticity magnitude at t=0.5", file=sys.stderr)
            print("- Save side-by-side FOM vs ROM U-mag at t=0.5", file=sys.stderr)
            print("- Save first 3 velocity modes (from results/low/modes_case)", file=sys.stderr)
            print("- (High) Save mesh whole + zoom, U-mag and vorticity at t=0.5", file=sys.stderr)
            print("- (High) Save side-by-side FOM-only vs adaptive (ROM) U-mag at t=0.5", file=sys.stderr)
            return

        # Ensure VTK for ROM case at t=0.5 and for modes.
        rom_case = (root / args.rom_case).resolve()
        try:
            _ensure_vorticity_field(rom_case, time_name="0.5")
        except Exception:
            pass
        rom_fields = "(U p vorticity)" if (rom_case / "0.5" / "vorticity").exists() else "(U p)"
        _ensure_vtk_for_case(
            rom_case,
            times=["0.5"],
            expected_globs=["rom_case_0.5*.vtk"],
            fields=rom_fields,
        )
        _ensure_vtk_for_case(
            (root / args.modes_case).resolve(),
            times=["1001", "1002", "1003"],
            expected_globs=["modes_case_1001*.vtk", "modes_case_1002*.vtk", "modes_case_1003*.vtk"],
        )
        # Ensure vorticity field and VTK at t=0.5 for the low case (used for the vorticity figure).
        low_case = (root / "cases/low").resolve()
        try:
            _ensure_vorticity_field(low_case, time_name="0.5")
        except Exception:
            pass
        low_fields = "(U p vorticity)" if (low_case / "0.5" / "vorticity").exists() else "(U p)"
        _ensure_vtk_for_case(
        low_case,
        times=["0.5"],
        expected_globs=["low_0.5*.vtk"],
        fields=low_fields,
    )
        # If VTK exists but is missing vorticity (e.g., previous export without the field), re-export.
        if "vorticity" in low_fields and not _vtk_has_field(low_case / "VTK", "low_0.5*.vtk", "vorticity"):
            _run(["foamToVTK", "-useTimeName", "-time", "0.5", "-fields", low_fields], cwd=low_case)

        # Ensure vorticity field and VTK at t=0.5 for the high case (used for the vorticity figure).
        high_case = (root / args.high_case).resolve()
        try:
            _ensure_vorticity_field(high_case, time_name="0.5")
        except Exception:
            pass
        high_fields = "(U p vorticity)" if (high_case / "0.5" / "vorticity").exists() else "(U p)"
        _ensure_vtk_for_case(
        high_case,
        times=["0.5"],
        expected_globs=["high_0.5*.vtk"],
        fields=high_fields,
    )
        if "vorticity" in high_fields and not _vtk_has_field(high_case / "VTK", "high_0.5*.vtk", "vorticity"):
            _run(["foamToVTK", "-useTimeName", "-time", "0.5", "-fields", high_fields], cwd=high_case)

        # Ensure VTK at t=0.5 for high-fidelity FOM-only and adaptive cases (for FOM-vs-ROM comparison).
        high_fom_case = (root / args.high_fom_case).resolve()
        high_adaptive_case = (root / args.high_adaptive_case).resolve()
        _ensure_vtk_for_case(
            high_fom_case,
            times=["0.5"],
            expected_globs=["fom_only_case_0.5*.vtk"],
            fields="(U p)",
        )
        _ensure_vtk_for_case(
            high_adaptive_case,
            times=["0.5"],
            expected_globs=["adaptive_case_0.5*.vtk"],
            fields="(U p)",
        )

        # Re-run self under pvpython (prefer offscreen; fall back to manual if unavailable).
        try:
            pv_cmd_off = [
                "pvpython",
                "--force-offscreen-rendering",
                "--disable-xdisplay-test",
                str(Path(__file__).resolve()),
                "--mode",
                "pv",
                "--figdir",
                str(figdir),
            ]
            _run(pv_cmd_off)
            print(f"[pv_screenshots] Wrote screenshots to: {figdir}")
            # Autocrop key screenshots so LaTeX subfigures fill their boxes even when the user
            # rebuilds the PDF without running scripts/07_make_report.sh.
            try:
                import subprocess

                crop_cmd = [
                    sys.executable,
                    str((root / "rom/python/autocrop.py").resolve()),
                    "--inplace",
                    "--pad",
                    "8",
                    "--tol",
                    "10",
                    "--collapse-vertical",
                    "--collapse-horizontal",
                    "--gap-min",
                    "80",
                    "--gap-pad",
                    "12",
                    str(figdir / "mesh_low.png"),
                    str(figdir / "mesh_low_zoom.png"),
                    str(figdir / "mesh_high.png"),
                    str(figdir / "mesh_high_zoom.png"),
                    str(figdir / "low_vort_mid.png"),
                    str(figdir / "high_vort_mid.png"),
                ]
                subprocess.run(crop_cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
        except Exception as exc:
            # Retry without offscreen flags (onscreen rendering).
            try:
                pv_cmd_on = [
                    "pvpython",
                    str(Path(__file__).resolve()),
                    "--mode",
                    "pv",
                    "--figdir",
                    str(figdir),
                ]
                _run(pv_cmd_on)
                print(f"[pv_screenshots] Wrote screenshots to: {figdir}")
                return
            except Exception as exc2:
                print(f"[pv_screenshots] pvpython screenshot generation failed: {exc2}", file=sys.stderr)
            print("[pv_screenshots] Manual screenshot checklist:", file=sys.stderr)
            print("- Open `cases/low/VTK/low_0.vtk` for mesh screenshots", file=sys.stderr)
            print("- Save mesh whole + zoom near cylinder", file=sys.stderr)
            print("- Save U-mag and p at steps 0,100,200", file=sys.stderr)
            print("- Save vorticity magnitude at step 100", file=sys.stderr)
            print("- Save side-by-side FOM (low_100.vtk) vs ROM (rom_case_0.5.vtk)", file=sys.stderr)
            print("- Save first 3 velocity modes (modes_case_1001/1002/1003.vtk)", file=sys.stderr)
            print("- (High) Open `cases/high/VTK/high_0.5.vtk` for high-fidelity views", file=sys.stderr)
            print("- (High) Save mesh whole + zoom, U-mag and vorticity at t=0.5", file=sys.stderr)
            print("- (High) Save side-by-side FOM-only vs adaptive at t=0.5", file=sys.stderr)
        return

    # pvpython mode
    try:
        from paraview.simple import (  # type: ignore
            Calculator,
            CellDatatoPointData,
            ColorBy,
            Delete,
            CreateView,
            GetActiveViewOrCreate,
            GetColorTransferFunction,
            GetLayout,
            Hide,
            Gradient,
            LegacyVTKReader,
            Render,
            UpdatePipeline,
            SaveScreenshot,
            Show,
            ResetCamera,
            GetScalarBar,
            Slice,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("This mode must be run under pvpython") from exc

    low_vtk_dir = (root / "cases/low/VTK").resolve()
    high_vtk_dir = (root / "cases/high/VTK").resolve()

    def low_file(step: int) -> Path:
        return low_vtk_dir / f"low_{step}.vtk"

    # Representative times for low case: 0, 0.5, 1.0 (dt=0.005 => steps 0,100,200)
    steps = {"t0": 0, "tmid": 100, "tend": 200}

    view = GetActiveViewOrCreate("RenderView")
    # Larger render size for report-quality figures (cropped later).
    view.ViewSize = [2200, 1200]
    view.CameraParallelProjection = 1
    view.OrientationAxesVisibility = 0
    view.UseColorPaletteForBackground = 0
    view.Background = [1.0, 1.0, 1.0]

    def setup_camera(bounds: tuple[float, float, float, float, float, float], zoom: float = 1.0) -> None:
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        cz = 0.5 * (zmin + zmax)
        view.CameraFocalPoint = [cx, cy, cz]
        view.CameraPosition = [cx, cy, cz + 10.0]
        view.CameraViewUp = [0.0, 1.0, 0.0]
        dx = max(xmax - xmin, 1e-12)
        dy = max(ymax - ymin, 1e-12)
        vw, vh = float(view.ViewSize[0]), float(view.ViewSize[1])
        aspect = (vw / vh) if vh > 0 else 1.0
        half_height = 0.5 * dy
        half_width = 0.5 * dx
        # ParaView's CameraParallelScale is half the visible height in world units.
        # Choose a scale that fits both x and y extents for the current aspect ratio.
        scale = max(half_height, half_width / max(aspect, 1e-12))
        view.CameraParallelScale = 1.05 * scale / max(zoom, 1e-6)

    def _bounds(reader) -> tuple[float, float, float, float, float, float]:
        # Bounds can be (0,0,0,0,0,0) until pipeline updates.
        try:
            UpdatePipeline(proxy=reader)
        except TypeError:
            # Older ParaView: UpdatePipeline() without keyword args.
            try:
                UpdatePipeline()
            except Exception:
                pass
        b = reader.GetDataInformation().GetBounds()
        return (float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4]), float(b[5]))

    def setup_camera_focus(bounds: tuple[float, float, float, float, float, float], margin: float = 2.5) -> None:
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        cz = 0.5 * (zmin + zmax)
        dx = max(xmax - xmin, 1e-12)
        dy = max(ymax - ymin, 1e-12)
        vw, vh = float(view.ViewSize[0]), float(view.ViewSize[1])
        aspect = (vw / vh) if vh > 0 else 1.0
        half_height = 0.5 * dy
        half_width = 0.5 * dx
        scale = max(half_height, half_width / max(aspect, 1e-12))
        view.CameraFocalPoint = [cx, cy, cz]
        view.CameraPosition = [cx, cy, cz + 10.0]
        view.CameraViewUp = [0.0, 1.0, 0.0]
        view.CameraParallelScale = float(margin) * scale

    def safe_color_by(disp, array_name: str) -> None:
        # Prefer point data, but fall back to cell data if needed.
        try:
            ColorBy(disp, ("POINTS", array_name))
            return
        except Exception:
            pass
        ColorBy(disp, ("CELLS", array_name))

    def array_range(proxy, array_name: str) -> tuple[float, float] | None:
        info = proxy.GetDataInformation()
        try:
            ai = info.GetPointDataInformation().GetArrayInformation(array_name)
            if ai is not None:
                lo, hi = ai.GetComponentRange(0)
                return float(lo), float(hi)
        except Exception:
            pass
        try:
            ai = info.GetCellDataInformation().GetArrayInformation(array_name)
            if ai is not None:
                lo, hi = ai.GetComponentRange(0)
                return float(lo), float(hi)
        except Exception:
            pass
        return None

    def has_array(reader, array_name: str) -> bool:
        info = reader.GetDataInformation()
        try:
            if info.GetPointDataInformation().GetArrayInformation(array_name) is not None:
                return True
        except Exception:
            pass
        try:
            if info.GetCellDataInformation().GetArrayInformation(array_name) is not None:
                return True
        except Exception:
            pass
        return False

    def show_scalar_bar(lut_name: str, title: str) -> None:
        lut = GetColorTransferFunction(lut_name)
        try:
            sb = GetScalarBar(lut, view)
            sb.Title = title
            sb.ComponentTitle = ""
            sb.Visibility = 1
            # Keep the scalar bar attached to the right side of the figure.
            # Use black text since the background is white.
            for attr, val in [
                ("TitleFontSize", 14),
                ("LabelFontSize", 12),
                ("WindowLocation", "AnyLocation"),
                ("Orientation", "Vertical"),
                # Place inside the view near the right edge and keep it short so it doesn't
                # force huge whitespace below the domain (cropping would otherwise keep the bar).
                ("Position", [0.92, 0.62]),
                ("ScalarBarLength", 0.30),
                ("ScalarBarThickness", 14),
                ("TitleColor", [0.0, 0.0, 0.0]),
                ("LabelColor", [0.0, 0.0, 0.0]),
            ]:
                try:
                    setattr(sb, attr, val)
                except Exception:
                    pass
        except Exception:
            pass

    # Mesh screenshots (use t0 file)
    r0 = LegacyVTKReader(FileNames=[str(low_file(steps["t0"]))])
    d0 = _bounds(r0)
    disp0 = Show(r0, view)
    disp0.Representation = "Surface With Edges"
    disp0.ColorArrayName = [None, ""]
    disp0.LineWidth = 2.0
    try:
        disp0.EdgeColor = [0.1, 0.1, 0.1]
        disp0.DiffuseColor = [0.85, 0.85, 0.85]
        disp0.AmbientColor = [0.85, 0.85, 0.85]
        disp0.Lighting = 0
    except Exception:
        pass
    setup_camera(d0, zoom=1.0)
    Render()
    SaveScreenshot(str(figdir / "mesh_low.png"), view)

    # Zoom near cylinder: use cylinder patch bounds if available to avoid zooming into the hole.
    cyl0 = low_vtk_dir / "cylinder" / "cylinder_0.vtk"
    if cyl0.exists():
        rc = LegacyVTKReader(FileNames=[str(cyl0)])
        # margin>1 zooms OUT; keep close to the patch bounds for a true zoom-in.
        setup_camera_focus(_bounds(rc), margin=2.0)
        Hide(rc, view)
    else:
        setup_camera(d0, zoom=3.0)
    Render()
    SaveScreenshot(str(figdir / "mesh_low_zoom.png"), view)
    Hide(r0, view)

    # ROM VTK at t=0.5 (used for side-by-side comparisons).
    rom_vtk = (root / "results/low/rom_case/VTK").resolve()
    rom_file = next(iter(rom_vtk.glob("rom_case_0.5*.vtk")), None)

    def render_Umag(step: int, out: Path, *, fixed_range: tuple[float, float] | None = None) -> None:
        r = LegacyVTKReader(FileNames=[str(low_file(step))])
        calc = Calculator(Input=r)
        calc.ResultArrayName = "Umagn"
        calc.Function = "mag(U)"
        disp = Show(calc, view)
        safe_color_by(disp, "Umagn")
        try:
            disp.SetScalarBarVisibility(view, True)
        except Exception:
            pass
        lut = GetColorTransferFunction("Umagn")
        if fixed_range is not None:
            try:
                lut.RescaleTransferFunction(float(fixed_range[0]), float(fixed_range[1]))
            except Exception:
                pass
        else:
            try:
                lut.RescaleTransferFunctionToDataRange(True, False)
            except TypeError:
                lut.RescaleTransferFunctionToDataRange()
        show_scalar_bar("Umagn", "|U|")
        setup_camera(r.GetDataInformation().GetBounds(), zoom=1.0)
        Render()
        SaveScreenshot(str(out), view)
        Hide(calc, view)
        Hide(r, view)
        try:
            Delete(calc)
            Delete(r)
        except Exception:
            pass

    def render_p(step: int, out: Path) -> None:
        r = LegacyVTKReader(FileNames=[str(low_file(step))])
        disp = Show(r, view)
        safe_color_by(disp, "p")
        try:
            disp.SetScalarBarVisibility(view, True)
        except Exception:
            pass
        lut = GetColorTransferFunction("p")
        try:
            lut.RescaleTransferFunctionToDataRange(True, False)
        except TypeError:
            lut.RescaleTransferFunctionToDataRange()
        show_scalar_bar("p", "p")
        setup_camera(r.GetDataInformation().GetBounds(), zoom=1.0)
        Render()
        SaveScreenshot(str(out), view)
        Hide(r, view)

    # Shared |U| scale for low FOM t=0.5 vs ROM t=0.5.
    shared_low_umag: tuple[float, float] | None = None
    if rom_file is not None:
        try:
            rA = LegacyVTKReader(FileNames=[str(low_file(steps["tmid"]))])
            cA = Calculator(Input=rA)
            cA.ResultArrayName = "Umagn"
            cA.Function = "mag(U)"
            UpdatePipeline(proxy=cA)
            ra = array_range(cA, "Umagn")

            rB = LegacyVTKReader(FileNames=[str(rom_file)])
            cB = Calculator(Input=rB)
            cB.ResultArrayName = "Umagn"
            cB.Function = "mag(U)"
            UpdatePipeline(proxy=cB)
            rb = array_range(cB, "Umagn")
            if ra is not None and rb is not None:
                shared_low_umag = (min(ra[0], rb[0]), max(ra[1], rb[1]))
            for pxy in [cA, rA, cB, rB]:
                try:
                    Delete(pxy)
                except Exception:
                    pass
        except Exception:
            shared_low_umag = None

    render_Umag(steps["t0"], figdir / "U_low_t0.png")
    render_Umag(steps["tmid"], figdir / "U_low_tmid.png", fixed_range=shared_low_umag)
    render_Umag(steps["tend"], figdir / "U_low_tend.png")

    render_p(steps["t0"], figdir / "p_low_t0.png")
    render_p(steps["tmid"], figdir / "p_low_tmid.png")
    render_p(steps["tend"], figdir / "p_low_tend.png")

    # Spanwise vorticity at mid time (prefer OpenFOAM-computed `vorticity` field; fall back to curl(U)).
    low_vtk_time = next(iter(low_vtk_dir.glob("low_0.5*.vtk")), None)
    if low_vtk_time is not None:
        r = LegacyVTKReader(FileNames=[str(low_vtk_time)])
        try:
            UpdatePipeline(proxy=r)
        except TypeError:
            try:
                UpdatePipeline()
            except Exception:
                pass
        has_vort = has_array(r, "vorticity")
    else:
        # Fallback: compute from U (may be less reliable depending on data association).
        r = LegacyVTKReader(FileNames=[str(low_file(steps["tmid"]))])
        try:
            UpdatePipeline(proxy=r)
        except TypeError:
            try:
                UpdatePipeline()
            except Exception:
                pass
        has_vort = False

    # Slice at mid-span so we always render a 2D plane (avoids accidental outer-surface views).
    b = _bounds(r)
    zmid = 0.5 * (float(b[4]) + float(b[5]))
    sl = Slice(Input=r)
    try:
        sl.SliceType.Origin = [0.0, 0.0, float(zmid)]
        sl.SliceType.Normal = [0.0, 0.0, 1.0]
    except Exception:
        pass
    c2p = CellDatatoPointData(Input=sl)

    omega_absmax_low: float | None = None
    if has_vort:
        calc = Calculator(Input=c2p)
        calc.ResultArrayName = "omegaZ"
        calc.Function = "vorticity_Z"
    else:
        # ParaView's calculator parser doesn't accept `curl(U)_Z`; compute curl(U) first, then take Z.
        calc0 = Calculator(Input=c2p)
        calc0.ResultArrayName = "curlVec"
        calc0.Function = "curl(U)"
        calc = Calculator(Input=calc0)
        calc.ResultArrayName = "omegaZ"
        calc.Function = "curlVec_Z"
    try:
        UpdatePipeline(proxy=calc)
    except TypeError:
        try:
            UpdatePipeline()
        except Exception:
            pass
    disp = Show(calc, view)
    safe_color_by(disp, "omegaZ")
    try:
        disp.Lighting = 0
    except Exception:
        pass
    try:
        disp.SetScalarBarVisibility(view, True)
    except Exception:
        pass
    lut = GetColorTransferFunction("omegaZ")
    try:
        lut.ApplyPreset("Cool to Warm", True)
    except Exception:
        pass
    r_omega = array_range(calc, "omegaZ")
    if r_omega is not None:
        lo, hi = float(r_omega[0]), float(r_omega[1])
        absmax = max(abs(lo), abs(hi))
        if absmax < 1e-8:
            absmax = 1.0
        omega_absmax_low = float(absmax)
        try:
            lut.RescaleTransferFunction(-absmax, absmax)
        except Exception:
            pass
    else:
        try:
            lut.RescaleTransferFunctionToDataRange(True, False)
        except TypeError:
            lut.RescaleTransferFunctionToDataRange()
    show_scalar_bar("omegaZ", "ω_z")
    setup_camera(_bounds(sl), zoom=1.0)
    Render()
    SaveScreenshot(str(figdir / "low_vort_mid.png"), view)
    # Back-compat name (older report builds).
    SaveScreenshot(str(figdir / "vort_low_tmid.png"), view)
    Hide(calc, view)
    Hide(c2p, view)
    Hide(sl, view)
    Hide(r, view)

    # ROM Umag at t=0.5 (for side-by-side comparison).
    if rom_file is not None:
        rR0 = LegacyVTKReader(FileNames=[str(rom_file)])
        cR0 = Calculator(Input=rR0)
        cR0.ResultArrayName = "Umagn"
        cR0.Function = "mag(U)"
        dR0 = Show(cR0, view)
        safe_color_by(dR0, "Umagn")
        try:
            dR0.SetScalarBarVisibility(view, True)
        except Exception:
            pass
        lut0 = GetColorTransferFunction("Umagn")
        if shared_low_umag is not None:
            try:
                lut0.RescaleTransferFunction(float(shared_low_umag[0]), float(shared_low_umag[1]))
            except Exception:
                pass
        else:
            try:
                lut0.RescaleTransferFunctionToDataRange(True, False)
            except TypeError:
                lut0.RescaleTransferFunctionToDataRange()
        show_scalar_bar("Umagn", "|U|")
        setup_camera(_bounds(rR0), zoom=1.0)
        Render()
        SaveScreenshot(str(figdir / "U_rom_tmid.png"), view)
        Hide(cR0, view)
        Hide(rR0, view)
        try:
            Delete(cR0)
            Delete(rR0)
        except Exception:
            pass

        # ROM spanwise vorticity at t=0.5 (same color scale as the low FOM vorticity when available).
        rRV = LegacyVTKReader(FileNames=[str(rom_file)])
        slR = Slice(Input=rRV)
        slR.SliceType = "Plane"
        slR.SliceType.Origin = [0.0, 0.0, 0.0]
        slR.SliceType.Normal = [0.0, 0.0, 1.0]
        c2pR = CellDatatoPointData(Input=slR)
        try:
            UpdatePipeline(proxy=c2pR)
        except TypeError:
            try:
                UpdatePipeline()
            except Exception:
                pass
        try:
            has_vort_R = has_array(rRV, "vorticity")
        except Exception:
            has_vort_R = False
        if has_vort_R:
            calcR = Calculator(Input=c2pR)
            calcR.ResultArrayName = "omegaZ"
            calcR.Function = "vorticity_Z"
        else:
            calcR0 = Calculator(Input=c2pR)
            calcR0.ResultArrayName = "curlVec"
            calcR0.Function = "curl(U)"
            calcR = Calculator(Input=calcR0)
            calcR.ResultArrayName = "omegaZ"
            calcR.Function = "curlVec_Z"
        try:
            UpdatePipeline(proxy=calcR)
        except TypeError:
            try:
                UpdatePipeline()
            except Exception:
                pass
        dispR = Show(calcR, view)
        safe_color_by(dispR, "omegaZ")
        try:
            dispR.Lighting = 0
        except Exception:
            pass
        try:
            dispR.SetScalarBarVisibility(view, True)
        except Exception:
            pass
        lutR = GetColorTransferFunction("omegaZ")
        try:
            lutR.ApplyPreset("Cool to Warm", True)
        except Exception:
            pass
        if omega_absmax_low is not None:
            try:
                lutR.RescaleTransferFunction(-float(omega_absmax_low), float(omega_absmax_low))
            except Exception:
                pass
        else:
            try:
                lutR.RescaleTransferFunctionToDataRange(True, False)
            except TypeError:
                lutR.RescaleTransferFunctionToDataRange()
        show_scalar_bar("omegaZ", "ω_z")
        setup_camera(_bounds(slR), zoom=1.0)
        Render()
        SaveScreenshot(str(figdir / "vort_rom_tmid.png"), view)
        Hide(calcR, view)
        Hide(c2pR, view)
        Hide(slR, view)
        Hide(rRV, view)
        try:
            Delete(calcR)
            Delete(c2pR)
            Delete(slR)
            Delete(rRV)
        except Exception:
            pass

    # High-fidelity screenshots (use VTK produced with -useTimeName at t=0.5 when available).
    high_time = next(iter(high_vtk_dir.glob("high_0.5*.vtk")), None)
    if high_time is not None:
        rh0 = LegacyVTKReader(FileNames=[str(high_time)])
        bh0 = _bounds(rh0)
        dh0 = Show(rh0, view)
        dh0.Representation = "Surface With Edges"
        dh0.ColorArrayName = [None, ""]
        dh0.LineWidth = 2.0
        try:
            dh0.EdgeColor = [0.1, 0.1, 0.1]
            dh0.DiffuseColor = [0.85, 0.85, 0.85]
            dh0.AmbientColor = [0.85, 0.85, 0.85]
            dh0.Lighting = 0
        except Exception:
            pass
        setup_camera(bh0, zoom=1.0)
        Render()
        SaveScreenshot(str(figdir / "mesh_high.png"), view)

        cylh = next(iter((high_vtk_dir / "cylinder").glob("cylinder_0.5*.vtk")), None)
        if cylh is None:
            cylh = next(iter((high_vtk_dir / "cylinder").glob("cylinder_0*.vtk")), None)
        if cylh is not None:
            rch = LegacyVTKReader(FileNames=[str(cylh)])
            setup_camera_focus(_bounds(rch), margin=2.0)
            Hide(rch, view)
        else:
            setup_camera(bh0, zoom=3.0)
        Render()
        SaveScreenshot(str(figdir / "mesh_high_zoom.png"), view)
        Hide(rh0, view)

        # High Umag at t=0.5
        rhU = LegacyVTKReader(FileNames=[str(high_time)])
        chU = Calculator(Input=rhU)
        chU.ResultArrayName = "Umagn"
        chU.Function = "mag(U)"
        dhU = Show(chU, view)
        safe_color_by(dhU, "Umagn")
        try:
            dhU.SetScalarBarVisibility(view, True)
        except Exception:
            pass
        lutU = GetColorTransferFunction("Umagn")
        try:
            lutU.RescaleTransferFunctionToDataRange(True, False)
        except TypeError:
            lutU.RescaleTransferFunctionToDataRange()
        show_scalar_bar("Umagn", "|U|")
        setup_camera(_bounds(rhU), zoom=1.0)
        Render()
        SaveScreenshot(str(figdir / "U_high_tmid.png"), view)
        Hide(chU, view)
        Hide(rhU, view)

        # High spanwise vorticity at t=0.5 (prefer OpenFOAM-computed `vorticity` field if present).
        rhV = LegacyVTKReader(FileNames=[str(high_time)])
        try:
            UpdatePipeline(proxy=rhV)
        except TypeError:
            try:
                UpdatePipeline()
            except Exception:
                pass
        has_vort_h = has_array(rhV, "vorticity")

        bh = _bounds(rhV)
        zmid_h = 0.5 * (float(bh[4]) + float(bh[5]))
        slh = Slice(Input=rhV)
        try:
            slh.SliceType.Origin = [0.0, 0.0, float(zmid_h)]
            slh.SliceType.Normal = [0.0, 0.0, 1.0]
        except Exception:
            pass
        c2ph = CellDatatoPointData(Input=slh)

        if has_vort_h:
            chV = Calculator(Input=c2ph)
            chV.ResultArrayName = "omegaZ"
            chV.Function = "vorticity_Z"
        else:
            ch0 = Calculator(Input=c2ph)
            ch0.ResultArrayName = "curlVec"
            ch0.Function = "curl(U)"
            chV = Calculator(Input=ch0)
            chV.ResultArrayName = "omegaZ"
            chV.Function = "curlVec_Z"
        try:
            UpdatePipeline(proxy=chV)
        except TypeError:
            try:
                UpdatePipeline()
            except Exception:
                pass
        dhV = Show(chV, view)
        safe_color_by(dhV, "omegaZ")
        try:
            dhV.Lighting = 0
        except Exception:
            pass
        try:
            dhV.SetScalarBarVisibility(view, True)
        except Exception:
            pass
        lutV = GetColorTransferFunction("omegaZ")
        try:
            lutV.ApplyPreset("Cool to Warm", True)
        except Exception:
            pass
        r_omega_h = array_range(chV, "omegaZ")
        if r_omega_h is not None:
            lo, hi = float(r_omega_h[0]), float(r_omega_h[1])
            absmax = max(abs(lo), abs(hi))
            if absmax < 1e-8:
                absmax = 1.0
            try:
                lutV.RescaleTransferFunction(-absmax, absmax)
            except Exception:
                pass
        else:
            try:
                lutV.RescaleTransferFunctionToDataRange(True, False)
            except TypeError:
                lutV.RescaleTransferFunctionToDataRange()
        show_scalar_bar("omegaZ", "ω_z")
        setup_camera(_bounds(slh), zoom=1.0)
        Render()
        SaveScreenshot(str(figdir / "high_vort_mid.png"), view)
        # Back-compat name (older report builds).
        SaveScreenshot(str(figdir / "vort_high_tmid.png"), view)
        Hide(chV, view)
        Hide(c2ph, view)
        Hide(slh, view)
        Hide(rhV, view)

    # High FOM-only vs adaptive (ROM) Umag at t=0.5 for external combination.
    high_fom_vtk = (root / "results/high/fom_only/fom_only_case/VTK").resolve()
    high_rom_vtk = (root / "results/high/adaptive/adaptive_case/VTK").resolve()
    high_fom_file = next(iter(high_fom_vtk.glob("fom_only_case_0.5*.vtk")), None)
    high_rom_file = next(iter(high_rom_vtk.glob("adaptive_case_0.5*.vtk")), None)
    shared_high_umag: tuple[float, float] | None = None
    if high_fom_file is not None and high_rom_file is not None:
        try:
            rA = LegacyVTKReader(FileNames=[str(high_fom_file)])
            cA = Calculator(Input=rA)
            cA.ResultArrayName = "Umagn"
            cA.Function = "mag(U)"
            UpdatePipeline(proxy=cA)
            ra = array_range(cA, "Umagn")

            rB = LegacyVTKReader(FileNames=[str(high_rom_file)])
            cB = Calculator(Input=rB)
            cB.ResultArrayName = "Umagn"
            cB.Function = "mag(U)"
            UpdatePipeline(proxy=cB)
            rb = array_range(cB, "Umagn")
            if ra is not None and rb is not None:
                shared_high_umag = (min(ra[0], rb[0]), max(ra[1], rb[1]))
            for pxy in [cA, rA, cB, rB]:
                try:
                    Delete(pxy)
                except Exception:
                    pass
        except Exception:
            shared_high_umag = None
    if high_fom_file is not None:
        rHF = LegacyVTKReader(FileNames=[str(high_fom_file)])
        cHF = Calculator(Input=rHF)
        cHF.ResultArrayName = "Umagn"
        cHF.Function = "mag(U)"
        dHF = Show(cHF, view)
        safe_color_by(dHF, "Umagn")
        try:
            dHF.SetScalarBarVisibility(view, True)
        except Exception:
            pass
        lutHF = GetColorTransferFunction("Umagn")
        if shared_high_umag is not None:
            try:
                lutHF.RescaleTransferFunction(float(shared_high_umag[0]), float(shared_high_umag[1]))
            except Exception:
                pass
        else:
            try:
                lutHF.RescaleTransferFunctionToDataRange(True, False)
            except TypeError:
                lutHF.RescaleTransferFunctionToDataRange()
        show_scalar_bar("Umagn", "|U|")
        setup_camera(_bounds(rHF), zoom=1.0)
        Render()
        SaveScreenshot(str(figdir / "U_high_fom_tmid.png"), view)
        Hide(cHF, view)
        Hide(rHF, view)
        try:
            Delete(cHF)
            Delete(rHF)
        except Exception:
            pass
    if high_rom_file is not None:
        rHR = LegacyVTKReader(FileNames=[str(high_rom_file)])
        cHR = Calculator(Input=rHR)
        cHR.ResultArrayName = "Umagn"
        cHR.Function = "mag(U)"
        dHR = Show(cHR, view)
        safe_color_by(dHR, "Umagn")
        try:
            dHR.SetScalarBarVisibility(view, True)
        except Exception:
            pass
        lutHR = GetColorTransferFunction("Umagn")
        if shared_high_umag is not None:
            try:
                lutHR.RescaleTransferFunction(float(shared_high_umag[0]), float(shared_high_umag[1]))
            except Exception:
                pass
        else:
            try:
                lutHR.RescaleTransferFunctionToDataRange(True, False)
            except TypeError:
                lutHR.RescaleTransferFunctionToDataRange()
        show_scalar_bar("Umagn", "|U|")
        setup_camera(_bounds(rHR), zoom=1.0)
        Render()
        SaveScreenshot(str(figdir / "U_high_rom_tmid.png"), view)
        Hide(cHR, view)
        Hide(rHR, view)
        try:
            Delete(cHR)
            Delete(rHR)
        except Exception:
            pass

    # POD modes (from modes_case VTK via -useTimeName)
    modes_vtk = (root / "results/low/modes_case/VTK").resolve()
    for k, tname in enumerate(["1001", "1002", "1003"], start=1):
        f = next(iter(modes_vtk.glob(f"modes_case_{tname}*.vtk")), None)
        if f is None:
            continue
        rM = LegacyVTKReader(FileNames=[str(f)])
        cM = Calculator(Input=rM)
        cM.ResultArrayName = "Umagn"
        cM.Function = "mag(U)"
        dM = Show(cM, view)
        try:
            ColorBy(dM, ("POINTS", "Umagn"))
        except Exception:
            ColorBy(dM, ("CELLS", "Umagn"))
        try:
            dM.SetScalarBarVisibility(view, True)
        except Exception:
            pass
        lutM = GetColorTransferFunction("Umagn")
        try:
            lutM.RescaleTransferFunctionToDataRange(True, False)
        except TypeError:
            lutM.RescaleTransferFunctionToDataRange()
        show_scalar_bar("Umagn", "|U|")
        setup_camera(rM.GetDataInformation().GetBounds(), zoom=4.0)
        Render()
        SaveScreenshot(str(figdir / f"mode{k}_U.png"), view)
        Hide(cM, view)
        Hide(rM, view)


if __name__ == "__main__":
    main()
