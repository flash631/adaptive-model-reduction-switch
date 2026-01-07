from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _read_image(path: Path) -> np.ndarray:
    try:
        from PIL import Image  # type: ignore

        img = Image.open(path)
        arr = np.asarray(img)
        return arr
    except Exception:
        import matplotlib.image as mpimg

        arr = mpimg.imread(str(path))
        if arr.dtype.kind == "f":
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        return arr


def _write_image(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image  # type: ignore

        Image.fromarray(arr).save(path)
        return
    except Exception:
        import matplotlib.pyplot as plt

        plt.imsave(str(path), arr)


def _compute_bg_mask(img: np.ndarray, *, tol: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (bg_rgb, mask) where mask is True for non-background pixels.
    """
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        return np.zeros((3,), dtype=np.int16), np.zeros((0, 0), dtype=bool)

    rgb = img[:, :, :3].astype(np.int16)
    h, w, _ = rgb.shape
    corners = np.stack(
        [
            rgb[0:10, 0:10, :].reshape(-1, 3),
            rgb[0:10, max(w - 10, 0) : w, :].reshape(-1, 3),
            rgb[max(h - 10, 0) : h, 0:10, :].reshape(-1, 3),
            rgb[max(h - 10, 0) : h, max(w - 10, 0) : w, :].reshape(-1, 3),
        ],
        axis=0,
    ).reshape(-1, 3)
    bg = np.median(corners, axis=0).astype(np.int16)
    diff = np.max(np.abs(rgb - bg[None, None, :]), axis=2)
    mask = diff > int(tol)
    return bg, mask


def _bbox_from_mask(mask: np.ndarray, *, pad: int) -> tuple[int, int, int, int] | None:
    if mask.ndim != 2 or not np.any(mask):
        return None
    h, w = mask.shape
    ys, xs = np.where(mask)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, h)
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, w)
    return y0, y1, x0, x1


def _collapse_largest_vertical_gap(img: np.ndarray, mask: np.ndarray, *, gap_min: int, gap_pad: int) -> np.ndarray:
    """
    If the non-background pixels form multiple vertical bands separated by a large empty gap
    (common when a colorbar is far from the field), remove most of that gap so the bands sit
    closer together.
    """
    if mask.ndim != 2 or not np.any(mask):
        return img
    rows = np.where(mask.any(axis=1))[0]
    if rows.size < 2:
        return img

    # Find contiguous runs of "content rows".
    runs: list[tuple[int, int]] = []
    start = int(rows[0])
    prev = int(rows[0])
    for r in rows[1:]:
        r = int(r)
        if r == prev + 1:
            prev = r
            continue
        runs.append((start, prev))
        start = r
        prev = r
    runs.append((start, prev))

    if len(runs) < 2:
        return img

    # Find the largest gap between consecutive runs.
    best = None
    best_gap = 0
    for (s0, e0), (s1, _e1) in zip(runs, runs[1:], strict=False):
        gap = int(s1 - e0 - 1)
        if gap > best_gap:
            best_gap = gap
            best = (e0, s1)
    if best is None or best_gap < int(gap_min):
        return img

    e0, s1 = best
    keep_gap = max(0, int(gap_pad))
    cut0 = int(e0) + 1
    cut1 = int(s1)
    cut_keep_end = min(cut0 + keep_gap, cut1)
    if cut_keep_end >= cut1:
        return img
    # Remove the (large) empty region between the two bands, leaving a small gap.
    return np.concatenate([img[:cut_keep_end, :, :], img[cut1:, :, :]], axis=0)


def _collapse_largest_horizontal_gap(img: np.ndarray, mask: np.ndarray, *, gap_min: int, gap_pad: int) -> np.ndarray:
    """
    If the non-background pixels form multiple horizontal bands separated by a large empty gap
    (common when a colorbar is far from the field), remove most of that gap so the bands sit
    closer together.
    """
    if mask.ndim != 2 or not np.any(mask):
        return img
    cols = np.where(mask.any(axis=0))[0]
    if cols.size < 2:
        return img

    # Find contiguous runs of "content columns".
    runs: list[tuple[int, int]] = []
    start = int(cols[0])
    prev = int(cols[0])
    for c in cols[1:]:
        c = int(c)
        if c == prev + 1:
            prev = c
            continue
        runs.append((start, prev))
        start = c
        prev = c
    runs.append((start, prev))

    if len(runs) < 2:
        return img

    # Find the largest gap between consecutive runs.
    best = None
    best_gap = 0
    for (s0, e0), (s1, _e1) in zip(runs, runs[1:], strict=False):
        gap = int(s1 - e0 - 1)
        if gap > best_gap:
            best_gap = gap
            best = (e0, s1)
    if best is None or best_gap < int(gap_min):
        return img

    e0, s1 = best
    keep_gap = max(0, int(gap_pad))
    cut0 = int(e0) + 1
    cut1 = int(s1)
    cut_keep_end = min(cut0 + keep_gap, cut1)
    if cut_keep_end >= cut1:
        return img
    # Remove the (large) empty region between the two bands, leaving a small gap.
    return np.concatenate([img[:, :cut_keep_end, :], img[:, cut1:, :]], axis=1)


def autocrop(
    path: Path,
    *,
    out: Path | None = None,
    pad: int = 10,
    tol: int = 8,
    collapse_vertical: bool = False,
    collapse_horizontal: bool = False,
    gap_min: int = 80,
    gap_pad: int = 10,
) -> Path:
    out = out or path
    img = _read_image(path)
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        return out

    # First pass: trim outer margins.
    _bg, mask = _compute_bg_mask(img, tol=int(tol))
    bbox = _bbox_from_mask(mask, pad=int(pad))
    if bbox is None:
        return out
    y0, y1, x0, x1 = bbox
    cropped = img[y0:y1, x0:x1, :]

    # Optional: collapse the largest vertical gap (useful for screenshots with distant colorbars).
    if collapse_vertical:
        _bg2, mask2 = _compute_bg_mask(cropped, tol=int(tol))
        if mask2.size:
            cropped = _collapse_largest_vertical_gap(cropped, mask2, gap_min=int(gap_min), gap_pad=int(gap_pad))
            # Re-crop after collapsing.
            _bg3, mask3 = _compute_bg_mask(cropped, tol=int(tol))
            bbox2 = _bbox_from_mask(mask3, pad=int(pad))
            if bbox2 is not None:
                y0, y1, x0, x1 = bbox2
                cropped = cropped[y0:y1, x0:x1, :]

    # Optional: collapse the largest horizontal gap (useful for screenshots with distant colorbars).
    if collapse_horizontal:
        _bg2, mask2 = _compute_bg_mask(cropped, tol=int(tol))
        if mask2.size:
            cropped = _collapse_largest_horizontal_gap(cropped, mask2, gap_min=int(gap_min), gap_pad=int(gap_pad))
            # Re-crop after collapsing.
            _bg3, mask3 = _compute_bg_mask(cropped, tol=int(tol))
            bbox2 = _bbox_from_mask(mask3, pad=int(pad))
            if bbox2 is not None:
                y0, y1, x0, x1 = bbox2
                cropped = cropped[y0:y1, x0:x1, :]

    _write_image(out, cropped)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="PNG paths to crop (glob externally)")
    ap.add_argument("--pad", type=int, default=10)
    ap.add_argument("--tol", type=int, default=8, help="Background tolerance in 0..255")
    ap.add_argument("--collapse-vertical", action="store_true", help="Collapse large vertical gaps between content bands")
    ap.add_argument("--collapse-horizontal", action="store_true", help="Collapse large horizontal gaps between content bands")
    ap.add_argument("--gap-min", type=int, default=80, help="Min gap (px) to collapse")
    ap.add_argument("--gap-pad", type=int, default=10, help="Gap padding (px) to keep after collapsing")
    ap.add_argument("--inplace", action="store_true", help="Overwrite input files")
    ap.add_argument("--outdir", default=None, help="Optional output directory (preserves filename)")
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve() if args.outdir else None
    for p in args.paths:
        path = Path(p).resolve()
        if not path.exists():
            continue
        out = None
        if not args.inplace:
            if outdir is None:
                out = path.with_name(path.stem + "_cropped" + path.suffix)
            else:
                out = outdir / path.name
        autocrop(
            path,
            out=out,
            pad=int(args.pad),
            tol=int(args.tol),
            collapse_vertical=bool(args.collapse_vertical),
            collapse_horizontal=bool(args.collapse_horizontal),
            gap_min=int(args.gap_min),
            gap_pad=int(args.gap_pad),
        )


if __name__ == "__main__":
    main()
