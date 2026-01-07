from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def combine_two_panel_png(left: Path, right: Path, out: Path, *, left_title: str = "FOM", right_title: str = "ROM") -> None:
    if not left.exists():
        raise FileNotFoundError(left)
    if not right.exists():
        raise FileNotFoundError(right)
    out.parent.mkdir(parents=True, exist_ok=True)

    img_l = mpimg.imread(str(left))
    img_r = mpimg.imread(str(right))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=150)
    for ax, img, title in [(axes[0], img_l, left_title), (axes[1], img_r, right_title)]:
        ax.imshow(img)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.tight_layout(pad=0.2)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", required=True, help="Left PNG path (FOM)")
    ap.add_argument("--right", required=True, help="Right PNG path (ROM)")
    ap.add_argument("--out", required=True, help="Output combined PNG path")
    ap.add_argument("--left-title", default="FOM")
    ap.add_argument("--right-title", default="ROM")
    args = ap.parse_args()

    combine_two_panel_png(
        Path(args.left),
        Path(args.right),
        Path(args.out),
        left_title=str(args.left_title),
        right_title=str(args.right_title),
    )


if __name__ == "__main__":
    main()

