"""Figure: apex frames from both cameras plus zoomed cue-ball crops across a flight.

    python scripts/make_frames_figure.py data/clips/jump_over_blocker paper/figures/frames_jump_over_blocker.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from jumpsim.detect import Clip, sphere_centre_image  # noqa: E402


def main(clip_dir: str, out: str):
    clips = {v: Clip.load(Path(clip_dir), v) for v in ("overhead", "broadcast")}
    gt = clips["overhead"].gt
    h = np.array(gt["height_max"])[:, 0]
    pos = np.array(gt["pos"])[:, 0]
    apex = int(np.argmax(h))
    crop_idx = np.linspace(clips["overhead"].n_pre - 1, apex + (apex - clips["overhead"].n_pre) * 2.2, 7).astype(int)
    fig = plt.figure(figsize=(12, 6.2))
    gs = fig.add_gridspec(3, 14, height_ratios=[3.3, 1, 1], hspace=0.12, wspace=0.05)
    for j, (v, c) in enumerate(clips.items()):
        ax = fig.add_subplot(gs[0, 7 * j:7 * j + 7])
        ax.imshow(c.frames[apex])
        ax.set_title(f"{v} camera, apex frame (h = {h[apex] * 1000:.0f} mm)", fontsize=9)
        ax.axis("off")
        for i, k in enumerate(crop_idx):
            uv = sphere_centre_image(c.cam, pos[k:k + 1])[0].astype(int)
            s = 34
            crop = c.frames[k][max(uv[1] - s, 0):uv[1] + s, max(uv[0] - s, 0):uv[0] + s]
            axc = fig.add_subplot(gs[1 + j, 2 * i:2 * i + 2])
            axc.imshow(crop, interpolation="nearest")
            axc.set_xticks([])
            axc.set_yticks([])
            if j == 0:
                axc.set_title(f"t={(k - c.n_pre) / c.fps:.2f}s  h={h[k] * 1000:.0f}mm", fontsize=7)
            if i == 0:
                axc.set_ylabel(v, fontsize=8)
    fig.savefig(out, dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main(*sys.argv[1:3])
