"""Signal available to each cue as a function of ball height (pure geometry).

For a ball raised by h above its on-cloth position, compute in pixels:
  size     - growth of the area-equivalent image radius;
  parallax - shift of the image centre away from where an on-cloth ball at the
             same table position would appear (the "motion" cue);
  shadow   - shift of each lamp's shadow centre away from the ball's image centre.

    python scripts/cue_geometry.py --figs paper/figures --out results
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from jumpsim.detect import silhouette_radius, sphere_centre_image  # noqa: E402
from jumpsim.physics import Params, Table  # noqa: E402
from jumpsim.render import broadcast_camera, default_lighting, overhead_camera  # noqa: E402

R = Params.R


def cues(cam, xy, hs, lamps):
    xy = np.asarray(xy, float)
    C = np.column_stack([np.repeat(xy[None], len(hs), 0), R + hs])
    r = silhouette_radius(cam, C, R)
    uv = sphere_centre_image(cam, C)
    size = r - r[0]
    parallax = np.linalg.norm(uv - uv[0], axis=1)
    sh = []
    for L in lamps:
        s = L[None] + (C - L[None]) * (L[2] / (L[2] - C[:, 2]))[:, None]
        s[:, 2] = 0
        suv, _ = cam.project(s)
        sep = np.linalg.norm(suv - uv, axis=1)
        sh.append(sep - sep[0])
    return size, parallax, np.max(sh, axis=0), np.min(sh, axis=0), r[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", type=Path, default=Path("paper/figures"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args()
    tb = Table()
    lamps = default_lighting().lamps
    hs = np.linspace(0, 0.12, 61)
    places = {"centre": (0.0, 0.0), "near corner": (1.0, 0.45)}
    cams = {
        "overhead 2.6 m, 720p": overhead_camera(tb, 2.6, 1280),
        "overhead 2.6 m, 4K": overhead_camera(tb, 2.6, 3840),
        "broadcast, 720p": broadcast_camera(tb, width=1280),
    }
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.9))
    numbers = {}
    for (cname, cam), col in zip(cams.items(), ("C0", "C2", "C1")):
        for (pname, xy), ls in zip(places.items(), ("-", "--")):
            size, par, sh_far, sh_near, r0 = cues(cam, xy, hs, lamps)
            lab = f"{cname} ({pname})"
            axes[0].plot(hs * 1000, size, ls, color=col, label=lab)
            axes[1].plot(hs * 1000, par, ls, color=col)
            axes[2].plot(hs * 1000, sh_far, ls, color=col)
            for h in (0.01, 0.03, 0.10):
                i = int(np.argmin(np.abs(hs - h)))
                numbers.setdefault(cname, {}).setdefault(pname, {})[f"{int(h*1000)}mm"] = dict(
                    r0_px=round(float(r0), 2), size_px=round(float(size[i]), 3), parallax_px=round(float(par[i]), 2),
                    shadow_far_lamp_px=round(float(sh_far[i]), 2), shadow_near_lamp_px=round(float(sh_near[i]), 2))
    for ax, title in zip(axes, ("(a) size: radius growth", "(b) parallax: centre shift", "(c) shadow: separation growth")):
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("ball height above cloth (mm)", fontsize=8)
        ax.set_ylabel("pixels", fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    axes[0].axhline(0.25, color="0.5", lw=0.8, ls=":")
    axes[0].text(2, 0.3, "~ size-estimator noise (720p)", fontsize=6, color="0.4")
    axes[0].legend(fontsize=6, loc="upper left")
    fig.tight_layout()
    a.figs.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.figs / "cue_geometry.pdf")
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(numbers, open(a.out / "cue_geometry.json", "w"), indent=1)
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
