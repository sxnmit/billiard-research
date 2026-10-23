"""Render synthetic clips with exact per-frame ground truth.

    python scripts/make_clips.py --showcase --out data/clips
    python scripts/make_clips.py --random 160 --seed 1 --randomize --out data/synth

Each clip directory gets ``<view>.mp4``, ``<view>_empty.jpg`` (the empty-table
reference frame a fixed camera can capture) and ``gt.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jumpsim.physics import Table  # noqa: E402
from jumpsim.render import (Lighting, Renderer, broadcast_camera,  # noqa: E402
                            overhead_camera, to_uint8)
from jumpsim.scenarios import random_shots, showcase  # noqa: E402

FPS = 60.0
PRE = 0.15          # seconds of still frames before the strike


def conditions(rng: np.random.Generator | None):
    """Camera/lighting/sensor settings; randomised when ``rng`` is given."""
    if rng is None:
        return dict(over_h=2.6, bc_dist=1.7, bc_h=1.55, lamp_h=0.95, lamp_sp=0.8, noise=0.008,
                    shutter=0.5, exposure=1.0)
    return dict(over_h=rng.uniform(2.2, 3.0), bc_dist=rng.uniform(1.3, 2.3), bc_h=rng.uniform(1.2, 1.9),
                lamp_h=rng.uniform(0.8, 1.1), lamp_sp=rng.uniform(0.6, 0.9), noise=rng.uniform(0.004, 0.016),
                shutter=rng.uniform(0.25, 1.0), exposure=rng.uniform(0.85, 1.15))


def write_mp4(path: Path, frames, w: int, h: int, crf: int = 20):
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
                          "-crf", str(crf), "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()


def render_job(job):
    shot, view, cond, out, seed = job
    table = Table()
    traj = shot.simulate(table, seed=seed)
    cam = overhead_camera(table, cond["over_h"]) if view == "overhead" else \
        broadcast_camera(table, cond["bc_dist"], cond["bc_h"])
    sp = cond["lamp_sp"]
    light = Lighting(np.array([[-sp, 0, cond["lamp_h"]], [0, 0, cond["lamp_h"]], [sp, 0, cond["lamp_h"]]]))
    rnd = Renderer(cam, table, light, seed=seed)
    times = np.arange(-PRE, traj.t[-1] - 1e-9, 1 / FPS)
    rng = np.random.default_rng(seed + 7)
    frames = (to_uint8(f, rng, cond["noise"], cond["exposure"])
              for f in rnd.render(traj, shot.ids, times, cond["shutter"], FPS))
    out.mkdir(parents=True, exist_ok=True)
    write_mp4(out / f"{view}.mp4", frames, cam.width, cam.height)
    import cv2
    cv2.imwrite(str(out / f"{view}_empty.jpg"),
                cv2.cvtColor(to_uint8(rnd.background, rng, cond["noise"], cond["exposure"]), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    return view, cam.to_dict(), light.lamps.tolist()


def ground_truth(shot, cond, seed, cams, lamps):
    traj = shot.simulate(Table(), seed=seed)
    times = np.arange(-PRE, traj.t[-1] - 1e-9, 1 / FPS)
    # ground truth at mid-exposure of each frame
    tm = np.clip(times + 0.5 * cond["shutter"] / FPS, 0, traj.t[-1])
    k = np.clip(np.searchsorted(traj.t, tm) - 1, 0, len(traj.t) - 2)
    a = ((tm - traj.t[k]) / (traj.t[k + 1] - traj.t[k]))[:, None, None]
    pos = (1 - a) * traj.pos[k] + a * traj.pos[k + 1]
    # a frame is airborne if the ball is off the cloth at any instant of the exposure
    t_hi = np.clip(times + cond["shutter"] / FPS, 0, traj.t[-1])
    air_full = traj.airborne()
    air = np.zeros((len(times), pos.shape[1]), bool)
    hmax = np.zeros((len(times), pos.shape[1]))
    for i, (lo, hi) in enumerate(zip(np.clip(times, 0, None), t_hi)):
        sel = (traj.t >= lo - 1e-9) & (traj.t <= hi + 1e-9)
        if sel.any():
            air[i] = air_full[sel].any(0)
            hmax[i] = traj.height[sel].max(0)
    return dict(
        name=shot.name, fps=FPS, pre=PRE, seed=seed, ids=shot.ids, R=traj.R, tags=shot.tags, note=shot.note,
        shot=dict(speed=shot.speed, heading_deg=shot.heading_deg, elevation_deg=shot.elevation_deg,
                  follow=shot.follow, side=shot.side, xy=np.asarray(shot.xy).tolist()),
        conditions=cond, cameras=cams, lamps=lamps,
        frame_times=times.round(6).tolist(),
        pos=pos.round(6).tolist(),                       # (T, N, 3) mid-exposure centres
        height_max=hmax.round(6).tolist(),               # (T, N) max height within exposure
        airborne=air.astype(int).tolist(),               # (T, N) off the cloth by > 1 mm
        active=traj.active[k].astype(int).tolist(),
        events=[dict(t=round(e.t, 5), kind=e.kind, a=int(e.a), b=int(e.b), info=float(e.info))
                for e in traj.events],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--showcase", action="store_true")
    ap.add_argument("--random", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--randomize", action="store_true", help="randomise camera/lighting/sensor")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    shots = showcase() if args.showcase else []
    shots += random_shots(rng, args.random, f"r{args.seed:02d}")
    conds = [conditions(rng if args.randomize else None) for _ in shots]
    jobs = [(s, v, c, args.out / s.name, args.seed * 100000 + i)
            for i, (s, c) in enumerate(zip(shots, conds)) for v in ("overhead", "broadcast")]
    jobs = [j for j in jobs if not (j[3] / "gt.json").exists()]  # resume an interrupted run
    print(f"{len(shots)} shots, {len(jobs)} render jobs", flush=True)
    results = {}
    with Pool(args.workers) as pool:
        for n, ((s, v, c, out, seed), res) in enumerate(zip(jobs, pool.imap(render_job, jobs))):
            results.setdefault(s.name, {})[v] = res
            if len(results[s.name]) == 2:
                cams = {r[0]: r[1] for r in results[s.name].values()}
                lamps = next(iter(results[s.name].values()))[2]
                (out / "gt.json").write_text(json.dumps(ground_truth(s, c, seed, cams, lamps)))
            print(f"[{n + 1}/{len(jobs)}] {s.name} {v}", flush=True)


if __name__ == "__main__":
    main()
