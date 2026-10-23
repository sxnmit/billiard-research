"""Track the cue ball and compute per-frame cues for every rendered clip.

    python scripts/extract_features.py data/clips data/synth --out data/features
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jumpsim.detect import Clip, CueTracker, frame_features  # noqa: E402


def job(args):
    clip_dir, view, out = args
    dst = out / f"{clip_dir.name}__{view}.npz"
    if dst.exists():
        return dst.name, "cached"
    try:
        clip = Clip.load(clip_dir, view)
        trk = CueTracker(clip).track()
        F, names = frame_features(clip, trk)
        gt = clip.gt
        T = min(len(F), len(gt["frame_times"]))
        np.savez_compressed(
            dst, F=F[:T], names=np.array(names), found=trk["found"][:T], uv=trk["uv"][:T],
            height=np.array(gt["height_max"])[:T, 0], airborne=np.array(gt["airborne"])[:T, 0],
            active=np.array(gt["active"])[:T, 0], obj_air=np.array(gt["airborne"])[:T, 1:].any(1),
            t=np.array(gt["frame_times"])[:T], clip=clip_dir.name, view=view,
            tags=np.array(gt["tags"]), cond=np.array([gt["conditions"][k] for k in sorted(gt["conditions"])]),
            cond_keys=np.array(sorted(gt["conditions"])))
        return dst.name, f"ok ({trk['found'][:T].mean():.0%} tracked)"
    except Exception:
        return dst.name, "FAILED\n" + traceback.format_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = [(d, v, a.out) for r in a.roots for d in sorted(r.iterdir()) if (d / "gt.json").exists()
            for v in ("overhead", "broadcast")]
    with Pool(a.workers) as pool:
        for i, (name, status) in enumerate(pool.imap_unordered(job, jobs)):
            print(f"[{i + 1}/{len(jobs)}] {name}: {status}", flush=True)


if __name__ == "__main__":
    main()
