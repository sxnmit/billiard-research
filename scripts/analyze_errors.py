"""Shot-level bootstrap intervals and false-alarm attribution for the all-cue models.

    python scripts/analyze_errors.py      # after scripts/evaluate.py

Writes results/bootstrap_ci.json and results/false_alarms.json.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate import H_AIR, H_POS, event_metrics, runs  # noqa: E402

VIEWS = ("overhead", "broadcast", "fused")


def bootstrap(z, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for v in VIEWS:
        p, h, g, thr = (z[f"{v}__{x}__all"] for x in ("p", "h", "g", "thr"))
        groups = np.unique(g)
        aps, recs = [], []
        for _ in range(n):
            pick = rng.choice(groups, len(groups), replace=True)
            idx = np.concatenate([np.where(g == q)[0] for q in pick])
            gg = np.concatenate([np.full((g == q).sum(), i) for i, q in enumerate(pick)])
            hh, pp = h[idx], p[idx]
            care = (hh > H_POS) | (hh <= H_AIR)
            aps.append(average_precision_score(hh[care] > H_POS, pp[care]))
            recs.append(event_metrics(pp, float(thr), hh, gg)[0].mean())
        out[v] = dict(ap_ci=[float(np.percentile(aps, 2.5)), float(np.percentile(aps, 97.5))],
                      recall_ci=[float(np.percentile(recs, 2.5)), float(np.percentile(recs, 97.5))])
    return out


def false_alarms(z, fps=60.0, window=0.12):
    """Label each false-alarm run by the nearest cue-ball event within ``window`` s."""
    clips = list(z["clips"])
    out = {}
    for v in VIEWS:
        p, h, g, k, thr = (z[f"{v}__{x}__all"] for x in ("p", "h", "g", "k", "thr"))
        cnt = collections.Counter()
        for gi in np.unique(g):
            s = g == gi
            hh, pp, kk = h[s], p[s] >= thr, k[s]
            air = hh > H_AIR
            near = np.convolve(air.astype(int), np.ones(5, int), "same") > 0
            gt = json.loads((ROOT / "data" / "synth" / clips[gi] / "gt.json").read_text())
            evs = [e for e in gt["events"] if e["a"] == 0 or e["b"] == 0]
            for a, b in runs(pp):
                if b - a < 2 or near[a:b].any():
                    continue
                t = kk[a] / fps - gt["pre"]
                dt, kind = min(((abs(e["t"] - t), e["kind"]) for e in evs), default=(np.inf, "none"))
                if dt < window:
                    cause = kind
                elif hh[max(a - 15, 0):b + 15].max() > H_AIR:
                    cause = "near flight"
                else:
                    cause = "rolling"
                cnt[cause] += 1
        out[v] = dict(cnt)
    return out


def main():
    z = np.load(ROOT / "results" / "test_predictions.npz")
    ci = bootstrap(z)
    fa = false_alarms(z)
    (ROOT / "results" / "bootstrap_ci.json").write_text(json.dumps(ci, indent=1))
    (ROOT / "results" / "false_alarms.json").write_text(json.dumps(fa, indent=1))
    print(json.dumps(dict(bootstrap=ci, false_alarms=fa), indent=1))


if __name__ == "__main__":
    main()
