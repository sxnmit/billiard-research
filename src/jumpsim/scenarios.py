"""Shot set-ups: the hand-designed showcase clips and a randomised generator.

A scenario is a layout (ball positions and identities) plus a cue strike. The
showcase clips each isolate one situation the detector must handle; the random
generator produces the training/evaluation pool.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .physics import Params, Simulator, Table

R = Params.R


@dataclass
class Shot:
    name: str
    xy: list                      # ball centres on the cloth
    ids: list                     # ball numbers (0 = cue ball)
    speed: float
    heading_deg: float
    elevation_deg: float = 0.0
    follow: float = 0.0
    side: float = 0.0
    duration: float = 2.5
    note: str = ""
    tags: list = field(default_factory=list)

    def simulate(self, table: Table | None = None, params: Params | None = None, seed: int = 0):
        sim = Simulator(self.xy, table, params)
        sim.randomize_orientations(np.random.default_rng(seed))
        sim.strike(0, self.speed, self.heading_deg, self.elevation_deg, self.follow, self.side)
        return sim.run(self.duration)


def heading(a, b) -> float:
    return float(np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0])))


def rack(apex=(0.635, 0.0), gap: float = 0.0003):
    """Standard 15-ball triangle, apex towards the head of the table."""
    d = 2 * R + gap
    xy = []
    for row in range(5):
        for k in range(row + 1):
            xy.append([apex[0] + row * d * np.sqrt(3) / 2, apex[1] + (k - row / 2) * d])
    ids = [1, 9, 2, 10, 8, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15]
    return xy, ids


def showcase() -> list[Shot]:
    """The fixed clip set used for qualitative figures and the paper's walk-through."""
    shots = []
    cue, tgt = (-0.85, -0.12), (0.25, -0.12)
    shots.append(Shot("stop_shot", [cue, tgt], [0, 3], 2.2, heading(cue, tgt), 3.0, 0.0,
                      note="Level stun shot, cue ball stops dead. Negative.", tags=["negative"]))

    cue, blk, tgt = (-0.75, 0.18), (-0.47, 0.19), (0.45, 0.23)
    shots.append(Shot("jump_over_blocker", [cue, blk, tgt], [0, 9, 1], 4.2, heading(cue, tgt), 42.0,
                      note="Cue ball jumps a blocker 28 cm away and hits the 1-ball. Positive.",
                      tags=["positive"]))

    cue, blk, tgt = (-0.70, -0.25), (-0.50, -0.25), (0.30, -0.27)
    shots.append(Shot("jump_clips_blocker", [cue, blk, tgt], [0, 2, 11], 3.0, heading(cue, tgt), 30.0,
                      note="Jump too low: cue ball strikes the top of the blocker, which hops. "
                           "Positive, with a secondary object-ball hop.", tags=["positive", "object_hop"]))

    cue, tgt = (-0.80, 0.30), (0.40, 0.30)
    blk = (-0.30, 0.30 + 2 * R + 0.004)
    shots.append(Shot("near_miss", [cue, blk, tgt], [0, 5, 7], 3.0, heading(cue, tgt), 5.0, 0.3,
                      note="Cue ball passes 4 mm beside a ball without leaving the cloth: "
                           "in 2-D the two balls almost touch. Hard negative.", tags=["negative", "hard"]))

    rxy, rids = rack()
    cue = (-0.635, 0.03)
    shots.append(Shot("break", [cue] + rxy, [0] + rids, 8.0, heading(cue, rxy[0]), 6.0, 0.0, duration=3.0,
                      note="Full-rack break with a slightly elevated cue: cue ball and object balls "
                           "hop off the cloth briefly.", tags=["mixed"]))

    cue, tgt = (0.95, -0.45), (-0.70, -0.40)
    shots.append(Shot("long_jump_off_axis", [cue, (0.70, -0.44), tgt], [0, 12, 4], 5.0, heading(cue, tgt), 32.0,
                      note="Long, fast jump along the side rail, far from the overhead camera's "
                           "optical axis (large parallax). Positive.", tags=["positive"]))

    cue, tgt = (-0.60, 0.0), (0.30, 0.05)
    shots.append(Shot("elevated_draw", [cue, tgt], [0, 6], 3.5, heading(cue, tgt), 14.0, -1.0,
                      note="Draw shot with an elevated cue (14 deg). Borderline: a ~1 cm hop.",
                      tags=["borderline"]))
    return shots


def random_shot(rng: np.random.Generator, name: str, table: Table | None = None) -> Shot:
    """A random layout and strike. Half the shots are jump attempts; a third of all
    shots place a ball close to the cue ball's path (as a blocker or near-miss)."""
    tb = table or Table()
    hl, hw = tb.length / 2 - 0.12, tb.width / 2 - 0.12

    def free(xy, p, gap=2.2 * R):
        return all(np.hypot(p[0] - q[0], p[1] - q[1]) > gap for q in xy)

    cue = np.array([rng.uniform(-hl, hl), rng.uniform(-hw, hw)])
    xy = [cue.tolist()]
    is_jump = rng.random() < 0.5
    # a target somewhere reasonably far away
    for _ in range(100):
        tgt = np.array([rng.uniform(-hl, hl), rng.uniform(-hw, hw)])
        if 0.4 < np.linalg.norm(tgt - cue) < 1.8:
            break
    xy.append(tgt.tolist())
    dirv = (tgt - cue) / np.linalg.norm(tgt - cue)
    nrm = np.array([-dirv[1], dirv[0]])
    tags = ["jump_attempt" if is_jump else "level"]
    if rng.random() < (0.8 if is_jump else 0.35):
        along = rng.uniform(0.12, 0.45) * np.linalg.norm(tgt - cue)
        if is_jump:
            off = rng.uniform(-1.2, 1.2) * R
        else:   # near-miss: just clear of the cue ball's path
            off = rng.choice([-1, 1]) * (2 * R + rng.uniform(0.002, 0.03))
        p = cue + along * dirv + off * nrm
        if abs(p[0]) < hl and abs(p[1]) < hw and free(xy, p):
            xy.append(p.tolist())
            tags.append("blocker" if is_jump else "near_miss")
    for _ in range(rng.integers(0, 4)):
        p = np.array([rng.uniform(-hl, hl), rng.uniform(-hw, hw)])
        if free(xy, p, 3 * R):
            xy.append(p.tolist())
    ids = [0] + list(rng.choice(np.arange(1, 16), len(xy) - 1, replace=False))
    hdg = heading(cue, tgt) + rng.normal(0, 1.0)
    if is_jump:
        # sample the first-hop apex (1.5-14 cm, the range of real jump shots) and
        # the cue elevation, then solve for the strike speed: v_up = e * v * sin(elev)
        apex, elev = rng.uniform(0.015, 0.14), rng.uniform(25, 55)
        speed = np.sqrt(2 * 9.81 * apex) / (Params.e_table * np.sin(np.radians(elev)))
        follow = rng.uniform(-0.3, 0.3)
    else:
        speed, elev = rng.uniform(0.6, 7.5), abs(rng.normal(0, 7))
        follow = rng.uniform(-1.2, 1.2)
    return Shot(name, xy, [int(i) for i in ids], float(speed), float(hdg), float(elev), float(follow),
                float(rng.uniform(-0.4, 0.4)), duration=2.0, tags=tags)


def random_shots(rng: np.random.Generator, n: int, prefix: str, max_height: float = 0.25) -> list[Shot]:
    """``n`` random shots, rejecting the rare ones where a ball ramps off another
    ball to an unrealistic height or leaves the table."""
    out = []
    while len(out) < n:
        s = random_shot(rng, f"{prefix}_{len(out):04d}")
        tr = s.simulate()
        if tr.height.max() <= max_height and not any(e.kind == "off_table" for e in tr.events):
            out.append(s)
    return out
