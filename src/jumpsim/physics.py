"""3D billiards physics with an airborne phase.

A time-stepped rigid-sphere model. Existing open simulators (e.g. pooltool)
keep every ball on the cloth, which makes them unusable for jump shots, so this
module adds the third dimension:

* on the cloth: sliding friction, rolling resistance and spin decay
  (the standard model of Leckie & Greenspan / Marlow);
* in the air: ballistic flight under gravity (air drag is negligible over the
  < 0.5 s flights involved);
* landings: a normal impulse with restitution ``e_table`` plus a Coulomb
  friction impulse at the contact point;
* ball-ball collisions resolved in 3D, so a ball that is high enough clears a
  blocker and a low one strikes its upper half and drives it into the slate;
* cushions as walls of height ``rail_height`` and six pockets.

Units are SI. The cloth is the plane z = 0, x runs along the table length and a
resting ball has its centre at z = R.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

G = 9.81


@dataclass
class Table:
    length: float = 2.54          # 9-ft table playing surface
    width: float = 1.27
    rail_height: float = 0.045    # cushion top above the cloth
    rail_width: float = 0.13
    corner_pocket_r: float = 0.058
    side_pocket_r: float = 0.062
    mouth: float = 0.075          # cushions are absent this close to a pocket

    def pocket_centres(self) -> np.ndarray:
        hl, hw = self.length / 2, self.width / 2
        o = 0.012
        return np.array([
            [-hl - o, -hw - o], [hl + o, -hw - o], [-hl - o, hw + o], [hl + o, hw + o],
            [0.0, -hw - 0.022], [0.0, hw + 0.022],
        ])


@dataclass
class Params:
    R: float = 0.028575           # ball radius (2.25 in diameter)
    m: float = 0.17
    mu_slide: float = 0.2         # ball-cloth sliding friction
    mu_roll: float = 0.01         # rolling resistance
    mu_spin: float = 0.01         # spin (about z) decay coefficient
    e_ball: float = 0.95
    e_table: float = 0.5          # ball-slate restitution under impact
    e_cushion: float = 0.8
    mu_land: float = 0.2          # friction coefficient during landing impulses
    v_settle: float = 0.04        # rebound speeds below this end the bounce sequence


def _rodrigues(w: np.ndarray, dt: float) -> np.ndarray:
    """Batched rotation matrices for angular velocities ``w`` (N,3) over ``dt``."""
    th = np.linalg.norm(w, axis=1) * dt
    k = np.zeros_like(w)
    nz = th > 1e-12
    k[nz] = w[nz] / (th[nz, None] / dt)
    K = np.zeros((len(w), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    s, c = np.sin(th)[:, None, None], np.cos(th)[:, None, None]
    return np.eye(3)[None] + s * K + (1 - c) * (K @ K)


@dataclass
class Event:
    t: float
    kind: str                     # strike | ball | cushion | land | pocket | off_table
    a: int
    b: int = -1
    info: float = 0.0             # e.g. impact speed


@dataclass
class Trajectory:
    t: np.ndarray                 # (T,)
    pos: np.ndarray               # (T, N, 3) ball centres
    rot: np.ndarray               # (T, N, 3, 3) body-to-world orientation
    active: np.ndarray            # (T, N) still on the table
    events: list = field(default_factory=list)
    R: float = Params.R

    @property
    def height(self) -> np.ndarray:
        """Height of each ball's lowest point above the cloth, (T, N)."""
        return self.pos[..., 2] - self.R

    def airborne(self, thresh: float = 1e-3) -> np.ndarray:
        return (self.height > thresh) & self.active


class Simulator:
    def __init__(self, xy, table: Table | None = None, params: Params | None = None):
        self.table = table or Table()
        self.p = params or Params()
        xy = np.asarray(xy, float)
        n = len(xy)
        self.pos = np.column_stack([xy, np.full(n, self.p.R)])
        self.vel = np.zeros((n, 3))
        self.w = np.zeros((n, 3))
        self.rot = np.repeat(np.eye(3)[None], n, axis=0)
        self.active = np.ones(n, bool)
        self.t = 0.0
        self.events: list[Event] = []

    def randomize_orientations(self, rng: np.random.Generator):
        q = rng.normal(size=(len(self.pos), 4))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        a, b, c, d = q.T
        self.rot = np.stack([
            np.stack([a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)], -1),
            np.stack([2*(b*c + a*d), a*a - b*b + c*c - d*d, 2*(c*d - a*b)], -1),
            np.stack([2*(b*d - a*c), 2*(c*d + a*b), a*a - b*b - c*c + d*d], -1),
        ], 1)

    def strike(self, ball: int, speed: float, heading_deg: float,
               elevation_deg: float = 0.0, follow: float = 0.0, side: float = 0.0):
        """Give ``ball`` its post-cue-impact state.

        ``elevation_deg`` is the cue elevation; the ball leaves the tip heading
        down into the slate at that angle and the first landing impulse turns
        that into the jump (the mechanism of a real jump shot). ``follow`` is
        the spin about the rolling axis as a fraction of natural roll
        (1 = rolling, 0 = stun, negative = draw); ``side`` is z-spin in the
        same units.
        """
        ph, th = np.radians(heading_deg), np.radians(elevation_deg)
        vh = speed * np.cos(th)
        self.vel[ball] = [vh * np.cos(ph), vh * np.sin(ph), -speed * np.sin(th)]
        axis = np.array([-np.sin(ph), np.cos(ph), 0.0])
        self.w[ball] = follow * vh / self.p.R * axis + np.array([0, 0, side * vh / self.p.R])
        self.events.append(Event(self.t, "strike", ball, info=speed))

    # ------------------------------------------------------------------ dynamics
    def _apply_cloth_friction(self, idx: np.ndarray, dt: float):
        p, R = self.p, self.p.R
        if len(idx) == 0:
            return
        v, w = self.vel[idx], self.w[idx]
        u = np.column_stack([v[:, 0] - R * w[:, 1], v[:, 1] + R * w[:, 0]])
        un = np.linalg.norm(u, axis=1)
        dec = 3.5 * p.mu_slide * G * dt
        sliding = un > dec
        if sliding.any():
            uh = u[sliding] / un[sliding, None]
            v[sliding, :2] -= p.mu_slide * G * dt * uh
            k = 5 * p.mu_slide * G * dt / (2 * R)
            w[sliding, 0] -= k * uh[:, 1]
            w[sliding, 1] += k * uh[:, 0]
        roll = ~sliding
        if roll.any():
            # finish the slide analytically (dv = -2/7 u), then roll
            v[roll, :2] -= (2 / 7) * u[roll]
            sp = np.linalg.norm(v[roll, :2], axis=1)
            dr = p.mu_roll * G * dt
            scale = np.where(sp > dr, (sp - dr) / np.maximum(sp, 1e-12), 0.0)
            v[roll, :2] *= scale[:, None]
            w[roll, 0] = -v[roll, 1] / R
            w[roll, 1] = v[roll, 0] / R
        dz = 5 * p.mu_spin * G * dt / (2 * R)
        w[:, 2] = np.sign(w[:, 2]) * np.maximum(np.abs(w[:, 2]) - dz, 0.0)
        v[:, 2] = 0.0
        self.vel[idx], self.w[idx] = v, w

    def _landings(self):
        p, R = self.p, self.p.R
        hit = np.where(self.active & (self.pos[:, 2] < R) & (self.vel[:, 2] < 0))[0]
        for i in hit:
            vz = -self.vel[i, 2]
            self.pos[i, 2] = R
            jn = (1 + p.e_table) * vz            # impulse per unit mass
            v, w = self.vel[i], self.w[i]
            u = np.array([v[0] - R * w[1], v[1] + R * w[0]])
            un = np.linalg.norm(u)
            if un > 1e-9:
                jt = min(p.mu_land * jn, (2 / 7) * un)
                uh = u / un
                v[:2] -= jt * uh
                k = 5 * jt / (2 * R)
                w[0] -= k * uh[1]
                w[1] += k * uh[0]
            v[2] = p.e_table * vz if p.e_table * vz > p.v_settle else 0.0
            if vz > 0.2:
                self.events.append(Event(self.t, "land", i, info=vz))

    def _ball_collisions(self):
        p, R = self.p, self.p.R
        act = np.where(self.active)[0]
        if len(act) < 2:
            return
        P = self.pos[act]
        d = P[:, None, :] - P[None, :, :]
        dist = np.linalg.norm(d, axis=2)
        ii, jj = np.where(np.triu(dist < 2 * R, 1))
        for a, b in zip(act[ii], act[jj]):
            n = self.pos[b] - self.pos[a]
            dn = np.linalg.norm(n)
            if dn < 1e-12:
                continue
            n /= dn
            vrel = np.dot(self.vel[a] - self.vel[b], n)
            if vrel <= 0:
                continue
            j = 0.5 * (1 + p.e_ball) * vrel
            self.vel[a] -= j * n
            self.vel[b] += j * n
            overlap = 2 * R - dn
            self.pos[a] -= 0.5 * overlap * n
            self.pos[b] += 0.5 * overlap * n
            for k in (a, b):  # a ball cannot be pushed into the slate
                self.pos[k, 2] = max(self.pos[k, 2], R)
            self.events.append(Event(self.t, "ball", int(a), int(b), info=vrel))

    def _cushions_and_pockets(self):
        tb, p, R = self.table, self.p, self.p.R
        hl, hw = tb.length / 2, tb.width / 2
        pockets = tb.pocket_centres()
        radii = np.array([tb.corner_pocket_r] * 4 + [tb.side_pocket_r] * 2)
        for i in np.where(self.active)[0]:
            x, y, z = self.pos[i]
            d = np.hypot(pockets[:, 0] - x, pockets[:, 1] - y)
            if (d < radii).any() and z < R + 0.02:
                self.active[i] = False
                self.events.append(Event(self.t, "pocket", i, info=float(np.argmin(d))))
                continue
            near_corner = abs(x) > hl - tb.mouth and abs(y) > hw - tb.mouth
            near_side = abs(x) < tb.mouth and abs(y) > hw - 2 * R
            if z < R + tb.rail_height and not (near_corner or near_side):
                for ax, lim in ((0, hl - R), (1, hw - R)):
                    c = self.pos[i, ax]
                    if abs(c) > lim and self.vel[i, ax] * np.sign(c) > 0:
                        self.pos[i, ax] = np.sign(c) * (2 * lim - abs(c))
                        self.events.append(Event(self.t, "cushion", i, info=abs(self.vel[i, ax])))
                        self.vel[i, ax] *= -p.e_cushion
            x, y = self.pos[i, :2]
            if abs(x) > hl + tb.rail_width or abs(y) > hw + tb.rail_width or \
                    ((near_corner or near_side) and (abs(x) > hl + R or abs(y) > hw + R)):
                self.active[i] = False
                self.events.append(Event(self.t, "off_table", i))

    def step(self, dt: float):
        R = self.p.R
        act = self.active
        on = act & (self.pos[:, 2] <= R + 1e-7) & (np.abs(self.vel[:, 2]) <= 1e-9)
        air = act & ~on
        self.vel[air, 2] -= G * dt
        self._apply_cloth_friction(np.where(on)[0], dt)
        self.pos[act] += self.vel[act] * dt
        self.rot[act] = _rodrigues(self.w[act], dt) @ self.rot[act]
        self.t += dt
        self._landings()
        self._ball_collisions()
        self._cushions_and_pockets()
        self.vel[~self.active] = 0
        self.w[~self.active] = 0

    def moving(self) -> bool:
        a = self.active
        return bool((np.abs(self.vel[a]).max(initial=0) > 1e-4) or
                    (self.pos[a, 2].max(initial=0) > self.p.R + 1e-7))

    def run(self, duration: float, rate: float = 240.0, substeps: int = 10,
            settle_tail: float = 0.25) -> Trajectory:
        """Simulate and record at ``rate`` Hz (``substeps`` physics steps per record).

        Stops early once everything has been still for ``settle_tail`` seconds.
        """
        dt = 1.0 / (rate * substeps)
        ts, ps, rs, acts = [], [], [], []
        still_since = None
        n_rec = int(round(duration * rate))
        for _ in range(n_rec + 1):
            ts.append(self.t)
            ps.append(self.pos.copy())
            rs.append(self.rot.copy())
            acts.append(self.active.copy())
            for _ in range(substeps):
                self.step(dt)
            if not self.moving():
                still_since = self.t if still_since is None else still_since
                if self.t - still_since > settle_tail:
                    break
            else:
                still_since = None
        return Trajectory(np.array(ts), np.array(ps), np.array(rs), np.array(acts),
                          list(self.events), self.p.R)
