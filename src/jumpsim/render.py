"""Ray-cast renderer for synthetic pool video with exact ground truth.

The scene is a 9-ft table lit by three hanging lamps (each an area light), seen
by a calibrated pinhole camera. Each pixel gets ``spp`` samples that are jittered
both inside the pixel (anti-aliasing) and inside the shutter interval (motion
blur). Balls carry textures (stripes, number discs) and rotate with the
simulated angular velocity, and they cast soft shadows on the cloth and on
each other. Only pixels near a ball or its shadows are re-rendered per frame;
the empty table is rendered once.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .physics import Table, Trajectory

BALL_COLORS = {
    0: (0.92, 0.91, 0.86), 1: (0.95, 0.72, 0.05), 2: (0.05, 0.15, 0.62), 3: (0.82, 0.06, 0.05),
    4: (0.32, 0.06, 0.46), 5: (0.95, 0.38, 0.05), 6: (0.04, 0.40, 0.16), 7: (0.45, 0.09, 0.05),
    8: (0.03, 0.03, 0.03),
}
WHITE = np.array((0.92, 0.91, 0.86))
CLOTH = np.array((0.04, 0.33, 0.14))
CUSHION = np.array((0.035, 0.27, 0.11))
WOOD = np.array((0.30, 0.15, 0.06))
FLOOR = np.array((0.08, 0.08, 0.09))


@dataclass
class Camera:
    pos: np.ndarray
    target: np.ndarray
    f: float                      # focal length in pixels
    width: int = 1280
    height: int = 720
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))

    def __post_init__(self):
        self.pos = np.asarray(self.pos, float)
        self.target = np.asarray(self.target, float)
        fwd = self.target - self.pos
        self.fwd = fwd / np.linalg.norm(fwd)
        r = np.cross(self.fwd, self.up)
        self.right = r / np.linalg.norm(r)
        self.down = np.cross(self.fwd, self.right)
        self.cx, self.cy = (self.width - 1) / 2, (self.height - 1) / 2

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.f, 0, self.cx], [0, self.f, self.cy], [0, 0, 1.0]])

    @property
    def Rcw(self) -> np.ndarray:
        """World-to-camera rotation (rows: right, down, forward)."""
        return np.stack([self.right, self.down, self.fwd])

    def project(self, X: np.ndarray):
        Xc = (np.asarray(X, float) - self.pos) @ self.Rcw.T
        z = Xc[..., 2]
        return np.stack([self.f * Xc[..., 0] / z + self.cx, self.f * Xc[..., 1] / z + self.cy], -1), z

    def rays(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        d = (self.fwd[None] + ((u - self.cx) / self.f)[:, None] * self.right[None]
             + ((v - self.cy) / self.f)[:, None] * self.down[None])
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def backproject_to_plane(self, uv: np.ndarray, z: float) -> np.ndarray:
        """Intersect pixel rays with the horizontal plane at height ``z``."""
        uv = np.atleast_2d(uv)
        d = self.rays(uv[:, 0], uv[:, 1])
        t = (z - self.pos[2]) / d[:, 2]
        return self.pos[None] + t[:, None] * d

    def to_dict(self) -> dict:
        return dict(pos=self.pos.tolist(), target=self.target.tolist(), f=self.f,
                    width=self.width, height=self.height, up=np.asarray(self.up).tolist(),
                    K=self.K.tolist(), R_world_to_cam=self.Rcw.tolist())


def overhead_camera(table: Table, height: float = 2.6, width: int = 1280, aspect=16 / 9,
                    margin: float = 1.08) -> Camera:
    """Camera looking straight down at the table centre, table long axis = image x."""
    span = (table.length + 2 * table.rail_width) * margin
    f = width * (height - 0.03) / span
    return Camera(np.array([0, 0, height]), np.zeros(3), f, width, int(round(width / aspect)),
                  up=np.array([0.0, 1.0, 0.0]))


def broadcast_camera(table: Table, dist: float = 1.7, height: float = 1.55, width: int = 1280,
                     aspect=16 / 9, f_scale: float = 1.0) -> Camera:
    """Elevated camera behind the foot rail, as in TV and streaming coverage."""
    pos = np.array([-table.length / 2 - dist, 0.0, height])
    tgt = np.array([-0.15, 0.0, 0.0])
    cam = Camera(pos, tgt, 1.0, width, int(round(width / aspect)))
    # choose f so the far rail and both side rails fit
    corners = np.array([[sx * (table.length / 2 + table.rail_width), sy * (table.width / 2 + table.rail_width), 0.04]
                        for sx in (-1, 1) for sy in (-1, 1)])
    Xc = (corners - cam.pos) @ cam.Rcw.T
    nx = np.abs(Xc[:, 0] / Xc[:, 2]).max()
    ny = np.abs(Xc[:, 1] / Xc[:, 2]).max()
    cam.f = f_scale * 0.96 * min(cam.cx / nx, cam.cy / ny)
    return cam


@dataclass
class Lighting:
    lamps: np.ndarray             # (n, 3) lamp centres
    size: float = 0.28            # side of each square area light
    grid: int = 2                 # samples per side
    ambient: float = 0.18
    gain: float = 1.0

    def samples(self) -> np.ndarray:
        g = (np.arange(self.grid) + 0.5) / self.grid - 0.5
        off = np.array([[a, b, 0.0] for a in g for b in g]) * self.size
        return (self.lamps[:, None, :] + off[None]).reshape(-1, 3)


def default_lighting(height: float = 0.95, spacing: float = 0.8) -> Lighting:
    return Lighting(np.array([[-spacing, 0, height], [0, 0, height], [spacing, 0, height]]))


def _ball_albedo(ident: int, q: np.ndarray) -> np.ndarray:
    """Albedo at body-frame unit vectors ``q`` (M,3) of ball ``ident``."""
    if ident == 0:
        return np.broadcast_to(WHITE, q.shape).copy()
    base = np.array(BALL_COLORS[ident if ident <= 8 else ident - 8])
    out = np.broadcast_to(base, q.shape).copy()
    if ident > 8:
        out[np.abs(q[:, 2]) > 0.52] = WHITE
    out[np.abs(q[:, 0]) > 0.9] = WHITE   # number discs on two sides
    return out


class Renderer:
    def __init__(self, cam: Camera, table: Table | None = None, light: Lighting | None = None,
                 R: float = 0.028575, spp: int = 4, seed: int = 0):
        self.cam, self.table = cam, table or Table()
        self.light = light or default_lighting()
        self.R, self.spp = R, spp
        self.rng = np.random.default_rng(seed)
        self.L = self.light.samples()
        lam = self.light.lamps
        # normalise so the cloth directly below the centre lamp has irradiance 1
        P0 = np.array([[0, 0, 0.0]])
        self.norm = 1.0 / self._irradiance(P0, np.array([[0, 0, 1.0]]), [], np.zeros((0, 3)))[0]
        self.background = self._render_full()

    # --------------------------------------------------------------- geometry
    def _irradiance(self, P, N, ids, C, skip=None):
        """Direct irradiance at points P with normals N, with ball shadows.

        Each area-light sample counts as a point source with 1/d^2 falloff."""
        tot = np.zeros(len(P))
        for Ls in self.L:
            D = Ls[None] - P
            dist = np.linalg.norm(D, axis=1)
            Dn = D / dist[:, None]
            cos = np.clip((Dn * N).sum(1), 0, None)
            vis = np.ones(len(P))
            for k, c in enumerate(C):
                oc = P - c[None]
                near = np.abs(oc[:, :2]).max(1) < 0.25
                if skip is not None:
                    near &= skip != k
                if not near.any():
                    continue
                b = (Dn[near] * oc[near]).sum(1)
                cc = (oc[near] ** 2).sum(1) - self.R ** 2
                disc = b * b - cc
                t = -b - np.sqrt(np.clip(disc, 0, None))
                blocked = (disc > 0) & (t > 1e-6) & (t < dist[near])
                vis[np.where(near)[0][blocked]] = 0.0
            tot += vis * cos / dist ** 2
        return tot

    def _shade(self, d: np.ndarray, ids, C, Rot) -> np.ndarray:
        """Colour of rays ``d`` (M,3) from the camera centre."""
        o = self.cam.pos
        M = len(d)
        tb, R = self.table, self.R
        hl, hw = tb.length / 2, tb.width / 2
        best = np.full(M, np.inf)
        kind = np.zeros(M, int)            # 0 floor, 1 cloth, 2 rail top, 3 cushion face, 4 ball
        which = np.full(M, -1)
        normal = np.zeros((M, 3))
        normal[:, 2] = 1.0

        def plane(z):
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (z - o[2]) / d[:, 2]
            return np.where(t > 0, t, np.inf)

        t = plane(-0.75)
        upd = t < best
        best[upd], kind[upd] = t[upd], 0
        for z, k, inside in ((0.0, 1, True), (tb.rail_height, 2, False)):
            t = plane(z)
            X = o[None] + t[:, None] * d
            inn = (np.abs(X[:, 0]) <= hl) & (np.abs(X[:, 1]) <= hw)
            out = (np.abs(X[:, 0]) <= hl + tb.rail_width) & (np.abs(X[:, 1]) <= hw + tb.rail_width)
            ok = inn if inside else (out & ~inn)
            upd = ok & (t < best)
            best[upd], kind[upd] = t[upd], k
        for ax, lim, other in ((0, hl, hw), (1, hw, hl)):
            for s in (-1, 1):
                with np.errstate(divide="ignore", invalid="ignore"):
                    t = (s * lim - o[ax]) / d[:, ax]
                X = o[None] + t[:, None] * d
                # inward-facing cushion face, seen by rays travelling towards it
                ok = (t > 0) & (d[:, ax] * s > 0) & (X[:, 2] >= 0) & \
                     (X[:, 2] <= tb.rail_height) & (np.abs(X[:, 1 - ax]) <= other)
                upd = ok & (t < best)
                best[upd], kind[upd] = t[upd], 3
                normal[upd] = 0.0
                normal[upd, ax] = -s
        for k, c in enumerate(C):
            oc = o - c
            b = d @ oc
            cc = oc @ oc - R * R
            disc = b * b - cc
            t = np.where(disc > 0, -b - np.sqrt(np.clip(disc, 0, None)), np.inf)
            upd = (t > 0) & (t < best)
            best[upd], kind[upd], which[upd] = t[upd], 4, k
        X = o[None] + np.where(np.isfinite(best), best, 0)[:, None] * d
        ball = kind == 4
        if ball.any():
            normal[ball] = (X[ball] - C[which[ball]]) / R
        alb = np.zeros((M, 3))
        alb[kind == 0] = FLOOR
        alb[kind == 1] = CLOTH
        alb[kind == 3] = CUSHION
        rail = kind == 2
        on_cushion = rail & ((np.abs(X[:, 0]) < hl + 0.05) & (np.abs(X[:, 1]) < hw + 0.05))
        alb[rail] = WOOD
        alb[on_cushion] = CUSHION
        # pockets: black discs on cloth and rail
        pk = tb.pocket_centres()
        pr = np.array([tb.corner_pocket_r] * 4 + [tb.side_pocket_r] * 2) * 1.05
        flat = (kind == 1) | (kind == 2) | (kind == 3)
        for (px, py), r in zip(pk, pr):
            alb[flat & (np.hypot(X[:, 0] - px, X[:, 1] - py) < r)] = 0.005
        for k in range(len(C)):
            sel = which == k
            if sel.any():
                q = (Rot[k].T @ normal[sel].T).T
                alb[sel] = _ball_albedo(ids[k], q)
        irr = self.norm * self._irradiance(X + 1e-5 * normal, normal, ids, C,
                                           skip=np.where(ball, which, -1))
        irr[kind == 0] = 0.3
        col = alb * (self.light.ambient + self.light.gain * irr)[:, None]
        if ball.any():
            # glossy phenolic balls: Blinn-Phong highlights of the lamps
            V = -d[ball]
            Nb = normal[ball]
            spec = np.zeros(ball.sum())
            for lc in self.light.lamps:
                Ld = lc[None] - X[ball]
                Ld /= np.linalg.norm(Ld, axis=1, keepdims=True)
                H = Ld + V
                H /= np.linalg.norm(H, axis=1, keepdims=True)
                spec += np.clip((H * Nb).sum(1), 0, None) ** 80
            col[ball] += 0.9 * spec[:, None]
        return col

    def _render_full(self) -> np.ndarray:
        cam = self.cam
        v, u = np.mgrid[0:cam.height, 0:cam.width]
        u, v = u.ravel().astype(float), v.ravel().astype(float)
        acc = np.zeros((len(u), 3))
        offs = [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)]
        for du, dv in offs:
            acc += self._shade(cam.rays(u + du, v + dv), [], np.zeros((0, 3)), np.zeros((0, 3, 3)))
        return (acc / len(offs)).reshape(cam.height, cam.width, 3)

    # ------------------------------------------------------------------ frames
    def _roi_mask(self, C: np.ndarray) -> np.ndarray:
        cam = self.cam
        mask = np.zeros((cam.height, cam.width), bool)
        R = self.R
        lz = self.light.lamps[:, 2].max()
        for c in C:
            pts = [c + np.array([sx, sy, sz]) * R * 1.15 for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
            for Ls in self.L:
                s = Ls + (c - Ls) * (Ls[2] / max(Ls[2] - c[2], 1e-3))
                s[2] = 0.0
                rs = R * lz / max(lz - c[2] - R, 1e-3) * 1.4
                pts += [s + np.array([sx * rs, sy * rs, 0.0]) for sx in (-1, 1) for sy in (-1, 1)]
            uv, z = cam.project(np.array(pts))
            if (z <= 0).any():
                continue
            u0, v0 = np.floor(uv.min(0)).astype(int) - 2
            u1, v1 = np.ceil(uv.max(0)).astype(int) + 3
            u0, v0 = max(u0, 0), max(v0, 0)
            u1, v1 = min(u1, cam.width), min(v1, cam.height)
            if u1 > u0 and v1 > v0:
                mask[v0:v1, u0:u1] = True
        return mask

    def render(self, traj: Trajectory, ids, frame_times: np.ndarray, shutter: float = 0.5,
               fps: float = 60.0):
        """Yield linear-RGB float frames for ``frame_times``.

        Each of the ``spp`` samples per pixel gets its own sub-pixel offset and
        its own instant inside the shutter window."""
        cam = self.cam
        tt = traj.t
        for tf in frame_times:
            mask_union = None
            samples = []
            for s in range(self.spp):
                ts = tf + shutter / fps * (s + self.rng.random()) / self.spp
                k = np.clip(np.searchsorted(tt, ts) - 1, 0, len(tt) - 2)
                a = np.clip((ts - tt[k]) / (tt[k + 1] - tt[k]), 0.0, 1.0)
                P = (1 - a) * traj.pos[k] + a * traj.pos[k + 1]
                act = traj.active[k + 1] & traj.active[k]
                Rot = traj.rot[k if a < 0.5 else k + 1]
                C, I, Rs = P[act], [ids[i] for i in np.where(act)[0]], Rot[act]
                samples.append((C, I, Rs))
                m = self._roi_mask(C)
                mask_union = m if mask_union is None else (mask_union | m)
            img = self.background.copy()
            vv, uu = np.nonzero(mask_union)
            if len(uu):
                acc = np.zeros((len(uu), 3))
                for C, I, Rs in samples:
                    du, dv = self.rng.random(2) - 0.5
                    acc += self._shade(cam.rays(uu + du, vv + dv), I, C, Rs)
                img[vv, uu] = acc / len(samples)
            yield img


def to_uint8(img: np.ndarray, rng: np.random.Generator, noise: float = 0.008,
             exposure: float = 1.0) -> np.ndarray:
    """Tone-map (gamma 2.2), add sensor noise and quantise."""
    x = np.clip(img * exposure, 0, 1) ** (1 / 2.2)
    x = x + rng.normal(0, noise, x.shape)
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)
