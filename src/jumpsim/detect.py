"""Cue-ball tracking and per-frame airborne cues from video pixels.

Nothing here reads ground truth: the inputs are the video, the empty-table
reference frame, the camera calibration and the ball radius. That's what a
fixed table camera has after a one-off calibration.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import brentq, least_squares

from .render import Camera

G = 9.81


def camera_from_dict(d: dict) -> Camera:
    return Camera(np.array(d["pos"]), np.array(d["target"]), d["f"], d["width"], d["height"],
                  up=np.array(d["up"]))


def silhouette_radius(cam: Camera, C: np.ndarray, R: float, n: int = 48) -> np.ndarray:
    """Area-equivalent image radius (px) of spheres centred at C (M,3)."""
    C = np.atleast_2d(C)
    D = C - cam.pos
    d = np.linalg.norm(D, axis=1)
    nh = D / d[:, None]
    cc = C - (R * R / d)[:, None] * nh
    rho = R * np.sqrt(1 - (R / d) ** 2)
    a = np.cross(nh, np.array([0.0, 0.0, 1.0]))
    bad = np.linalg.norm(a, axis=1) < 1e-6
    a[bad] = np.cross(nh[bad], np.array([1.0, 0, 0]))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = np.cross(nh, a)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    P = cc[:, None] + rho[:, None, None] * (np.cos(th)[None, :, None] * a[:, None] + np.sin(th)[None, :, None] * b[:, None])
    uv, _ = cam.project(P)
    x, y = uv[..., 0], uv[..., 1]
    area = 0.5 * np.abs((x * np.roll(y, -1, 1) - np.roll(x, -1, 1) * y).sum(1))
    return np.sqrt(area / np.pi)


def sphere_centre_image(cam: Camera, C: np.ndarray) -> np.ndarray:
    """Image position of a sphere's centre (good to <0.1 px for a 2.25 in ball)."""
    uv, _ = cam.project(np.atleast_2d(C))
    return uv


@dataclass
class Clip:
    path: Path
    view: str
    frames: np.ndarray            # (T, H, W, 3) uint8 RGB
    empty: np.ndarray             # (H, W, 3) uint8 RGB
    cam: Camera
    R: float
    fps: float
    n_pre: int                    # still frames before the strike
    gt: dict

    @classmethod
    def load(cls, path: Path, view: str) -> "Clip":
        path = Path(path)
        gt = json.loads((path / "gt.json").read_text())
        cap = cv2.VideoCapture(str(path / f"{view}.mp4"))
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        empty = cv2.cvtColor(cv2.imread(str(path / f"{view}_empty.jpg")), cv2.COLOR_BGR2RGB)
        n_pre = int(round(gt["pre"] * gt["fps"]))
        return cls(path, view, np.array(frames), empty, camera_from_dict(gt["cameras"][view]),
                   gt["R"], gt["fps"], n_pre, gt)


def _lum(img):
    return img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114


class CueTracker:
    """Find the cue ball in each frame and measure its centre, radius and shadow."""

    def __init__(self, clip: Clip):
        self.c = clip
        self.bg = clip.empty.astype(np.float32) / 255
        self.Ybg = _lum(self.bg)
        self.bg_chroma = self.bg / (self.bg.sum(2, keepdims=True) + 1e-3)
        cam = clip.cam
        # expected on-cloth radius map, coarse grid, for gating
        self.r_centre = float(silhouette_radius(cam, np.array([[0, 0, clip.R]]), clip.R)[0])

    def _white_mask(self, img):
        Y = _lum(img)
        mx, mn = img.max(2), img.min(2)
        sat = (mx - mn) / (mx + 1e-3)
        return ((Y - self.Ybg) > 0.18) & (sat < 0.3) & (Y > 0.55)

    def _expected_r(self, uv):
        X = self.c.cam.backproject_to_plane(uv, self.c.R)
        return float(silhouette_radius(self.c.cam, X, self.c.R)[0])

    def _alpha(self, img, uv, rad, mode):
        """Soft ball coverage in a disc of radius ``rad`` around ``uv``.

        ``lum``: brighter-than-cloth, scaled by the local peak (good when the
        visible side is lit). ``chroma``: loss of the cloth's green chromaticity,
        which is unaffected by shading and by shadows on the cloth."""
        H, W = self.Ybg.shape
        u0, u1 = int(max(uv[0] - rad - 2, 0)), int(min(uv[0] + rad + 3, W))
        v0, v1 = int(max(uv[1] - rad - 2, 0)), int(min(uv[1] + rad + 3, H))
        vv, uu = np.mgrid[v0:v1, u0:u1]
        inside = (uu - uv[0]) ** 2 + (vv - uv[1]) ** 2 < rad ** 2
        sub = img[v0:v1, u0:u1]
        if mode == "lum":
            d = _lum(sub) - self.Ybg[v0:v1, u0:u1]
            peak = np.percentile(d[inside], 97) if inside.any() else 0
            if peak < 0.1:
                return None
            a = np.clip(d / (0.5 * peak), 0, 1)
        else:
            gb = self.bg_chroma[v0:v1, u0:u1, 1]
            g = sub[..., 1] / (sub.sum(2) + 1e-3)
            a = np.clip((gb - g) / (gb - 1 / 3), 0, 1)
        return a * inside, uu, vv

    @staticmethod
    def _moments(a, uu, vv):
        m = a.sum()
        cu, cv = (a * uu).sum() / m, (a * vv).sum() / m
        du, dv = uu - cu, vv - cv
        cov = np.array([[(a * du * du).sum(), (a * du * dv).sum()],
                        [(a * du * dv).sum(), (a * dv * dv).sum()]]) / m
        lam = np.linalg.eigvalsh(cov)
        # a uniform disc of radius r has variance r^2/4 along every axis; motion
        # blur only adds variance along the direction of travel
        return np.array([cu, cv]), np.sqrt(m / np.pi), 2 * np.sqrt(max(lam[0], 0)), \
            2 * np.sqrt(max(lam[1] - lam[0], 0))

    def _refine(self, img, uv, r_exp):
        """Centroid plus area-, and blur-robust (minor-axis), radii for both masks."""
        for _ in range(3):
            res = self._alpha(img, uv, 1.6 * r_exp, "lum")
            if res is None or res[0].sum() < 1:
                return None
            uv, r_area, r_perp, blur = self._moments(*res)
        out = dict(uv=uv, r_area=r_area, r_perp=r_perp, blur=blur)
        out["rc_area"] = out["rc_perp"] = np.nan
        for _ in range(2):
            res = self._alpha(img, out["uv"], 1.6 * r_exp, "chroma")
            if res is None or res[0].sum() < 0.3 * np.pi * r_exp ** 2:
                break
            # the chroma centroid is not pulled towards the lit side of the ball
            out["uv"], out["rc_area"], out["rc_perp"], _ = self._moments(*res)
        return out

    def _shadow(self, img, uv, r):
        """Shadow mass, centroid offset and spread around the ball (units of r)."""
        H, W = self.Ybg.shape
        rad = 4.5 * r
        u0, u1 = int(max(uv[0] - rad, 0)), int(min(uv[0] + rad + 1, W))
        v0, v1 = int(max(uv[1] - rad, 0)), int(min(uv[1] + rad + 1, H))
        if u1 <= u0 or v1 <= v0:
            return np.zeros(6)
        vv, uu = np.mgrid[v0:v1, u0:u1]
        du, dv = (uu - uv[0]) / r, (vv - uv[1]) / r
        rr = np.hypot(du, dv)
        sub = img[v0:v1, u0:u1]
        bg = self.bg[v0:v1, u0:u1]
        Yr = _lum(sub) / (_lum(bg) + 1e-3)
        chroma = sub / (sub.sum(2, keepdims=True) + 1e-3)
        same_hue = np.abs(chroma - self.bg_chroma[v0:v1, u0:u1]).sum(2) < 0.12
        a = np.clip((1 - Yr) / 0.35, 0, 1) * same_hue * (rr > 1.12) * (rr < 4.5)
        m = a.sum()
        if m < 1e-3:
            return np.zeros(6)
        cu, cv = (a * du).sum() / m, (a * dv).sum() / m
        spread = np.sqrt((a * ((du - cu) ** 2 + (dv - cv) ** 2)).sum() / m)
        near = (a * (rr < 1.6)).sum() / (np.pi * (1.6 ** 2 - 1.12 ** 2) + 1e-6)
        return np.array([m / np.pi, cu, cv, np.hypot(cu, cv), spread, near])

    def track(self):
        c = self.c
        T = len(c.frames)
        out = dict(found=np.zeros(T, bool), uv=np.full((T, 2), np.nan), r_exp=np.full(T, np.nan),
                   shadow=np.zeros((T, 6)),
                   **{k: np.full(T, np.nan) for k in ("r_area", "r_perp", "rc_area", "rc_perp", "blur")})
        prev, vel, missed = None, np.zeros(2), 0
        for t in range(T):
            img = c.frames[t].astype(np.float32) / 255
            mask = self._white_mask(img).astype(np.uint8)
            n, lab, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
            cand = [(cents[i], stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)
                    if stats[i, cv2.CC_STAT_AREA] > 0.25 * np.pi * self.r_centre ** 2]
            missed += 1
            if not cand:
                continue
            if prev is None:
                # initialise on the largest white blob (stripes show only partial white)
                cen, _ = max(cand, key=lambda ca: ca[1])
            else:
                # the gate grows while the ball is lost, so a hard strike (tens of
                # pixels per frame from a standstill) is re-acquired
                pred = prev + vel * missed
                gate = max(4 * self.r_centre, 1.5 * np.linalg.norm(vel) + 2 * self.r_centre) + \
                    6 * self.r_centre * (missed - 1)
                dists = [np.linalg.norm(ca[0] - pred) for ca in cand]
                k = int(np.argmin(dists))
                if dists[k] > gate:
                    continue
                cen = cand[k][0]
            r_exp = self._expected_r(cen)
            res = self._refine(img, np.array(cen, float), r_exp)
            if res is None:
                continue
            uv = res["uv"]
            if prev is not None:
                vel = 0.6 * (uv - prev) / missed + 0.4 * vel
            prev, missed = uv, 0
            out["found"][t] = True
            out["uv"][t] = uv
            for k in ("r_area", "r_perp", "rc_area", "rc_perp", "blur"):
                out[k][t] = res[k]
            out["r_exp"][t] = self._expected_r(uv)
            out["shadow"][t] = self._shadow(img, uv, out["r_exp"][t])
        return out


# ---------------------------------------------------------------- model fits
def _window_fit(cam: Camera, R: float, uv: np.ndarray, r: np.ndarray, t: np.ndarray,
                sig_c: float = 0.35, sig_r: float = 0.12, use_radius: bool = False):
    """Fit a rolling (on-cloth, constant 2-D acceleration) and a ballistic model to
    a short window of observations. Returns (cost_roll, cost_ball, z_ball_mid, vz).

    With ``use_radius`` the apparent radius is fitted too; by default only the
    centre track is used so the motion cue stays separable from the size cue."""
    X0 = cam.backproject_to_plane(uv, R)
    tc = t - t.mean()
    A = np.column_stack([np.ones_like(tc), tc])
    coef, *_ = np.linalg.lstsq(A, X0[:, :2], rcond=None)

    def image_res(P):
        out = [((sphere_centre_image(cam, P) - uv) / sig_c).ravel()]
        if use_radius:
            out.append((silhouette_radius(cam, P, R) - r) / sig_r)
        return out

    def res_roll(p):
        # on the cloth: deceleration along the direction of travel bounded by
        # sliding friction, plus a small lateral term for swerve
        x0, y0, vx, vy, a_lon, a_lat = p
        sp = np.hypot(vx, vy) + 1e-9
        ex, ey = vx / sp, vy / sp
        ax, ay = -a_lon * ex - a_lat * ey, -a_lon * ey + a_lat * ex
        P = np.column_stack([x0 + vx * tc + 0.5 * ax * tc ** 2, y0 + vy * tc + 0.5 * ay * tc ** 2,
                             np.full(len(t), R)])
        return np.concatenate(image_res(P))

    def res_ball(p):
        x = p[0] + p[3] * tc
        y = p[1] + p[4] * tc
        z = p[2] + p[5] * tc - 0.5 * G * tc ** 2
        P = np.column_stack([x, y, z])
        pen = np.maximum(R - z, 0) / 0.002          # cannot be below the cloth
        return np.concatenate(image_res(P) + [pen])

    lo = [-np.inf] * 4 + [-0.3, -0.6]
    hi = [np.inf] * 4 + [2.5, 0.6]
    p0 = np.r_[coef[0], coef[1], 0.5, 0.0]
    p_roll = least_squares(res_roll, p0, bounds=(lo, hi), x_scale=[0.1, 0.1, 1, 1, 1, 1])
    best = None
    for z0 in (R + 0.005, R + 0.04, R + 0.1):
        p0 = np.r_[coef[0, 0], coef[0, 1], z0, coef[1, 0], coef[1, 1], 0.0]
        sol = least_squares(res_ball, p0, method="trf", x_scale=[0.1, 0.1, 0.02, 1, 1, 0.5])
        if best is None or sol.cost < best.cost:
            best = sol
    return p_roll.cost, best.cost, best.x[2] - R, best.x[5]


def _win(x: np.ndarray, half: int, fn) -> np.ndarray:
    import warnings
    pad = np.pad(x, half, constant_values=np.nan)
    w = np.lib.stride_tricks.sliding_window_view(pad, 2 * half + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(w, 1)


def frame_features(clip: Clip, trk: dict, half: int = 4) -> tuple[np.ndarray, list[str]]:
    """Per-frame cue vector. NaN where the cue ball was not found.

    Size and shadow cues are expressed relative to the still frames before the
    strike, which absorbs the per-installation bias of the radius estimator and
    the room's lamp layout."""
    cam, R = clip.cam, clip.R
    found = trk["found"]
    T = len(found)
    pre = found.copy()
    pre[clip.n_pre:] = False
    if not pre.any():
        pre = found
    t = np.arange(T) / clip.fps
    X = np.full((T, 3), np.nan)
    X[found] = cam.backproject_to_plane(trk["uv"][found], R)
    feats, names = [], []

    def add(name, v):
        names.append(name)
        feats.append(np.asarray(v, float))

    # size: blur-robust minor-axis radius from two masks, relative to an on-cloth ball here
    for key, tag in (("r_perp", "lum"), ("rc_perp", "chroma")):
        ratio = trk[key] / trk["r_exp"]
        rr = ratio / np.nanmedian(ratio[pre]) - 1
        add(f"size_{tag}", rr)
        add(f"size_{tag}_win_mean", _win(rr, half, np.nanmean))
        add(f"size_{tag}_win_max", _win(rr, half, np.nanmax))
    sh = trk["shadow"].copy()
    sh[~found] = np.nan
    base = np.nanmedian(sh[pre], 0)
    for i, n in enumerate(["mass", "du", "dv", "off", "spread", "near"]):
        add(f"shadow_{n}", sh[:, i])
        add(f"shadow_{n}_delta", sh[:, i] - base[i])
    add("shadow_off_delta_win_max", _win(sh[:, 3] - base[3], half, np.nanmax))
    add("shadow_near_delta_win_min", _win(sh[:, 5] - base[5], half, np.nanmin))
    # motion: table-plane kinematics under the on-cloth assumption
    vel = np.gradient(X[:, :2], t, axis=0)
    acc = np.gradient(vel, t, axis=0)
    add("speed", np.linalg.norm(vel, axis=1))
    add("accel", np.linalg.norm(acc, axis=1))
    add("blur", trk["blur"] / trk["r_exp"])
    # where the ball is: shadow and size baselines vary with position under the lamps
    add("pos_x", X[:, 0])
    add("pos_y", X[:, 1])
    # rolling vs ballistic fits on three 7-frame windows (ending at, centred on and
    # starting at frame k); a window that straddles a bounce fits neither model,
    # so keep the window that favours flight the most
    cr, cb, zb, vz = (np.full(T, np.nan) for _ in range(4))
    for k in range(T):
        if not found[k]:
            continue
        best = None
        for lo in (k - 6, k - 3, k):
            lo, hi = max(lo, 0), min(lo + 7, T)
            sel = np.arange(lo, hi)[found[lo:hi]]
            if len(sel) < 5:
                continue
            if np.ptp(trk["uv"][sel], axis=0).max() < 0.5:   # still ball: nothing to fit
                fit = (0.0, 0.0, 0.0, 0.0)
            else:
                fit = _window_fit(cam, R, trk["uv"][sel], trk["r_perp"][sel], t[sel])
            if best is None or np.log1p(fit[0]) - np.log1p(fit[1]) > np.log1p(best[0]) - np.log1p(best[1]):
                best = fit
        if best is not None:
            cr[k], cb[k], zb[k], vz[k] = best
    add("fit_cost_roll", np.log1p(cr))
    add("fit_cost_ballistic", np.log1p(cb))
    lr = np.log1p(cr) - np.log1p(cb)
    add("fit_log_ratio", lr)
    add("fit_log_ratio_win_max", _win(lr, half, np.nanmax))
    add("fit_height", zb)
    add("fit_vz", vz)
    return np.column_stack(feats), names


def feature_groups(names: list[str]) -> dict[str, list[str]]:
    return {g: [n for n in names if n.startswith(p)] for g, p in
            (("size", "size_"), ("shadow", "shadow_"))} | \
        {"motion": [n for n in names if not n.startswith(("size_", "shadow_"))]}
