"""Train and evaluate the per-frame airborne classifier; write tables and figures.

    python scripts/evaluate.py --features data/features --out results --figs paper/figures

Protocol
--------
* Unit: frames in which the cue ball is still on the table.
* Positive: the cue ball is > 5 mm off the cloth at some instant of the
  exposure. Frames with a 1-5 mm hop are "don't care" for frame metrics
  (no camera here can resolve them; see the cue-geometry analysis).
* Split by shot: random shots 0-139 train, 140-199 test. The showcase clips
  are never trained on.
* Thresholds come from out-of-fold predictions on the training shots only.
* Raw speed and blur length are left out of every cue set: in any finite shot
  generator they correlate with the label through the shot mix (jump shots are
  struck hard), and a detector should not learn that prior.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

H_POS, H_AIR = 0.005, 0.001
CONTEXT = ["pos_x", "pos_y"]


def cue_sets(names):
    size = [n for n in names if n.startswith("size_")]
    shadow = [n for n in names if n.startswith("shadow_")]
    motion = ["accel"] + [n for n in names if n.startswith("fit_")]
    return {
        "size": size + CONTEXT, "shadow": shadow + CONTEXT, "motion": motion + CONTEXT,
        "size+shadow": size + shadow + CONTEXT, "size+motion": size + motion + CONTEXT,
        "shadow+motion": shadow + motion + CONTEXT, "all": size + shadow + motion + CONTEXT,
    }


def load(feat_dir: Path):
    data = defaultdict(dict)
    for p in sorted(feat_dir.glob("*.npz")):
        z = np.load(p, allow_pickle=False)
        data[str(z["clip"])][str(z["view"])] = {k: z[k] for k in z.files}
    return data


def table(data, clips, view, cols):
    """Stack frames of ``clips`` for ``view`` (or 'fused') into X, y, h, groups, frame idx."""
    Xs, hs, gs, ks, fs = [], [], [], [], []
    for ci, c in enumerate(clips):
        views = ["overhead", "broadcast"] if view == "fused" else [view]
        d0 = data[c][views[0]]
        T = min(len(data[c][v]["F"]) for v in views)
        parts = []
        for v in views:
            d = data[c][v]
            ix = [list(d["names"]).index(n) for n in cols]
            Fv = d["F"][:T, ix].astype(float)
            Fv[~d["found"][:T]] = np.nan
            parts.append(Fv)
        X = np.hstack(parts)
        found = np.any([data[c][v]["found"][:T] for v in views], axis=0)
        act = d0["active"][:T].astype(bool)
        Xs.append(X[act])
        hs.append(d0["height"][:T][act])
        gs.append(np.full(act.sum(), ci))
        ks.append(np.arange(T)[act])
        fs.append(found[act])
    return np.vstack(Xs), np.concatenate(hs), np.concatenate(gs), np.concatenate(ks), np.concatenate(fs)


def model():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=0)


def smooth(p, g):
    out = p.copy()
    for gi in np.unique(g):
        s = g == gi
        q = p[s]
        if len(q) >= 3:
            out[s] = np.median(np.stack([np.r_[q[:1], q[:-1]], q, np.r_[q[1:], q[-1:]]]), 0)
    return out


def runs(mask):
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def event_metrics(p, thr, h, g, fps=60.0):
    """GT event: a run of > 1 mm frames (bounces merged across <= 3-frame gaps)
    whose apex exceeds 5 mm. Detected if >= 2 frames inside (+-1) fire.
    False alarm: a >= 2-frame predicted run touching no > 1 mm frame (+-2)."""
    det, apex, fa, minutes = [], [], 0, 0.0
    for gi in np.unique(g):
        s = g == gi
        hh, pp = h[s], p[s] >= thr
        minutes += len(hh) / fps / 60
        air = hh > H_AIR
        for a, b in runs(air):
            pass
        # merge short gaps
        merged = air.copy()
        for a, b in runs(~air):
            if 0 < a and b < len(air) and b - a <= 3:
                merged[a:b] = True
        for a, b in runs(merged):
            if hh[a:b].max() <= H_POS:
                continue
            lo, hi = max(a - 1, 0), min(b + 1, len(pp))
            det.append(pp[lo:hi].sum() >= 2)
            apex.append(hh[a:b].max())
        near = np.convolve(air.astype(int), np.ones(5, int), "same") > 0
        for a, b in runs(pp):
            if b - a >= 2 and not near[a:b].any():
                fa += 1
    return np.array(det), np.array(apex), fa, minutes


def shot_scores(p, h, g):
    ys, ss = [], []
    for gi in np.unique(g):
        s = g == gi
        ys.append(h[s].max() > H_POS)
        ss.append(p[s].max())
    return np.array(ys), np.array(ss)


def fit_eval(data, train, test, view, cols):
    Xtr, htr, gtr, _, _ = table(data, train, view, cols)
    Xte, hte, gte, kte, fte = table(data, test, view, cols)
    care_tr = (htr > H_POS) | (htr <= H_AIR)
    ytr = htr > H_POS
    oof = np.zeros(len(ytr))
    for tr_i, va_i in GroupKFold(5).split(Xtr, ytr, gtr):
        m = model().fit(Xtr[tr_i][care_tr[tr_i]], ytr[tr_i][care_tr[tr_i]])
        oof[va_i] = m.predict_proba(Xtr[va_i])[:, 1]
    oof = smooth(oof, gtr)
    cand = np.linspace(0.05, 0.95, 91)
    thr = float(cand[np.argmax([f1_score(ytr[care_tr], oof[care_tr] >= c) for c in cand])])
    m = model().fit(Xtr[care_tr], ytr[care_tr])
    p = smooth(m.predict_proba(Xte)[:, 1], gte)
    p[~fte] = 0.0                                   # untracked frames count as "not airborne"
    care = (hte > H_POS) | (hte <= H_AIR)
    yte = hte > H_POS
    det, apex, fa, minutes = event_metrics(p, thr, hte, gte)
    ys, ss = shot_scores(p, hte, gte)
    res = dict(view=view, n_features=len(cols), thr=thr,
               frame_ap=float(average_precision_score(yte[care], p[care])),
               frame_auroc=float(roc_auc_score(yte[care], p[care])),
               frame_f1=float(f1_score(yte[care], p[care] >= thr)),
               event_recall=float(det.mean()), n_events=int(len(det)),
               false_alarms_per_min=fa / minutes, shot_auroc=float(roc_auc_score(ys, ss)),
               tracked=float(fte.mean()))
    return res, m, dict(p=p, h=hte, g=gte, k=kte, det=det, apex=apex, thr=thr)


def height_regression(data, train, test, view, cols):
    Xtr, htr, *_ = table(data, train, view, cols)
    Xte, hte, *_ = table(data, test, view, cols)
    a, b = htr > H_POS, hte > H_POS
    reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(Xtr[a], htr[a])
    pred = reg.predict(Xte[b])
    return float(np.mean(np.abs(pred - hte[b]))), float(np.median(np.abs(pred - hte[b]) / hte[b]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=Path("data/features"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--figs", type=Path, default=Path("paper/figures"))
    ap.add_argument("--n-train", type=int, default=140)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    a.figs.mkdir(parents=True, exist_ok=True)
    data = load(a.features)
    data = {c: v for c, v in data.items() if len(v) == 2}
    rand = sorted(c for c in data if c.startswith("r"))
    show = sorted(c for c in data if not c.startswith("r"))
    train, test = rand[:a.n_train], rand[a.n_train:]
    names = list(next(iter(data.values()))["overhead"]["names"])
    sets = cue_sets(names)
    print(f"{len(train)} train shots, {len(test)} test shots, {len(show)} showcase clips")

    # label balance
    _, htr, *_ = table(data, train, "overhead", CONTEXT)
    _, hte, *_ = table(data, test, "overhead", CONTEXT)
    summary = dict(train_shots=len(train), test_shots=len(test),
                   train_frames=int(len(htr)), test_frames=int(len(hte)),
                   train_pos_frac=float((htr > H_POS).mean()), test_pos_frac=float((hte > H_POS).mean()),
                   train_dontcare_frac=float(((htr > H_AIR) & (htr <= H_POS)).mean()))
    print(summary)

    results, keep = [], {}
    for view in ("overhead", "broadcast", "fused"):
        for sname, cols in sets.items():
            res, m, extra = fit_eval(data, train, test, view, cols)
            res["cues"] = sname
            results.append(res)
            keep[(view, sname)] = (m, extra, cols)
            print(f"{view:9s} {sname:14s} AP={res['frame_ap']:.3f} AUROC={res['frame_auroc']:.3f} "
                  f"F1={res['frame_f1']:.3f} evR={res['event_recall']:.3f} FA/min={res['false_alarms_per_min']:.2f} "
                  f"shotAUC={res['shot_auroc']:.3f}", flush=True)
    heights = {}
    for view in ("overhead", "broadcast", "fused"):
        mae, rel = height_regression(data, train, test, view, sets["all"])
        heights[view] = dict(mae_mm=mae * 1000, median_rel_err=rel)
        print(f"height regression {view}: MAE {mae * 1000:.1f} mm, median rel. err {rel:.2f}")

    # ---- showcase: per-clip predictions from the all-cue models (trained on random shots only)
    showcase = {}
    for view in ("overhead", "broadcast", "fused"):
        m, _, cols = keep[(view, "all")]
        thr = keep[(view, "all")][1]["thr"]
        for c in show:
            X, h, g, k, f = table(data, [c], view, cols)
            p = smooth(m.predict_proba(X)[:, 1], g)
            p[~f] = 0
            det, apex, fa, _ = event_metrics(p, thr, h, g)
            showcase.setdefault(c, {})[view] = dict(p=p.tolist(), h=h.tolist(), k=k.tolist(), thr=thr,
                                                    detected=det.tolist(), apex_mm=(apex * 1000).tolist(),
                                                    false_alarms=fa, max_p=float(p.max()))

    json.dump(dict(summary=summary, results=results, height=heights,
                   showcase={c: {v: {k: x for k, x in d.items() if k not in ("p", "h", "k")}
                                 for v, d in vs.items()} for c, vs in showcase.items()}),
              open(a.out / "metrics.json", "w"), indent=1)
    json.dump(showcase, open(a.out / "showcase_predictions.json", "w"))
    write_table(results, heights, a.out / "table_ablation.tex")
    fig_recall_vs_apex(keep, a.figs / "recall_vs_apex.pdf")
    fig_showcase(showcase, data, a.figs / "showcase_timeseries.pdf")
    np.savez_compressed(a.out / "test_predictions.npz", clips=np.array(test),
                        **{f"{v}__{k}__{c}": keep[(v, c)][1][k] for v in ("overhead", "broadcast", "fused")
                           for c in ("all",) for k in ("p", "h", "g", "k", "thr")})
    imp = importance(data, test, keep)
    json.dump(imp, open(a.out / "importance.json", "w"), indent=1)


def importance(data, test, keep, n_rep=3):
    """Permutation importance (drop in frame AP) of each feature, all-cue models."""
    from sklearn.inspection import permutation_importance
    out = {}
    for view in ("overhead", "broadcast"):
        m, _, cols = keep[(view, "all")]
        X, h, *_ = table(data, test, view, cols)
        care = (h > H_POS) | (h <= H_AIR)
        r = permutation_importance(m, X[care], h[care] > H_POS, scoring="average_precision",
                                   n_repeats=n_rep, random_state=0)
        order = np.argsort(-r.importances_mean)
        out[view] = [(cols[i], float(r.importances_mean[i])) for i in order[:10]]
        print(view, "top features:", out[view][:5])
    return out


def write_table(results, heights, path):
    order = ["size", "shadow", "motion", "size+shadow", "size+motion", "shadow+motion", "all"]
    by = {(r["view"], r["cues"]): r for r in results}
    lines = [r"\begin{tabular}{l" + "rrr" * 3 + "}", r"\toprule",
             r" & \multicolumn{3}{c}{Overhead} & \multicolumn{3}{c}{Broadcast} & \multicolumn{3}{c}{Both views} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
             r"Cues & AP & Ev.\,rec. & FA/min & AP & Ev.\,rec. & FA/min & AP & Ev.\,rec. & FA/min \\", r"\midrule"]
    for cue in order:
        row = [cue]
        for v in ("overhead", "broadcast", "fused"):
            r = by[(v, cue)]
            row += [f"{r['frame_ap']:.2f}", f"{r['event_recall']:.2f}", f"{r['false_alarms_per_min']:.1f}"]
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\midrule", "Height MAE (mm) & \\multicolumn{3}{c}{" + f"{heights['overhead']['mae_mm']:.0f}" +
              "} & \\multicolumn{3}{c}{" + f"{heights['broadcast']['mae_mm']:.0f}" + "} & \\multicolumn{3}{c}{" +
              f"{heights['fused']['mae_mm']:.0f}" + r"} \\", r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def fig_recall_vs_apex(keep, path):
    bins = np.array([5, 15, 30, 60, 100, 250]) / 1000
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8), sharey=True)
    for ax, view in zip(axes, ("overhead", "broadcast", "fused")):
        for cue, style in (("size", ":"), ("shadow", "--"), ("motion", "-."), ("all", "-")):
            ex = keep[(view, cue)][1]
            idx = np.digitize(ex["apex"], bins) - 1
            rec = [ex["det"][idx == i].mean() if (idx == i).any() else np.nan for i in range(len(bins) - 1)]
            ax.plot(range(len(bins) - 1), rec, style, marker="o", ms=3, label=cue,
                    color="k" if cue == "all" else None)
        ax.set_xticks(range(len(bins) - 1))
        idx = np.digitize(keep[(view, "all")][1]["apex"], bins) - 1
        ax.set_xticklabels([f"{int(bins[i]*1000)}-{int(bins[i+1]*1000)}\n(n={(idx == i).sum()})"
                            for i in range(len(bins) - 1)], fontsize=7)
        ax.set_title({"fused": "both views"}.get(view, view), fontsize=9)
        ax.set_xlabel("event apex height (mm)", fontsize=8)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("event recall", fontsize=8)
    axes[-1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_showcase(showcase, data, path):
    clips = list(showcase)
    fig, axes = plt.subplots(len(clips), 1, figsize=(7, 1.35 * len(clips)), sharex=True)
    for ax, c in zip(np.atleast_1d(axes), clips):
        d = showcase[c]
        k = np.array(d["overhead"]["k"]) / 60.0 - 0.15
        ax.fill_between(k, 0, np.array(d["overhead"]["h"]) * 1000 / 100, color="0.85", lw=0, label="GT height (/100 mm)")
        for v, col in (("overhead", "C0"), ("broadcast", "C1"), ("fused", "k")):
            ax.plot(np.array(d[v]["k"]) / 60.0 - 0.15, d[v]["p"], color=col, lw=1, label=f"P(airborne) {v}")
        ax.set_ylim(0, 1.05)
        ax.text(0.99, 0.85, c.replace("_", " "), transform=ax.transAxes, ha="right", fontsize=8)
        ax.tick_params(labelsize=7)
    np.atleast_1d(axes)[0].legend(fontsize=6, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.45))
    np.atleast_1d(axes)[-1].set_xlabel("time since strike (s)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
