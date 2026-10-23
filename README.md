# billiard-research

**Did it leave the cloth?** Detecting airborne balls (jump shots, break hops,
balls riding over balls) in pool video, and finding out which visual cues make
that possible from an **overhead** table camera vs. an elevated **broadcast**
camera.

No billiards dataset labels ball height, so the project starts with a
simulator and renderer that produce video with exact per-frame height ground
truth. A detector built on interpretable pixel cues is evaluated on that video,
and a protocol for real clips is set out to test transfer.

| | |
|---|---|
| Paper draft | [`paper/main.tex`](paper/main.tex) (build: `cd paper && latexmk -pdf main.tex`) |
| Scenario clips | [`data/clips/`](data/clips): 7 shots × 2 cameras, mp4 + `gt.json` |
| Real-clip protocol | [`data/real/README.md`](data/real/README.md) |
| Results | [`results/`](results): metrics, ablation table, cue geometry |

## The scenario clips

| Clip | What it tests |
|---|---|
| `stop_shot` | level stun shot (negative) |
| `jump_over_blocker` | clean jump over a ball, ~10 cm apex (positive) |
| `jump_clips_blocker` | jump too low, rides up the blocker, which hops (positive) |
| `near_miss` | cue ball passes 4 mm beside a ball, which looks like contact in 2-D (hard negative) |
| `break` | 16-ball break with a slightly elevated cue (small hops) |
| `long_jump_off_axis` | long jump near the rail, far from the overhead camera's axis (positive) |
| `elevated_draw` | 14° elevated draw, ~1 cm hop (borderline) |

## Layout

```
src/jumpsim/
  physics.py     3-D billiards physics with an airborne phase (flight, slate bounces, 3-D collisions)
  render.py      ray-cast renderer: cameras, lamps with soft shadows, motion blur, textured spinning balls
  scenarios.py   the scenario clips and the randomised shot generator
  detect.py      cue-ball tracker and per-frame size / shadow / motion cues (no ground truth used)
scripts/
  make_clips.py          render clips (+ ground truth)
  extract_features.py    track + compute cues for every clip
  evaluate.py            train/test the classifier, cue ablations, figures
  cue_geometry.py        pixels of signal per mm of height, per cue and camera
  make_frames_figure.py  qualitative figure
  make_paper_numbers.py  results/numbers.tex for the paper
```

## Reproduce

```bash
pip install -r requirements.txt
python scripts/make_clips.py --showcase --out data/clips                        # ~10 min on 4 cores
python scripts/make_clips.py --random 200 --seed 1 --randomize --out data/synth  # ~40 min on 4 cores
python scripts/extract_features.py data/clips data/synth --out data/features
python scripts/cue_geometry.py
python scripts/evaluate.py
python scripts/make_paper_numbers.py
python scripts/make_frames_figure.py data/clips/jump_over_blocker paper/figures/frames_jump_over_blocker.png
cd paper && latexmk -pdf main.tex
```

`data/synth/` and `data/features/` are not committed (regenerate them with the commands above).
