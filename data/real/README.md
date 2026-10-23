# Real clips: collection protocol

The synthetic study shows *what* to measure. Whether it transfers to real video
can only be tested on real clips with **height ground truth**. Broadcast and
YouTube footage has no height labels and almost never comes with an overhead
view, so we film our own. Public footage is still useful later as an unlabelled
check of false-alarm rates (see the end of this file).

## Camera rig (three phones are enough)

| View | Placement | Settings | Purpose |
|---|---|---|---|
| **Overhead** | Fixed above the table centre, ~2.5 m above the cloth, looking straight down. A light-fixture or ceiling mount, or a boom stand. | 1080p or 4K, 60 fps, locked exposure/focus | The view under test |
| **Broadcast** | Behind the foot rail, ~1.5 m high, ~1.7 m back, framing the whole table | 1080p, 60 fps, locked | The view under test |
| **Height (ground truth)** | At cloth level, perpendicular to the shot line, 1–1.5 m from the rail. Tape a printed mm scale to the rail facing it. | 240 fps slow motion | Measures ball height directly |

- **Sync:** one hand clap or phone-flash burst in view of all cameras at the
  start of every take.
- **Calibration:** per session, one still of the empty table from each
  camera (needed by the detector, and used to calibrate the camera from the
  table corners), plus a few seconds of a ball resting at 5 known spots.
- **Lighting:** do not change the lamps during a session. Photograph the lamp
  layout and measure lamp height above the cloth; the shadow cue depends on it.

## Shot list (mirrors the synthetic scenario clips)

Aim for at least 10 takes of each; vary position on the table, especially
under the overhead camera vs near the corners (overhead parallax is zero at
the centre).

1. `stop_shot`: level stun shot into an object ball (negative)
2. `jump_over_blocker`: blocker 15–40 cm from the cue ball (positive)
3. `jump_clips_blocker`: deliberately too-low jump that hits the blocker's top (positive, object-ball hop)
4. `near_miss`: cue ball passes within a few mm of a ball (hard negative)
5. `break`: full rack (mixed; small hops)
6. `long_jump_off_axis`: long jump near a side rail (positive)
7. `elevated_draw` / `elevated_follow`: cue raised 10–20° (borderline hops)
8. Level shots at all speeds, including very hard ones (negatives; tests that speed is not mistaken for height)

## Labelling

From the 240 fps side view, mark per take:
- launch frame, apex frame, apex height (mm, from the rail scale), each
  landing frame, and the final landing frame;
- the frames where any *object* ball leaves the cloth.

Convert to the 60 fps overhead/broadcast timelines with the sync event. Store
one `labels.json` per take next to the videos, with the same fields as the
synthetic `gt.json` where possible (`frame_times`, `height_max`, `airborne`
for the cue ball).

## Directory layout

```
data/real/<session>/<take>/
    overhead.mp4  broadcast.mp4  side240.mp4
    overhead_empty.jpg  broadcast_empty.jpg
    calib.json      # intrinsics + pose per camera, lamp positions
    labels.json
```

Large videos should not be committed to git: keep them in shared storage and
commit only `labels.json` and `calib.json`.

## Public footage (unlabelled)

Once network access to video sites is enabled for this environment, gather
clips that show jump shots, trick shots or breaks from a fixed overhead or
behind-the-rail camera. Use them only to measure false alarms per minute on
normal play and for qualitative figures, and check each source's licence
before reusing frames in the paper.
