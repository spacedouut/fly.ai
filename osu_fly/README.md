# A fly-connectome cursor for osu!lazer

Two motor experiments using the full MaleCNS network in `flybrain`: the original
**replay-assisted** controller below, and a **screenshot-driven circle prototype**
described in [Vision](#vision). In replay-assisted mode, the teacher supplies cursor
coordinates (including slider paths and spinner motion) and button timing from an
osu! beatmap. A frozen spiking network processes those signals, and a trained
linear decoder produces cursor coordinates and a button score.

```
beatmap → teacher trajectory → 9,201 visual projection neurons
        → 166,700-neuron / 25,582,938-connection frozen network
        → undriven postsynaptic + descending neurons → fitted decoder → XTest
```

The 48 input populations are assigned randomly, with a fixed seed; this is an
engineering interface, not a model of the fly's eyes. The decoder observes clipped
membrane potentials and filtered spikes. No target coordinates or clock enter the
decoder directly. The original network's transmitter signs, normalization and
point-neuron approximations still apply.

## Install

From this repository root, Python 3.12 on Linux/X11:

```sh
python3 -m venv .venv
.venv/bin/pip install -r osu_fly/requirements.txt
.venv/bin/python -m flybrain download
```

The downloaded network files are checksum-verified by `flybrain`. They occupy about
260 MB in `~/fly-data`. The simulator source used for this experiment was inspected
at upstream commit `1fd161aaa4a525c61e63647f6db892eb3535aa89`.

For playback, install osu!lazer's official Linux AppImage, `ffmpeg`, and PulseAudio
utilities (`pactl`, `parec`). The desktop must use X11; Wayland does not expose this
XTest interface to native clients.

## Train and evaluate

The exact bundled tutorial is **nekodex – new beginnings [tutorial]**, mapped by
pishifat, beatmap 2116202, set 1011011. Download it from osu!'s official asset host:

```sh
curl -fL https://assets.ppy.sh/client-resources/bundled/1011011.osz -o tutorial.osz
NUMBA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m osu_fly.train tutorial.osz
```

This trains on the complete tutorial and writes an ignored `osu_fly/output/`
directory containing the decoder, neural replay outputs, and `metrics.json`.
An archive must contain one difficulty; otherwise extract and pass an `.osu` file.
Implemented path types: linear, Bezier (including repeated control points), and
perfect circular arcs. Catmull curves are rejected. Stacking is not implemented.

Metrics distinguish the training replay from a **new noise seed on the same map**.
The latter is not an unseen-map test. The same fitted decoder is also evaluated
with all synapses disconnected and with zero neural features. These controls can
show whether this controller depends on neural activity; they cannot establish
that biological wiring is better than a random network.

## Play locally

Import the OSZ in lazer, stay signed out, select the tutorial, use normal speed,
and disable mouse-button input so cursor movement cannot accidentally click.
Turn off high-precision mouse input so absolute XTest coordinates are respected.
Keep a separate copy of the OSZ: importing it can consume the source archive.
Do not enable Autoplay, Relax, Autopilot, or other assist mods.

Extract `audio.mp3` from the OSZ, then start the controller **before** starting the
map. Pause the song-selection preview and leave the game focused. It listens to
the desktop audio monitor and requires two confident, time-consistent matches
within the first 15 seconds of the original track. Start it before playback;
late starts time out. Escape or switching to another app stops gameplay input and
releases the held key.

```sh
NUMBA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m osu_fly.play \
  --live osu_fly/output/decoder.npz --map tutorial.osz --audio audio.mp3
```

For playback of previously computed neural outputs (no simulation during play):

```sh
.venv/bin/python -m osu_fly.play \
  --replay osu_fly/output/validation.npz --audio audio.mp3
```

Use `--replay osu_fly/output/training.npz` for the fitted noise-seed replay.
These modes play the decoded network output, not the ideal teacher coordinates.
Without `--audio`, F8 marks song time zero; `--start-delay` can compensate for
starting the client later. `--offset-ms` advances output relative to the detected
song time. `--rect X Y WIDTH HEIGHT` sets the playfield in native desktop pixels;
the default assumes fullscreen with the standard centered playfield.

`--seed` and `--lead-ms` must match the intended experiment (defaults 65 and 60 ms;
training uses seed 64). Live simulation runs at a 20 ms model timestep, with a
60 ms teacher lead to compensate for propagation. Its timing and the client's
score must be checked in a real attempt; offline cursor errors are not scores.

## Measured gameplay

Tested on osu!lazer **2026.804.2**, Linux/X11, fullscreen 1600×1200, as Guest,
with **no mods**, normal speed, zero timing offset, and automatic audio alignment.
The trained decoder and teacher lead were unchanged across these attempts.

| Mode | Grade | Accuracy | Score | Max combo | Misses |
|---|---|---:|---:|---:|---:|
| Live network, fitted noise seed 64 | A | 93.52% | 698,780 | 47/73 | 2 |
| Live network, new noise seed 65 | D | 31.14% | 21,138 | 13/73 | 23 |
| Precomputed neural output, seed 65 | C | 70.72% | 217,588 | 26/73 | 12 |

The fitted-seed live attempt hit 26/27 slider ticks, all 12 slider ends, and all
13 spinner spins. It demonstrates actual cursor and button control through the
network, but the weaker new-seed results show that this is not a robust controller.

Live seed 65 also showed controller-loop stalls: after song time 15 seconds,
11 emission intervals exceeded 100 ms, with a maximum of 689 ms. The precomputed
replay's maximum interval was 36.9 ms. These are loop timestamps, not measured
end-to-end input latency; both decoder sensitivity and scheduling need attention.
Visual perception and unseen-map generalization were not tested.

Escape and focus loss released a held Z and stopped the cursor in neural-replay
checks. Abort responsiveness during live catch-up or audio synchronization remains
unverified.

## Vision

```text
MSS screen capture → OpenCV concentric-circle detector → shrinking-ring tracker
  → estimated X/Y and tap signal → same frozen MaleCNS → synthetic-trained decoder
  → XTest cursor and Z
```

This is engineered visual feature extraction, not a biological retina or learned
image understanding inside the connectome. OpenCV supplies object geometry; a
linear fit to recent ring radii estimates contact time. Only neural features
enter the motor decoder. The live vision command reads no beatmap, audio, or
replay, and has no song clock. Captured pixels are cropped to the playfield and
resized to 512×384.

Train on randomly generated movements and button pulses, using four noise seeds.
The held-out validation uses new positions, a new neural seed, and a disconnected
network control. A 60 ms target delay accounts for neural response time.

```sh
NUMBA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m osu_fly.train_vision
NUMBA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m osu_fly.vision_play \
  --decoder osu_fly/output/vision/decoder.npz --seed 100
```

Wait for “Ready”, focus the game, start the map, and press F8 after loading. F8
only arms screen capture; its timing relative to the song is unimportant. Use the same fullscreen
and mouse settings as above. Escape or focus loss stops input and releases Z.
The default runtime is 160 seconds; `--seconds` changes it. `--rect` overrides the
playfield crop, and `--lead-ms` adjusts visual anticipation for neural latency.
Alternatively, `--arm-after 10` arms automatically ten seconds after “Ready”;
this avoids opening lazer chat by pressing F8 during loading. The game must have
focus when the countdown expires. `--frames-dir` saves sampled perception images
for debugging (cropped images, not a fullscreen gameplay recording).

On our CPU-only llvmpipe desktop, restarting lazer at 1024×768 fullscreen with
VSync substantially reduced frozen perception frames and audio-clock warnings.
Recording at 15 fps with two encoder threads reduced recording load. These
settings are a tested workaround, not a guarantee of smooth timing; the first
few seconds can still stall. Inspect captured frames as well as loop timings.

For a separately labelled **non-neural diagnostic baseline**:

```sh
.venv/bin/python -m osu_fly.vision_play --detector-only
```

This mode sends the visual controller's output directly and must not be presented
as fly gameplay. Comparing it with neural mode can separate perception failures
from decoder failures.

Each frame runs one simulation step, with no accumulated catch-up loop. The
target period is 20 ms; slow frames stretch simulated time rather than trigger
bursts of stale input. Logs include per-frame capture/detection/neural timings,
detected circles, visual signals and neural outputs. These timings do not measure
physical display-to-input latency.
Countdown fits expire after 150 ms without a valid shrinking-ring estimate.
Tracking preserves the initial hit-circle radius as the approach ring merges
with it, rather than changing the predicted contact boundary.

**Current scope:** circles and slider heads, with a 120 ms tap hold. No slider-path
following, spinner recognition, or stacked-circle disambiguation. Skin effects,
tutorial storyboard demonstrations, HUD circles, occlusion and low capture rates
can confuse the detector. It is not yet a general osu! player. Offline motor
validation and synthetic circle tests do not establish real gameplay performance.

### Recorded vision results

Revision `1566899`, osu!lazer 2026.804.2, Guest, no mods, normal speed,
1024×768 fullscreen/VSync. Both neural attempts used the same synthetic decoder
and seed 100, with no map-derived controller input.

| Mode and map | Outcome | Accuracy | Score | Max combo | Misses |
|---|---|---:|---:|---:|---:|
| Detector-only tutorial | D, completed | 29.27% | 28,417 | 8/73 | 6 |
| Live neural vision tutorial | D, completed | 19.90% | 18,906 | 8/73 | 7 |
| Live neural vision, Faded Winter [Beginner] | Failed at 0:35 (71%) | 14.89% partial | 6,856 partial | 4 | Unavailable |

The tutorial has zero HP drain, so completion alone is weak evidence. Both modes
scored zero slider ticks, slider ends, and spinner spins. Real visual input did
produce neural cursor movement and registered hits, but performance remains poor.
The detector baseline also struggled, so neural decoding is not the only problem.

The held-out attempt included about 12 seconds of song-select input before
gameplay; its startup was not cleanly aligned to the intended countdown. The
failure overlay did not provide a miss count. It was not retrained or retried.
No renewed audio warning appeared in these runs. Changing cursor pixels mean
the absence of identical neural screenshots cannot establish smooth rendering.

Separate live checks confirmed Z was held before Escape and focus loss, then
released within 2.9 ms and 19.7 ms respectively, with controller exit and no further
cursor movement. These observations do not guarantee a latency bound.

## Checks

```sh
.venv/bin/ruff check osu_fly
.venv/bin/mypy --follow-imports=silent --ignore-missing-imports osu_fly
.venv/bin/python -m pytest osu_fly/tests -q
```

## Sources

- [MaleCNS data and attribution (CC BY 4.0)](https://male-cns.janelia.org/download/)
- [fly.ai simulator (MIT)](https://github.com/alextitonis/fly.ai)
- [Beat Saber creator's clarification](https://www.unrollnow.com/status/2097527368919470162):
  describes replay distillation and teacher input; this implementation is separate.
- [Official lazer releases](https://github.com/ppy/osu/releases)
