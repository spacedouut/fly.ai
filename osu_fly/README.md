# A fly-connectome cursor for osu!lazer

A **replay-assisted motor experiment**, using the full MaleCNS network in `flybrain`.
This is not an autonomous fly watching the screen. The teacher supplies cursor
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
Do not enable Autoplay, Relax, Autopilot, or other assist mods.

Extract `audio.mp3` from the OSZ, then start the controller **before** starting the
map. Leave the game focused. It listens to the desktop audio monitor and aligns
it with the original track. Escape or switching to another app stops input and
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
