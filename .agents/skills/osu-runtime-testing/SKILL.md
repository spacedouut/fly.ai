---
name: osu-runtime-testing
description: Install official osu!lazer on Linux/X11 and measure the fly.ai replay-assisted neural cursor in real signed-out tutorial attempts, with audio/video evidence and input-release checks.
---

# Scope and interpretation

Run from the fly.ai checkout. Read `osu_fly/README.md` first. This experiment uses beatmap-supplied teacher signals through a frozen connectome and decoder; it is not screenshot perception. Distinguish `--live` simulation from `--replay` precomputed neural output. Never substitute ideal teacher coordinates, game autoplay, or assist mods for score evidence. Tutorial HP0 means reaching results alone proves little: collect grade, accuracy, score, max combo, misses, slider ticks/ends and spinner spins.

## Devin Secrets Needed

None for official public downloads and signed-out local play. Do not sign in or submit scores online.

## Linux desktop setup

Verified on Ubuntu22.04/X11, software-rendered llvmpipe, eight CPUs:

1. Verify parent directories and existing tools before installation. `ffmpeg`, `wmctrl`, and the repository `.venv` may already exist. Install missing `pulseaudio pulseaudio-utils libasound2 libgl1` with the available package manager; do not assume a particular `libicu` version exists.
2. Download `https://github.com/ppy/osu/releases/latest/download/osu.AppImage`, chmod executable, then execute `osu.AppImage --appimage-extract` in its intended install directory if FUSE is unavailable. Launch `squashfs-root/AppRun`. Record the actual client version because latest changes; this session used2026.804.2.
3. Start `pulseaudio --start --exit-idle-time=-1` before launching lazer. `pactl info`, `pactl list short sink-inputs`, and `pactl list short sources` should show the game stream and a monitor. A headless desktop commonly uses `auto_null`/`auto_null.monitor`; this is sufficient for correlation and audio recording.
4. Download `https://assets.ppy.sh/client-resources/bundled/1011011.osz`. Extract the reference with `unzip -p tutorial.osz audio.mp3 > audio.mp3`. Import a **copy** of the OSZ by passing its path to AppRun: lazer may delete the imported source. Keep a separate source OSZ for the Python controller. Tutorial archive SHA256 observed: `5ad9dda93befc2d8033c9e1eee3951b26cd48ce7720ef90e09926ede8a9e19df`.
5. Use `xrandr` to record native desktop dimensions. Set fullscreen; maximize before recording using `wmctrl -r osu! -b add,maximized_vert,maximized_horz`.
6. Remain Guest. The first-run wizard may be closed with Escape.

## Input and UI configuration

- Ctrl+O opens settings; type `mouse`. Turn **High precision mouse OFF**, sensitivity1.0, and **Disable mouse buttons during gameplay ON**. These permit absolute XTest movement and keyboard Z hits.
- If the cursor does not move under absolute automation, close lazer cleanly (Alt+F4, Enter if confirmation appears). It writes `~/.local/share/osu/input.json`. While stopped, set the MouseHandler's `UseRelativeMode` to `false`, restart, and verify visually. Do not change prototype code to work around client input configuration.
- Config files are under `~/.local/share/osu/`: `input.json`, `framework.ini`, `game.ini`; client diagnostics are under `logs/`.
- GUI mouse actions may need move→200ms wait→150–200ms held click. Instant press/release may be missed. The main menu's expanded buttons time out after about6seconds: click the logo, Play, then Solo promptly in one sequence.
- Select `nekodex - new beginnings [tutorial]` by pishifat, beatmap2116202. UI should report20circles,12sliders,2spinners,130BPM,HP0,AR2,OD0.
- F1 opens mods. Capture no selections, disabled Deselect All,1.00× multiplier and130BPM. No Autoplay, Relax, Autopilot, NoFail or speed mods.
- Song-select preview must be paused before audio synchronization: top music icon→pause→Escape to close player. Repeat after every return to song selection. Enter launches selected tutorial; do not skip the intro.
- At native1600×1200 the unmodified controller default rect is `[160,120,1280,960]`. Only use `--rect` for documented native-pixel calibration; do not confuse scaled screenshot coordinates with native pixels.

## Real attempts

From repository root, use separate stdout and controller JSON logs for every attempt:

```sh
NUMBA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 .venv/bin/python -u -m osu_fly.play \
  --live osu_fly/output/decoder.npz --map /absolute/path/tutorial.osz \
  --seed 64 --audio /absolute/path/audio.mp3 \
  --log osu_fly/output/live64.json > /tmp/live64.log 2>&1
```

Wait for `Brain warm` and audio listening, then press Enter in the focused tutorial selection. Do not move the mouse or manually hit during play. Capture first sync positions/confidences; menu noise or preview must not advance the clock. Current revised synchronizer searches first15seconds and requires two half-second windows at confidence>=0.6 with consistent epochs. In real attempts with storyboard voiceover it locked around3–4seconds; do not expect the first possible window necessarily to qualify.

Repeat with seed65 for new stochastic activity on the **same fitted map**, not unseen-map evaluation. If live scheduling is questionable, explicitly label the authorized neural replay comparison:

```sh
OPENBLAS_NUM_THREADS=2 .venv/bin/python -u -m osu_fly.play \
  --replay osu_fly/output/validation.npz --seed 65 \
  --audio /absolute/path/audio.mp3 \
  --log osu_fly/output/replay65.json > /tmp/replay65.log 2>&1
```

`training.npz` corresponds to fitted-seed neural output; `validation.npz` is new-noise output. F8 without `--audio`, `--offset-ms`, and `--start-delay` are supported diagnostic alternatives; never disguise a manual-sync or replay attempt as live audio-synchronized inference.

At results, capture the entire fullscreen scoreboard. Compare controller emission intervals from `frames[:,0]`, separating initial catch-up from active play. Intervals are loop timestamps, not precise XTest emission/hit latency. Poor score can reflect both readout sensitivity and scheduling; replay comparison reduces but does not eliminate confounds.

## Evidence with audio

Use annotated screen recordings for readable test milestones. If built-in capture lacks audio, additionally record monitor audio plus X11, adjusting size/display/sink to actual values:

```sh
ffmpeg -y -f x11grab -video_size 1600x1200 -framerate 30 -i :0.0 \
  -f pulse -i auto_null.monitor -c:v libx264 -preset ultrafast \
  -crf 23 -threads 2 -c:a aac /absolute/path/attempt-with-audio.mp4
```

Stop ffmpeg with SIGINT so MP4 finalizes. Verify it has both video/audio streams via ffprobe. Recording adds CPU load; report this when interpreting live scheduling. Keep raw stdout/controller JSON and screenshots, not only edited video.

## Held-input release

Check Escape and focus loss separately while Z is genuinely held. An external Xlib observer can wait for200ms continuous hold, press Escape or focus Chrome with `wmctrl`, then measure release and cursor stability. This instrumentation must not generate gameplay coordinates or Z.

To isolate safety without waiting for a full map, a clearly labelled **non-scoring** neural replay may seek to a known held-slider interval via `--offset-ms` and F8 while the real game is running. Locate a sustained output score>0.5 in the actual replay rather than hardcoding times across datasets. Capture that Z was held before the action, is released afterward, controller prints `Stopped: Escape or focus changed`, process exits, and pointer stops. Do not claim this covers abort responsiveness during a long live catch-up loop or while waiting for audio.
