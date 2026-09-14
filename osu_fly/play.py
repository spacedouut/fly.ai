from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from Xlib import X, XK, display
from Xlib.ext import xtest

from .audio import synchronize
from .beatmap import Beatmap, FloatArray
from .neural import CursorBrain, Decoder


class Desktop:
    def __init__(self, rect: list[float] | None):
        self.connection = display.Display()
        self.root = self.connection.screen().root
        width = self.connection.screen().width_in_pixels
        height = self.connection.screen().height_in_pixels
        scale = min(width / 640, height / 480)
        self.rect = rect or [
            (width - 512 * scale) / 2,
            (height - 384 * scale) / 2,
            512 * scale,
            384 * scale,
        ]
        self.key = self.connection.keysym_to_keycode(XK.string_to_keysym("z"))
        self.escape = self.connection.keysym_to_keycode(XK.string_to_keysym("Escape"))
        self.start_key = self.connection.keysym_to_keycode(XK.string_to_keysym("F8"))
        self.down = False

    def key_down(self, key: int) -> bool:
        state = self.connection.query_keymap()
        return bool(state[key // 8] & (1 << (key % 8)))

    def focused_on_game(self) -> bool:
        focus = self.connection.get_input_focus().focus
        if isinstance(focus, int):
            return False
        for _ in range(5):
            name = focus.get_wm_name() or ""
            if "osu!" in name.lower():
                return True
            parent = focus.query_tree().parent
            if parent.id == focus.id or parent.id == self.root.id:
                break
            focus = parent
        return False

    def emit(self, output: FloatArray, threshold: float) -> None:
        left, top, width, height = self.rect
        xtest.fake_input(
            self.connection,
            X.MotionNotify,
            x=int(left + np.clip(output[0], 0, 1) * width),
            y=int(top + np.clip(output[1], 0, 1) * height),
        )
        pressed = bool(output[2] > threshold)
        if pressed != self.down:
            xtest.fake_input(
                self.connection, X.KeyPress if pressed else X.KeyRelease, self.key
            )
            self.down = pressed
        self.connection.flush()

    def release(self) -> None:
        if self.down:
            xtest.fake_input(self.connection, X.KeyRelease, self.key)
        self.down = False
        self.connection.sync()

    def close(self) -> None:
        self.release()
        self.connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Drive local osu!lazer through XTest.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--replay", type=Path, help="Previously decoded neural output NPZ"
    )
    mode.add_argument(
        "--live", type=Path, help="Decoder NPZ; simulate the brain during play"
    )
    parser.add_argument("--map", type=Path, help="Required with --live")
    parser.add_argument(
        "--audio", type=Path, help="Auto-sync using song MP3 and PulseAudio"
    )
    parser.add_argument(
        "--rect", nargs=4, type=float, metavar=("X", "Y", "WIDTH", "HEIGHT")
    )
    parser.add_argument(
        "--offset-ms",
        type=float,
        default=0,
        help="Positive advances output relative to detected song time",
    )
    parser.add_argument("--lead-ms", type=float, default=60)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=65)
    parser.add_argument(
        "--start-delay",
        type=float,
        default=0,
        help="Without --audio, song starts this many seconds after F8",
    )
    parser.add_argument("--log", type=Path, default=Path("osu_fly/output/play.json"))
    args = parser.parse_args()
    desktop = Desktop(args.rect)
    rows = []
    brain = None
    decoder = None
    stimulus = None
    outputs = None
    cursor = 0
    try:
        if args.live:
            if not args.map:
                parser.error("--live requires --map")
            beatmap = Beatmap.load(args.map)
            times: FloatArray = np.arange(
                -2, beatmap.objects[-1].end + 1, 0.020, dtype=np.float64
            )
            stimulus = beatmap.targets(times + args.lead_ms / 1000)
            decoder = Decoder.load(args.live)
            brain = CursorBrain(seed=args.seed)
            while times[cursor] < 0:
                brain.step(stimulus[cursor])
                cursor += 1
            print("Brain warm; ready for playback.", flush=True)
        else:
            with np.load(args.replay, allow_pickle=False) as data:
                times, outputs = data["times"], data["output"]
            if outputs.shape != (len(times), 3) or not np.isfinite(outputs).all():
                raise ValueError(
                    "Replay must contain finite times and three outputs per frame."
                )
        print(
            f"Playfield rectangle: {desktop.rect}. Escape or focus loss stops input.",
            flush=True,
        )
        if args.audio:
            epoch = synchronize(args.audio)
        else:
            print(
                "Press F8 at song time zero (or --start-delay before it).", flush=True
            )
            while not desktop.key_down(desktop.start_key):
                if desktop.key_down(desktop.escape):
                    return
                time.sleep(0.01)
            epoch = time.monotonic() + args.start_delay
        print("Playing", flush=True)
        if not desktop.focused_on_game():
            raise RuntimeError(
                "Focus osu! before starting; refusing input into another window."
            )
        output = np.array([0.5, 0.5, 0.0])
        while True:
            now = time.monotonic() - epoch + args.offset_ms / 1000
            if now > times[-1]:
                break
            if desktop.key_down(desktop.escape) or not desktop.focused_on_game():
                print("Stopped: Escape or focus changed.", flush=True)
                break
            if brain is not None and decoder is not None and stimulus is not None:
                while cursor < len(times) and times[cursor] <= now:
                    output = decoder.predict(brain.step(stimulus[cursor]))
                    cursor += 1
            elif outputs is not None:
                output = np.array(
                    [np.interp(now, times, outputs[:, i]) for i in range(3)]
                )
            desktop.emit(output, args.threshold)
            rows.append([now, *output.tolist()])
            time.sleep(0.002 if brain is not None else 0.008)
        print(f"Stopped at song time {now:.3f}s; {len(rows)} outputs sent.", flush=True)
    finally:
        desktop.close()
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(
            json.dumps(
                {
                    "mode": "live" if args.live else "neural_replay",
                    "seed": args.seed,
                    "offset_ms": args.offset_ms,
                    "rect": desktop.rect,
                    "frames": rows,
                }
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
