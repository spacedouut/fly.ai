from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import mss
import numpy as np

from .neural import CursorBrain, Decoder
from .play import Desktop
from .vision import CircleDetector, VisionController


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use screen pixels for targets and approach timing."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--decoder", type=Path, help="Run live MaleCNS with this decoder")
    mode.add_argument(
        "--detector-only",
        action="store_true",
        help="Explicit non-neural diagnostic baseline",
    )
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--rect", nargs=4, type=float)
    parser.add_argument("--seconds", type=float, default=160)
    parser.add_argument("--lead-ms", type=float, default=80)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--arm-after", type=float, help="Arm after N seconds instead of pressing F8"
    )
    parser.add_argument(
        "--frames-dir", type=Path, help="Save sampled perception frames for debugging"
    )
    parser.add_argument(
        "--log", type=Path, default=Path("osu_fly/output/vision-play.json")
    )
    args = parser.parse_args()
    desktop = Desktop(args.rect)
    detector = CircleDetector()
    controller = VisionController(0 if args.detector_only else args.lead_ms / 1000)
    brain = CursorBrain(seed=args.seed) if args.decoder else None
    decoder = Decoder.load(args.decoder) if args.decoder else None
    rows = []
    saved_at = 0.0
    if args.frames_dir:
        args.frames_dir.mkdir(parents=True, exist_ok=True)
    try:
        if brain is not None:
            for _ in range(100):
                brain.step(np.array([0.5, 0.5, 0.0]))
        print(
            f"Ready: {'DETECTOR ONLY' if args.detector_only else 'LIVE NEURAL VISION'}. "
            f"Rect {desktop.rect}. "
            + (
                f"Arming in {args.arm_after} seconds; "
                if args.arm_after is not None
                else "Press F8 in osu! to start; "
            )
            + "Escape/focus loss stops.",
            flush=True,
        )
        arm_at = (
            time.monotonic() + args.arm_after if args.arm_after is not None else None
        )
        while (
            time.monotonic() < arm_at
            if arm_at is not None
            else not desktop.key_down(desktop.start_key)
        ):
            if desktop.key_down(desktop.escape):
                return
            time.sleep(0.01)
        if not desktop.focused_on_game():
            raise RuntimeError("Focus osu! before starting.")
        start = time.monotonic()
        left, top, width, height = map(int, desktop.rect)
        monitor = {"left": left, "top": top, "width": width, "height": height}
        with mss.mss() as capture:
            while time.monotonic() - start < args.seconds:
                begin = time.monotonic()
                if desktop.key_down(desktop.escape) or not desktop.focused_on_game():
                    print("Stopped: Escape or focus changed.", flush=True)
                    break
                frame = np.asarray(capture.grab(monitor))
                captured = time.monotonic()
                image = cv2.resize(
                    frame[:, :, :3], (512, 384), interpolation=cv2.INTER_AREA
                ).astype(np.uint8)
                circles = detector.detect(image)
                signal = controller.update(circles, captured)
                detected = time.monotonic()
                output = (
                    decoder.predict(brain.step(signal))
                    if brain is not None and decoder is not None
                    else signal
                )
                if desktop.key_down(desktop.escape) or not desktop.focused_on_game():
                    break
                desktop.emit(output, args.threshold)
                emitted = time.monotonic()
                rows.append(
                    {
                        "time": captured - start,
                        "input": signal.tolist(),
                        "output": output.tolist(),
                        "circles": [asdict(circle) for circle in circles],
                        "taps": controller.taps,
                        "capture_ms": (captured - begin) * 1000,
                        "vision_ms": (detected - captured) * 1000,
                        "neural_emit_ms": (emitted - detected) * 1000,
                    }
                )
                if args.frames_dir and captured - saved_at >= 0.1:
                    cv2.imwrite(
                        str(args.frames_dir / f"{captured - start:.3f}.png"), image
                    )
                    saved_at = captured
                time.sleep(max(0, 0.020 - (time.monotonic() - begin)))
        print(
            f"Finished: {len(rows)} frames; {controller.taps} visual tap events.",
            flush=True,
        )
    finally:
        desktop.close()
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(
            json.dumps(
                {
                    "mode": "detector_only"
                    if args.detector_only
                    else "live_neural_vision",
                    "seed": args.seed,
                    "rect": desktop.rect,
                    "input_source": "X11 screenshot pixels only; no map, audio, or replay",
                    "frames": rows,
                }
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
