from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.signal import fftconvolve

RATE = 8000


def locate(
    reference: NDArray[np.float64], sample: NDArray[np.float64]
) -> tuple[float, float]:
    sample = sample - sample.mean()
    correlation = fftconvolve(reference, sample[::-1], mode="valid")
    sums = np.r_[0, np.cumsum(reference)]
    squares = np.r_[0, np.cumsum(reference**2)]
    length = len(sample)
    energy = (
        squares[length:]
        - squares[:-length]
        - (sums[length:] - sums[:-length]) ** 2 / length
    )
    normalized = correlation / np.sqrt(np.maximum(energy * (sample @ sample), 1e-20))
    best = int(np.argmax(np.abs(normalized)))
    return best / RATE, float(abs(normalized[best]))


def synchronize(audio: Path, timeout: float = 90) -> float:
    decoded = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(audio),
            "-ac",
            "1",
            "-ar",
            str(RATE),
            "-f",
            "s16le",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    reference = np.frombuffer(decoded.stdout, dtype="<i2").astype(float) / 32768
    sink = subprocess.check_output(["pactl", "get-default-sink"], text=True).strip()
    process = subprocess.Popen(
        [
            "parec",
            "--raw",
            "--format=s16le",
            f"--rate={RATE}",
            "--channels=1",
            "--latency-msec=10",
            f"--device={sink}.monitor",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    print("Waiting for song audio on the desktop monitor…", flush=True)
    deadline = time.monotonic() + timeout
    try:
        if process.stdout is None:
            raise RuntimeError("Could not open audio capture.")
        while time.monotonic() < deadline:
            raw = process.stdout.read(RATE * 2 * 2)
            end = time.monotonic() - 0.010
            if len(raw) != RATE * 2 * 2:
                raise RuntimeError("Audio capture ended before synchronization.")
            sample = np.frombuffer(raw, dtype="<i2").astype(float) / 32768
            position, confidence = locate(reference, sample)
            print(
                f"Audio match: t={position:.3f}s, confidence={confidence:.3f}",
                flush=True,
            )
            if confidence > 0.12:
                return end - 2 - position
        raise TimeoutError(
            "No matching song audio. Check volume and the PulseAudio monitor."
        )
    finally:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    args = parser.parse_args()
    start = synchronize(args.audio)
    print(f"Song is at {time.monotonic() - start:.3f}s", flush=True)


if __name__ == "__main__":
    main()
