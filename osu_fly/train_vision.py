from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .neural import Array, CursorBrain, Decoder
from .train import metrics


def synthetic_targets(seed: int, seconds: float) -> Array:
    rng = np.random.default_rng(seed)
    targets = np.zeros((int(seconds / 0.02), 3))
    position = np.array([0.5, 0.5])
    start = 0
    while start < len(targets):
        duration = int(rng.integers(45, 90))
        end = min(start + duration, len(targets))
        destination = rng.uniform(0.06, 0.94, 2)
        move = int(rng.integers(6, 18))
        progress = np.clip(np.arange(end - start) / move, 0, 1)[:, None]
        progress = progress * progress * (3 - 2 * progress)
        targets[start:end, :2] = position + (destination - position) * progress
        hit = start + move + int(rng.integers(12, 22))
        hold = int(rng.choice([6, 6, 6, 15, 25]))
        targets[hit : min(hit + hold, end), 2] = 1
        start, position = end, destination
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit on synthetic motor signals, without beatmaps."
    )
    parser.add_argument("--output", type=Path, default=Path("osu_fly/output/vision"))
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--seeds", type=int, nargs="+", default=[70, 71, 72, 73])
    parser.add_argument("--ridge", type=float, default=100)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    features, targets = [], []
    for seed in args.seeds:
        stimulus = synthetic_targets(seed + 1000, args.seconds)
        desired = np.vstack([np.repeat(stimulus[:1], 3, axis=0), stimulus[:-3]])
        features.append(CursorBrain(seed=seed).collect(stimulus))
        targets.append(desired)
    decoder = Decoder.fit(np.vstack(features), np.vstack(targets), ridge=args.ridge)
    decoder.save(args.output / "decoder.npz")
    validation = synthetic_targets(2000, 60)
    desired = np.vstack([np.repeat(validation[:1], 3, axis=0), validation[:-3]])
    brain = CursorBrain(seed=100)
    prediction = decoder.predict(brain.collect(validation))
    disconnected = decoder.predict(
        CursorBrain(seed=100, disconnected=True).collect(validation)
    )
    report = {
        "training": "Synthetic random positions and button pulses; no map files or recordings",
        "training_seeds": args.seeds,
        "validation_seed": 100,
        "desired_delay_ms": 60,
        "validation": metrics(prediction, desired),
        "disconnected": metrics(disconnected, desired),
        "limits": "Motor channel validation only; vision and game scores require real testing",
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        args.output / "validation.npz", target=desired, output=prediction
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
