from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .beatmap import Beatmap, FloatArray
from .neural import CursorBrain, Decoder


def metrics(prediction: FloatArray, target: FloatArray) -> dict[str, float]:
    error = np.linalg.norm((prediction[:, :2] - target[:, :2]) * [512, 384], axis=1)
    active = target[:, 2] > 0.5
    pressed = prediction[:, 2] > 0.5
    return {
        "cursor_mean_error_osu_px": float(error.mean()),
        "cursor_active_mean_error_osu_px": float(error[active].mean()),
        "cursor_active_p95_error_osu_px": float(np.quantile(error[active], 0.95)),
        "button_precision": float(np.sum(pressed & active) / max(1, pressed.sum())),
        "button_recall": float(np.sum(pressed & active) / max(1, active.sum())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a replay-assisted neural cursor decoder."
    )
    parser.add_argument("map", type=Path)
    parser.add_argument("--output", type=Path, default=Path("osu_fly/output"))
    parser.add_argument("--seed", type=int, default=64)
    parser.add_argument("--ridge", type=float, default=10)
    parser.add_argument("--lead-ms", type=float, default=60)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    beatmap = Beatmap.load(args.map)
    times: FloatArray = np.arange(
        -2, beatmap.objects[-1].end + 1, 0.020, dtype=np.float64
    )
    targets = beatmap.targets(times)
    stimulus = beatmap.targets(times + args.lead_ms / 1000)
    brain = CursorBrain(seed=args.seed)
    print(
        f"{brain.brain.n:,} neurons; {len(brain.brain.weights):,} connections",
        flush=True,
    )
    start = time.perf_counter()
    train_features = brain.collect(stimulus)
    simulation_seconds = time.perf_counter() - start
    print("Fitting decoder", flush=True)
    decoder = Decoder.fit(train_features, targets, ridge=args.ridge)
    decoder.save(args.output / "decoder.npz")
    training = decoder.predict(train_features)
    np.savez_compressed(args.output / "training.npz", times=times, output=training)
    print("Held-out noise seed (same map)", flush=True)
    validation_features = CursorBrain(seed=args.seed + 1).collect(stimulus)
    validation = decoder.predict(validation_features)
    np.savez_compressed(args.output / "validation.npz", times=times, output=validation)
    print("Disconnected synapses control", flush=True)
    disconnected = decoder.predict(
        CursorBrain(seed=args.seed + 1, disconnected=True).collect(stimulus)
    )
    np.savez_compressed(
        args.output / "disconnected.npz", times=times, output=disconnected
    )
    zero = decoder.predict(np.zeros_like(validation_features))
    report = {
        "experiment": "Replay-assisted teacher input; frozen connectome; trained linear decoder",
        "map": beatmap.metadata,
        "map_sha256": hashlib.sha256(args.map.read_bytes()).hexdigest(),
        "neurons": brain.brain.n,
        "connections": len(brain.brain.weights),
        "directly_stimulated_neurons": len(brain.inputs),
        "readout_neurons": len(brain.read_indices),
        "readout_overlap_with_inputs": len(
            np.intersect1d(brain.inputs, brain.read_indices)
        ),
        "dt": 0.020,
        "lead_ms": args.lead_ms,
        "seed": args.seed,
        "ridge": args.ridge,
        "simulation_seconds": simulation_seconds,
        "simulated_seconds": len(times) * 0.020,
        "training": metrics(training, targets),
        "held_out_noise_seed_same_map": metrics(validation, targets),
        "disconnected_synapses": metrics(disconnected, targets),
        "zero_neural_features": metrics(zero, targets),
        "ideal_teacher": metrics(targets, targets),
        "limits": [
            "Map coordinates, slider paths, spinner motion and button timing are teacher inputs.",
            "No screenshot perception, reinforcement learning or plasticity.",
            "Validation is a new noise seed on the training map, not an unseen map.",
            "Controls test dependence, not superiority to a random network.",
            "Offline error metrics are not an osu! score; use a real client attempt.",
        ],
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
