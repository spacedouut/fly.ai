from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from flybrain import FlyBrain

Array = NDArray[np.float64]


class CursorBrain:
    def __init__(self, seed: int = 64, disconnected: bool = False):
        self.brain = FlyBrain(seed=seed, device="cpu")
        brain = self.brain
        self.inputs = brain.cells(["visual_projection"])
        rng = np.random.default_rng(2026)
        self.groups = np.array_split(rng.permutation(self.inputs), 48)
        self.projection = rng.normal(size=(48, 3))
        self.bias = rng.uniform(-0.8, 0.8, 48)
        matrix = sparse.csc_matrix(
            (brain.weights, brain.indices, brain.indptr), shape=(brain.n, brain.n)
        )
        strengths = np.asarray(abs(matrix[:, self.inputs]).sum(axis=1)).ravel()
        strengths[self.inputs] = -1
        strongest = np.argsort(strengths)[-1024:]
        self.read_indices = np.unique(
            np.r_[strongest, brain.cells(["descending_neuron"])]
        )
        self.trace = np.zeros(brain.n, dtype=np.float32)
        if disconnected:
            brain.weights = np.zeros_like(brain.weights)
        self.steps = 0

    def step(self, target: Array, silent: bool = False) -> Array:
        signal = np.tanh(self.projection @ (target * 2 - 1) + self.bias)
        amounts = 0.45 * (signal + 1)
        fired = self.brain.step(
            inject=[] if silent else list(zip(self.groups, amounts, strict=True))
        )
        self.trace *= 0.6
        self.trace[fired] += 1
        self.steps += 1
        return np.r_[
            np.clip(self.brain.v[self.read_indices, 0], -3, 1),
            self.trace[self.read_indices],
        ].astype(float)

    def collect(self, targets: Array, silent: bool = False) -> Array:
        features = []
        begin = time.perf_counter()
        for i, target in enumerate(targets):
            features.append(self.step(target, silent=silent))
            if i % 1000 == 0:
                print(
                    f"  step {i}/{len(targets)}, {time.perf_counter() - begin:.1f}s",
                    flush=True,
                )
        return np.stack(features)


@dataclass
class Decoder:
    mean: Array
    scale: Array
    weights: Array
    bias: Array

    @classmethod
    def fit(cls, features: Array, targets: Array, ridge: float = 10.0) -> Decoder:
        mean, scale = features.mean(axis=0), features.std(axis=0) + 0.05
        normal = (features - mean) / scale
        bias = targets.mean(axis=0)
        weights = np.linalg.solve(
            normal.T @ normal + ridge * np.eye(normal.shape[1]),
            normal.T @ (targets - bias),
        )
        return cls(mean, scale, weights, bias)

    def predict(self, features: Array) -> Array:
        return np.clip(
            (features - self.mean) / self.scale @ self.weights + self.bias, 0, 1
        )

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path, mean=self.mean, scale=self.scale, weights=self.weights, bias=self.bias
        )

    @classmethod
    def load(cls, path: Path) -> Decoder:
        with np.load(path, allow_pickle=False) as data:
            return cls(data["mean"], data["scale"], data["weights"], data["bias"])
