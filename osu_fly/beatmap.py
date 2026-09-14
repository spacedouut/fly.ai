from __future__ import annotations

import math
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def bezier(points: FloatArray) -> FloatArray:
    t = np.linspace(0, 1, max(100, len(points) * 80))[:, None]
    degree = len(points) - 1
    return sum(
        (
            math.comb(degree, i) * (1 - t) ** (degree - i) * t**i * p
            for i, p in enumerate(points)
        ),
        np.zeros((len(t), 2)),
    )


def curve(kind: str, points: FloatArray, length: float) -> FloatArray:
    if kind == "L":
        path = points
    elif kind == "P" and len(points) == 3:
        a, b, c = points
        matrix = 2 * np.array([b - a, c - a])
        if abs(np.linalg.det(matrix)) < 1e-8:
            path = points
        else:
            center = np.linalg.solve(matrix, np.array([b @ b - a @ a, c @ c - a @ a]))
            start, middle, end = np.arctan2(
                points[:, 1] - center[1], points[:, 0] - center[0]
            )
            sweep = (end - start) % (2 * np.pi)
            if (middle - start) % (2 * np.pi) > sweep:
                sweep -= 2 * np.pi
            angle = np.linspace(start, start + sweep, 600)
            path = center + np.linalg.norm(a - center) * np.column_stack(
                [np.cos(angle), np.sin(angle)]
            )
    elif kind in ("B", "P"):
        segments: list[FloatArray] = []
        first = 0
        for i in range(1, len(points)):
            if np.array_equal(points[i], points[i - 1]):
                segments.append(bezier(points[first:i]))
                first = i
        segments.append(bezier(points[first:]))
        path = np.concatenate(segments)
    else:
        raise ValueError(f"Unsupported slider curve: {kind}")
    distances = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    keep = np.r_[True, np.diff(distances) > 1e-8]
    path, distances = path[keep], distances[keep]
    if len(path) < 2:
        return np.repeat(path[:1], 2, axis=0)
    if length > distances[-1]:
        direction = (path[-1] - path[-2]) / (distances[-1] - distances[-2])
        path = np.vstack([path, path[-1] + direction * (length - distances[-1])])
        distances = np.r_[distances, length]
    samples = np.linspace(0, length, max(2, int(length * 2)))
    return np.column_stack(
        [np.interp(samples, distances, path[:, axis]) for axis in (0, 1)]
    )


@dataclass
class HitObject:
    start: float
    end: float
    position: FloatArray
    kind: str
    path: FloatArray
    repeats: int = 1

    def at(self, time: float) -> FloatArray:
        if self.kind == "spinner":
            angle = (time - self.start) * 2 * np.pi * 4
            return np.array([256 + 75 * np.cos(angle), 192 + 75 * np.sin(angle)])
        if self.kind != "slider":
            return self.position.copy()
        progress = float(np.clip((time - self.start) / (self.end - self.start), 0, 1))
        span = progress * self.repeats
        if progress == 1:
            fraction = float(self.repeats % 2)
        else:
            fraction = span % 1 if int(span) % 2 == 0 else 1 - span % 1
        index = fraction * (len(self.path) - 1)
        lo = min(int(index), len(self.path) - 2)
        return self.path[lo] * (1 - (index - lo)) + self.path[lo + 1] * (index - lo)


@dataclass
class Beatmap:
    metadata: dict[str, str]
    objects: list[HitObject]
    circle_size: float

    @classmethod
    def load(cls, source: Path) -> Beatmap:
        if source.suffix.lower() == ".osz":
            with zipfile.ZipFile(source) as archive:
                maps = [n for n in archive.namelist() if n.endswith(".osu")]
                if len(maps) != 1:
                    raise ValueError(
                        "Archive must contain one difficulty; pass an .osu file."
                    )
                text = archive.read(maps[0]).decode("utf-8-sig")
        else:
            text = source.read_text(encoding="utf-8-sig")
        return cls.parse(text)

    @classmethod
    def parse(cls, text: str) -> Beatmap:
        sections: dict[str, list[str]] = {}
        section = ""
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1]
                sections[section] = []
            elif section:
                sections[section].append(line)
        values = {}
        for name in ("General", "Metadata", "Difficulty"):
            values.update(dict(line.split(":", 1) for line in sections.get(name, [])))
        values = {k.strip(): v.strip() for k, v in values.items()}
        if int(values.get("Mode", "0")) != 0:
            raise ValueError("Only osu!standard maps are supported.")
        timing = [line.split(",") for line in sections.get("TimingPoints", [])]
        objects = []
        for line in sections.get("HitObjects", []):
            fields = line.split(",")
            x, y, time, flags = map(int, fields[:4])
            position = np.array([x, y], dtype=float)
            start = time / 1000
            if flags & 1:
                obj = HitObject(
                    start, start + 0.060, position, "circle", position[None]
                )
            elif flags & 8:
                obj = HitObject(
                    start, float(fields[5]) / 1000, position, "spinner", position[None]
                )
            elif flags & 2:
                beat_length, velocity = 500.0, 1.0
                for row in timing:
                    if float(row[0]) > time:
                        break
                    value = float(row[1])
                    if value > 0:
                        beat_length, velocity = value, 1.0
                    else:
                        velocity = float(np.clip(-100 / value, 0.1, 10))
                descriptor = fields[5].split("|")
                points = np.array(
                    [[x, y], *[list(map(float, p.split(":"))) for p in descriptor[1:]]]
                )
                repeats, length = int(fields[6]), float(fields[7])
                duration = (
                    length
                    * repeats
                    * beat_length
                    / (100 * float(values.get("SliderMultiplier", "1.4")) * velocity)
                    / 1000
                )
                obj = HitObject(
                    start,
                    start + duration,
                    position,
                    "slider",
                    curve(descriptor[0], points, length),
                    repeats,
                )
            else:
                raise ValueError(f"Unsupported hit object flags: {flags}")
            objects.append(obj)
        if not objects:
            raise ValueError("Map has no hit objects.")
        return cls(values, objects, float(values.get("CircleSize", "5")))

    def targets(self, times: FloatArray) -> FloatArray:
        result = np.zeros((len(times), 3))
        previous = np.array([256.0, 192.0])
        previous_end = -10.0
        cursor = 0
        for i, time in enumerate(times):
            while cursor < len(self.objects) and time > self.objects[cursor].end:
                obj = self.objects[cursor]
                previous, previous_end = obj.at(obj.end), obj.end
                cursor += 1
            if cursor == len(self.objects):
                result[i, :2] = previous
                continue
            obj = self.objects[cursor]
            if time >= obj.start:
                result[i, :2], result[i, 2] = obj.at(time), 1
            else:
                begin = max(previous_end + 0.020, obj.start - 0.250)
                blend = float(
                    np.clip((time - begin) / max(obj.start - begin, 1e-6), 0, 1)
                )
                blend = blend * blend * (3 - 2 * blend)
                result[i, :2] = previous * (1 - blend) + obj.at(obj.start) * blend
        result[:, :2] /= [512, 384]
        return result
