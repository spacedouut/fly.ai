from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks

from .neural import Array

Image = NDArray[np.uint8]


@dataclass
class Circle:
    x: float
    y: float
    radius: float
    approach: float | None


class CircleDetector:
    """Detect concentric rings in a 512×384 playfield image."""

    def __init__(self) -> None:
        cv2.setNumThreads(1)
        angles = np.linspace(0, 2 * np.pi, 128, endpoint=False)
        self.dx = np.cos(angles)
        self.dy = np.sin(angles)

    def detect(self, image: Image) -> list[Circle]:
        value = image[:, :, :3].max(axis=2)
        gray = cv2.GaussianBlur(value, (3, 3), 0.6)
        found = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            1,
            28,
            param1=55,
            param2=23,
            minRadius=22,
            maxRadius=62,
        )
        if found is None:
            return []
        result = []
        for x, y, radius in found[0]:
            radii = np.arange(16, 220, dtype=float)
            xx = x + radii[:, None] * self.dx
            yy = y + radii[:, None] * self.dy
            valid = (xx >= 0) & (xx < 512) & (yy >= 0) & (yy < 384)
            samples = gray[
                np.clip(yy.astype(int), 0, 383), np.clip(xx.astype(int), 0, 511)
            ].astype(float)
            gradient = np.abs(samples[2:] - samples[:-2])
            gradient[~(valid[2:] & valid[:-2])] = np.nan
            supported = np.sum(np.isfinite(gradient), axis=1) > 40
            profile = np.zeros(len(gradient))
            profile[supported] = np.nanmedian(gradient[supported], axis=1)
            peaks, _ = find_peaks(profile, height=4, distance=5)
            peaks = peaks + 1
            base = peaks[(radii[peaks] >= 22) & (radii[peaks] <= 62)]
            if not len(base):
                continue
            radius = float(radii[base[-1]])
            outer = peaks[(radii[peaks] > radius + 6) & (radii[peaks] < radius * 4.5)]
            approach = None
            if len(outer):
                strongest = outer[np.argmax(profile[outer - 1])]
                approach = float(radii[strongest])
            result.append(Circle(float(x), float(y), radius, approach))
        return result


@dataclass
class Track:
    circle: Circle
    seen: float
    history: list[tuple[float, float]] = field(default_factory=list)
    due: float | None = None
    triggered: bool = False

    def update(self, circle: Circle, now: float) -> None:
        self.circle, self.seen = circle, now
        if circle.approach is None:
            return
        self.history.append((now, circle.approach))
        self.history = [(t, r) for t, r in self.history if t >= now - 0.4]
        if len(self.history) < 3 or now - self.history[0][0] < 0.07:
            return
        points = np.array(self.history)
        t = points[:, 0] - now
        design = np.column_stack([t, np.ones(len(t))])
        speed, radius = np.linalg.lstsq(design, points[:, 1], rcond=None)[0]
        residual = np.sqrt(np.mean((design @ [speed, radius] - points[:, 1]) ** 2))
        if -600 < speed < -12 and residual < 3:
            due = now + (radius - circle.radius) / -speed
            if now - 0.08 < due < now + 2:
                self.due = float(due)


class VisionController:
    def __init__(self, lead: float = 0.06) -> None:
        self.lead = lead
        self.tracks: list[Track] = []
        self.position = np.array([0.5, 0.5])
        self.release_at = 0.0
        self.taps = 0

    def update(self, circles: list[Circle], now: float) -> Array:
        self.tracks = [track for track in self.tracks if now - track.seen < 0.25]
        unmatched = list(self.tracks)
        for circle in circles:
            matches = [
                track
                for track in unmatched
                if np.hypot(track.circle.x - circle.x, track.circle.y - circle.y) < 9
                and abs(track.circle.radius - circle.radius) < 30
            ]
            if matches:
                track = min(
                    matches,
                    key=lambda t: np.hypot(
                        t.circle.x - circle.x, t.circle.y - circle.y
                    ),
                )
                unmatched.remove(track)
            else:
                track = Track(circle, now)
                self.tracks.append(track)
            track.update(circle, now)
        candidates = [
            track
            for track in self.tracks
            if track.due is not None
            and not track.triggered
            and now - track.seen < 0.15
            and track.due > now - 0.12
        ]
        if candidates and now >= self.release_at:
            target = min(
                candidates, key=lambda t: float("inf") if t.due is None else t.due
            )
            self.position = np.array([target.circle.x / 512, target.circle.y / 384])
            if target.due is not None and now >= target.due - self.lead:
                self.release_at = now + 0.12
                target.triggered = True
                self.taps += 1
        return np.r_[self.position, float(now < self.release_at)]
