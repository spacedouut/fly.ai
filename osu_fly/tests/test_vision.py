import cv2
import numpy as np
import pytest

from osu_fly.vision import Circle, CircleDetector, VisionController


@pytest.mark.parametrize("center", [(310, 140), (480, 200)])
def test_rendered_countdown_including_clipped_rings(center: tuple[int, int]) -> None:
    detector, controller = CircleDetector(), VisionController(lead=0)
    taps = []
    for now in np.arange(0, 1.31, 0.02):
        image = np.zeros((384, 512, 3), dtype=np.uint8)
        cv2.circle(image, center, 40, (100, 130, 160), 3)
        radius = round(40 + 80 * (1 - now))
        if radius > 40:
            cv2.circle(image, center, radius, (140, 180, 220), 2)
        output = controller.update(detector.detect(image), float(now))
        if output[2]:
            taps.append(now)
    assert controller.taps == 1
    assert 0.88 <= taps[0] <= 1.12
    np.testing.assert_allclose(output[:2] * [512, 384], center, atol=5)


def test_detects_concentric_target_at_arbitrary_position():
    image = np.zeros((384, 512, 3), dtype=np.uint8)
    cv2.circle(image, (310, 140), 40, (100, 130, 160), 3)
    cv2.circle(image, (310, 140), 90, (140, 180, 220), 2)
    detections = CircleDetector().detect(image)
    target = min(detections, key=lambda c: np.hypot(c.x - 310, c.y - 140))
    assert np.hypot(target.x - 310, target.y - 140) < 3
    assert 37 < target.radius < 44
    assert target.approach is not None and abs(target.approach - 90) < 4


def test_hit_circle_decoration_does_not_replace_the_approach_ring() -> None:
    image = np.zeros((384, 512, 3), dtype=np.uint8)
    cv2.circle(image, (310, 140), 40, (220, 220, 220), 2)
    cv2.circle(image, (310, 140), 35, (160, 160, 160), -1)
    cv2.circle(image, (310, 140), 28, (90, 90, 90), -1)
    cv2.circle(image, (310, 140), 100, (110, 110, 110), 2)
    detections = CircleDetector().detect(image)
    target = min(detections, key=lambda c: np.hypot(c.x - 310, c.y - 140))
    assert 37 < target.radius < 44
    assert target.approach is not None
    assert abs(target.approach - 100) < 4


def test_visual_countdown_taps_once_without_song_clock():
    controller = VisionController(lead=0.06)
    taps = []
    for now in np.arange(100, 101.4, 0.02):
        radius = 40 + 80 * (101 - now)
        output = controller.update(
            [Circle(310, 140, 40, radius if radius > 46 else None)], now
        )
        if output[2]:
            taps.append(now)
    assert controller.taps == 1
    assert 100.91 < taps[0] < 100.99
    assert taps[-1] - taps[0] <= 0.13
    np.testing.assert_allclose(output[:2], [310 / 512, 140 / 384])


def test_static_or_expanding_decoration_never_taps():
    controller = VisionController()
    for now in np.arange(0, 2, 0.02):
        output = controller.update(
            [
                Circle(100, 100, 40, 90),
                Circle(400, 280, 40, 70 + now * 20),
            ],
            now,
        )
        assert output[2] == 0
    assert controller.taps == 0


def test_lost_target_does_not_trigger_predicted_future_tap():
    controller = VisionController()
    for now in np.arange(0, 0.4, 0.02):
        controller.update([Circle(250, 190, 40, 120 - now * 80)], now)
    for now in np.arange(0.4, 2, 0.02):
        assert controller.update([], now)[2] == 0


def test_frozen_countdown_expires_and_resumes_without_an_early_tap():
    controller = VisionController(lead=0)
    for now in np.arange(0, 0.4, 0.02):
        controller.update([Circle(250, 190, 40, 120 - now * 80)], float(now))
    for now in np.arange(0.4, 2, 0.02):
        assert controller.update([Circle(250, 190, 40, 88)], float(now))[2] == 0
    taps = []
    for now in np.arange(2, 2.8, 0.02):
        radius = 88 - (now - 2) * 80
        output = controller.update(
            [Circle(250, 190, 40, radius if radius > 46 else None)], float(now)
        )
        if output[2]:
            taps.append(now)
    assert controller.taps == 1
    assert 2.56 <= taps[0] <= 2.65


def test_approach_ring_merging_with_base_preserves_contact_time():
    controller = VisionController(lead=0)
    taps = []
    for now in np.arange(0, 1.3, 0.02):
        radius = 120 - now * 80
        circle = (
            Circle(250, 190, 40, radius)
            if radius > 62
            else Circle(250, 190, max(40, radius), None)
        )
        if controller.update([circle], float(now))[2]:
            taps.append(now)
    assert controller.taps == 1
    assert 0.97 <= taps[0] <= 1.06
