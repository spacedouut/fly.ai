import numpy as np

from osu_fly.audio import RATE, WINDOW, SongClock, locate


def test_clock_rejects_weak_or_inconsistent_audio_then_locks():
    clock = SongClock()
    assert clock.observe(200, 4.220, 0.107) is None
    assert clock.observe(200.5, 3.440, 0.150) is None
    assert clock.observe(201, 8, 0.9) is None
    assert clock.observe(201.5, 0, 0.95) is None
    assert clock.observe(202, 0.5, 0.95) == 201
    assert clock.observe(202.5, 1, 0.1) is None
    assert clock.observe(203, 1.5, 0.95) is None


def test_short_windows_lock_before_first_object():
    rng = np.random.default_rng(11)
    reference = rng.normal(size=RATE * 15)
    clock = SongClock()
    for position in (0.25, 0.75):
        sample = reference[int(position * RATE) : int((position + WINDOW) * RATE)]
        found, confidence = locate(reference, 0.2 * sample)
        epoch = clock.observe(100 + position + WINDOW, found, confidence)
    assert epoch == 100
