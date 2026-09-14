import numpy as np

from osu_fly.audio import RATE, locate


def test_audio_sync_finds_excerpt_despite_gain_noise_and_dc():
    rng = np.random.default_rng(32)
    reference = rng.normal(size=RATE * 8)
    sample = reference[3 * RATE : 5 * RATE] * 0.2 + rng.normal(0, 0.03, RATE * 2) + 0.3
    position, confidence = locate(reference, sample)
    assert position == 3
    assert confidence > 0.9
