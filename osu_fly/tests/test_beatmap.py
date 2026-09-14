import numpy as np
import pytest

from osu_fly.beatmap import Beatmap, HitObject, curve


def test_perfect_curve_direction_and_length():
    path = curve("P", np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]), np.pi)
    assert path[len(path) // 2, 1] > 0.9
    np.testing.assert_allclose(path[-1], [-1, 0], atol=2e-4)
    assert np.linalg.norm(np.diff(path, axis=0), axis=1).sum() == pytest.approx(
        np.pi, rel=0.02
    )


def test_repeated_slider_returns_to_start():
    path = curve("L", np.array([[0.0, 0.0], [100.0, 0.0]]), 100)
    obj = HitObject(1, 3, path[0], "slider", path, 2)
    np.testing.assert_allclose(obj.at(1), [0, 0])
    np.testing.assert_allclose(obj.at(2), [100, 0])
    np.testing.assert_allclose(obj.at(3), [0, 0])


def test_inherited_velocity_and_timing_reset():
    beatmap = Beatmap.parse(
        "[Difficulty]\nSliderMultiplier:1\n[TimingPoints]\n"
        "0,500,4,1,0,100,1,0\n1000,-50,4,1,0,100,0,0\n"
        "3000,400,4,1,0,100,1,0\n[HitObjects]\n"
        "0,0,1500,2,0,L|100:0,2,100\n"
        "0,0,3500,2,0,L|100:0,1,100\n"
    )
    assert beatmap.objects[0].end == 2
    assert beatmap.objects[1].end == 3.9
    target = beatmap.targets(np.array([1.0, 1.5, 1.75, 2.01]))
    np.testing.assert_allclose(target[:, 2], [0, 1, 1, 0])


def test_bezier_red_anchor_and_length_extension():
    path = curve(
        "B", np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 0.0], [10.0, 10.0]]), 25
    )
    np.testing.assert_allclose(path[0], [0, 0])
    np.testing.assert_allclose(path[-1], [10, 15])
