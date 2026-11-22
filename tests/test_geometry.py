import numpy as np
from m3.geometry.camera import hfov_deg, face_px, eff_res, MODES


def test_hfov_datasheet():
    assert abs(hfov_deg(*MODES["B"]) - 62.2) < 0.5
    assert abs(hfov_deg(*MODES["A"]) - 39.0) < 0.2


def test_reference_values():
    exp = {"A": [190, 86, 56, 38], "B": [95, 43, 28, 19], "C": [74, 34, 22, 15], "D": [37, 17, 11, 7], "E": [111, 51, 33, 22]}
    for m, v in exp.items():
        got = face_px([2, 4.4, 6.8, 10], MODES[m][1]); assert np.all(np.abs(got - v) <= 1), (m, got)
    assert list(eff_res([2, 4.4, 6.8, 10], MODES["B"][1])) == [48, 48, 33, 22]
