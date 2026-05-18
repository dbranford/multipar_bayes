from multipar_bayes.measurements import MeasurementLossBayesianUpdate, MeasurementLossGeneral
import numpy as np


def test_measurements():
    ρ0 = np.diag([0.25, 0.25, 0.25, 0.25])
    ρ1s = [np.diag([0.5, 0, 0, 0]), np.diag([0, 0.5, 0, 0])]

    povms = [
        np.diag([1, 0, 0, 0]),
        np.diag([0, 1, 0, 0]),
        np.diag([0, 0, 1, 0]),
        np.diag([0, 0, 0, 1]),
    ]

    m1 = MeasurementLossBayesianUpdate(ρ0, ρ1s, povms)

    assert m1.m1s is not None
    assert m1.m2s is not None

    m2 = MeasurementLossGeneral(ρ0, ρ1s, m1.m1s, m1.m2s)

    assert (m1.matrix_pseudo_gain() == m2.matrix_pseudo_gain()).all()
