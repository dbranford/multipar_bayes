import numpy as np
import pytest

from multipar_bayes.bounds import MatrixBound, ConvexBound, ScalarBound
from multipar_bayes.lower import SPMBound, RPMBound, SqPMBound
from multipar_bayes.pgm import PGMBound


@pytest.mark.parametrize(
    "bound_class,kwargs",
    [
        (MatrixBound, {"matrix_pseudo_gain": np.diag([1, 1])}),
        (SPMBound, {}),
        (RPMBound, {}),
        (SqPMBound, {}),
        (PGMBound, {}),
    ],
)
def test_second_moment(bound_class, kwargs):
    ρ0 = np.array([[1]])
    ρ1s = [np.array([[1]]) for _ in range(2)]

    b = bound_class(ρ0, ρ1s, **kwargs)

    assert b.prior_second_moment is None

    assert (
        b.weighted_prior_second_moment(
            weight_matrix=np.identity(2), prior_second_moment=np.diag([1, 2])
        )
        == 3
    )

    b = bound_class(ρ0, ρ1s, weight_matrix=np.identity(2), **kwargs)

    assert b.prior_second_moment is None

    assert b.weighted_prior_second_moment(prior_second_moment=np.diag([1, 2])) == 3

    # Directly supplied weight_matrix is preferred
    assert (
        b.weighted_prior_second_moment(
            weight_matrix=2 * np.identity(2), prior_second_moment=np.diag([1, 2])
        )
        == 6
    )


@pytest.mark.parametrize(
    "bound_class,kwargs",
    [
        (ScalarBound, {"scalar_pseudo_gain": 2}),
        (ConvexBound, {"scalar_pseudo_gain": 2, "cvxpy_problem": None}),
    ],
)
def test_second_moment2(bound_class, kwargs):
    ρ0 = np.array([[1]])
    ρ1s = [np.array([[1]]) for _ in range(2)]
    weight_matrix = np.identity(2)

    b = bound_class(ρ0, ρ1s, weight_matrix=weight_matrix, **kwargs)

    assert b.prior_second_moment is None

    assert b.weighted_prior_second_moment(prior_second_moment=np.diag([1, 2])) == 3
