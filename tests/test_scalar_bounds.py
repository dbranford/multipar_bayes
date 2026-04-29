import numpy as np
import pytest

from multipar_bayes.bounds import MatrixBound, ConvexBound, ScalarBound
from multipar_bayes.convex import HolevoNagaokaBound, NagaokaHayashiBound
from multipar_bayes.lower import SPMBound, RPMBound, SqPMBound
from multipar_bayes.pgm import PGMBound


@pytest.mark.parametrize(
    "bound_class,kwargs",
    [
        (MatrixBound, {"matrix_pseudo_gain": np.diag([1, 1])}),
        (SPMBound, {}),
        (RPMBound, {}),
        (SqPMBound, {}),
    ],
)
def test_scalar_bound_matrix(bound_class, kwargs):
    ρ0 = np.diag([0.25, 0.25, 0.25, 0.25])
    ρ1s = [np.diag([0.5, 0, 0, 0]), np.diag([0, 0.5, 0, 0])]

    weight_matrix = np.identity(2)

    b = bound_class(ρ0, ρ1s, **kwargs)

    assert b.weight_matrix is None
    assert b.prior_second_moment is None
    assert b.weighted_prior_second_moment() is None

    assert b.scalar_pseudo_gain(weight_matrix=weight_matrix) == 2

    prior_second_moment = np.diag([5, 2])

    assert (
        b.scalar_bound(weight_matrix=weight_matrix, prior_second_moment=prior_second_moment) == 5
    )

    b = bound_class(ρ0, ρ1s, weight_matrix=weight_matrix, **kwargs)

    assert b.prior_second_moment is None

    assert b.scalar_bound(prior_second_moment=prior_second_moment) == 5

    # Directly supplied weight_matrix is preferred
    assert (
        b.scalar_bound(weight_matrix=np.diag([1, 1.5]), prior_second_moment=prior_second_moment)
        == 5.5
    )


@pytest.mark.parametrize("bound_class,kwargs", [(PGMBound, {})])
def test_scalar_bound_pgm(bound_class, kwargs):
    ρ0 = np.diag([0.25, 0.25, 0.25, 0.25])
    ρ1s = [np.diag([0.5, 0, 0, 0]), np.diag([0, 0.5, 0, 0])]

    weight_matrix = np.identity(2)

    b = bound_class(ρ0, ρ1s, **kwargs)

    assert b.weight_matrix is None
    assert b.prior_second_moment is None
    assert b.weighted_prior_second_moment() is None

    prior_second_moment = np.diag([5, 2])

    assert (
        b.scalar_pseudo_gain(weight_matrix=weight_matrix, prior_second_moment=prior_second_moment)
        == -3
    )

    b = bound_class(ρ0, ρ1s, weight_matrix=weight_matrix, **kwargs)

    assert b.prior_second_moment is None

    assert b.scalar_bound(prior_second_moment=prior_second_moment) == 10

    # Directly supplied weight_matrix is preferred
    assert (
        b.scalar_bound(weight_matrix=np.diag([1, 1.5]), prior_second_moment=prior_second_moment)
        == 11
    )


@pytest.mark.parametrize(
    "bound_class,kwargs",
    [
        (ScalarBound, {"scalar_pseudo_gain": 2}),
        (ConvexBound, {"scalar_pseudo_gain": 2, "cvxpy_problem": None}),
        (NagaokaHayashiBound, {}),
        (HolevoNagaokaBound, {}),
    ],
)
def test_second_moment2(bound_class, kwargs):
    ρ0 = np.diag([0.25, 0.25, 0.25, 0.25])
    ρ1s = [np.diag([0.5, 0, 0, 0]), np.diag([0, 0.5, 0, 0])]
    weight_matrix = np.identity(2)

    b = bound_class(ρ0, ρ1s, weight_matrix=weight_matrix, **kwargs)

    assert b.prior_second_moment is None
    assert b.weighted_prior_second_moment() is None

    assert np.allclose(b.scalar_pseudo_gain, 2)

    prior_second_moment = np.diag([5, 2])

    assert np.allclose(b.scalar_bound(prior_second_moment=prior_second_moment), 5)
