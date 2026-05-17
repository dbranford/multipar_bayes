import cvxpy as cp
from dataclasses import dataclass
import itertools
import numpy as np
from collections.abc import Iterable, Sequence
from functools import cached_property
from typing import Self

from multipar_bayes.bounds import MatrixBound, BoundMSL

__all__ = ["MeasurementLossGeneral", "MeasurementLossBayesianUpdate", "seesaw_optimized_povm"]


class MeasurementLossGeneral(MatrixBound):
    m1s: np.typing.NDArray
    r"""
    Array of \\\( M_j \\\) operators, with shape [param_num, hilbert_dim, hilbert_dim].
    This may not be available for subclasses of `MeasurementLossGeneral` in which case they will be `None`.
    """
    m2s: np.typing.NDArray
    r"""
    Array of \\\( M_{j,k} \\\) operators, with shape [param_num, param_num, hilbert_dim, hilbert_dim]
    This may not be available for subclasses of `MeasurementLossGeneral` in which case they will be `None`.
    """
    # Probably with itertools.accumulate m1s and m2s can be made compulsary by evaluating simultaneously with matrix_pseudo_gain in POVM-based subclasses
    eps: float
    r"""Epsilon value below which \\( \mathrm{Tr}(\rho M(x)) \\) is taken to be zero"""

    def __init__(
        self,
        rho0: np.typing.ArrayLike,
        rho1s: list[np.typing.ArrayLike],
        meas1s: list[np.typing.ArrayLike],
        meas2s: list[list[np.typing.ArrayLike]] | np.typing.ArrayLike,
        weight_matrix: np.typing.ArrayLike | None = None,
        prior_second_moment: np.typing.ArrayLike | None = None,
        weighted_prior_second_moment: float | None = None,
        eps: float = 1e-15,
    ):
        r"""
        Compute the gain from a given measurement[1]_<sup>,</sup>[2]_.

        Parameters
        ----------
        rho0 : ndarray
            Average density matrix of the ensemble.
        rho1s : list of ndarray
            List of first moment operators with respect to each parameter.
        meas1s : list of ndarray
            List of first moment measurements with respect to each parameter.
        meas2s : ndarray or list of list of ndarray
            second moment measurements with respect to all parameter pairs, shape should be [para_num, param_num, hilbert_size, hilbert_size].
        weight_matrix : ndarray, optional
            Weight matrix. Defaults to the identity matrix
        prior_second_moment : ndarray, optional
            Second moment of the prior (\\( \Lambda \\))
        weighted_prior_second_moment : float, optional
            Scalar weighted second moment of the prior (\\( \lambda \\)). This is ignored if both `weight_matrix` and `prior_second_moment` are provided
        eps : float, optional
            Minimum value of \\( \mathrm{Tr}(\rho_0 \Pi) \\) to consider as \\( 0 \\)

        Returns
        -------
        matrix_bound : MeasurementLossGeneral
            Bound

        References
        ----------
        .. [1] J. Suzuki, Bayesian Nagaoka-Hayashi Bound for Multiparameter Quantum-State Estimation Problem, IEICE Trans. Fundam. Electron. Commun. Comput. Sci. E107.A, 510 (2024), [arXiv:2302.14223](https://arxiv.org/abs/2302.14223).
        .. [2] F. Albarelli, D. Branford, J. Rubio, Measurement incompatibility in Bayesian multiparameter quantum estimation, [arXiv:2511.16645](https://arxiv.org/abs/2511.16645).
        """
        self.m1s = np.asarray(meas1s)
        self.m2s = np.asarray(meas2s)
        self.eps = eps

        BoundMSL.__init__(
            self,
            rho0=rho0,
            rho1s=rho1s,
            weight_matrix=weight_matrix,
            weighted_prior_second_moment=weighted_prior_second_moment,
            prior_second_moment=prior_second_moment,
        )

    @cached_property
    def _matrix_pseudo_gain(self) -> np.typing.NDArray:
        m2ρ0 = np.einsum("jkmn,nm->jk", self.m2s, self.rho0)
        m1ρ1 = np.einsum("jmn,knm->jk", self.m1s, self.rho1s)
        m1ρ1 = m1ρ1 + np.transpose(m1ρ1)

        return m2ρ0 - m1ρ1


class MeasurementLossBayesianUpdate(MeasurementLossGeneral):
    povms: list[np.typing.ArrayLike]
    """POVMs supplied, if povms where supplied as an iterator rather than an e.g. list, this will be an empty list"""

    def __init__(
        self,
        rho0: np.typing.ArrayLike,
        rho1s: Sequence[np.typing.ArrayLike],
        povms: Iterable[np.typing.ArrayLike],
        weight_matrix: np.typing.ArrayLike | None = None,
        prior_second_moment: np.typing.ArrayLike | None = None,
        weighted_prior_second_moment: float | None = None,
        eps: float = 1e-15,
    ):
        r"""
        Compute the gain from a given measurement, where the Bayesian update rule is used[1]_.

        Parameters
        ----------
        rho0 : ndarray
            Average density matrix of the ensemble.
        rho1s : list of ndarray
            List of first moment operators with respect to each parameter.
        povms : iterable of ndarray
            List of POVM elements
        weight_matrix : ndarray, optional
            Weight matrix. Defaults to the identity matrix.
        prior_second_moment : ndarray, optional
            Second moment of the prior (\\( \Lambda \\))
        weighted_prior_second_moment : float, optional
            Scalar weighted second moment of the prior (\\( \lambda \\)). This is ignored if both `weight_matrix` and `prior_second_moment` are provided
        eps : float, optional
            Minimum value of \\( \mathrm{Tr}(\rho_0 \Pi) \\) to consider as \\( 0 \\)

        Returns
        -------
        bound : MeasurementLossBayesianUpdate
            Bound instance

        References
        ----------
        .. [1] J. Rubio and J. Dunningham, Bayesian multiparameter quantum metrology with limited data, [Phys. Rev. A 101, 032114 (2020)](https://doi.org/10.1103/PhysRevA.101.032114).
        """
        BoundMSL.__init__(
            self,
            rho0=rho0,
            rho1s=rho1s,
            weight_matrix=weight_matrix,
            weighted_prior_second_moment=weighted_prior_second_moment,
            prior_second_moment=prior_second_moment,
        )

        self.eps = eps
        self._povms = povms
        self._matrix_pseudo_gain
        self.povms = list(self._povms)
        del self._povms

    @cached_property
    def _matrix_pseudo_gain(self) -> np.typing.NDArray:
        num_params = len(self.rho1s)
        matrix_pseudo_gain = np.zeros((num_params, num_params))
        for povm in self._povms:
            povm = np.asarray(povm)
            povm_rho0 = _prob_povm(self.rho0, povm)
            if np.abs(povm_rho0) < self.eps:
                continue
            povm_rho1s = [_prob_povm(rho1, povm) for rho1 in self.rho1s]
            matrix_pseudo_gain += np.outer(povm_rho1s, povm_rho1s) / povm_rho0
        return matrix_pseudo_gain

    @property
    def m1s(self) -> np.typing.NDArray | None:
        if len(self.povms) == 0:
            return None
        num_params = len(self.rho1s)
        dim = self.rho0.shape[0]
        m1 = np.zeros((num_params, dim, dim))
        for povm in self.povms:
            povm = np.asarray(povm)
            povm_rho0 = _prob_povm(self.rho0, povm)
            if np.abs(povm_rho0) < self.eps:
                continue
            povm_m1s = np.array([povm * _prob_povm(rho1, povm) / povm_rho0 for rho1 in self.rho1s])
            m1 += povm_m1s
        return m1

    @property
    def m2s(self) -> np.typing.NDArray | None:
        if len(self.povms) == 0:
            return None
        num_params = len(self.rho1s)
        dim = self.rho0.shape[0]
        m2 = np.zeros((num_params, num_params, dim, dim))
        for povm in self.povms:
            povm = np.asarray(povm)
            povm_rho0 = _prob_povm(self.rho0, povm)
            if np.abs(povm_rho0) < self.eps:
                continue
            povm_rho1s = [_prob_povm(rho1, povm) for rho1 in self.rho1s]
            povm_m2s = np.tensordot(np.outer(povm_rho1s, povm_rho1s) / povm_rho0, povm, axes=0)
            m2 += povm_m2s
        return m2

    @classmethod
    def local_measurement(
        cls,
        rho0: np.typing.ArrayLike,
        rho1s: list[np.typing.ArrayLike],
        povms: tuple[Iterable[np.typing.ArrayLike], int] | list[Iterable[np.typing.ArrayLike]],
        weight_matrix: np.typing.ArrayLike | None = None,
        prior_second_moment: np.typing.ArrayLike | None = None,
        weighted_prior_second_moment: float | None = None,
        eps: float = 1e-15,
    ) -> Self:
        r"""
        Compute `finite_measurement_bayesian_update` where the POVM is of the form \\( \Pi_{\vec{k}} = \bigotimes_j \Pi^{(j)}_{k_j} \\)  where each \\( \Pi^{(j)} \\) forms a valid POVM

        Parameters
        ----------
        rho0 : ndarray
            Average density matrix of the ensemble.
        rho1s : list of ndarray
            List of first moment operators with respect to each parameter.
        povms : (list of ndarray, int) or list of list of ndarray
            List of POVM elements
        weight_matrix : ndarray, optional
            Weight matrix. Defaults to the identity matrix.
        eps : float, optional
            Minimum value of \\( \mathrm{Tr}(\rho_0 \Pi) \\) to consider as \\( 0 \\)

        Returns
        -------
        bound : MeasurementLossBayesianUpdate
            MeasurementLossBayesianUpdate instance

        Notes
        -----
        The resulting object will not have `povms` stored in `MeasurementLossBayesianUpdate.povms` on the basis that the full POVM set is unwieldly to initialise as a list.
        """
        match povms:
            case (povm, num):
                _povms = itertools.product(povm, repeat=num)
            case [*local_povms]:
                _povms = itertools.product(*local_povms)
            case _:
                raise ValueError(
                    "povms must be a (list of arrays, int) tuple, or a list of lists of arrays"
                )

        _povms = itertools.starmap(_kron_all, _povms)

        return cls(
            rho0,
            rho1s,
            _povms,
            weight_matrix=weight_matrix,
            weighted_prior_second_moment=weighted_prior_second_moment,
            prior_second_moment=prior_second_moment,
            eps=eps,
        )


def _kron_all(*mats: np.typing.ArrayLike) -> np.typing.NDArray:
    res = np.identity(1)
    for mat in mats:
        res = np.kron(res, np.asarray(mat))
    return res


def _prob_povm(ρ: np.typing.NDArray, povm: np.typing.NDArray) -> float:
    return np.real(np.einsum("ij,ji->", ρ, povm))


def _product_trace(a: np.typing.ArrayLike, b: np.typing.ArrayLike) -> float | complex:
    a = np.asarray(a)
    b = np.asarray(b)
    return np.einsum("ij,ji->", a, b)


def _optimize_povm_given_estimators(
    rho0: np.typing.NDArray,
    rho1s: list[np.typing.NDArray],
    f_hats: np.typing.NDArray,
    weight_matrix: np.typing.NDArray,
    solver: str = "SCS",
    verbose: bool = False,
) -> tuple[list[np.typing.NDArray], float, str]:
    """
    SDP step for fixed estimators in f-space.
    """
    _rho1s = np.asarray(rho1s)
    f_hats = np.asarray(f_hats, dtype=float)
    weight_matrix = np.asarray(weight_matrix)

    dim = rho0.shape[0]
    num_outcomes, num_params = f_hats.shape
    if _rho1s.shape[-1] != num_params:
        raise ValueError("rho1s and estimator dimensions do not match.")

    xis = [
        2.0 * np.einsum("jmn,jk,k->mn", _rho1s, weight_matrix, f_hats_i)
        - np.einsum("jk,j,k->", weight_matrix, f_hats_i, f_hats_i) * rho0
        for f_hats_i in f_hats
    ]

    povms = [cp.Variable((dim, dim), hermitian=True) for _ in range(num_outcomes)]
    constraints = [m >> 0 for m in povms]
    constraints += [sum(povms) == np.eye(dim)]

    objective = cp.sum([cp.real(cp.trace(m @ xi)) for m, xi in zip(povms, xis)])
    prob = cp.Problem(cp.Maximize(objective), constraints)
    prob.solve(solver=solver, verbose=verbose)

    return [m.value for m in povms], objective.value, prob.status


def _update_estimators_posterior_mean(
    rho0: np.typing.NDArray,
    rho1s: list[np.typing.NDArray],
    povms: list[np.typing.NDArray],
    eps: float = 1e-12,
) -> np.typing.NDArray:
    """
    Closed-form Bayesian update for quadratic loss in f-space.
    """
    num_params = len(rho1s)
    num_povms = len(povms)

    f_hats = np.zeros((num_povms, num_params), dtype=float)
    for i, m in enumerate(povms):
        m = np.asarray(m)
        pi = _prob_povm(rho0, m)
        if pi <= eps:
            continue
        f_hats[i, :] = [_prob_povm(rho1, m) / pi for rho1 in rho1s]
    return f_hats


def seesaw_optimized_povm(
    rho0: np.typing.ArrayLike,
    rho1s: list[np.typing.ArrayLike],
    weight_matrix: np.typing.ArrayLike,
    prior_second_moment: np.typing.ArrayLike | None = None,
    weighted_prior_second_moment: float | None = None,
    num_outcomes: int | None = None,
    num_rounds: int = 20,
    eps: float = 1e-12,
    solver: str = "SCS",
    verbose: bool = False,
    seed: np.random.Generator | int | None = None,
    init_box: np.typing.ArrayLike | None = None,
) -> tuple[MeasurementLossBayesianUpdate, np.typing.NDArray, str]:
    r"""
    Implement numerical approach to find POVM[1]_<sup>,</sup>[2]_

    Parameters
    ----------
    rho0 : ndarray
        Average density matrix of the ensemble.
    rho1s : list of ndarray
        List of first moment operators with respect to each parameter.
    weight_matrix : ndarray
        Weight matrix
    prior_second_moment : ndarray, optional
        Second moment of the prior (\\( \Lambda \\)). Required in lieu of `weighted_prior_second_moment`.
    weighted_prior_second_moment : float, optional
        Scalar weighted second moment of the prior (\\( \lambda \\)). This is ignored if `prior_second_moment` are provided
    num_outcomes : int, optional
        Number of outcomes to consider, defaults to the square of the corresponding Hilbert space dimension.
    num_rounds : int, optional
        Number of rounds of optimisation to use, defaults to 20.
    eps : float, optional
        Minimum value of \\( \mathrm{Tr}(\rho_0 \Pi) \\) to consider as \\( 0 \\).
    solver : str, optional
        CVXPY solver to use (e.g., 'SCS', 'MOSEK').
    verbose : bool, optional
        Print additional detail (also passed to `cvxpy.Problem`), defaults to False.
    seed :  np.random.Generator, int, optional
        Seed random
    init_box : arraylike, optional
        Initial values for


    Returns
    -------
    bound : MeasurementLossBayesianUpdate
        Bound instance
    f_hats : array
        Matrix of estimates associated with given outcomes
    status : str
        Status of the corresponding `cvxpy.Problem` which led to the outcome

    References
    ----------
    .. [1] J. Bavaresco, P. Lipka-Bartosik, P. Sekatski, M. Mehboudi, Designing optimal protocols in Bayesian quantum parameter estimation with higher-order operations, [Phys. Rev. Research 6, 023305 (2024)](https://doi.org/10.1103/PhysRevResearch.6.023305).
    .. [2] F. Albarelli, D. Branford, J. Rubio, Measurement incompatibility in Bayesian multiparameter quantum estimation, [arXiv:2511.16645](https://arxiv.org/abs/2511.16645).
    """
    rho0 = np.asarray(rho0)
    _rho1s = [np.asarray(r) for r in rho1s]

    weight_matrix = np.asarray(weight_matrix)

    dim = rho0.shape[0]
    num_params = len(rho1s)

    if num_outcomes is None:
        num_outcomes = dim * dim

    if init_box is None:
        init_box = np.ones(num_params, dtype=float)
    else:
        init_box = np.asarray(init_box, dtype=float)

        if init_box.shape != (num_params,):
            raise ValueError(f"init_box must have shape {(num_params,)}, got {init_box.shape}.")

    rng = np.random.default_rng(seed)
    f_hats = rng.uniform(low=-init_box, high=init_box, size=(num_outcomes, num_params))

    best_msl = np.inf

    for round_idx in range(num_rounds):
        povms, score, status = _optimize_povm_given_estimators(
            rho0=rho0,
            rho1s=_rho1s,
            f_hats=f_hats,
            weight_matrix=weight_matrix,
            solver=solver,
            verbose=verbose,
        )
        f_hats = _update_estimators_posterior_mean(rho0=rho0, rho1s=_rho1s, povms=povms, eps=eps)
        bound = MeasurementLossBayesianUpdate(
            rho0,
            _rho1s,
            povms,
            weight_matrix=weight_matrix,
            prior_second_moment=prior_second_moment,
            weighted_prior_second_moment=weighted_prior_second_moment,
            eps=eps,
        )
        msl = bound.scalar_bound()

        if msl < best_msl:
            best_msl = msl
            best = _SeesawStrategy(povms=povms, f_hats=f_hats, status=status, bound=bound)

        if verbose:
            print(f"[round {round_idx:02d}] status={status:>20} score={score: .8e} msl={msl: .8e}")

    return best.bound, best.f_hats, best.status


@dataclass
class _SeesawStrategy:
    povms: list[np.typing.NDArray]
    f_hats: np.typing.NDArray
    status: str
    bound: MeasurementLossBayesianUpdate
