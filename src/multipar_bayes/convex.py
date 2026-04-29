import cvxpy as cp
import numpy as np
import numpy.typing as npt
import scipy as sp
import itertools
from collections.abc import Iterable
from scipy.sparse import coo_array, dia_array, eye_array

from multipar_bayes.bounds import ConvexBound, BoundMSL


def su_n_generators(n: int) -> tuple[list[coo_array], list[coo_array], list[dia_array]]:
    u = [
        coo_array(([1 / np.sqrt(2), 1 / np.sqrt(2)], ([i, j], [j, i])), shape=(n, n))
        for i in range(1, n)
        for j in range(i)
    ]
    v = [
        coo_array(([1j / np.sqrt(2), -1j / np.sqrt(2)], ([i, j], [j, i])), shape=(n, n))
        for i in range(1, n)
        for j in range(i)
    ]
    w = [
        np.sqrt(1 / (i * (i + 1)))
        * dia_array(([1 if j < i else -j for j in range(i + 1)], 0), shape=(n, n))
        for i in range(1, n)
    ]
    return u, v, w


def _uxu_sparse(c1: tuple[int, int], c2: tuple[int, int], shape: tuple[int, int]) -> coo_array:
    match (c1[0] == c2[0], c1[1] == c2[1], c1[0] == c2[1], c1[1] == c2[0]):
        case (True, True, _, _):
            return coo_array(([0.5, 0.5], (c2, c1)), shape=shape)
        case (True, False, _, _):
            return coo_array(([0.5], ([c2[1]], [c1[1]])), shape=shape)
        case (False, True, _, _):
            return coo_array(([0.5], ([c2[0]], [c1[0]])), shape=shape)
        case (False, False, True, _):
            return coo_array(([0.5], ([c2[0]], [c1[1]])), shape=shape)
        case (False, False, _, True):
            return coo_array(([0.5], ([c2[1]], [c1[0]])), shape=shape)
        case (False, False, False, False):
            return coo_array(shape)
    raise ValueError("Invalid arguments to _uxu_sparse, this should be unreachable")


def _uxv_sparse(c1: tuple[int, int], c2: tuple[int, int], shape: tuple[int, int]) -> coo_array:
    match (c1[0] == c2[0], c1[1] == c2[1], c1[0] == c2[1], c1[1] == c2[0]):
        case (True, True, _, _):
            return coo_array(([0.5j, -0.5j], (c2, c1)), shape=shape)
        case (True, False, _, _):
            return coo_array(([-0.5j], ([c2[1]], [c1[1]])), shape=shape)
        case (False, True, _, _):
            return coo_array(([0.5j], ([c2[0]], [c1[0]])), shape=shape)
        case (False, False, True, _):
            return coo_array(([-0.5j], ([c2[0]], [c1[1]])), shape=shape)
        case (False, False, _, True):
            return coo_array(([0.5j], ([c2[1]], [c1[0]])), shape=shape)
        case (False, False, False, False):
            return coo_array(shape)
    raise ValueError("Invalid arguments to _uxv_sparse, this should be unreachable")


def _vxu_sparse(c1: tuple[int, int], c2: tuple[int, int], shape: tuple[int, int]) -> coo_array:
    match (c1[0] == c2[0], c1[1] == c2[1], c1[0] == c2[1], c1[1] == c2[0]):
        case (True, True, _, _):
            return coo_array(([-0.5j, 0.5j], (c2, c1)), shape=shape)
        case (True, False, _, _):
            return coo_array(([0.5j], ([c2[1]], [c1[1]])), shape=shape)
        case (False, True, _, _):
            return coo_array(([-0.5j], ([c2[0]], [c1[0]])), shape=shape)
        case (False, False, True, _):
            return coo_array(([-0.5j], ([c2[0]], [c1[1]])), shape=shape)
        case (False, False, _, True):
            return coo_array(([0.5j], ([c2[1]], [c1[0]])), shape=shape)
        case (False, False, False, False):
            return coo_array(shape)
    raise ValueError("Invalid arguments to _vxu_sparse, this should be unreachable")


def _vxv_sparse(c1: tuple[int, int], c2: tuple[int, int], shape: tuple[int, int]) -> coo_array:
    match (c1[0] == c2[0], c1[1] == c2[1], c1[0] == c2[1], c1[1] == c2[0]):
        case (True, True, _, _):
            return coo_array(([0.5, 0.5], (c2, c1)), shape=shape)
        case (True, False, _, _):
            return coo_array(([0.5], ([c2[1]], [c1[1]])), shape=shape)
        case (False, True, _, _):
            return coo_array(([0.5], ([c2[0]], [c1[0]])), shape=shape)
        case (False, False, True, _):
            return coo_array(([-0.5], ([c2[0]], [c1[1]])), shape=shape)
        case (False, False, _, True):
            return coo_array(([-0.5], ([c2[1]], [c1[0]])), shape=shape)
        case (False, False, False, False):
            return coo_array(shape)
    raise ValueError("Invalid arguments to _vxv_sparse, this should be unreachable")


def u_n_generators_product(n: int) -> Iterable[coo_array | dia_array]:
    u, v, w = su_n_generators(n)
    u = [ui.astype(np.float64) for ui in u]
    w = [wi.astype(np.float64) for wi in w]
    iden = eye_array(n, format="dia") / np.sqrt(n)

    coords = [(i, j) for j in range(1, n) for i in range(j)]

    prod_iter = [iden] + u + v + w
    prod_iter = map(lambda x: x / np.sqrt(n), prod_iter)
    uix = [
        [ui / np.sqrt(n)]
        + [_uxu_sparse((j, k), c, shape=(n, n)) for k in range(1, n) for j in range(k)]
        + [_uxv_sparse((j, k), c, shape=(n, n)) for k in range(1, n) for j in range(k)]
        + [
            coo_array(([wi[c[0]] / np.sqrt(2), wi[c[1]] / np.sqrt(2)], (c[::-1], c)), shape=(n, n))
            for wi in map(lambda x: x.diagonal(), w)
        ]
        for c, ui in zip(coords, u)
    ]
    vix = [
        [vi / np.sqrt(n)]
        + [_vxu_sparse((j, k), c, shape=(n, n)) for k in range(1, n) for j in range(k)]
        + [_vxv_sparse((j, k), c, shape=(n, n)) for k in range(1, n) for j in range(k)]
        + [
            coo_array(
                ([1j * wi[c[0]] / np.sqrt(2), -1j * wi[c[1]] / np.sqrt(2)], (c[::-1], c)),
                shape=(n, n),
            )
            for wi in map(lambda x: x.diagonal(), w)
        ]
        for c, vi in zip(coords, v)
    ]
    wix = [
        [wi / np.sqrt(n)]
        + [
            coo_array(
                ([wi_diag[c[1]] / np.sqrt(2), wi_diag[c[0]] / np.sqrt(2)], (c[::-1], c)),
                shape=(n, n),
            )
            for c, ui in zip(coords, u)
        ]
        + [
            coo_array(
                (
                    [-1j * wi_diag[c[0]] / np.sqrt(2), 1j * wi_diag[c[1]] / np.sqrt(2)],
                    (c, c[::-1]),
                ),
                shape=(n, n),
            )
            for c, vi in zip(coords, v)
        ]
        + [wi @ wii for wii in w]
        for i, (wi, wi_diag) in enumerate(map(lambda x: (x, x.diagonal()), w))
    ]

    for uixi in uix:
        prod_iter = itertools.chain(prod_iter, uixi)
    for vixi in vix:
        prod_iter = itertools.chain(prod_iter, vixi)
    for wixi in wix:
        prod_iter = itertools.chain(prod_iter, wixi)

    return prod_iter


class HolevoNagaokaBound(ConvexBound):
    _x_opt: npt.NDArray
    rho0: np.typing.NDArray
    rho1s: list[np.typing.NDArray]
    weight_matrix: np.typing.NDArray
    prior_second_moment: np.typing.NDArray | None
    cvx_problem: cp.Problem
    scalar_pseudo_gain: float

    def __init__(
        self,
        rho0: npt.ArrayLike,
        rho1s: list[npt.ArrayLike],
        weight_matrix: npt.ArrayLike,
        solver: str | None = "SCS",
        prior_second_moment: npt.ArrayLike | None = None,
        weighted_prior_second_moment: float | None = None,
    ):
        r"""
        Calculation of the Bayesian Holevo-Nagaoka bound (NHB)[1]_ via semidefinite programming.

        Parameters
        ----------
        rho0 : ndarray
            Density matrix, average state of the ensemble.
        rho1s : list of ndarrays
            List of first moment operators wrt to each parameter
        weight_matrix : ndarray
            Weight matrix
        solver : str, optional
            CVXPY solver to use (e.g., 'SCS', 'MOSEK').
        prior_second_moment : ndarray, optional
            Second moment of the prior (\\( \Lambda \\))
        weighted_prior_second_moment : float, optional
            Scalar weighted second moment of the prior (\\( \lambda \\)). This is ignored if both `weight_matrix` and `prior_second_moment` are provided

        References
        ----------
        .. [1] J. Suzuki, Bayesian Nagaoka-Hayashi Bound for Multiparameter Quantum-State Estimation Problem, [IEICE Trans. Fundam. Electron. Commun. Comput. Sci. E107.A, 510 (2024)](https://doi.org/10.1587/transfun.2023TAP0014), [arXiv:2302.14223](https://arxiv.org/abs/2302.14223).
        """

        rho0 = np.asarray(rho0)
        dim = rho0.shape[0]
        num = dim * dim
        para_num = len(rho1s)

        if weight_matrix is None:
            raise ValueError("HolevoNagaokaBound requires a weight_matrix to be explicitly passed")

        BoundMSL.__init__(
            self,
            rho0=rho0,
            rho1s=rho1s,
            weight_matrix=weight_matrix,
            prior_second_moment=prior_second_moment,
            weighted_prior_second_moment=weighted_prior_second_moment,
        )

        u, v, w = su_n_generators(dim)
        lambdas = [eye_array(dim, format="dia") / np.sqrt(dim)] + u + v + w

        generator_products = u_n_generators_product(dim)
        s = np.array([(self.rho0 @ m).trace() for m in generator_products]).reshape((num, num))

        lu, d, _ = sp.linalg.ldl(s)
        r = np.dot(lu, sp.linalg.sqrtm(d)).conj().T

        v = cp.Variable((para_num, para_num), symmetric=True)
        x = cp.Variable((num, para_num))

        constraints = [cp.bmat([[v, x.T @ r.conj().T], [r @ x, np.identity(num)]]) >> 0]

        vec_rho1s = np.array(
            [[np.real((rho1j @ λ).trace()) for λ in lambdas] for rho1j in self.rho1s]
        )

        objective = cp.trace(weight_matrix @ v) - 2 * cp.trace(x @ weight_matrix @ vec_rho1s)

        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve(solver=solver)

        self._scalar_pseudo_gain = -objective.value
        self._cvxpy_problem = prob
        self._x_opt = x.value

    @property
    def x_opt(self) -> npt.NDArray:
        """Solution to the variable X"""
        return self._x_opt


class NagaokaHayashiBound(ConvexBound):
    _l_opt: npt.NDArray
    _x_opt: list[npt.NDArray]
    rho0: np.typing.NDArray
    rho1s: list[np.typing.NDArray]
    weight_matrix: np.typing.NDArray
    prior_second_moment: np.typing.NDArray | None
    cvx_problem: cp.Problem
    scalar_pseudo_gain: float

    def __init__(
        self,
        rho0: npt.ArrayLike,
        rho1s: list[npt.ArrayLike],
        weight_matrix: npt.ArrayLike | None = None,
        solver: str | None = "SCS",
        prior_second_moment: npt.ArrayLike | None = None,
        weighted_prior_second_moment: float | None = None,
    ):
        r"""
        Calculation of the Bayesian Nagaoka-Hayashi bound (NHB)[1]_ via semidefinite programming.

        Parameters
        ----------
        rho0 : ndarray
            Density matrix, average state of the ensemble.
        rho1s : list of ndarrays
            List of first moment operators wrt to each parameter
        weight_matrix : ndarray
            Weight matrix
        solver : str, optional
            CVXPY solver to use (e.g., 'SCS', 'MOSEK').
        prior_second_moment : ndarray, optional
            Second moment of the prior (\\( \Lambda \\))
        weighted_prior_second_moment : float, optional
            Scalar weighted second moment of the prior (\\( \lambda \\)). This is ignored if both `weight_matrix` and `prior_second_moment` are provided

        References
        ----------
        .. [1] J. Suzuki, Bayesian Nagaoka-Hayashi Bound for Multiparameter Quantum-State Estimation Problem, [IEICE Trans. Fundam. Electron. Commun. Comput. Sci. E107.A, 510 (2024)](https://doi.org/10.1587/transfun.2023TAP0014), [arXiv:2302.14223](https://arxiv.org/abs/2302.14223).
        """
        rho0 = np.asarray(rho0)
        dim = rho0.shape[0]
        para_num = len(rho1s)

        if weight_matrix is None:
            raise ValueError(
                "NagaokaHayashiBound requires a weight_matrix to be explicitly passed"
            )

        weight_matrix = np.asarray(weight_matrix)

        BoundMSL.__init__(
            self,
            rho0=rho0,
            rho1s=rho1s,
            weight_matrix=weight_matrix,
            prior_second_moment=prior_second_moment,
            weighted_prior_second_moment=weighted_prior_second_moment,
        )

        lvars = [
            cp.Variable((dim, dim), hermitian=True) for j in range(para_num) for _ in range(j + 1)
        ]

        u = [
            coo_array(([1, 1], ([i, j], [j, i])), shape=(para_num, para_num))
            for i in range(1, para_num)
            for j in range(i)
        ] + [coo_array(([1], ([i], [i])), shape=(para_num, para_num)) for i in range(para_num)]

        L = sum(cp.kron(u_i, l_i) for u_i, l_i in zip(u, lvars))

        # X_j variables for the optimization
        X = [cp.Variable((dim, dim), hermitian=True) for _ in range(para_num)]

        # Positive semidefinite constraint
        constraints = [cp.bmat([[L, cp.vstack(X)], [cp.hstack(X), np.identity(dim)]]) >> 0]

        # Objective function
        D = np.kron(weight_matrix, np.eye(dim)) @ cp.vstack(self.rho1s)
        objective = cp.real(
            cp.trace(cp.kron(weight_matrix, self.rho0) @ L) - 2 * cp.trace(D @ cp.hstack(X))
        )

        # Solve the problem
        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve(solver=solver)

        self._scalar_pseudo_gain = -objective.value
        self._cvxpy_problem = prob
        self._x_opt = [x_i.value for x_i in X]
        self._l_opt = L.value

    @property
    def x_opt(self) -> list[npt.NDArray]:
        """Solution to the variable X"""
        return self._x_opt

    @property
    def l_opt(self) -> npt.NDArray:
        """Solution to the variable L"""
        return self._l_opt
