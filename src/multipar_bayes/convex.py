import cvxpy as cp
import numpy as np
import scipy as sp
import itertools
from collections.abc import Iterable
from scipy.sparse import coo_array, dia_array, eye_array


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


def _uxu_sparse(c1, c2, shape) -> coo_array:
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


def _uxv_sparse(c1, c2, shape) -> coo_array:
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


def _vxu_sparse(c1, c2, shape) -> coo_array:
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


def _vxv_sparse(c1, c2, shape) -> coo_array:
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


def hn_fun(
    rho0: np.typing.ArrayLike,
    rho1s: list[np.typing.ArrayLike],
    weight_matrix: np.typing.ArrayLike | None = None,
    solver: str | None = "SCS",
) -> tuple[float, np.typing.NDArray, cp.Problem]:
    """
    Calculation of the Bayesian Holevo-Nagaoka bound (NHB)[1]_ via semidefinite programming.

    Parameters
    ----------
    rho0 : ndarray
        Density matrix, average state of the ensemble.
    rho1s : list of ndarrays
        List of first moment operators wrt to each parameter
    weight_matrix : ndarray, optional
        Weight matrix (defaults to identity).
    solver : str, optional
        CVXPY solver to use (e.g., 'SCS', 'MOSEK').

    Returns
    -------
    gain : float
        The pseudo-gain of the objective (Nagaoka–Hayashi bound).
    x_opt : ndarray
        Matrix assembled from auxiliary variables X.
    prob : cp.Problem
        The CVXPY problem object (for further inspection).

    References
    ----------
    .. [1] J. Suzuki, Bayesian Nagaoka-Hayashi Bound for Multiparameter Quantum-State Estimation Problem, [IEICE Trans. Fundam. Electron. Commun. Comput. Sci. E107.A, 510 (2024)](https://doi.org/10.1587/transfun.2023TAP0014), [arXiv:2302.14223](https://arxiv.org/abs/2302.14223).
    """

    if not isinstance(rho1s, list):
        raise TypeError("Please make sure rho1 is a list!")

    dim = len(rho0)
    num = dim * dim
    para_num = len(rho1s)

    if weight_matrix is None:
        weight_matrix = np.identity(para_num)

    u, v, w = su_n_generators(dim)
    lambdas = [eye_array(dim, format="dia") / np.sqrt(dim)] + u + v + w

    generator_products = u_n_generators_product(dim)
    S = np.array([(rho0 @ m).trace() for m in generator_products]).reshape((num, num))

    lu, d, _ = sp.linalg.ldl(S)
    R = np.dot(lu, sp.linalg.sqrtm(d)).conj().T

    V = cp.Variable((para_num, para_num), symmetric=True)
    X = cp.Variable((num, para_num))

    constraints = [cp.bmat([[V, X.T @ R.conj().T], [R @ X, np.identity(num)]]) >> 0]

    vec_rho1s = np.array([[np.real((rho1j @ λ).trace()) for λ in lambdas] for rho1j in rho1s])

    objective = cp.trace(weight_matrix @ V) - 2 * cp.trace(X @ weight_matrix @ vec_rho1s)

    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(solver=solver)

    # Extract the results
    x_opt = X.value
    bound = objective.value

    return -bound, x_opt, prob


def nh_fun(
    rho0: np.typing.ArrayLike,
    rho1s: list[np.typing.ArrayLike],
    weight_matrix: np.typing.ArrayLike | None = None,
    solver: str | None = "SCS",
) -> tuple[float, np.typing.NDArray, list[np.typing.NDArray], cp.Problem]:
    """
    Calculation of the Bayesian Nagaoka-Hayashi bound (NHB)[1]_ via semidefinite programming.

    Parameters
    ----------
    rho0 : ndarray
        Density matrix, average state of the ensemble.
    rho1s : list of ndarrays
        List of first moment operators wrt to each parameter
    weight_matrix : ndarray, optional
        Weight matrix (defaults to identity).
    solver : str, optional
        CVXPY solver to use (e.g., 'SCS', 'MOSEK').

    Returns
    -------
    gain : float
        The gain lower-bound of the Nagaoka–Hayashi bound.
    l_opt : ndarray
        Matrix assembled from SDP variables v.
    x_opt : ndarray
        Matrix assembled from auxiliary variables X.
    prob : cp.Problem
        The CVXPY problem object (for further inspection).

    References
    ----------
    .. [1] J. Suzuki, Bayesian Nagaoka-Hayashi Bound for Multiparameter Quantum-State Estimation Problem, [IEICE Trans. Fundam. Electron. Commun. Comput. Sci. E107.A, 510 (2024)](https://doi.org/10.1587/transfun.2023TAP0014), [arXiv:2302.14223](https://arxiv.org/abs/2302.14223).
    """
    dim = len(rho0)
    para_num = len(rho1s)

    if weight_matrix is None:
        weight_matrix = np.eye(para_num)

    L_tp = [[[] for i in range(para_num)] for j in range(para_num)]
    for para_i in range(para_num):
        for para_j in range(para_i, para_num):
            L_tp[para_i][para_j] = cp.Variable((dim, dim), hermitian=True)
            L_tp[para_j][para_i] = L_tp[para_i][para_j]
    L = cp.vstack([cp.hstack(L_tp_i) for L_tp_i in L_tp])

    # X_j variables for the optimization
    X = [cp.Variable((dim, dim), hermitian=True) for _ in range(para_num)]

    # Positive semidefinite constraint
    constraints = [cp.bmat([[L, cp.vstack(X)], [cp.hstack(X), np.identity(dim)]]) >> 0]

    # Objective function
    D = np.kron(weight_matrix, np.eye(dim)) @ cp.vstack(rho1s)
    objective = cp.real(
        cp.trace(cp.kron(weight_matrix, rho0) @ L) - 2 * cp.trace(D @ cp.hstack(X))
    )

    # Solve the problem
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(solver=solver)

    # Extract the results
    l_opt = L.value
    x_opt = [x_i.value for x_i in X]
    gain = -objective.value

    return gain, l_opt, x_opt, prob
