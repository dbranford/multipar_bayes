import itertools
import numpy as np
from collections.abc import Iterable


def general_measurement(
    rho0: np.typing.ArrayLike,
    rho1s: list[np.typing.ArrayLike],
    meas1s: list[np.typing.ArrayLike],
    meas2s: list[list[np.typing.ArrayLike]] | np.typing.ArrayLike,
    weight_matrix: np.typing.ArrayLike | None = None,
) -> tuple[float, np.typing.NDArray, list[np.typing.NDArray]]:
    """
    Compute the gain from a given measurement[1]_.

    Parameters
    ----------
    rho0 : ndarray
        Average density matrix of the ensemble.
    rho1s : list of ndarray
        List of first moment operators with respect to each parameter.
    meas1s : list of ndarray
        List of first moment measurements with respect to each parameter.
    meas2s : ndarray or list of list of ndarray
        second moment measurements with respect to all parameter pairs, shape should be [para_num,param_num,hilbert_size,hilbert_size].
    weight_matrix : ndarray, optional
        Weight matrix. Defaults to the identity matrix.

    Returns
    -------
    scalar_gain : float
        Scalar gain.
    matrix_gain : ndarray
        Matrix SPM gain.

    References
    ----------
    .. [1] F. Albarelli, D. Branford, J. Rubio, Measurement incompatibility in Bayesian multiparameter quantum estimation, arXiv:2511.XXXXX.
    """
    if weight_matrix is None:
        weight_matrix = np.eye(len(rho1s))

    rho1s = np.asarray(rho1s)
    meas1s = np.asarray(meas1s)
    meas2s = np.asarray(meas2s)

    m2ρ0 = np.trace(np.tensordot(meas2s, rho0))
    m1ρ1 = np.trace(np.moveaxis(np.tensordot(meas1s, np.moveaxis(rho1s, 0, -1)), -1, 1))
    m1ρ1 = m1ρ1 + np.transpose(m1ρ1)

    # Result
    matrix_gain = m2ρ0 - m1ρ1
    scalar_gain = np.real(np.trace(weight_matrix @ matrix_gain))

    return scalar_gain, matrix_gain


def finite_measurement_bayesian_update(
    rho0: np.typing.ArrayLike,
    rho1s: list[np.typing.ArrayLike],
    povms: Iterable[np.typing.ArrayLike],
    weight_matrix: np.typing.ArrayLike | None = None,
) -> tuple[float, np.typing.NDArray, list[np.typing.NDArray]]:
    """
    Compute the gain from a given measurement, where the Bayesian update rule is used[1]_.

    Parameters
    ----------
    rho0 : ndarray
        Average density matrix of the ensemble.
    rho1s : list of ndarray
        List of first moment operators with respect to each parameter.
    povms : list of ndarray
        List of POVM elements
    weight_matrix : ndarray, optional
        Weight matrix. Defaults to the identity matrix.

    Returns
    -------
    scalar_gain : float
        Scalar gain.
    matrix_gain : ndarray
        Matrix SPM gain.

    References
    ----------
    .. [1] J. Rubio and J. Dunningham, Bayesian multiparameter quantum metrology with limited data, Phys. Rev. A 101, 032114 (2020)
    """
    num_params = len(rho1s)
    if weight_matrix is None:
        weight_matrix = np.eye(num_params)

    povm_terms = map(lambda povm: _finite_bound_povm_terms(povm, rho0, rho1s), povms)

    matrix_gain = np.zeros((num_params, num_params))
    for t in povm_terms:
        matrix_gain += t

    scalar_gain = np.trace(weight_matrix @ matrix_gain)

    return scalar_gain, matrix_gain


def finite_local_measurement_bayesian_update(
    rho0: np.typing.ArrayLike,
    rho1s: list[np.typing.ArrayLike],
    povms: tuple[Iterable[np.typing.ArrayLike], int] | list[Iterable[np.typing.ArrayLike]],
    weight_matrix: np.typing.ArrayLike | None = None,
) -> tuple[float, np.typing.NDArray, list[np.typing.NDArray]]:
    r"""
    Compute `finite_measurement_bayesian_update` where the POVM is of the form \\( \Pi_{\vec{k}} = \bigotimes_j \Pi^{(j)}_{k_j} \\)  where each \\( \Pi^{(j)} \\) forms a valid POVM

    Parameters
    ----------
    rho0 : ndarray
        Average density matrix of the ensemble.
    rho1s : list of ndarray
        List of first moment operators with respect to each parameter.
    povms : list of ndarray
        List of POVM elements
    weight_matrix : ndarray, optional
        Weight matrix. Defaults to the identity matrix.

    Returns
    -------
    scalar_gain : float
        Scalar pseudo-gain.
    matrix_gain : ndarray
        Matrix pseudo-gain.

    Notes
    -----
    This mainly exists to allow lazy loading of the POVM as a map, that then gets passed to `finite_measurement_bayesian_update`
    """
    match povms:
        case (povm, num):
            povms = itertools.product(povm, repeat=num)
        case [*local_povms]:
            povms = itertools.product(*local_povms)

    povms = itertools.starmap(_kron_all, povms)

    return finite_measurement_bayesian_update(rho0, rho1s, povms, weight_matrix)


def _kron_all(*mats: np.typing.ArrayLike) -> np.typing.NDArray:
    res = np.identity(1)
    for mat in mats:
        res = np.kron(res, np.asarray(mat))
    return res


def _product_trace(a: np.typing.ArrayLike, b: np.typing.ArrayLike) -> float | complex:
    b = np.asarray(b)
    return np.sum(a * b.transpose())


def _finite_bound_povm_terms(
    povm: np.typing.ArrayLike, rho0: np.typing.ArrayLike, rho1s: list[np.typing.ArrayLike]
) -> np.typing.NDArray:
    povm = np.asarray(povm)
    povm_rho0 = np.real(_product_trace(povm, rho0))
    povm_rho1s = [np.real(_product_trace(povm, rho1)) for rho1 in rho1s]
    return np.outer(povm_rho1s, povm_rho1s) / povm_rho0
