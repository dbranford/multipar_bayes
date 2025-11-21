import numpy as np
from scipy.linalg import solve_continuous_lyapunov, sqrtm, fractional_matrix_power


def spm_fun(
    rho0: np.typing.ArrayLike,
    rho1s: list[np.typing.ArrayLike],
    weight_matrix: np.typing.ArrayLike | None = None,
) -> tuple[float, np.typing.NDArray, list[np.typing.NDArray]]:
    """
    Compute the Symmetric Posterior Mean (SPM) gain [1]_.

    Parameters
    ----------
    rho0 : ndarray
        Average density matrix of the ensemble.
    rho1s : list of ndarray
        List of first moment operators with respect to each parameter.
    weight_matrix : ndarray, optional
        Weight matrix. Defaults to the identity matrix.

    Returns
    -------
    spm_gain : float
        SPM gain.
    matrix_gain : ndarray
        Matrix SPM gain.
    spms : list of ndarray
        List of SPM operators.

    References
    ----------
    .. [1] J. Rubio and J. Dunningham, Bayesian multiparameter quantum metrology with limited data, Phys. Rev. A 101, 032114 (2020)
    """

    if weight_matrix is None:
        n = len(rho1s)
        weight_matrix = np.eye(n)

    # Solve the Sylvester equations
    spms = [solve_continuous_lyapunov(rho0, 2 * rho1) for rho1 in rho1s]

    matrix_gain = np.array([[np.trace(r1 @ spm) for spm in spms] for r1 in rho1s])

    # Result
    spm_gain = np.real(np.trace(weight_matrix @ matrix_gain))

    return spm_gain, matrix_gain, spms


def rpm_fun(
    rho0: np.typing.ArrayLike,
    rho1s: list[np.typing.ArrayLike],
    weight_matrix: np.typing.ArrayLike | None = None,
) -> tuple[float, np.typing.NDArray, list[np.typing.NDArray]]:
    """
    Compute the Right Posterior Mean (RPM) gain[1]_<sup>,</sup>[2]_.

    Parameters
    ----------
    rho0 : ndarray
        Average density matrix of the ensemble.
    rho1s : list of ndarray
        List of first moment operators with respect to each parameter.
    weight_matrix : ndarray, optional
        Weight matrix. Defaults to the identity matrix.

    Returns
    -------
    rpm_gain : float
        RPM gain.
    matrix_gain : ndarray
        Matrix RPM gain.
    rpms : list of ndarray
        List of RPM operators.

    References
    ----------
    .. [1] J. Suzuki, Bayesian Nagaoka-Hayashi Bound for Multiparameter Quantum-State Estimation Problem, [IEICE Trans. Fundam. Electron. Commun. Comput. Sci. E107.A, 510 (2024)](https://doi.org/10.1587/transfun.2023TAP0014), arXiv:2302.14223.
    .. [2] F. Albarelli, D. Branford, J. Rubio, Measurement incompatibility in Bayesian multiparameter quantum estimation, arXiv:2511.XXXXX.
    """
    if weight_matrix is None:
        n = len(rho1s)
        weight_matrix = np.eye(n)

    rho0inv = np.linalg.inv(rho0)

    # Calculate the RPM operators
    rpms = [rho0inv @ rho1 for rho1 in rho1s]

    matrix_gain = np.array([[np.trace(r1 @ rpm) for rpm in rpms] for r1 in rho1s])

    # Result
    sqrt_weight = sqrtm(weight_matrix)
    rpm_gain = np.trace(weight_matrix @ np.real(matrix_gain)) - np.sum(
        np.abs(np.linalg.eigvals(sqrt_weight @ np.imag(matrix_gain) @ sqrt_weight))
    )

    return rpm_gain, matrix_gain, rpms


def sqpm_fun(
    r0: np.typing.ArrayLike,
    rho1s: list[np.typing.ArrayLike],
    weight_matrix: np.typing.ArrayLike | None = None,
) -> tuple[float, np.typing.NDArray, list[np.typing.NDArray]]:
    """
    Computes the square-root posterior mean gain[1]_

    Parameters
    ----------
    rho0 : ndarray
        Average density matrix of the ensemble.
    rho1s : list of ndarray
        List of first moment operators with respect to each parameter.
    weight_matrix : ndarray, optional
        Weight matrix. Defaults to the identity matrix.

    Returns
    -------
    sqpm_gain : float
        SQPM gain.
    matrix_gain: ndarray
        Matrix SQPM gain.
    sqpms : list of ndarray
        List of SQPM operators.

    See Also
    --------
    pgm_fun : Corresponding lower bound and posterior mean operators

    References
    ----------
    .. [1] F. Albarelli, D. Branford, J. Rubio, Measurement incompatibility in Bayesian multiparameter quantum estimation, arXiv:2511.XXXXX.
    """
    if weight_matrix is None:
        n = len(rho1s)
        weight_matrix = np.eye(n)

    r0_inv_sqrt = fractional_matrix_power(r0, -0.5)

    # Calculate the RPM operators
    sqpms = [r0_inv_sqrt @ rho1 @ r0_inv_sqrt for rho1 in rho1s]

    # Compute the matrix precision gain
    matrix_gain = np.array([[np.trace(r1 @ sqpm) for sqpm in sqpms] for r1 in rho1s])

    # Result
    pgm_gain = np.real(np.trace(weight_matrix @ matrix_gain))

    return pgm_gain, matrix_gain, sqpms
