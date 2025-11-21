import numpy as np
from .lower import sqpm_fun


def pgm_fun(
    rho0: np.typing.ArrayLike,
    rho1s: list[np.typing.ArrayLike],
    λ: float | np.typing.ArrayLike,
    weight_matrix: np.typing.ArrayLike | None = None,
) -> tuple[float, np.typing.NDArray | None]:
    """
    Computes the pretty good measurement (PGM) gain[1]_.

    Parameters
    ----------
    rho0 : ndarray
        Average density matrix of the ensemble.
    rho1s : list of ndarray
        List of first moment operators with respect to each parameter.
    λ : float, ndarray
        Second moment of the parameters
    weight_matrix : ndarray, optional
        Weight matrix. Defaults to the identity matrix.

    Returns
    -------
    pgm_gain : float
        PGM gain.
    matrix_gain: ndarray, None
        Matrix PGM gain, if a matrix λ is supplied, else None

    See Also
    --------
    sqpm_fun : Corresponding lower bound and posterior mean operators

    References
    ----------
    .. [1] F. Albarelli, D. Branford, J. Rubio, Measurement incompatibility in Bayesian multiparameter quantum estimation, arXiv:2511.XXXXX.
    """
    if weight_matrix is None:
        weight_matrix = np.identity(len(rho1s))
    scalar_gain, matrix_gain, _ = sqpm_fun(rho0, rho1s, weight_matrix)
    if isinstance(λ, float):
        return 2 * (λ - scalar_gain), None
    λ = np.asarray(λ)
    if np.prod(λ.shape) == 1:
        return 2 * (λ.item() - scalar_gain), None
    return 2 * (np.trace(weight_matrix @ λ) - scalar_gain), 2 * (λ - matrix_gain)
