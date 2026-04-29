"""
Posterior mean lower bounds
"""

from collections.abc import Sequence
import numpy as np
from scipy.linalg import solve_continuous_lyapunov, fractional_matrix_power

from multipar_bayes.bounds import PMBound, RealPMBound, BoundMSL


class SPMBound(RealPMBound):
    _spms: list[np.typing.NDArray]
    rho0: np.typing.NDArray
    rho1s: list[np.typing.NDArray]

    def __init__(
        self,
        rho0: np.typing.ArrayLike,
        rho1s: Sequence[np.typing.ArrayLike],
        weight_matrix: np.typing.ArrayLike | None = None,
        prior_second_moment: np.typing.ArrayLike | None = None,
        weighted_prior_second_moment: float | None = None,
    ):
        r"""
        Compute the Symmetric Posterior Mean (SPM) gain [1]_.

        Parameters
        ----------
        rho0 : ndarray
            Average density matrix of the ensemble
        rho1s : list of ndarray
            List of first moment operators with respect to each parameter
        weight_matrix : ndarray, optional
            Weight matrix. Defaults to the identity matrix
        prior_second_moment : ndarray, optional
            Second moment of the prior (\\( \Lambda \\))
        weighted_prior_second_moment : float, optional
            Scalar weighted second moment of the prior (\\( \lambda \\)). This is ignored if both `weight_matrix` and `prior_second_moment` are provided

        References
        ----------
        .. [1] J. Rubio and J. Dunningham, Bayesian multiparameter quantum metrology with limited data, Phys. Rev. A 101, 032114 (2020)
        """
        BoundMSL.__init__(
            self,
            rho0=rho0,
            rho1s=rho1s,
            weight_matrix=weight_matrix,
            prior_second_moment=prior_second_moment,
            weighted_prior_second_moment=weighted_prior_second_moment,
        )

        self._spms = [solve_continuous_lyapunov(self.rho0, 2 * rho1) for rho1 in self.rho1s]
        self._posterior_mean_operators = self._spms


class RPMBound(PMBound):
    _rpms: list[np.typing.NDArray]
    rho0: np.typing.NDArray
    rho1s: list[np.typing.NDArray]

    def __init__(
        self,
        rho0: np.typing.ArrayLike,
        rho1s: Sequence[np.typing.ArrayLike],
        weight_matrix: np.typing.ArrayLike | None = None,
        prior_second_moment: np.typing.ArrayLike | None = None,
        weighted_prior_second_moment: float | None = None,
    ):
        r"""
        Compute the Right Posterior Mean (RPM) bound[1]_<sup>,</sup>[2]_.

        Parameters
        ----------
        rho0 : ndarray
            Average density matrix of the ensemble
        rho1s : list of ndarray
            List of first moment operators with respect to each parameter
        weight_matrix : ndarray, optional
            Weight matrix. Defaults to the identity matrix
        prior_second_moment : ndarray, optional
            Second moment of the prior (\\( \Lambda \\))
        weighted_prior_second_moment : float, optional
            Scalar weighted second moment of the prior (\\( \lambda \\)). This is ignored if both `weight_matrix` and `prior_second_moment` are provided

        References
        ----------
        .. [1] J. Suzuki, Bayesian Nagaoka-Hayashi Bound for Multiparameter Quantum-State Estimation Problem, [IEICE Trans. Fundam. Electron. Commun. Comput. Sci. E107.A, 510 (2024)](https://doi.org/10.1587/transfun.2023TAP0014), arXiv:2302.14223.
        .. [2] F. Albarelli, D. Branford, J. Rubio, Measurement incompatibility in Bayesian multiparameter quantum estimation, [arXiv:2511.16645](https://arxiv.org/abs/2511.16645).
        """
        BoundMSL.__init__(
            self,
            rho0=rho0,
            rho1s=rho1s,
            weight_matrix=weight_matrix,
            prior_second_moment=prior_second_moment,
            weighted_prior_second_moment=weighted_prior_second_moment,
        )

        rho0inv = np.linalg.inv(self.rho0)

        self._rpms = [rho0inv @ rho1 for rho1 in self.rho1s]
        self._posterior_mean_operators = self._rpms


class SqPMBound(RealPMBound):
    _sqpms: list[np.typing.NDArray]
    rho0: np.typing.NDArray
    rho1s: list[np.typing.NDArray]

    def __init__(
        self,
        rho0: np.typing.ArrayLike,
        rho1s: Sequence[np.typing.ArrayLike],
        weight_matrix: np.typing.ArrayLike | None = None,
        prior_second_moment: np.typing.ArrayLike | None = None,
        weighted_prior_second_moment: float | None = None,
    ):
        r"""
        Computes the square-root posterior mean bound[1]_

        Parameters
        ----------
        rho0 : ndarray
            Average density matrix of the ensemble
        rho1s : list of ndarray
            List of first moment operators with respect to each parameter
        weight_matrix : ndarray, optional
            Weight matrix. Defaults to the identity matrix
        prior_second_moment : ndarray, optional
            Second moment of the prior (\\( \Lambda \\))
        weighted_prior_second_moment : float, optional
            Scalar weighted second moment of the prior (\\( \lambda \\)). This is ignored if both `weight_matrix` and `prior_second_moment` are provided


        See Also
        --------
        PGMBound : Corresponding upper bound and posterior mean operators

        References
        ----------
        .. [1] F. Albarelli, D. Branford, J. Rubio, Measurement incompatibility in Bayesian multiparameter quantum estimation, [arXiv:2511.16645](https://arxiv.org/abs/2511.16645).
        """
        BoundMSL.__init__(
            self,
            rho0=rho0,
            rho1s=rho1s,
            weight_matrix=weight_matrix,
            prior_second_moment=prior_second_moment,
            weighted_prior_second_moment=weighted_prior_second_moment,
        )

        rho0_inv_sqrt = fractional_matrix_power(self.rho0, -0.5)
        self._sqpms = [rho0_inv_sqrt @ rho1 @ rho0_inv_sqrt for rho1 in self.rho1s]
        self._posterior_mean_operators = self._sqpms
