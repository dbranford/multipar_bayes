import numpy as np
from multipar_bayes.bounds import BoundMSL
from multipar_bayes.lower import SqPMBound


class PGMBound(SqPMBound):
    rho0: np.typing.NDArray
    rho1s: list[np.typing.NDArray]
    weight_matrix: np.typing.NDArray
    prior_second_moment: np.typing.NDArray | None
    sqpm_bound: SqPMBound
    """The corresponding `SqPMBound` used for the `PGMBound`."""

    def __init__(
        self,
        rho0: np.typing.ArrayLike,
        rho1s: list[np.typing.ArrayLike],
        weight_matrix: np.typing.ArrayLike | None = None,
        prior_second_moment: np.typing.ArrayLike | None = None,
        weighted_prior_second_moment: float | None = None,
    ):
        r"""
        Computes the pretty good measurement (PGM) gain[1]_.

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
        SqPMBound : Corresponding lower bound and posterior mean operators

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
        self.sqpm_bound = SqPMBound(rho0=self.rho0, rho1s=self.rho1s)

    def matrix_pseudo_gain(
        self, prior_second_moment: np.typing.ArrayLike | None = None
    ) -> np.typing.NDArray:
        r"""
        Pseudo-gain \\( \Lambda - \mathcal{M}_{\mathrm{B}} \\)

        Parameters
        ----------
        prior_second_moment : ndarray, optional
            Prior second moment \\( \Lambda = \int \mathrm{d}\boldsymbol{\theta}\, p(\boldsymbol{\theta}) \boldsymbol{\theta} \boldsymbol{\theta}^T \\)

        Returns
        -------
        pseudo_gain : ndarray
            The pseudo-gain of the bound
        """
        matrix_pseudo_gain = self.sqpm_bound.matrix_pseudo_gain()
        if prior_second_moment is not None:
            prior_second_moment = np.asarray(prior_second_moment)
        else:
            prior_second_moment = self.prior_second_moment
        if prior_second_moment is None:
            raise ValueError(
                "PGMBound.matrix_pseudo_gain requires prior_second_moment to be passed as an argument or set in the PGMBound object"
            )
        return 2 * matrix_pseudo_gain - prior_second_moment

    def matrix_bound(
        self, prior_second_moment: np.typing.ArrayLike | None = None
    ) -> np.typing.NDArray:
        match (self.prior_second_moment, prior_second_moment):
            case (None, None):
                raise ValueError(
                    "MatrixBound.matrix_bound requires prior_second_moment to be set in the bound or passed as an argument"
                )
            case (Λ, None) | (None, Λ):
                prior_second_moment = np.asarray(Λ)
        return prior_second_moment - self.sqpm_bound.matrix_pseudo_gain()

    def scalar_pseudo_gain(
        self,
        weight_matrix: np.typing.ArrayLike | None = None,
        prior_second_moment: np.typing.ArrayLike | None = None,
        weighted_prior_second_moment: float | None = None,
    ) -> float:
        r"""
        Scalar pseudo-gain \\( \lambda - \mathcal{L}_{\mathrm{PGM}} \\)

        Parameters
        ----------
        weight_matrix : ndarray, optional
            Weight matrix, defaults to identity if not specified as argument or in the Bound object
        prior_second_moment : ndarray, optional
            Prior second moment (\\( \Lambda \\))
        weighted_prior_second_moment : float, optional
            Scalar weighted second moment of the prior (\\( \lambda \\)). This is ignored if both `weight_matrix` and `prior_second_moment` are provided

        Returns
        -------
        pseudo_gain : float
            The pseudo-gain
        """
        if weight_matrix is None:
            weight_matrix = self.weight_matrix

        sqpm_scalar_pseudo_gain = self.sqpm_bound.scalar_pseudo_gain(weight_matrix=weight_matrix)

        if weighted_prior_second_moment is None:
            weighted_prior_second_moment = self.weighted_prior_second_moment(
                weight_matrix, prior_second_moment
            )
            if weighted_prior_second_moment is None:
                raise ValueError(
                    "PGMBound.scalar_pseudo_gain requires weighted_prior_second_moment (or prior_second_moment along with access to weight_matrix) to be passed as an argument or set in the PGMBound object"
                )
        return 2 * sqpm_scalar_pseudo_gain - weighted_prior_second_moment

    def scalar_bound(
        self,
        weight_matrix: np.typing.ArrayLike | None = None,
        prior_second_moment: np.typing.ArrayLike | None = None,
        weighted_prior_second_moment: float | None = None,
    ) -> float:
        if weighted_prior_second_moment is None:
            weighted_prior_second_moment = self.weighted_prior_second_moment(
                weight_matrix, prior_second_moment
            )
        if weighted_prior_second_moment is None:
            raise ValueError(
                "scalar_bound requires weighted_prior_second_moment to be set, passed, or calculable"
            )
        return weighted_prior_second_moment - self.scalar_pseudo_gain(
            weight_matrix=weight_matrix,
            prior_second_moment=prior_second_moment,
            weighted_prior_second_moment=weighted_prior_second_moment,
        )
