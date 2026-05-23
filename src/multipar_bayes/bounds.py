"""
Generic classes which exist primarily to be subclassed.
"""

from collections.abc import Sequence
import cvxpy
import numpy as np
from scipy.linalg import sqrtm
from functools import cached_property


class BoundMSL:
    _ρ0: np.typing.NDArray
    _ρ1s: list[np.typing.NDArray]
    _scalar_λ: float | None
    _matrix_λ: np.typing.NDArray | None
    _weight_matrix: np.typing.NDArray | None

    def __init__(
        self,
        rho0: np.typing.ArrayLike,
        rho1s: Sequence[np.typing.ArrayLike],
        weight_matrix: np.typing.ArrayLike | None = None,
        prior_second_moment: np.typing.ArrayLike | None = None,
        weighted_prior_second_moment: float | None = None,
    ):
        r"""
        Base class for bounds

        Parameters
        ----------
        rho0 : ndarray
            Average density matrix of the ensemble
        rho1s : list of ndarray
            List of first moment operators with respect to each parameter
        weight_matrix : ndarray, optional
            Weight matrix. Defaults to the identity matrix
        prior_second_moment : ndarray, optional
            Prior second moment \\( \Lambda = \int \mathrm{d}\boldsymbol{\theta}\, p(\boldsymbol{\theta}) \boldsymbol{\theta} \boldsymbol{\theta}^T \\)
        weighted_prior_second_moment : float, optional
            Scalar weighted second moment of the prior (\\( \lambda \\)). This is ignored if both `weight_matrix` and `prior_second_moment` are provided
        """

        self._ρ0 = np.asarray(rho0)

        if not isinstance(rho1s, Sequence):
            raise TypeError("rho1s must be a list")

        self._ρ1s = [np.asarray(ρ1) for ρ1 in rho1s]

        if weight_matrix is not None:
            weight_matrix = np.asarray(weight_matrix)
        self._weight_matrix = weight_matrix
        self._scalar_λ = weighted_prior_second_moment
        if prior_second_moment is not None:
            prior_second_moment = np.asarray(prior_second_moment)
        self._matrix_λ = prior_second_moment

    @property
    def rho0(self) -> np.typing.NDArray:
        r"""Zeroth state moment \\( \rho_0 = \int \mathrm{d}\boldsymbol{\theta}\, p(\boldsymbol{\theta}) \rho(\boldsymbol{\theta}) \\)."""
        return self._ρ0

    @property
    def rho1s(self) -> list[np.typing.NDArray]:
        r"""First state moments \\( \int \mathrm{d}\boldsymbol{\theta}\, p(\boldsymbol{\theta}) \rho(\boldsymbol{\theta}) \boldsymbol{\theta} \\)."""
        return self._ρ1s

    @property
    def weight_matrix(self) -> np.typing.NDArray | None:
        r"""Weight matrix \\( W \\) (if set)."""
        return self._weight_matrix

    @property
    def prior_second_moment(self) -> np.typing.NDArray | None:
        r"""
        Prior second moment \\( \Lambda = \int \mathrm{d}\boldsymbol{\theta}\, p(\boldsymbol{\theta}) \boldsymbol{\theta} \boldsymbol{\theta}^T \\) (if set)
        """
        return self._matrix_λ

    @prior_second_moment.setter
    def prior_second_moment(self, value: np.typing.NDArray | None):
        if value is not None:
            value = np.asarray(value)
        self._matrix_λ = value

    @prior_second_moment.deleter
    def prior_second_moment(self):
        self._matrix_λ = None


class ScalarBound(BoundMSL):
    _scalar_pseudo_gain: float
    _weight_matrix: np.typing.NDArray

    def __init__(
        self,
        rho0: np.typing.ArrayLike,
        rho1s: Sequence[np.typing.ArrayLike],
        scalar_pseudo_gain: float,
        weight_matrix: np.typing.ArrayLike,
        prior_second_moment: np.typing.ArrayLike | None = None,
        weighted_prior_second_moment: float | None = None,
    ):
        self._scalar_pseudo_gain = scalar_pseudo_gain
        BoundMSL.__init__(
            self,
            rho0=rho0,
            rho1s=rho1s,
            weight_matrix=weight_matrix,
            prior_second_moment=prior_second_moment,
            weighted_prior_second_moment=weighted_prior_second_moment,
        )

    @property
    def weight_matrix(self) -> np.typing.NDArray:
        r"""Weight matrix"""
        return self._weight_matrix

    @property
    def scalar_pseudo_gain(self) -> float:
        r"""
        Pseudo-gain (\\( \lambda - \mathcal{L}_{\mathrm{B}} \\)) of the coresponding bound
        """
        return self._scalar_pseudo_gain

    def scalar_bound(
        self,
        prior_second_moment: np.typing.ArrayLike | None = None,
        weighted_prior_second_moment: float | None = None,
    ) -> float:
        r"""
        Scalar bound \\( \mathcal{L}_{\mathrm{B}} \\)

        Parameters
        ----------
        prior_second_moment : ndarray, optional
            Prior second moment \\( \Lambda = \int \mathrm{d}\boldsymbol{\theta}\, p(\boldsymbol{\theta}) \boldsymbol{\theta} \boldsymbol{\theta}^T \\)
        weighted_prior_second_moment : float, optional
            Weighted prior second moment (\\( \lambda \\))

        Returns
        -------
        scalar_bound : float
            The scalar bound

        Raises
        ------
        ValueError
            If `weighted_prior_second_moment` is not provided or calculable from other arguments or the Bound object itself
        """
        if weighted_prior_second_moment is None:
            weighted_prior_second_moment = self.weighted_prior_second_moment(
                prior_second_moment=prior_second_moment
            )
            if weighted_prior_second_moment is None:
                raise ValueError(
                    "scalar_bound cannot be calculated without a (calculable) weighted_prior_second_moment"
                )
        return weighted_prior_second_moment - self.scalar_pseudo_gain

    def weighted_prior_second_moment(
        self,
        prior_second_moment: np.typing.ArrayLike | None = None,
    ) -> float | None:
        r"""
        Scalar prior second moment \\( \int \mathrm{d}\boldsymbol{\theta}\, p(\boldsymbol{\theta}) \boldsymbol{\theta}^T W \boldsymbol{\theta} \\) (if calculable from `prior_second_moment` and `weight_matrix` (preferable) or directly set)

        Parameters
        ----------
        prior_second_moment : ndarray, optional
            Prior second moment (\\( \Lambda \\))

        Returns
        -------
        weighted_prior_second_moment : float
            Weighted prior second moment (\\( \lambda \\))
        """
        match (
            self.weight_matrix,
            prior_second_moment,
            self.prior_second_moment,
            self._scalar_λ,
        ):
            case (None, _, _, λ) | (_, None, None, λ):
                return λ
            case (_, _, None, λ) | (_, None, _, λ) if λ is not None:
                return λ
            case (w, None, m, _) | (w, m, _, _):
                m = np.asarray(m)
                w = np.asarray(w)
                return np.trace(w @ m)
            case _:
                return None


class ConvexBound(ScalarBound):
    _cvxpy_problem: cvxpy.Problem

    def __init__(
        self,
        rho0: np.typing.ArrayLike,
        rho1s: Sequence[np.typing.ArrayLike],
        scalar_pseudo_gain: float,
        cvxpy_problem: cvxpy.Problem,
        weight_matrix: np.typing.ArrayLike,
        prior_second_moment: np.typing.ArrayLike | None = None,
        weighted_prior_second_moment: float | None = None,
    ):
        self._cvxpy_problem = cvxpy_problem
        ScalarBound.__init__(
            self,
            rho0=rho0,
            rho1s=rho1s,
            scalar_pseudo_gain=scalar_pseudo_gain,
            weight_matrix=weight_matrix,
            prior_second_moment=prior_second_moment,
            weighted_prior_second_moment=weighted_prior_second_moment,
        )

    @property
    def cvx_problem(self) -> cvxpy.Problem:
        """The cvxpy problem solved to arrive at the bound"""
        return self._cvxpy_problem


class MatrixBound(BoundMSL):
    _matrix_pseudo_gain: np.typing.NDArray

    def __init__(
        self,
        rho0: np.typing.ArrayLike,
        rho1s: Sequence[np.typing.ArrayLike],
        matrix_pseudo_gain: np.typing.ArrayLike,
        weight_matrix: np.typing.ArrayLike | None = None,
        prior_second_moment: np.typing.ArrayLike | None = None,
        weighted_prior_second_moment: float | None = None,
    ):
        matrix_pseudo_gain = np.asarray(matrix_pseudo_gain)
        if np.isreal(matrix_pseudo_gain).all():
            matrix_pseudo_gain = np.real(matrix_pseudo_gain)
        self._matrix_pseudo_gain = matrix_pseudo_gain
        BoundMSL.__init__(
            self,
            rho0=rho0,
            rho1s=rho1s,
            weight_matrix=weight_matrix,
            prior_second_moment=prior_second_moment,
            weighted_prior_second_moment=weighted_prior_second_moment,
        )

    def matrix_pseudo_gain(self) -> np.typing.NDArray:
        r"""
        Pseudo-gain (\\( \Lambda - \mathcal{K}_{\mathrm{B}} \\)) of the coresponding matrix bound

        Returns
        -------
        pseudo_gain : ndarray
            The pseudo-gain of the bound
        """
        return self._matrix_pseudo_gain

    def matrix_bound(
        self, prior_second_moment: np.typing.ArrayLike | None = None
    ) -> np.typing.NDArray:
        r"""
        The matrix bound \\( \mathcal{K}_{\mathrm{B}} \\)

        Parameters
        ----------
        prior_second_moment : ndarray, optional
            Prior second moment \\( \Lambda = \int \mathrm{d}\boldsymbol{\theta}\, p(\boldsymbol{\theta}) \boldsymbol{\theta} \boldsymbol{\theta}^T \\)

        Returns
        -------
        matrix_bound : ndarray
            The matrix bound

        Raises
        ------
        ValueError
            If the prior_second_moment was not provided on creation of the bound or at calling of the bound
        """
        match (self.prior_second_moment, prior_second_moment):
            case (None, None):
                raise ValueError(
                    "MatrixBound.matrix_bound requires prior_second_moment to be set in the bound or passed as an argument"
                )
            case (Λ, None) | (_, Λ):
                prior_second_moment = np.asarray(Λ)
        return prior_second_moment - self.matrix_pseudo_gain()

    def scalar_pseudo_gain(self, weight_matrix: np.typing.ArrayLike | None = None) -> float:
        r"""
        Scalar pseudo-gain \\( \lambda - \mathcal{L}_{\mathrm{B}} \\)

        Parameters
        ----------
        weight_matrix : ndarray, optional
            Weight matrix, defaults to identity if not specified as argument or in the Bound object

        Returns
        -------
        pseudo_gain : float
            The pseudo-gain
        """
        matrix_pseudo_gain = self.matrix_pseudo_gain()

        if weight_matrix is None:
            weight_matrix = self.weight_matrix

        if np.isreal(matrix_pseudo_gain).all():
            return np.trace(weight_matrix @ matrix_pseudo_gain)

        sqrt_weight = sqrtm(weight_matrix)
        return np.trace(weight_matrix @ np.real(matrix_pseudo_gain)) - np.sum(
            np.abs(np.linalg.eigvals(sqrt_weight @ np.imag(matrix_pseudo_gain) @ sqrt_weight))
        )

    def scalar_bound(
        self,
        weight_matrix: np.typing.ArrayLike | None = None,
        prior_second_moment: np.typing.ArrayLike | None = None,
        weighted_prior_second_moment: float | None = None,
    ) -> float:
        r"""
        Scalar bound \\( \mathcal{L}_{\mathrm{B}} \\)

        Parameters
        ----------
        weight_matrix : ndarray, optional
            Weight matrix, defaults to identity if not specified as argument or in the Bound object
        prior_second_moment : ndarray, optional
            Prior second moment \\( \Lambda = \int \mathrm{d}\boldsymbol{\theta}\, p(\boldsymbol{\theta}) \boldsymbol{\theta} \boldsymbol{\theta}^T \\)
        weighted_prior_second_moment : float, optional
            Weighted prior second moment (\\( \lambda \\))

        Returns
        -------
        scalar_bound : float
            The scalar bound

        Raises
        ------
        ValueError
            If `weighted_prior_second_moment` is not provided or calculable from other arguments or the Bound object itself
        """
        if weighted_prior_second_moment is None:
            weighted_prior_second_moment = self.weighted_prior_second_moment(
                weight_matrix=weight_matrix, prior_second_moment=prior_second_moment
            )
            if weighted_prior_second_moment is None:
                raise ValueError(
                    "scalar_bound cannot be calculated without a (calculable) weighted_prior_second_moment"
                )
        return weighted_prior_second_moment - self.scalar_pseudo_gain(weight_matrix=weight_matrix)

    def weighted_prior_second_moment(
        self,
        weight_matrix: np.typing.ArrayLike | None = None,
        prior_second_moment: np.typing.ArrayLike | None = None,
    ) -> float | None:
        r"""
        Scalar prior second moment \\( \int \mathrm{d}\boldsymbol{\theta}\, p(\boldsymbol{\theta}) \boldsymbol{\theta}^T W \boldsymbol{\theta} \\) (if calculable from `prior_second_moment` and `weight_matrix` (preferable) or directly set)

        Parameters
        ----------
        weight_matrix : ndarray, optional
            Weight matrix
        prior_second_moment : ndarray, optional
            Prior second moment (\\( \Lambda \\))

        Returns
        -------
        weighted_prior_second_moment : float, None
            Weighted prior second moment (\\( \lambda \\)), where calculable
        """
        match (
            weight_matrix,
            self.weight_matrix,
            prior_second_moment,
            self.prior_second_moment,
            self._scalar_λ,
        ):
            case (None, None, _, _, λ) | (_, _, None, None, λ):
                return λ
            case (None, _, _, None, λ) | (_, None, None, _, λ) if λ is not None:
                return λ
            case (None, w, None, m, _) | (None, w, m, _, _) | (w, _, None, m, _) | (w, _, m, _, _):
                m = np.asarray(m)
                w = np.asarray(w)
                return np.trace(w @ m)
            case _:
                return None


class PMBound(MatrixBound):
    _posterior_mean_operators: list[np.typing.NDArray]

    def __init__(
        self,
        rho0: np.typing.ArrayLike,
        rho1s: Sequence[np.typing.ArrayLike],
        pmos: Sequence[np.typing.ArrayLike],
        weight_matrix: np.typing.ArrayLike | None = None,
        prior_second_moment: np.typing.ArrayLike | None = None,
        weighted_prior_second_moment: float | None = None,
    ):
        BoundMSL.__init__(
            self,
            rho0=rho0,
            rho1s=rho1s,
            weight_matrix=weight_matrix,
            prior_second_moment=prior_second_moment,
            weighted_prior_second_moment=weighted_prior_second_moment,
        )
        self._posterior_mean_operators = [np.asarray(pmo) for pmo in pmos]

    @property
    def posterior_mean_operators(self) -> list[np.typing.NDArray]:
        return self._posterior_mean_operators

    @cached_property
    def _matrix_pseudo_gain(self) -> np.typing.NDArray:
        matrix_pseudo_gain = np.real(
            np.einsum("jmn,knm->jk", self.rho1s, self.posterior_mean_operators)
        )
        if np.isreal(matrix_pseudo_gain).all():
            matrix_pseudo_gain = np.real(matrix_pseudo_gain)
        return matrix_pseudo_gain


class RealPMBound(PMBound):
    # Forced real matrix bounds to account for complex numerical noise
    @property
    def posterior_mean_operators(self) -> list[np.typing.NDArray]:
        return self._posterior_mean_operators

    @cached_property
    def _matrix_pseudo_gain(self) -> np.typing.NDArray:
        return np.real(np.einsum("jmn,knm->jk", self.rho1s, self.posterior_mean_operators))
