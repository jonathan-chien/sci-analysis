from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence, runtime_checkable

import numpy as np 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .linear_model import OLS


class PCA(BaseEstimator):
    """ 
    Performs PCA on arbitrarily batched matrix shaped data, i.e. on input of 
    shape (..., m, n). Also allows plug in of arbitrary callback for estimating
    number of components.
    """

    def __init__(
        self, 
        n_components: int | NComponentsSelector | None = None,
        fill_value: float = np.nan
    ):
        self.n_components = n_components
        self.fill_value = fill_value
        
    def fit(
        self,
        x: np.ndarray,
        y=None
    ):
        if x.ndim < 2:
            raise ValueError(
                f"Input data array x must have ndim at least 2, but got x.ndim={x.ndim}."
            )
        if not np.all(np.isfinite(x)):
            raise ValueError(
                "Non-finite values detected in input array x."
            )
        
        self.batch_shape_ = x.shape[:-2]
        self.mean_ = np.mean(x, axis=-2, keepdims=True)
        self.dof_ = x.shape[-2] - 1

        x_centered = x - self.mean_
        u, s, vh = np.linalg.svd(x_centered, full_matrices=False)

        self.u_ = u
        self.s_ = s
        self.v_ = vh.mT

        self._set_n_components(x)

        return self

    def _set_n_components(self, x: np.ndarray):
        """ 
        """
        check_is_fitted(self)

        if (
            isinstance(self.n_components, (int, np.integer)) 
            and not isinstance(self.n_components, bool)
        ):
            if self.n_components < 0:
                raise ValueError(
                    "If self.n_components is an int, it must be positive, " 
                    f"but got {self.n_components}."
                )
            n_components = self.n_components
        elif isinstance(self.n_components, NComponentsSelector):
            pca_environment = PCAEnvironment(pca=self, x=x) # Pass in un-centered data, since some cross-val routines need to center within folds
            n_components = self.n_components.fit_from_pca_environment(pca_environment).n_components_
        elif self.n_components is None:
            n_components = (
                x.shape[-1] if x.shape[-2] > x.shape[-1] 
                else x.shape[-2] - 1
            )
        else:
            raise ValueError(
                "self.n_components must be a positive int, an NComponentsSelector object, or "
                f"None, but got {self.n_components}."
            )
        
        self.n_components_ = np.broadcast_to(n_components, self.batch_shape_)

        if np.any(self.n_components_ < 0): 
            raise RuntimeError(
                "At least one estimated value for the number of components is invalid. "
                "Values must be non-negative integers, but got a negative number."
            )

        if np.any(self.n_components_ > np.min(x.shape[-2:])):
            raise RuntimeError(
                "At least one estimated value for the number of components is "
                "invalid. Values cannot be greater than min(m, n) for data "
                "consisting of m observations and n variables, but got "
                f"{self.n_components_[self.n_components_ > np.min(x.shape[-2:])]}."
            )
        
    def transform(self, x):
        """ 
        """
        check_is_fitted(self)

        x = x - self.mean_

        max_n_components = int(self.n_components_.max())

        return np.where(
            self._get_n_components_mask(max_n_components),
            x @ self.v_[..., :max_n_components],
            self.fill_value
        )

    def project(self, x):
        """ 
        """
        check_is_fitted(self)
        max_n_components = int(self.n_components_.max())
        mask = self._get_n_components_mask(max_n_components)
        return (
            x 
            @ self.v_[..., :max_n_components] 
            * mask
            @ self.v_[..., :max_n_components].mT
        )

    def reconstruct(self, x):
        """ 
        """
        return self.project(x - self.mean_) + self.mean_
       
    def fit_transform(
        self,
        x,
        y=None
    ):
        """ 
        Overwrites sklearn's method for direct computation of PCs via US,
        skipping dense multiplication XV in self.transform.
        """
        self.fit(x)

        max_n_components = int(self.n_components_.max())

        return np.where(
            self._get_n_components_mask(max_n_components),
            self.u_[..., :max_n_components] * self.s_[..., np.newaxis, :max_n_components],
            self.fill_value
        ) 

    def _get_n_components_mask(self, max_n_components: int):
        """ 
        """
        if not hasattr(self, 'n_components_'):
            raise RuntimeError(
                "Attempting to access self.n_components_, but this attribute " 
                "has not been set yet."
            )
        
        return (
            np.arange(max_n_components) 
            < self.n_components_[..., np.newaxis]
        )[..., np.newaxis, :]
    
    def compute_eigenspectrum(self):
        """ 
        """
        check_is_fitted(self)
        return self.s_**2 / self.dof_
    
    def compute_pve(self):
        """ 
        """
        check_is_fitted(self)
        spectrum = self.compute_eigenspectrum()
        return np.cumsum(spectrum, axis=-1) / np.sum(spectrum, axis=-1, keepdims=True)


@dataclass
class PCAEnvironment: 
    """ 
    Common, controlled interface for calling NComponentSelector objects.
    """
    pca: PCA
    x: np.ndarray


@runtime_checkable
class NComponentsSelector(Protocol):
    """ 
    Callback that can be used to select number of components in PCA based on
    input data x and a PCA fit to x.
    """
    def fit_from_pca_environment(self, pca_environment: PCAEnvironment) -> NComponentsSelector: ...


class PCAPVE(BaseEstimator):
    """ 
    Select number of components based on minimum number needed to achieve 
    target proportion of variance explained.
    """
    def __init__(self, target_pve: float):
        self.target_pve = target_pve

    def fit_from_pca_environment(self, pca_environment: PCAEnvironment):
        """ 
        """
        target_pve = np.asarray(self.target_pve)
        invalid = (target_pve <= 0) | (target_pve > 1)
        if np.any(invalid):
            raise ValueError(
                "Invalid values detected in self.target_pve. Values must be "
                f"in (0, 1] but got {target_pve[invalid]}"
            )
        
        pve = pca_environment.pca.compute_pve()
        self.n_components_ = np.minimum(
            np.sum(pve < target_pve[..., np.newaxis], axis=-1) + 1,
            pve.shape[-1]
        )
        # self.n_components_ = np.argmax(pve >= self.target_pve, axis=-1) + 1
        return self


def _raise_value_error(msg):
    raise ValueError(msg)


class BiCrossValSVD(BaseEstimator):
    """ 
    Select number of SVD components based on cross-validation as described in 
    Owen & Perry 2009 (using Type I residuals).
    """
    def __init__(
        self, 
        n_components_to_test: Sequence[int], 
        n_row_blocks: int, 
        n_col_blocks: int, 
        n_permutations: int = 1,
        center_blocks: bool = False,
        store_perm_errors: bool = False,
        perm_reduction: Callable = np.median,
        seed: int | np.random.SeedSequence | np.random.Generator | None = None,
        bhv_if_block_too_small: Callable = _raise_value_error
    ):
        """ 
        """
        self.n_components_to_test = n_components_to_test
        self.n_row_blocks = n_row_blocks
        self.n_col_blocks = n_col_blocks
        self.n_permutations = n_permutations
        self.center_blocks = center_blocks
        self.store_perm_errors = store_perm_errors
        self.perm_reduction = perm_reduction
        self.seed = seed
        self.bhv_if_block_too_small = bhv_if_block_too_small

    def fit(self, x: np.ndarray, y=None):
        """ 
        """
        # Validate input data.
        if x.ndim < 2:
            raise ValueError(
                f"Input data array x must have ndim at least 2, but got x.ndim={x.ndim}."
            )
        if not np.all(np.isfinite(x)):
            raise ValueError(
                "Non-finite values detected in input array x."
            )

        # Validate number of components to test.
        n_components_to_test_arr = np.asarray(self.n_components_to_test)
        if n_components_to_test_arr.size == 0:
            raise ValueError("self.n_components_to_test is empty.")
        elif (
            not np.issubdtype(n_components_to_test_arr.dtype, np.integer) 
            or np.any(n_components_to_test_arr < 0)
        ):
            raise ValueError(
                "self.n_components_to_test must contain non-negative integers " 
                f"but got {n_components_to_test_arr}."
            )

        # Parse input data shape and validate row and col block numbers.
        self.n_rows_, self.n_cols_ = x.shape[-2:]
        if self.n_row_blocks > self.n_rows_:
            raise ValueError(
                "self.n_row_blocks should be at most self.n_rows_, but got "
                f"{self.n_row_blocks} and {self.n_rows_}, respectively."
            )
        if self.n_col_blocks > self.n_cols_:
            raise ValueError(
                "self.n_col_blocks should be at most self.n_cols_, but got "
                f"{self.n_col_blocks} and {self.n_cols_}, respectively."
            )
        if self.n_row_blocks < 2:
            raise ValueError(
                f"self.n_row_blocks must be at least 2 but got {self.n_row_blocks}."
            )
        if self.n_col_blocks < 2:
            raise ValueError(
                f"self.n_col_blocks must be at least 2 but got {self.n_col_blocks}."
            )

        # Get edge indices for row and column blocks.
        row_block_edges = np.linspace(0, self.n_rows_, self.n_row_blocks+1).astype(int)
        col_block_edges = np.linspace(0, self.n_cols_, self.n_col_blocks+1).astype(int)

        # Ensure that none of the numbers of components to test exceeds rank of smallest D.
        smallest_d_row_size = self.n_rows_ - np.max(np.diff(row_block_edges, n=1))
        smallest_d_col_size = self.n_cols_ - np.max(np.diff(col_block_edges, n=1))
        max_n_components = np.max(self.n_components_to_test)
        if max_n_components > smallest_d_row_size:
            msg = (
                f"Row block size of smallest D block is {smallest_d_row_size}, "
                f"but largest number of components to test is {max_n_components}, "
                "which thus exceeds the rank of D."
            )
            self.bhv_if_block_too_small(msg)
        if max_n_components > smallest_d_col_size:
            msg = (
                f"Col block size of smallest D block is {smallest_d_col_size}, "
                f"but largest number of components to test is {max_n_components}, "
                "which thus exceeds the rank of D."
            )
            self.bhv_if_block_too_small(msg)
        
        # Get permutation of row and column indices, separately.
        rng = np.random.default_rng(self.seed)
        perm_row_ind = rng.permuted(np.tile(np.arange(self.n_rows_), (self.n_permutations, 1)), axis=1)
        perm_col_ind = rng.permuted(np.tile(np.arange(self.n_cols_), (self.n_permutations, 1)), axis=1)

        # Compute error on held out block across cross-validation folds.
        ssr = np.empty(
            x.shape[:-2]
            + (self.n_permutations,)
            + (len(self.n_components_to_test),)
            + (self.n_row_blocks, self.n_col_blocks)
        )
        for i_row_block in range(self.n_row_blocks):
            for i_col_block in range(self.n_col_blocks):
                # Get row and col indices for fitting block and held out block.
                (
                    row_block_ind, 
                    col_block_ind,
                    row_block_ind_complement,
                    col_block_ind_complement
                ) = self._get_block_ind(
                    perm_row_ind=perm_row_ind,
                    perm_col_ind=perm_col_ind,
                    row_block_head=row_block_edges[i_row_block],
                    row_block_tail=row_block_edges[i_row_block+1],
                    col_block_head=col_block_edges[i_col_block],
                    col_block_tail=col_block_edges[i_col_block+1]
                )

                # Compute ssr for current fold across batch, permutations, and
                # number of components to test.
                ssr[..., i_row_block, i_col_block] = self._compute_ssr(
                    x=x,
                    n_components_to_test=self.n_components_to_test,
                    row_block_ind=row_block_ind,
                    col_block_ind=col_block_ind,
                    row_block_ind_complement=row_block_ind_complement,
                    col_block_ind_complement=col_block_ind_complement,
                    center_blocks=self.center_blocks
                )

        # Reduce across blocks (optionally saving) to get errors across permutations.
        perm_errors = np.sum(ssr, axis=(-2, -1))
        if self.store_perm_errors:
            self.perm_errors_ = perm_errors

        # Reduce across permutations to get final errors, which are stored.
        errors = self.perm_reduction(perm_errors, axis=-2)
        self.errors_ = errors

        self.n_components_ = np.asarray(self.n_components_to_test)[np.argmin(errors, axis=-1)]

        return self

    @staticmethod
    def _get_block_ind(
        perm_row_ind, 
        perm_col_ind, 
        row_block_head: int,
        row_block_tail: int,
        col_block_head: int,
        col_block_tail: int
    ):
        # row_block_ind has shape n_permutations x n_block_size.
        row_block_ind = perm_row_ind[:, row_block_head:row_block_tail]
        col_block_ind = perm_col_ind[:, col_block_head:col_block_tail]

        # Get row and column indices not in block A.
        row_block_ind_complement = np.concatenate(
            (
                perm_row_ind[:, :row_block_head],
                perm_row_ind[:, row_block_tail:]
            ),
            axis=1
        )
        col_block_ind_complement = np.concatenate(
            (
                perm_col_ind[:, :col_block_head],
                perm_col_ind[:, col_block_tail:]
            ),
            axis=1
        )

        return (
            row_block_ind, 
            col_block_ind, 
            row_block_ind_complement, 
            col_block_ind_complement
        )

    @staticmethod
    def _compute_ssr(
        x: np.ndarray, 
        n_components_to_test: Sequence[int],
        row_block_ind: np.ndarray, 
        col_block_ind: np.ndarray, 
        row_block_ind_complement: np.ndarray, 
        col_block_ind_complement: np.ndarray,
        center_blocks: bool
    ):
        
        # Get blocks A, B, C, and D.
        a = x[
            ..., 
            row_block_ind[:, :, np.newaxis], 
            col_block_ind[:, np.newaxis, :]
        ]
        b = x[
            ...,
            row_block_ind[:, :, np.newaxis],
            col_block_ind_complement[:, np.newaxis, :]
        ]
        c = x[
            ...,
            row_block_ind_complement[:, :, np.newaxis],
            col_block_ind[:, np.newaxis, :]
        ] 
        d = x[
            ...,
            row_block_ind_complement[:, :, np.newaxis],
            col_block_ind_complement[:, np.newaxis, :]
        ]

        if center_blocks:
            # Compute two sets of means, one for the columns that include A and
            # another for those that don't. Use only rows not shared with A.
            mean_test_cols = np.mean(c, axis=-2, keepdims=True)
            mean_fit_cols = np.mean(d, axis=-2, keepdims=True)
            a, b = a - mean_test_cols, b - mean_fit_cols
            c, d = c - mean_test_cols, d - mean_fit_cols

        # Compute pseudoinverse of D and residuals for each number of components.
        return np.sum(
            (
                a[..., np.newaxis, :, :] 
                - (
                    b[..., np.newaxis, :, :] 
                    @ BiCrossValSVD.truncated_pinv(
                        d, 
                        n_components=n_components_to_test, 
                        stack_axis=-3
                    ) 
                    @ c[..., np.newaxis, :, :]
                )
            )**2,
            axis=(-2, -1)
        )

    @staticmethod
    def truncated_pinv(
        x: np.ndarray, 
        n_components: Sequence[int], 
        rtol: float =1e-5, 
        stack_axis: int =-1
    ):
        """ 
        """
        u, s, vh = np.linalg.svd(x, full_matrices=False)

        s_inv = np.zeros(s.shape)
        nonzero_ind = s > rtol * s[..., :1]
        s_inv[nonzero_ind] = 1 / s[nonzero_ind]

        return np.stack(
            tuple(
                vh.mT[..., :k] 
                * s_inv[..., np.newaxis, :k] 
                @ u[..., :k].mT
                for k in n_components
            ),
            axis=stack_axis
        )


class BiCrossValPCA(BiCrossValSVD):
    """ 
    Subclasses BiCrossValSVD but with center_blocks=True, and adds a
    fit_from_pca_environment matching the NComponentsSelector protocol.
    """
    def __init__(
        self,
        n_components_to_test: Sequence[int], 
        n_row_blocks: int, 
        n_col_blocks: int, 
        n_permutations: int = 1,
        store_perm_errors: bool = False,
        perm_reduction: Callable = np.median,
        seed: int | np.random.SeedSequence | np.random.Generator | None = None,
        bhv_if_block_too_small: Callable = _raise_value_error
    ):
        super().__init__(
            n_components_to_test=n_components_to_test,
            n_row_blocks=n_row_blocks,
            n_col_blocks=n_col_blocks,
            n_permutations=n_permutations,
            center_blocks=True,
            store_perm_errors=store_perm_errors,
            perm_reduction=perm_reduction,
            seed=seed,
            bhv_if_block_too_small=bhv_if_block_too_small
        )

    def fit_from_pca_environment(self, pca_environment: PCAEnvironment):
        """ 
        """
        self.fit(pca_environment.x)
        return self

    
class TDR(TransformerMixin, BaseEstimator):

    def __init__(
        self,
        ols_args: dict | None = None,
        pca_args: dict | None = None,
    ):
        self.ols_args = ols_args
        self.pca_args = pca_args

    def fit(
        self,
        x: np.ndarray, # Neural data
        y: np.ndarray # Stimuli information
    ):
        """ 
        """
        self.ols = OLS(**(self.ols_args or {}))
        self.ols.fit(y, x) # Regress neural data on stimuli
        self.pca = PCA(**(self.pca_args or {}))
        self.pca.fit(x)

        # Project coefficients.
        self.coef_denoised_ = self.pca.project(
            self.ols.coef_[..., :-1, :] if self.ols.add_intercept
            else self.ols.coef_
        )

        # Orthogonalize via QR.
        self.coef_orthogonalized_, self.r_ = np.linalg.qr(
            self.coef_denoised_.mT, 
            mode='reduced'
        )

        return self

    def transform(self, x):
        """ 
        """
        check_is_fitted(self)
        return x @ self.coef_orthogonalized_
    