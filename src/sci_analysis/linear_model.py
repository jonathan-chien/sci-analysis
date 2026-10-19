from abc import ABC, abstractmethod

import numpy as np 
from scipy import linalg
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


class LeastSquares(RegressorMixin, BaseEstimator, ABC):
    def __init__(self, add_intercept: bool = False):
        self.add_intercept = add_intercept

    @abstractmethod
    def fit(self, x, y):
        """ 
        """
        pass

    @staticmethod
    def _add_intercept(x: np.ndarray):
        return np.append(x, np.ones(x.shape[:-1] + (1,)), axis=-1)
            
    def predict(self, x: np.ndarray):
        check_is_fitted(self)
        return (
            self._add_intercept(x) @ self.coef_ if self.add_intercept
            else x @ self.coef_
        )
    

class OLS(LeastSquares):
    def __init__(self, add_intercept: bool = False, lstsq_args: dict | None = None):
        super().__init__(add_intercept=add_intercept)
        self.lstsq_args = lstsq_args

    def fit(self, x, y):
        """ 
        """
        if self.add_intercept:
            x = self._add_intercept(x)

        self.coef_ = linalg.lstsq(x, y, **(self.lstsq_args or {}))[0]

        return self


class GLS(LeastSquares):

    def __init__(self, add_intercept: bool = False):
        super().__init__(add_intercept=add_intercept)

    def fit(self, x, y, g):
        """ 
        g here should be inverse covariance. TODO: switch to supporting g as covariance instead.
        """
        if self.add_intercept:
            x = self._add_intercept(x)

        l = np.linalg.cholesky(g)
        q, r = np.linalg.qr(l.mT @ x)
        self.coef_ = linalg.solve_triangular(r, q.mT @ l.mT @ y)

        return self
    