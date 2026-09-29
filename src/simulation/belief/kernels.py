from abc import ABC, abstractmethod

import numpy as np


class Kernel(ABC):
    @abstractmethod
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray: ...
    @abstractmethod
    def diag(self, X: np.ndarray) -> np.ndarray: ...


class RBFKernel(Kernel):
    r"""Squared-exponential kernel. `length_scales` length = input dimension."""

    def __init__(self, sigma0: float, length_scales: list[float]):
        self.sigma0_sq = sigma0**2
        self.length_scales = np.asarray(length_scales, float)

    def __call__(self, X1, X2):
        X1s = np.atleast_2d(X1) / self.length_scales
        X2s = np.atleast_2d(X2) / self.length_scales
        sq = np.sum(X1s**2, 1)[:, None] + np.sum(X2s**2, 1) - 2 * X1s @ X2s.T
        return self.sigma0_sq * np.exp(-0.5 * np.maximum(sq, 0.0))

    def diag(self, X):
        return np.full(np.atleast_2d(X).shape[0], self.sigma0_sq)


class LaplacianKernel(Kernel):
    r"""Matern-1/2 kernel (Ilut-Vachev original). `length_scales` length = input dim."""

    def __init__(self, sigma0: float, length_scales: list[float]):
        self.sigma0_sq = sigma0**2
        self.length_scales = np.asarray(length_scales, float)

    def __call__(self, X1, X2):
        X1s = np.atleast_2d(X1) / self.length_scales
        X2s = np.atleast_2d(X2) / self.length_scales
        sq = np.sum(X1s**2, 1)[:, None] + np.sum(X2s**2, 1) - 2 * X1s @ X2s.T
        return self.sigma0_sq * np.exp(-np.sqrt(np.maximum(sq, 0.0)))

    def diag(self, X):
        return np.full(np.atleast_2d(X).shape[0], self.sigma0_sq)
