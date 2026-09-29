from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Menu:
    r"""A quadrature rule over the feasible action set at one state.

    Immutable object of action menu at a given state. Immutability allows safe sharing and caching.

    Row :math:`j` is a cell of the continuous action space, of Lebesgue
    measure :math:`\Delta_j`, action :math:`a_j` sits at its midpoint. A sum
    weighted by ``measures`` approximates an integral,

    .. math::
        \sum_j \Delta_j\, g(a_j) \approx \int g(a)\, \mathrm{d}a ,

    and :math:`L = \sum_j \Delta_j` is the Lebesgue measure of the region the menu covers.
    These are the weights the planner takes as its ``areas``.

    Args:
        state (tuple): The ``(z, omega)`` this menu was built at.
        actions (np.ndarray): Cell midpoints, shape ``(N, 2)``; rows ``(k', b')``.
        measures (np.ndarray): Cell measures :math:`\Delta_j`, shape ``(N,)``.
            Equal on a fixed product grid, row-dependent on the adaptive menu.
        points (np.ndarray): Menu features, shape ``(N, d)``; rows ``(z, omega, k', b')``.
            Used to query the GP belief. Built from featurized state and action information in the environment.
    """

    state: tuple[float, float]
    actions: np.ndarray
    measures: np.ndarray
    points: np.ndarray

    def __len__(self) -> int:
        return len(self.actions)

    @property
    def measure(self) -> float:
        r"""Total measure :math:`L = \sum_j \Delta_j` of the region the menu covers."""
        return float(np.sum(self.measures))
