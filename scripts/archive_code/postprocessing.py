from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def evaluate_firm_beliefs(
    firm_agents: list,
    eval_k_grid: np.ndarray,
    delta: float,
    z: float = 1.0,
    b: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate expected policies, Q-values, and GP uncertainty for each agent.

    Parameters
    ----------
    firm_agents : list[ExperienceReasoningAgent]
        Agents with ``get_expected_action`` and ``.gp.predict``.
    eval_k_grid : np.ndarray
        Capital values to evaluate on.
    delta : float
        Depreciation rate (needed to compute i from k and k').
    z, b : float
        Productivity and debt at which to evaluate.

    Returns
    -------
    kp_arr : np.ndarray, shape (N, G)
        Expected next-period capital for each agent at each grid point.
    q_arr : np.ndarray, shape (N, G)
        GP posterior mean (Q-value) at the expected action.
    std_arr : np.ndarray, shape (N, G)
        GP posterior std at the expected action.
    """
    N = len(firm_agents)
    G = len(eval_k_grid)
    kp_arr = np.zeros((N, G))
    q_arr = np.zeros((N, G))
    std_arr = np.zeros((N, G))

    for j, agent in enumerate(firm_agents):
        for i_k, kv in enumerate(eval_k_grid):
            kp, _ = agent.get_expected_action(z=z, k=float(kv), b=b)
            kp_arr[j, i_k] = kp
            i_val = kp - (1.0 - delta) * kv
            q_mean, q_std = agent.gp.predict(
                np.array([[z, kv, i_val]]), return_std=True
            )
            q_arr[j, i_k] = q_mean[0]
            std_arr[j, i_k] = q_std[0]

    return kp_arr, q_arr, std_arr


def cluster_firm_policies(
    firm_agents: list,
    eval_k_grid: np.ndarray,
    z: float = 1.0,
    b: float = 0.0,
    n_clusters: int | None = None,
    max_clusters: int = 8,
    **kmeans_kwargs,
) -> tuple[np.ndarray, np.ndarray, int, KMeans]:
    """Extract expected policies from a list of GP-based agents and cluster them.

    Parameters
    ----------
    firm_agents : list[ExperienceReasoningAgent]
        Population of agents whose ``get_expected_action`` will be evaluated.
    eval_k_grid : np.ndarray
        Capital values at which to evaluate each firm's policy.
    z, b : float
        Productivity and debt at which policies are evaluated.
    n_clusters : int | None
        Fixed number of clusters.  If ``None``, the optimal count is chosen
        automatically via silhouette score over ``range(2, max_clusters+1)``.
    max_clusters : int
        Upper bound when searching for the optimal cluster count.
    **kmeans_kwargs
        Forwarded to ``sklearn.cluster.KMeans`` (e.g. ``random_state``,
        ``n_init``, ``max_iter``).

    Returns
    -------
    firm_policies : np.ndarray, shape (N_firms, len(eval_k_grid))
        Matrix of evaluated expected policies.
    cluster_labels : np.ndarray, shape (N_firms,)
        Integer cluster assignment for each firm.
    n_clusters : int
        Number of clusters used (useful when auto-selected).
    kmeans : KMeans
        Fitted KMeans object.
    """
    N = len(firm_agents)
    G = len(eval_k_grid)
    firm_policies = np.zeros((N, G))

    for j, agent in enumerate(firm_agents):
        for i_k, k_val in enumerate(eval_k_grid):
            kp, _ = agent.get_expected_action(z=z, k=float(k_val), b=b)
            firm_policies[j, i_k] = kp

    kmeans_kwargs.setdefault("random_state", 42)

    if n_clusters is not None:
        km = KMeans(n_clusters=n_clusters, **kmeans_kwargs)
        labels = km.fit_predict(firm_policies)
        return firm_policies, labels, n_clusters, km

    # Auto-select via silhouette score
    best_k, best_score = 2, -1.0
    best_km = None
    upper = min(max_clusters, N - 1) + 1
    for k in range(2, upper):
        km = KMeans(n_clusters=k, **kmeans_kwargs)
        labels = km.fit_predict(firm_policies)
        score = float(silhouette_score(firm_policies, labels))
        if score > best_score:
            best_k, best_score, best_km = k, score, km

    labels = best_km.predict(firm_policies)  # type: ignore[union-attr]
    return firm_policies, labels, best_k, best_km  # type: ignore[return-value]
