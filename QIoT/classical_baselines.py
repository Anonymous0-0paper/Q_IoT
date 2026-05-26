"""
classical_baselines.py — Classical optimisation solvers for the IoT task
assignment problem used as baselines against Q-IoT.

All solvers return (solution_dict) with keys:
  assignment   : np.ndarray (n_tasks, n_resources)  binary assignment
  objective    : float                               cost value (lower is better)
  runtime_ms   : float                               wall-clock ms
"""

import time
import random
import numpy as np
from config import RANDOM_SEED

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _objective(assignment: np.ndarray, cost_matrix: np.ndarray) -> float:
    """Compute assignment cost Σ c_{ik} x_{ik}."""
    return float(np.sum(assignment * cost_matrix))


def _random_onehot(n_tasks: int, n_resources: int, rng) -> np.ndarray:
    """Generate a random one-hot assignment matrix."""
    X = np.zeros((n_tasks, n_resources), dtype=float)
    for i in range(n_tasks):
        k = rng.integers(0, n_resources)
        X[i, k] = 1.0
    return X


def _neighbour(X: np.ndarray, rng) -> np.ndarray:
    """Return a neighbour by reassigning one random task."""
    X_new = X.copy()
    n_tasks, n_res = X.shape
    t = int(rng.integers(0, n_tasks))
    k_new = int(rng.integers(0, n_res))
    X_new[t, :] = 0
    X_new[t, k_new] = 1
    return X_new


# ─────────────────────────────────────────────────────────────────────────────
# 1. Greedy heuristic
# ─────────────────────────────────────────────────────────────────────────────

def greedy(cost_matrix: np.ndarray, n_tasks: int, n_resources: int) -> dict:
    """
    Greedy heuristic: assign each task to the cheapest resource.

    Parameters
    ----------
    cost_matrix : np.ndarray, shape (n_tasks, n_resources)
    n_tasks : int
    n_resources : int

    Returns
    -------
    dict
        Keys: 'assignment', 'objective', 'runtime_ms'.
    """
    t0 = time.perf_counter()
    X = np.zeros((n_tasks, n_resources), dtype=float)
    for i in range(n_tasks):
        k = int(np.argmin(cost_matrix[i]))
        X[i, k] = 1.0
    obj = _objective(X, cost_matrix)
    return {"assignment": X, "objective": obj,
            "runtime_ms": (time.perf_counter() - t0) * 1000}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Simulated Annealing
# ─────────────────────────────────────────────────────────────────────────────

def simulated_annealing(
    cost_matrix: np.ndarray,
    n_tasks: int,
    n_resources: int,
    T_init: float = 10.0,
    T_min: float = 1e-4,
    alpha: float = 0.98,
    max_steps: int = 5_000,
    seed: int = RANDOM_SEED,
) -> dict:
    """
    Simulated Annealing for task assignment.

    Parameters
    ----------
    cost_matrix : np.ndarray, shape (n_tasks, n_resources)
    n_tasks : int
    n_resources : int
    T_init : float
        Initial temperature.
    T_min : float
        Stopping temperature.
    alpha : float
        Geometric cooling rate.
    max_steps : int
        Maximum SA steps.
    seed : int

    Returns
    -------
    dict
        Keys: 'assignment', 'objective', 'runtime_ms'.
    """
    t0  = time.perf_counter()
    rng = np.random.default_rng(seed)

    X_cur = _random_onehot(n_tasks, n_resources, rng)
    E_cur = _objective(X_cur, cost_matrix)
    X_best, E_best = X_cur.copy(), E_cur

    T = T_init
    step = 0
    while T > T_min and step < max_steps:
        X_new = _neighbour(X_cur, rng)
        E_new = _objective(X_new, cost_matrix)
        dE = E_new - E_cur
        if dE < 0 or rng.random() < np.exp(-dE / T):
            X_cur, E_cur = X_new, E_new
            if E_cur < E_best:
                X_best, E_best = X_cur.copy(), E_cur
        T *= alpha
        step += 1

    return {"assignment": X_best, "objective": E_best,
            "runtime_ms": (time.perf_counter() - t0) * 1000}


# ─────────────────────────────────────────────────────────────────────────────
# 3. Ant Colony Optimisation
# ─────────────────────────────────────────────────────────────────────────────

def ant_colony_optimization(
    cost_matrix: np.ndarray,
    n_tasks: int,
    n_resources: int,
    n_ants: int = 20,
    n_iter: int = 100,
    alpha_aco: float = 1.0,
    beta_aco: float = 2.0,
    rho: float = 0.1,
    Q: float = 1.0,
    seed: int = RANDOM_SEED,
) -> dict:
    """
    Ant Colony Optimisation for task-to-resource assignment.

    Parameters
    ----------
    cost_matrix : np.ndarray, shape (n_tasks, n_resources)
    n_tasks, n_resources : int
    n_ants : int
        Colony size.
    n_iter : int
        Number of ACO iterations.
    alpha_aco : float
        Pheromone exponent.
    beta_aco : float
        Heuristic (inverse cost) exponent.
    rho : float
        Evaporation rate.
    Q : float
        Pheromone deposit constant.
    seed : int

    Returns
    -------
    dict
        Keys: 'assignment', 'objective', 'runtime_ms'.
    """
    t0  = time.perf_counter()
    rng = np.random.default_rng(seed)

    # Pheromone matrix τ[i, k]
    tau   = np.ones((n_tasks, n_resources))
    # Heuristic η[i, k] = 1 / (c_{ik} + ε)
    eta   = 1.0 / (cost_matrix + 1e-6)

    X_best = None
    E_best = np.inf

    for _ in range(n_iter):
        ant_solutions = []
        ant_costs     = []

        for _ in range(n_ants):
            X = np.zeros((n_tasks, n_resources), dtype=float)
            for i in range(n_tasks):
                prob = (tau[i] ** alpha_aco) * (eta[i] ** beta_aco)
                prob /= prob.sum()
                k = int(rng.choice(n_resources, p=prob))
                X[i, k] = 1.0
            cost = _objective(X, cost_matrix)
            ant_solutions.append(X)
            ant_costs.append(cost)
            if cost < E_best:
                E_best = cost
                X_best = X.copy()

        # Pheromone update
        tau *= (1.0 - rho)
        for X_ant, c_ant in zip(ant_solutions, ant_costs):
            tau += Q / (c_ant + 1e-6) * X_ant

    return {"assignment": X_best, "objective": E_best,
            "runtime_ms": (time.perf_counter() - t0) * 1000}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Particle Swarm Optimisation
# ─────────────────────────────────────────────────────────────────────────────

def particle_swarm_optimization(
    cost_matrix: np.ndarray,
    n_tasks: int,
    n_resources: int,
    n_particles: int = 30,
    n_iter: int = 100,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    seed: int = RANDOM_SEED,
) -> dict:
    """
    Particle Swarm Optimisation for task assignment (discrete PSO via sigmoid).

    Each particle is a continuous vector in R^{n_tasks * n_resources};
    assignment is decoded by argmax per task.

    Parameters
    ----------
    cost_matrix : np.ndarray, shape (n_tasks, n_resources)
    n_tasks, n_resources : int
    n_particles : int
    n_iter : int
    w : float
        Inertia weight.
    c1, c2 : float
        Cognitive and social coefficients.
    seed : int

    Returns
    -------
    dict
        Keys: 'assignment', 'objective', 'runtime_ms'.
    """
    t0  = time.perf_counter()
    rng = np.random.default_rng(seed)
    dim = n_tasks * n_resources

    # Initialise positions and velocities
    pos = rng.uniform(0, 1, (n_particles, dim))
    vel = rng.uniform(-0.5, 0.5, (n_particles, dim))

    def decode(p: np.ndarray) -> np.ndarray:
        """Decode continuous particle to binary assignment via softmax-argmax."""
        X = np.zeros((n_tasks, n_resources), dtype=float)
        for i in range(n_tasks):
            row = p[i * n_resources:(i + 1) * n_resources]
            k   = int(np.argmax(row))
            X[i, k] = 1.0
        return X

    def eval_particle(p: np.ndarray) -> float:
        return _objective(decode(p), cost_matrix)

    # Personal best
    pbest_pos = pos.copy()
    pbest_val = np.array([eval_particle(pos[i]) for i in range(n_particles)])

    gbest_idx = int(np.argmin(pbest_val))
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]

    for _ in range(n_iter):
        r1 = rng.random((n_particles, dim))
        r2 = rng.random((n_particles, dim))
        vel = (w * vel
               + c1 * r1 * (pbest_pos - pos)
               + c2 * r2 * (gbest_pos - pos))
        pos = np.clip(pos + vel, 0, 1)

        vals = np.array([eval_particle(pos[i]) for i in range(n_particles)])
        improved = vals < pbest_val
        pbest_pos[improved] = pos[improved].copy()
        pbest_val[improved] = vals[improved]

        gbest_idx = int(np.argmin(pbest_val))
        if pbest_val[gbest_idx] < gbest_val:
            gbest_val = pbest_val[gbest_idx]
            gbest_pos = pbest_pos[gbest_idx].copy()

    X_best = decode(gbest_pos)
    return {"assignment": X_best, "objective": gbest_val,
            "runtime_ms": (time.perf_counter() - t0) * 1000}
