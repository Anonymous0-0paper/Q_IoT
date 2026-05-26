"""
qaoa_circuit.py — QAOA circuit construction with cold-start and warm-start
initialization, plus adaptive shot scheduling for the Q-IoT framework.
"""

import numpy as np
import random
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from config import (
    QAOA_DEPTH, N_SHOTS_BASE, N_SHOTS_EXTRA, ADAPTIVE_LAM, RANDOM_SEED,
    EMA_ALPHA,
)

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _apply_cost_layer(qc: QuantumCircuit, gamma: float,
                      h_local: np.ndarray, J_coupling: np.ndarray) -> None:
    """Apply the cost unitary U_C(γ) = exp(-iγ H_C)."""
    n = qc.num_qubits
    for i in range(n):
        if abs(h_local[i]) > 1e-10:
            qc.rz(2 * gamma * h_local[i], i)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(J_coupling[i, j]) > 1e-10:
                qc.cx(i, j)
                qc.rz(2 * gamma * J_coupling[i, j], j)
                qc.cx(i, j)


def _apply_mixer_layer(qc: QuantumCircuit, beta: float) -> None:
    """Apply the mixer unitary U_M(β) = exp(-iβ Σ X_i)."""
    for q in range(qc.num_qubits):
        qc.rx(2 * beta, q)


# ─────────────────────────────────────────────────────────────────────────────
# Warm-start angle computation (Equations 23–25)
# ─────────────────────────────────────────────────────────────────────────────

def compute_warm_start_angles(
    s0: np.ndarray,
    p: int,
    ema_alpha: float = EMA_ALPHA,
) -> np.ndarray:
    """
    Compute warm-start QAOA angles from a classical solution s0.

    Based on equations (23)–(25) of the paper:
      θ_i = arcsin(√s0_i)                   (eq. 23 — rotation angle per qubit)
      γ_0 = mean(θ) / π                     (eq. 24 — initial γ from solution bias)
      β_0 = (1 - ema_alpha) * π/4           (eq. 25 — conservative mixer angle)

    Parameters
    ----------
    s0 : np.ndarray, shape (n_qubits,)
        Classical solution bitstring (binary 0/1 per qubit), relaxed to [0,1].
    p : int
        QAOA depth.
    ema_alpha : float
        EMA weight, controls how strongly the classical solution biases angles.

    Returns
    -------
    np.ndarray, shape (2 * p,)
        Interleaved [γ_0, β_0, γ_1, β_1, …, γ_{p-1}, β_{p-1}].
    """
    s_clamped = np.clip(s0.astype(float), 1e-6, 1 - 1e-6)
    theta = np.arcsin(np.sqrt(s_clamped))          # eq. 23

    gamma0 = float(np.mean(theta) / np.pi)         # eq. 24
    beta0  = (1.0 - ema_alpha) * (np.pi / 4.0)     # eq. 25

    # Build p-layer parameter array with mild linear schedule
    params = np.zeros(2 * p)
    for layer in range(p):
        frac = (layer + 1) / p
        params[2 * layer]     = gamma0 * frac          # γ grows with depth
        params[2 * layer + 1] = beta0 * (1.0 - 0.5 * frac)  # β shrinks
    return params


# ─────────────────────────────────────────────────────────────────────────────
# Circuit builder
# ─────────────────────────────────────────────────────────────────────────────

def build_qaoa_circuit(
    h_local: np.ndarray,
    J_coupling: np.ndarray,
    p: int,
    params: np.ndarray,
    s0: np.ndarray | None = None,
    warm_start: bool = False,
) -> QuantumCircuit:
    """
    Construct a QAOA circuit for the Ising Hamiltonian defined by (h, J).

    Parameters
    ----------
    h_local : np.ndarray, shape (n_qubits,)
        Single-body Ising coefficients.
    J_coupling : np.ndarray, shape (n_qubits, n_qubits)
        Two-body coupling matrix (upper triangular).
    p : int
        Number of QAOA layers (depth).
    params : np.ndarray, shape (2 * p,)
        Interleaved [γ_0, β_0, …, γ_{p-1}, β_{p-1}].
    s0 : np.ndarray or None
        Classical solution for warm-start state preparation. Ignored if
        warm_start=False.
    warm_start : bool
        If True, initialise qubits into Ry(θ_i)|0⟩ from s0 (eq. 23).
        If False, apply H gates for uniform superposition.

    Returns
    -------
    QuantumCircuit
        Fully parameterised QAOA circuit (measurements included).
    """
    n = len(h_local)
    qc = QuantumCircuit(n, n)

    # ── Initial state ──────────────────────────────────────────────────────────
    if warm_start and s0 is not None:
        s_clamped = np.clip(s0.astype(float), 1e-6, 1 - 1e-6)
        theta = np.arcsin(np.sqrt(s_clamped))   # eq. 23
        for q in range(n):
            qc.ry(2.0 * theta[q], q)
    else:
        qc.h(range(n))   # uniform superposition (cold-start)

    # ── QAOA layers ────────────────────────────────────────────────────────────
    for layer in range(p):
        gamma = params[2 * layer]
        beta  = params[2 * layer + 1]
        _apply_cost_layer(qc, gamma, h_local, J_coupling)
        _apply_mixer_layer(qc, beta)

    qc.measure(range(n), range(n))
    return qc


def cold_start_params(p: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Sample random initial QAOA parameters for cold-start.

    Parameters
    ----------
    p : int
        QAOA depth.
    rng : np.random.Generator, optional
        Random number generator. Uses global seed if None.

    Returns
    -------
    np.ndarray, shape (2 * p,)
        Random γ ∈ [0, π], β ∈ [0, π/2].
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    gammas = rng.uniform(0, np.pi,     p)
    betas  = rng.uniform(0, np.pi / 2, p)
    return np.stack([gammas, betas], axis=1).ravel()


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive shot scheduling
# ─────────────────────────────────────────────────────────────────────────────

def N_shots_adaptive(
    iteration: int,
    delta_param: float,
    N_base: int = N_SHOTS_BASE,
    N_extra: int = N_SHOTS_EXTRA,
    lam: float = ADAPTIVE_LAM,
) -> int:
    """
    Compute the adaptive number of measurement shots for a given iteration.

    Formula:
        N_shots = N_base + N_extra * exp(-λ * |Δparam|)

    Early iterations (large |Δparam|) use fewer shots; later iterations
    (small |Δparam| near convergence) use more shots for low variance.

    Parameters
    ----------
    iteration : int
        Current optimisation iteration (0-indexed).
    delta_param : float
        L∞ norm of parameter update from the previous iteration.
    N_base : int
        Baseline shots (always allocated).
    N_extra : int
        Additional shots at full convergence.
    lam : float
        Decay rate λ.

    Returns
    -------
    int
        Number of shots to use.
    """
    n = N_base + int(N_extra * np.exp(-lam * abs(delta_param)))
    return max(N_base, min(n, N_base + N_extra))


# ─────────────────────────────────────────────────────────────────────────────
# Expectation value from shot counts
# ─────────────────────────────────────────────────────────────────────────────

def expectation_from_counts(
    counts: dict,
    h_local: np.ndarray,
    J_coupling: np.ndarray,
    E0: float = 0.0,
) -> float:
    """
    Estimate ⟨H⟩ from measurement counts using diagonal Ising Hamiltonian.

    Parameters
    ----------
    counts : dict
        Bitstring measurement counts from Qiskit (key = bitstring, val = int).
    h_local : np.ndarray, shape (n_qubits,)
        Single-body Ising coefficients.
    J_coupling : np.ndarray, shape (n_qubits, n_qubits)
        Two-body coupling matrix.
    E0 : float
        Constant energy offset.

    Returns
    -------
    float
        Estimated energy ⟨H⟩.
    """
    total_shots = sum(counts.values())
    energy = 0.0
    for bitstring, count in counts.items():
        # Qiskit uses little-endian: bitstring[-1] = qubit 0
        bits = np.array([int(b) for b in reversed(bitstring)], dtype=float)
        spins = 1 - 2 * bits   # {0,1} → {+1,-1}
        e = E0
        e += np.dot(h_local, spins)
        # Upper triangular J
        n = len(spins)
        for i in range(n):
            for j in range(i + 1, n):
                e += J_coupling[i, j] * spins[i] * spins[j]
        energy += (count / total_shots) * e
    return energy
