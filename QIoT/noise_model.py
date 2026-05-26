"""
noise_model.py — Realistic Qiskit AerSimulator noise models and fidelity /
quantum-advantage metrics for the Q-IoT framework.
"""

import numpy as np
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
    ReadoutError,
)
from config import QPU_PROFILES, DELTA_THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
# Noise model construction
# ─────────────────────────────────────────────────────────────────────────────

def build_noise_model(qpu_name: str) -> NoiseModel:
    """
    Build a Qiskit AerSimulator noise model for a named QPU profile.

    The model combines:
      - Depolarising error on single-qubit gates.
      - Depolarising error on two-qubit (CX) gates.
      - Thermal relaxation (T1, T2) for both gate types.
      - Readout (measurement) error.

    Parameters
    ----------
    qpu_name : str
        One of 'IBM', 'IonQ', 'AWS'.

    Returns
    -------
    NoiseModel
        Configured Qiskit noise model.
    """
    spec = QPU_PROFILES[qpu_name]
    nm = NoiseModel()

    t1_ns = spec["t1_us"] * 1_000    # μs → ns
    t2_ns = spec["t2_us"] * 1_000
    t_1q  = spec["gate_time_1q_ns"]
    t_cx  = spec["gate_time_cx_ns"]

    # ── Single-qubit gate errors ───────────────────────────────────────────────
    dep_1q   = depolarizing_error(spec["depolar_1q"], 1)
    therm_1q = thermal_relaxation_error(t1_ns, t2_ns, t_1q)
    err_1q   = dep_1q.compose(therm_1q)
    nm.add_all_qubit_quantum_error(err_1q, ["h", "rx", "ry", "rz", "x", "u1", "u2", "u3"])

    # ── Two-qubit gate errors ──────────────────────────────────────────────────
    dep_2q   = depolarizing_error(spec["depolar_2q"], 2)
    therm_2q = thermal_relaxation_error(t1_ns, t2_ns, t_cx).expand(
               thermal_relaxation_error(t1_ns, t2_ns, t_cx))
    err_2q   = dep_2q.compose(therm_2q)
    nm.add_all_qubit_quantum_error(err_2q, ["cx", "cz", "ecr"])

    # ── Readout error ──────────────────────────────────────────────────────────
    p_ro = 1.0 - spec["f_ro"]
    ro_err = ReadoutError([[1 - p_ro, p_ro], [p_ro, 1 - p_ro]])
    nm.add_all_qubit_readout_error(ro_err)

    return nm


def build_all_noise_models() -> dict:
    """
    Return noise models for all three QPU profiles.

    Returns
    -------
    dict
        Mapping qpu_name -> NoiseModel.
    """
    return {name: build_noise_model(name) for name in QPU_PROFILES}


# ─────────────────────────────────────────────────────────────────────────────
# Fidelity calculation — Equation (17)
# ─────────────────────────────────────────────────────────────────────────────

def compute_fidelity(qpu_spec: dict, circuit_depth: int) -> float:
    """
    Compute the effective circuit fidelity F(q, t) using Equation (17).

    F(q,t) = f_1Q^{N_1Q} * f_CX^{N_CX} * f_RO^{N_qubits}
             * exp(-t_total / T_eff)

    where:
      N_1Q ≈ 2 * n_qubits * p    (single-qubit gates per layer per qubit)
      N_CX ≈ n_qubits * (n_qubits - 1) / 2 * p  (CNOT-equivalent two-qubit gates)
      t_total = N_1Q * t_1q + N_CX * t_cx  (total gate time in μs)
      T_eff = harmonic mean of T1 and T2

    Parameters
    ----------
    qpu_spec : dict
        A single QPU profile dict from config.QPU_PROFILES.
    circuit_depth : int
        QAOA depth p (number of alternating layers).

    Returns
    -------
    float
        Estimated fidelity in [0, 1].
    """
    n  = qpu_spec["n_qubits"]
    p  = circuit_depth
    f1 = qpu_spec["f_1q"]
    fcx= qpu_spec["f_cx"]
    fro= qpu_spec["f_ro"]
    t1 = qpu_spec["t1_us"]
    t2 = qpu_spec["t2_us"]
    t_1q_us = qpu_spec["gate_time_1q_ns"] / 1_000
    t_cx_us = qpu_spec["gate_time_cx_ns"] / 1_000

    N_1q = 2 * n * p          # rough single-qubit gate count
    N_cx = int(n * (n - 1) / 2 * p)   # rough CNOT count for dense QAOA

    F_gate = (f1 ** N_1q) * (fcx ** N_cx) * (fro ** n)

    t_total_us = N_1q * t_1q_us + N_cx * t_cx_us
    T_eff = 2.0 * t1 * t2 / (t1 + t2 + 1e-12)   # harmonic mean
    F_decohere = np.exp(-t_total_us / T_eff)

    return float(np.clip(F_gate * F_decohere, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Quantum advantage metric — Equation (22)
# ─────────────────────────────────────────────────────────────────────────────

def compute_advantage_metric(
    T_classical: float,
    T_quantum: float,
    rho_quality: float,
    fidelity: float = 1.0,
) -> float:
    """
    Compute the quantum advantage metric Θ from Equation (22).

    Θ = (T_classical / T_quantum) * ρ_quality * F(q,t)

    where:
      T_classical : wall-clock time of best classical solver (ms)
      T_quantum   : wall-clock time of quantum solver including overhead (ms)
      ρ_quality   : solution quality ratio (classical / quantum objective, ≥1 is better)
      fidelity    : F(q,t) in [0,1]

    Θ > δ_threshold → quantum execution preferred.
    Θ ≤ δ_threshold → classical fallback.

    Parameters
    ----------
    T_classical : float
        Classical solver runtime in milliseconds.
    T_quantum : float
        Quantum solver runtime in milliseconds.
    rho_quality : float
        Quality ratio: classical_objective / quantum_objective (> 1 means quantum is better).
    fidelity : float
        Circuit fidelity F(q,t).

    Returns
    -------
    float
        Quantum advantage metric Θ.
    """
    if T_quantum <= 0:
        return 0.0
    theta = (T_classical / T_quantum) * rho_quality * fidelity
    return float(theta)


def fidelity_to_scenario(fidelity: float) -> str:
    """
    Map a fidelity value to one of the three degradation scenarios.

    Parameters
    ----------
    fidelity : float

    Returns
    -------
    str
        'high', 'medium', or 'low'.
    """
    if fidelity >= 0.95:
        return "high"
    elif fidelity >= 0.85:
        return "medium"
    else:
        return "low"


def adaptive_penalty(
    P_base: float,
    fidelity: float,
    gamma: float = 3.0,
) -> float:
    """
    Compute the fidelity-adaptive penalty coefficient.

    P_adaptive = P_base * (1 + γ * (1 - F))

    Parameters
    ----------
    P_base : float
        Baseline penalty coefficient.
    fidelity : float
        Current circuit fidelity F ∈ [0, 1].
    gamma : float
        Amplification factor γ ∈ [2, 5].

    Returns
    -------
    float
        Adapted penalty coefficient.
    """
    return P_base * (1.0 + gamma * (1.0 - fidelity))
