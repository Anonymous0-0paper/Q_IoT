"""
qubo_encoder.py — QUBO / Ising Hamiltonian construction for IoT task scheduling.

Encodes the NP-hard task-to-resource assignment problem as a QUBO and then
transforms it to an Ising spin model suitable for QAOA.
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _z_op(n_qubits: int, qubit_idx: int) -> SparsePauliOp:
    """Return a single Z_i Pauli operator on an n_qubit register."""
    pauli_str = ["I"] * n_qubits
    pauli_str[qubit_idx] = "Z"
    return SparsePauliOp("".join(reversed(pauli_str)))


def _zz_op(n_qubits: int, i: int, j: int) -> SparsePauliOp:
    """Return a Z_i ⊗ Z_j Pauli operator."""
    pauli_str = ["I"] * n_qubits
    pauli_str[i] = "Z"
    pauli_str[j] = "Z"
    return SparsePauliOp("".join(reversed(pauli_str)))


def _identity(n_qubits: int) -> SparsePauliOp:
    return SparsePauliOp("I" * n_qubits)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_cost_hamiltonian(
    n_tasks: int,
    n_resources: int,
    cost_matrix: np.ndarray,
) -> SparsePauliOp:
    """
    Build the cost Hamiltonian for task-to-resource assignment.

    Binary variable x_{i,k} = 1 iff task i is assigned to resource k.
    The cost term is:
        H_cost = Σ_{i,k} c_{i,k} * x_{i,k}

    In Ising form (x = (1 - Z) / 2):
        H_cost = Σ_{i,k} c_{i,k} * (I - Z_{ik}) / 2

    Parameters
    ----------
    n_tasks : int
        Number of tasks T.
    n_resources : int
        Number of resources R.
    cost_matrix : np.ndarray, shape (n_tasks, n_resources)
        c_{i,k}: cost of assigning task i to resource k.

    Returns
    -------
    SparsePauliOp
        The cost Hamiltonian over n_tasks * n_resources qubits.
    """
    n_qubits = n_tasks * n_resources
    terms = []
    coeffs = []

    for i in range(n_tasks):
        for k in range(n_resources):
            qubit = i * n_resources + k
            c_ik = float(cost_matrix[i, k])
            # constant term c/2 * I
            terms.append(_identity(n_qubits))
            coeffs.append(c_ik / 2.0)
            # Z term -c/2 * Z_{ik}
            terms.append(_z_op(n_qubits, qubit))
            coeffs.append(-c_ik / 2.0)

    H_cost = sum(c * t for c, t in zip(coeffs, terms))
    return H_cost.simplify()


def build_penalty_hamiltonian(
    n_tasks: int,
    n_resources: int,
    P: float,
) -> SparsePauliOp:
    """
    Build the constraint penalty Hamiltonian enforcing one-hot assignment.

    Each task i must be assigned to exactly one resource:
        H_pen = P * Σ_i (Σ_k x_{i,k} - 1)^2

    In Ising form this expands to cross-product Z_i Z_j terms.

    Parameters
    ----------
    n_tasks : int
        Number of tasks T.
    n_resources : int
        Number of resources R.
    P : float
        Penalty strength coefficient.

    Returns
    -------
    SparsePauliOp
        The penalty Hamiltonian over n_tasks * n_resources qubits.
    """
    n_qubits = n_tasks * n_resources
    terms = []
    coeffs = []

    for i in range(n_tasks):
        qubits_i = [i * n_resources + k for k in range(n_resources)]

        # (Σ_k x_{i,k} - 1)^2 = Σ_k x_{i,k}^2 + 2 Σ_{k<l} x_{i,k}x_{i,l} - 2 Σ_k x_{i,k} + 1
        # x^2 = x for binary, x_{ik}x_{il} = (I-Z_ik)/2 * (I-Z_il)/2

        # Diagonal x_{i,k}^2 = x_{i,k} → already handled below
        # Cross terms x_{i,k} * x_{i,l}
        for k_idx, qk in enumerate(qubits_i):
            for l_idx, ql in enumerate(qubits_i):
                if k_idx >= l_idx:
                    continue
                # 2 * x_k * x_l = 2 * (I-Zk)/2 * (I-Zl)/2 = (I - Zk - Zl + ZkZl)/2
                coeff = P / 2.0
                terms.append(_identity(n_qubits));  coeffs.append(coeff)
                terms.append(_z_op(n_qubits, qk));  coeffs.append(-coeff)
                terms.append(_z_op(n_qubits, ql));  coeffs.append(-coeff)
                terms.append(_zz_op(n_qubits, qk, ql)); coeffs.append(coeff)

        # Linear terms: (-2 + n_resources) * Σ_k x_{i,k} + constant
        # combined from the -2Σx and Σx^2=Σx parts → (-1)*Σx_{ik}
        for qk in qubits_i:
            # -1 * x_{ik} = -1 * (I-Zk)/2
            terms.append(_identity(n_qubits)); coeffs.append(-P / 2.0)
            terms.append(_z_op(n_qubits, qk)); coeffs.append(P / 2.0)

        # Constant: P * 1  (from the -2*Σx + Σx^2 + 1 = ... after full expansion)
        terms.append(_identity(n_qubits)); coeffs.append(P)

    H_pen = sum(c * t for c, t in zip(coeffs, terms))
    return H_pen.simplify()


def transform_to_ising(
    H_cost: SparsePauliOp,
    H_pen: SparsePauliOp,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Combine cost and penalty Hamiltonians and extract local fields h and
    coupling matrix J in the Ising form H = Σ_i h_i Z_i + Σ_{i<j} J_{ij} Z_i Z_j + E0.

    Parameters
    ----------
    H_cost : SparsePauliOp
        The cost Hamiltonian.
    H_pen : SparsePauliOp
        The penalty Hamiltonian.

    Returns
    -------
    h_local : np.ndarray, shape (n_qubits,)
        Single-body Ising coefficients.
    J_coupling : np.ndarray, shape (n_qubits, n_qubits)
        Two-body Ising coupling matrix (upper triangular).
    E0 : float
        Constant energy offset.
    """
    H_total = (H_cost + H_pen).simplify()
    n_qubits = H_total.num_qubits

    h_local = np.zeros(n_qubits)
    J_coupling = np.zeros((n_qubits, n_qubits))
    E0 = 0.0

    for pauli, coeff in zip(H_total.paulis, H_total.coeffs):
        pauli_str = pauli.to_label()  # e.g. "IIZZI"
        coeff_r = float(np.real(coeff))

        z_positions = [idx for idx, ch in enumerate(reversed(pauli_str)) if ch == "Z"]

        if len(z_positions) == 0:
            E0 += coeff_r
        elif len(z_positions) == 1:
            h_local[z_positions[0]] += coeff_r
        elif len(z_positions) == 2:
            i, j = sorted(z_positions)
            J_coupling[i, j] += coeff_r
        # Higher-order terms are dropped (shouldn't appear for QUBO)

    return h_local, J_coupling, E0
