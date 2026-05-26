"""
orchestrator.py — Implementations of Algorithms 1–4 from the Q-IoT paper.

Algorithm 1: Dynamic Quantinuum Orchestration
Algorithm 2: Fidelity-Aware Qubit Mapping
Algorithm 3: Warm-Started Quantum Routing (WS-QR)
Algorithm 4: Elastic QoS-Aware Resource Allocation
"""

import logging
import time
import numpy as np
from scipy.optimize import minimize

from qiskit_aer import AerSimulator

from config import (
    QAOA_DEPTH, N_SHOTS_COLD, N_SHOTS_WARM, PENALTY_BASE,
    EMA_ALPHA, FIDELITY_HIGH, FIDELITY_MED_LO, DELTA_THRESHOLD,
    CONVERGENCE_TOL, MAX_ITER, QOS_TARGET, RANDOM_SEED,
    PENALTY_GAMMA_LO, PENALTY_GAMMA_HI,
)
from qubo_encoder import build_cost_hamiltonian, build_penalty_hamiltonian, transform_to_ising
from qaoa_circuit import (
    build_qaoa_circuit, cold_start_params, compute_warm_start_angles,
    N_shots_adaptive, expectation_from_counts,
)
from noise_model import (
    compute_fidelity, compute_advantage_metric, adaptive_penalty, build_noise_model,
)
from qiot_logger import (
    QAOAProgressTracker,
    log_fidelity_check, log_routing_decision, log_shot_allocation,
    log_qos_result, log_qubo_build, log_advantage_metric,
    QAOA_ITER,
)

log = logging.getLogger(__name__)

np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def _decode_counts(counts: dict, n_tasks: int, n_resources: int) -> np.ndarray:
    """Return the most-probable assignment matrix from measurement counts."""
    best_bs  = max(counts, key=counts.get)
    bits     = np.array([int(b) for b in reversed(best_bs)])
    return bits.reshape(n_tasks, n_resources)


def _assignment_objective(X: np.ndarray, cost_matrix: np.ndarray) -> float:
    """Compute Σ_{i,k} c_{ik} x_{ik} for binary assignment matrix X."""
    return float(np.sum(X * cost_matrix))


def _greedy_assignment(cost_matrix: np.ndarray) -> np.ndarray:
    """Greedy one-hot assignment: each task → cheapest available resource."""
    n_tasks, n_resources = cost_matrix.shape
    X = np.zeros((n_tasks, n_resources), dtype=int)
    for i in range(n_tasks):
        k = int(np.argmin(cost_matrix[i]))
        X[i, k] = 1
    return X


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm 2 — Fidelity-Aware Qubit Mapping
# ─────────────────────────────────────────────────────────────────────────────

def fidelity_aware_qubit_mapping(
    n_tasks: int,
    n_resources: int,
    qpu_spec: dict,
    circuit_depth: int,
) -> dict:
    """
    Algorithm 2: Fidelity-Aware Qubit Mapping.

    Determines the required number of qubits, estimates circuit fidelity,
    and decides whether quantum execution is viable.

    Parameters
    ----------
    n_tasks : int
        Number of IoT tasks.
    n_resources : int
        Number of compute resources.
    qpu_spec : dict
        QPU hardware profile (from config.QPU_PROFILES).
    circuit_depth : int
        QAOA depth p.

    Returns
    -------
    dict
        Keys: 'n_qubits_required', 'fidelity', 'viable', 'mapped_qubits'.
    """
    n_qubits_required = n_tasks * n_resources
    fidelity = compute_fidelity(qpu_spec, circuit_depth)
    viable   = (fidelity >= FIDELITY_MED_LO) and (n_qubits_required <= qpu_spec["n_qubits"])

    log.info("Qubit mapping: need %d qubits, QPU has %d",
             n_qubits_required, qpu_spec["n_qubits"])
    log_fidelity_check(log, fidelity, qpu_spec.get("name", "QPU"),
                       circuit_depth, viable)

    mapped_qubits = list(range(min(n_qubits_required, qpu_spec["n_qubits"])))
    return {
        "n_qubits_required": n_qubits_required,
        "fidelity":          fidelity,
        "viable":            viable,
        "mapped_qubits":     mapped_qubits,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm 3 — Warm-Started Quantum Routing (WS-QR)
# ─────────────────────────────────────────────────────────────────────────────

def warm_started_quantum_routing(
    cost_matrix: np.ndarray,
    n_tasks: int,
    n_resources: int,
    fidelity: float,
    noise_model,
    p: int = QAOA_DEPTH,
    n_shots_base: int = N_SHOTS_WARM,
    max_iter: int = MAX_ITER,
) -> dict:
    """
    Algorithm 3: Warm-Started Quantum Routing (WS-QR).

    1. Solve the relaxed problem classically (greedy) to get s0.
    2. Compute warm-start angles from s0 (eq. 23–25).
    3. Build QUBO with fidelity-adaptive penalty.
    4. Run QAOA with COBYLA optimisation using adaptive shot counts.
    5. Decode the best bitstring as the assignment solution.

    Parameters
    ----------
    cost_matrix : np.ndarray, shape (n_tasks, n_resources)
        Assignment cost matrix.
    n_tasks : int
        Number of tasks.
    n_resources : int
        Number of resources.
    fidelity : float
        Current QPU fidelity.
    noise_model : NoiseModel or None
        Qiskit noise model; None for ideal simulation.
    p : int
        QAOA depth.
    n_shots_base : int
        Base number of shots.
    max_iter : int
        Maximum COBYLA iterations.

    Returns
    -------
    dict
        Keys: 'assignment', 'objective', 'iterations', 'total_shots',
              'convergence_curve', 'runtime_ms'.
    """
    t_start = time.perf_counter()

    # ── Step 1: Classical warm-start solution ──────────────────────────────────
    X0     = _greedy_assignment(cost_matrix)
    s0_flat = X0.ravel().astype(float)

    # ── Step 2: Warm-start angles ─────────────────────────────────────────────
    gamma = adaptive_penalty(PENALTY_BASE, fidelity, gamma=PENALTY_GAMMA_LO)
    params = compute_warm_start_angles(s0_flat, p)

    # ── Step 3: Build QUBO Hamiltonians ───────────────────────────────────────
    H_cost = build_cost_hamiltonian(n_tasks, n_resources, cost_matrix)
    H_pen  = build_penalty_hamiltonian(n_tasks, n_resources, gamma)
    h, J, E0 = transform_to_ising(H_cost, H_pen)
    n_z  = int(np.count_nonzero(h))
    n_zz = int(np.count_nonzero(J))
    log_qubo_build(log, n_tasks * n_resources, n_z, n_zz, gamma)

    # ── Step 4: QAOA optimisation ─────────────────────────────────────────────
    simulator = AerSimulator(noise_model=noise_model)
    convergence_curve = []
    total_shots = 0
    iteration_counter = [0]
    prev_params = params.copy()

    tracker = QAOAProgressTracker(
        algorithm="WS-QAOA", n_tasks=n_tasks,
        max_iter=max_iter, fidelity=fidelity,
    )

    def objective_fn(current_params):
        nonlocal total_shots
        delta = float(np.max(np.abs(current_params - prev_params)))
        n_shots = N_shots_adaptive(iteration_counter[0], delta, N_base=n_shots_base)
        total_shots += n_shots
        iteration_counter[0] += 1

        qc = build_qaoa_circuit(h, J, p, current_params, s0=s0_flat, warm_start=True)
        job = simulator.run(qc, shots=n_shots)
        counts = job.result().get_counts()
        energy = expectation_from_counts(counts, h, J, E0)
        convergence_curve.append(energy)

        tracker.update(iteration_counter[0], energy, n_shots, delta)
        log.log(QAOA_ITER,
                "WS-QAOA  iter=%3d  E=%+.4f  delta=%.4f  shots=%d  total=%d",
                iteration_counter[0], energy, delta, n_shots, total_shots)
        log_shot_allocation(log, iteration_counter[0], delta, n_shots, total_shots)

        prev_params[:] = current_params
        return energy

    result = minimize(
        objective_fn, params,
        method="COBYLA",
        options={"maxiter": max_iter, "rhobeg": 0.5, "catol": CONVERGENCE_TOL},
    )
    opt_params = result.x
    converged  = bool(result.status == 1 or
                      (len(convergence_curve) >= 2 and
                       abs(convergence_curve[-1] - convergence_curve[-2]) < CONVERGENCE_TOL))

    # ── Step 5: Decode best bitstring ─────────────────────────────────────────
    qc_final = build_qaoa_circuit(h, J, p, opt_params, s0=s0_flat, warm_start=True)
    job_final = simulator.run(qc_final, shots=n_shots_base * 3)
    counts_final = job_final.result().get_counts()
    total_shots += n_shots_base * 3

    X_out = _decode_counts(counts_final, n_tasks, n_resources)
    obj   = _assignment_objective(X_out, cost_matrix)
    runtime_ms = (time.perf_counter() - t_start) * 1000

    tracker.finish(converged=converged, total_shots=total_shots,
                   objective=obj, runtime_ms=runtime_ms)
    log.info("WS-QR done: obj=%.4f, iters=%d, shots=%d, runtime=%.1f ms",
             obj, iteration_counter[0], total_shots, runtime_ms)

    return {
        "assignment":       X_out,
        "objective":        obj,
        "iterations":       iteration_counter[0],
        "total_shots":      total_shots,
        "convergence_curve": convergence_curve,
        "runtime_ms":       runtime_ms,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cold-start QAOA (for comparison)
# ─────────────────────────────────────────────────────────────────────────────

def cold_start_qaoa(
    cost_matrix: np.ndarray,
    n_tasks: int,
    n_resources: int,
    fidelity: float,
    noise_model,
    p: int = QAOA_DEPTH,
    n_shots_base: int = N_SHOTS_COLD,
    max_iter: int = MAX_ITER,
) -> dict:
    """
    Cold-start QAOA baseline: uniform superposition + random initial angles.

    Parameters mirror warm_started_quantum_routing.

    Returns
    -------
    dict
        Same keys as warm_started_quantum_routing.
    """
    t_start = time.perf_counter()

    gamma = adaptive_penalty(PENALTY_BASE, fidelity, gamma=PENALTY_GAMMA_HI)
    H_cost = build_cost_hamiltonian(n_tasks, n_resources, cost_matrix)
    H_pen  = build_penalty_hamiltonian(n_tasks, n_resources, gamma)
    h, J, E0 = transform_to_ising(H_cost, H_pen)
    log_qubo_build(log, n_tasks * n_resources,
                   int(np.count_nonzero(h)), int(np.count_nonzero(J)), gamma)

    rng = np.random.default_rng(RANDOM_SEED)
    params = cold_start_params(p, rng)

    simulator = AerSimulator(noise_model=noise_model)
    convergence_curve = []
    total_shots = 0
    iteration_counter = [0]
    prev_params = params.copy()

    tracker = QAOAProgressTracker(
        algorithm="Cold-QAOA", n_tasks=n_tasks,
        max_iter=max_iter, fidelity=fidelity,
    )

    def objective_fn(current_params):
        nonlocal total_shots
        delta = float(np.max(np.abs(current_params - prev_params)))
        n_shots = N_shots_adaptive(iteration_counter[0], delta, N_base=n_shots_base)
        total_shots += n_shots
        iteration_counter[0] += 1

        qc = build_qaoa_circuit(h, J, p, current_params, warm_start=False)
        job = simulator.run(qc, shots=n_shots)
        counts = job.result().get_counts()
        energy = expectation_from_counts(counts, h, J, E0)
        convergence_curve.append(energy)

        tracker.update(iteration_counter[0], energy, n_shots, delta)
        log.log(QAOA_ITER,
                "Cold-QAOA iter=%3d  E=%+.4f  delta=%.4f  shots=%d  total=%d",
                iteration_counter[0], energy, delta, n_shots, total_shots)

        prev_params[:] = current_params
        return energy

    result = minimize(
        objective_fn, params,
        method="COBYLA",
        options={"maxiter": max_iter, "rhobeg": 1.0, "catol": CONVERGENCE_TOL},
    )
    opt_params = result.x
    converged  = bool(result.status == 1 or
                      (len(convergence_curve) >= 2 and
                       abs(convergence_curve[-1] - convergence_curve[-2]) < CONVERGENCE_TOL))

    qc_final = build_qaoa_circuit(h, J, p, opt_params, warm_start=False)
    job_final = simulator.run(qc_final, shots=n_shots_base)
    counts_final = job_final.result().get_counts()
    total_shots += n_shots_base

    X_out = _decode_counts(counts_final, n_tasks, n_resources)
    obj   = _assignment_objective(X_out, cost_matrix)
    runtime_ms = (time.perf_counter() - t_start) * 1000

    tracker.finish(converged=converged, total_shots=total_shots,
                   objective=obj, runtime_ms=runtime_ms)
    log.info("Cold-QAOA done: obj=%.4f, iters=%d, shots=%d, runtime=%.1f ms",
             obj, iteration_counter[0], total_shots, runtime_ms)

    return {
        "assignment":       X_out,
        "objective":        obj,
        "iterations":       iteration_counter[0],
        "total_shots":      total_shots,
        "convergence_curve": convergence_curve,
        "runtime_ms":       runtime_ms,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm 4 — Elastic QoS-Aware Resource Allocation
# ─────────────────────────────────────────────────────────────────────────────

def elastic_qos_resource_allocation(
    assignment: np.ndarray,
    cost_matrix: np.ndarray,
    resource_capacities: np.ndarray,
    task_priorities: np.ndarray,
    fidelity: float,
    qos_target: float = QOS_TARGET,
) -> dict:
    """
    Algorithm 4: Elastic QoS-Aware Resource Allocation.

    Post-processes a quantum assignment by:
    1. Checking QoS compliance for each task (latency budget).
    2. Elastically reallocating overloaded resources to maintain QoS.
    3. Triggering classical fallback for resources below fidelity threshold.

    Parameters
    ----------
    assignment : np.ndarray, shape (n_tasks, n_resources)
        Binary assignment matrix.
    cost_matrix : np.ndarray, shape (n_tasks, n_resources)
        Effective latency / cost matrix.
    resource_capacities : np.ndarray, shape (n_resources,)
        Maximum load each resource can handle.
    task_priorities : np.ndarray, shape (n_tasks,)
        Priority weights for tasks (higher = more important).
    fidelity : float
        Current quantum fidelity (decides classical fallback).
    qos_target : float
        Minimum QoS compliance rate threshold.

    Returns
    -------
    dict
        Keys: 'final_assignment', 'qos_compliance', 'fallback_triggered',
              'objective', 'avg_latency_ms'.
    """
    n_tasks, n_resources = assignment.shape
    X = assignment.copy().astype(float)

    # ── Resource load check ───────────────────────────────────────────────────
    resource_load = X.sum(axis=0)
    overloaded = np.where(resource_load > resource_capacities)[0]

    # ── Elastic reallocation for overloaded resources ─────────────────────────
    for r_over in overloaded:
        task_indices = np.where(X[:, r_over] == 1)[0]
        # Sort by ascending priority (reallocate low-priority tasks first)
        task_indices = task_indices[np.argsort(task_priorities[task_indices])]
        for t in task_indices:
            if resource_load[r_over] <= resource_capacities[r_over]:
                break
            # Find cheapest alternative resource with capacity
            alt_costs = cost_matrix[t].copy()
            alt_costs[r_over] = np.inf
            for alt_r in np.argsort(alt_costs):
                if resource_load[alt_r] < resource_capacities[alt_r]:
                    X[t, r_over] = 0
                    X[t, alt_r] = 1
                    resource_load[r_over] -= 1
                    resource_load[alt_r]  += 1
                    break

    # ── QoS compliance computation ────────────────────────────────────────────
    latencies = (X * cost_matrix).sum(axis=1)   # effective latency per task
    latency_budget = np.percentile(latencies, 95) * 1.1   # 10% slack on 95th-pct
    compliant = (latencies <= latency_budget).astype(float)
    qos_compliance = float(compliant.mean())

    fallback_triggered = (fidelity < FIDELITY_MED_LO) or (qos_compliance < qos_target)

    log.info("QoS-Alloc: compliance=%.3f, fallback=%s, avg_latency=%.2f ms",
             qos_compliance, fallback_triggered, float(latencies.mean()))

    return {
        "final_assignment":  X,
        "qos_compliance":    qos_compliance,
        "fallback_triggered": fallback_triggered,
        "objective":         float(np.sum(X * cost_matrix)),
        "avg_latency_ms":    float(latencies.mean()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm 1 — Dynamic Quantinuum Orchestration (top-level loop)
# ─────────────────────────────────────────────────────────────────────────────

def dynamic_quantinuum_orchestration(
    cost_matrix: np.ndarray,
    qpu_name: str = "IBM",
    p: int = QAOA_DEPTH,
    max_iter: int = MAX_ITER,
    resource_capacities: np.ndarray | None = None,
    task_priorities: np.ndarray | None = None,
) -> dict:
    """
    Algorithm 1: Dynamic Quantinuum Orchestration.

    Top-level controller that:
    1. Runs Algorithm 2 (qubit mapping / fidelity check).
    2. If fidelity high → Runs Algorithm 3 (WS-QR) on QPU.
    3. If fidelity medium → Runs warm-start with classical fallback backup.
    4. If fidelity low → Runs best classical solver directly.
    5. Runs Algorithm 4 (elastic QoS allocation) on the result.
    6. Computes quantum advantage metric Θ (eq. 22).

    Parameters
    ----------
    cost_matrix : np.ndarray, shape (n_tasks, n_resources)
        Assignment cost matrix.
    qpu_name : str
        QPU profile name ('IBM', 'IonQ', 'AWS').
    p : int
        QAOA depth.
    max_iter : int
        Maximum optimiser iterations.
    resource_capacities : np.ndarray or None
        Per-resource capacity limits. Defaults to 2 tasks per resource.
    task_priorities : np.ndarray or None
        Per-task priorities. Defaults to uniform.

    Returns
    -------
    dict
        Full orchestration result including solver used, objective, QoS metrics,
        quantum advantage Θ, and timing.
    """
    from classical_baselines import simulated_annealing

    n_tasks, n_resources = cost_matrix.shape
    t_orch_start = time.perf_counter()

    if resource_capacities is None:
        resource_capacities = np.full(n_resources, 2, dtype=int)
    if task_priorities is None:
        task_priorities = np.ones(n_tasks)

    from config import QPU_PROFILES
    qpu_spec = QPU_PROFILES[qpu_name]
    noise_model = build_noise_model(qpu_name)

    # ── Algorithm 2: Fidelity-aware mapping ───────────────────────────────────
    mapping = fidelity_aware_qubit_mapping(n_tasks, n_resources, qpu_spec, p)
    fidelity = mapping["fidelity"]
    viable   = mapping["viable"]

    solver_used = None
    quantum_result = None
    t_classical_ms = None

    # ── Classical baseline time (for Θ computation) ───────────────────────────
    sa_result = simulated_annealing(cost_matrix, n_tasks, n_resources)
    t_classical_ms = sa_result["runtime_ms"]

    # ── Routing decision ──────────────────────────────────────────────────────
    if viable and fidelity >= FIDELITY_HIGH:
        log_routing_decision(log, "WS-QAOA", fidelity, reason="F ≥ F_high threshold")
        quantum_result = warm_started_quantum_routing(
            cost_matrix, n_tasks, n_resources, fidelity, noise_model, p, max_iter=max_iter)
        solver_used = "WS-QAOA"

    elif viable and fidelity >= FIDELITY_MED_LO:
        log_routing_decision(log, "WS-QAOA", fidelity, reason="F_med ≤ F < F_high — with SA backup")
        quantum_result = warm_started_quantum_routing(
            cost_matrix, n_tasks, n_resources, fidelity, noise_model, p, max_iter=max_iter)
        # Verify against SA; take better
        if quantum_result["objective"] > sa_result["objective"] * 1.1:
            log_routing_decision(log, "SA-fallback", fidelity,
                                 reason="quantum obj > 110% of SA — taking SA")
            log.info("Quantum result worse than SA — using SA fallback.")
            assignment_final = sa_result["assignment"]
            objective_final  = sa_result["objective"]
            solver_used = "SA-fallback"
        else:
            assignment_final = quantum_result["assignment"]
            objective_final  = quantum_result["objective"]
            solver_used = "WS-QAOA"

    else:
        log_routing_decision(log, "SA-fallback", fidelity,
                             reason="F < F_med or qubit shortage")
        assignment_final = sa_result["assignment"]
        objective_final  = sa_result["objective"]
        solver_used = "SA-fallback"
        quantum_result = {
            "assignment": assignment_final, "objective": objective_final,
            "iterations": 0, "total_shots": 0,
            "convergence_curve": [], "runtime_ms": sa_result["runtime_ms"],
        }

    if solver_used in ("WS-QAOA",):
        assignment_final = quantum_result["assignment"]
        objective_final  = quantum_result["objective"]

    # ── Algorithm 4: QoS allocation ───────────────────────────────────────────
    qos_result = elastic_qos_resource_allocation(
        assignment_final, cost_matrix,
        resource_capacities, task_priorities, fidelity,
    )

    # ── Quantum advantage Θ ────────────────────────────────────────────────────
    t_quantum_ms = quantum_result["runtime_ms"] if quantum_result else 1e6
    rho_quality  = (sa_result["objective"] / (objective_final + 1e-9))
    theta = compute_advantage_metric(t_classical_ms, t_quantum_ms, rho_quality, fidelity)

    log_qos_result(log,
                   qos_result["qos_compliance"],
                   qos_result["avg_latency_ms"],
                   qos_result["fallback_triggered"],
                   qos_result["objective"])
    log_advantage_metric(log, theta, DELTA_THRESHOLD, t_classical_ms, t_quantum_ms)

    t_total_ms = (time.perf_counter() - t_orch_start) * 1000
    log.info("Orchestration done: solver=%s, obj=%.4f, Θ=%.3f, total=%.1f ms",
             solver_used, objective_final, theta, t_total_ms)

    return {
        "solver_used":        solver_used,
        "fidelity":           fidelity,
        "viable":             viable,
        "assignment":         qos_result["final_assignment"],
        "objective":          qos_result["objective"],
        "qos_compliance":     qos_result["qos_compliance"],
        "fallback_triggered": qos_result["fallback_triggered"],
        "avg_latency_ms":     qos_result["avg_latency_ms"],
        "theta":              theta,
        "iterations":         quantum_result.get("iterations", 0),
        "total_shots":        quantum_result.get("total_shots", 0),
        "convergence_curve":  quantum_result.get("convergence_curve", []),
        "runtime_ms":         t_total_ms,
        "t_classical_ms":     t_classical_ms,
    }
