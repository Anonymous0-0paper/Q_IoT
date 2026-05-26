"""
config.py — Global hyperparameters, hardware specs, and simulation settings
for the Q-IoT Hybrid Quantum-Classical Framework simulation.
"""

import numpy as np

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Problem sizes ──────────────────────────────────────────────────────────────
TASK_SIZES       = [10, 20, 30, 40, 50]
N_RESOURCES_MIN  = 2
N_RESOURCES_MAX  = 8
DEFAULT_N_TASKS  = 30
DEFAULT_N_RES    = 6

# ── QAOA hyperparameters ────────────────────────────────────────────────────────
QAOA_DEPTH       = 3          # p (number of QAOA layers)
N_SHOTS_COLD     = 10_000     # shots per iteration, cold-start
N_SHOTS_WARM     = 3_500      # shots per iteration, warm-start
N_SHOTS_BASE     = 1_000      # N_base for adaptive shot formula
N_SHOTS_EXTRA    = 9_000      # N_extra for adaptive shot formula
ADAPTIVE_LAM     = 2.0        # λ in N_shots_adaptive

# ── Warm-start ─────────────────────────────────────────────────────────────────
EMA_ALPHA        = 0.7        # exponential moving average weight
PENALTY_GAMMA_LO = 2.0        # γ lower bound (penalty amplification)
PENALTY_GAMMA_HI = 5.0        # γ upper bound
PENALTY_BASE     = 10.0       # P_base

# ── Fidelity thresholds (Table II) ─────────────────────────────────────────────
FIDELITY_HIGH    = 0.95       # F > 0.95 → quantum execution
FIDELITY_MED_LO  = 0.85       # 0.85 ≤ F ≤ 0.95 → adaptive warm-start
FIDELITY_LOW     = 0.85       # F < 0.85 → classical fallback
DELTA_THRESHOLD  = 1.5        # Θ threshold for quantum-advantage decision

# ── Convergence ────────────────────────────────────────────────────────────────
CONVERGENCE_TOL  = 0.05       # δ convergence tolerance
MAX_ITER         = 100        # maximum optimiser iterations
QOS_TARGET       = 0.99       # 99 % QoS compliance target

# ── Benchmark repetitions ──────────────────────────────────────────────────────
N_TRIALS         = 10         # independent runs per (algorithm, problem size)

# ── QPU hardware profiles ──────────────────────────────────────────────────────
# Keys: t1_us (μs), t2_us (μs), f_1q (single-qubit gate fidelity),
#       f_cx (two-qubit gate fidelity), f_ro (readout fidelity),
#       gate_time_1q_ns, gate_time_cx_ns, n_qubits
QPU_PROFILES = {
    "IBM": {
        "name":            "IBM Quantum (Eagle-class)",
        "n_qubits":        27,
        "t1_us":           150.0,
        "t2_us":           100.0,
        "f_1q":            0.9995,
        "f_cx":            0.990,
        "f_ro":            0.975,
        "gate_time_1q_ns": 50,
        "gate_time_cx_ns": 300,
        "depolar_1q":      5e-4,
        "depolar_2q":      1e-2,
    },
    "IonQ": {
        "name":            "IonQ Harmony (trapped-ion)",
        "n_qubits":        11,
        "t1_us":           10_000.0,
        "t2_us":           1_000.0,
        "f_1q":            0.9993,
        "f_cx":            0.9660,
        "f_ro":            0.9770,
        "gate_time_1q_ns": 10_000,
        "gate_time_cx_ns": 600_000,
        "depolar_1q":      7e-4,
        "depolar_2q":      3.4e-2,
    },
    "AWS": {
        "name":            "AWS Braket (superconducting)",
        "n_qubits":        34,
        "t1_us":           90.0,
        "t2_us":           60.0,
        "f_1q":            0.9985,
        "f_cx":            0.985,
        "f_ro":            0.965,
        "gate_time_1q_ns": 60,
        "gate_time_cx_ns": 350,
        "depolar_1q":      1.5e-3,
        "depolar_2q":      1.5e-2,
    },
}

# ── Noise / fidelity scenario labels ──────────────────────────────────────────
FIDELITY_SCENARIOS = {
    "high":   (0.95, 1.00),
    "medium": (0.85, 0.95),
    "low":    (0.70, 0.85),
}

# ── Figure export paths ────────────────────────────────────────────────────────
FIG_DIR    = "figures"
TABLE_DIR  = "tables"
RESULT_DIR = "."

# ── IEEE figure style defaults ─────────────────────────────────────────────────
IEEE_FONT_FAMILY  = "serif"
IEEE_FONT_SIZE    = 8
IEEE_TICK_SIZE    = 7
IEEE_LEGEND_SIZE  = 7
IEEE_LINEWIDTH    = 1.0
IEEE_MARKERSIZE   = 4
IEEE_FIG_SINGLE   = (3.5, 2.6)
IEEE_FIG_DOUBLE   = (7.2, 2.8)
IEEE_DPI_EXPORT   = 600
IEEE_DPI_PNG      = 300

COLORS = {
    "blue":   "#1f77b4",
    "red":    "#d62728",
    "green":  "#2ca02c",
    "orange": "#ff7f0e",
    "purple": "#9467bd",
    "brown":  "#8c564b",
}
