"""
qiot_logger.py — Real-time structured logger for the Q-IoT simulation framework.

Provides:
  • Coloured, columnar console output with module-aligned fields
  • Custom log levels: QAOA_ITER, FIDELITY, DECISION, SHOT, QOS_EVENT
  • QAOAProgressTracker  — in-place overwriting progress bar per QAOA iteration
  • BenchmarkProgressTracker — live table showing per-algorithm results as they land
  • Plain-text file log (qiot_run.log) mirroring all console output
  • setup_logging()  — single call that replaces all existing handlers globally
  • print_final_summary(results) — boxed summary table at end of run

Usage (in run_all.py, BEFORE other Q-IoT imports):
    import qiot_logger
    qiot_logger.setup_logging()          # installs globally; honours --quiet flag

Usage in any module (unchanged from stdlib):
    import logging
    log = logging.getLogger(__name__)
    log.info("...")                       # automatically uses QIoT formatter

Specialised helpers (optional, in orchestrator / benchmark_runner):
    from qiot_logger import QAOAProgressTracker, BenchmarkProgressTracker
"""

from __future__ import annotations

import logging
import os
import sys
import time
import threading
from datetime import datetime
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# ANSI escape sequences (no external dependencies)
# ─────────────────────────────────────────────────────────────────────────────

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"

# Foreground colours
_BLACK   = "\033[30m"
_RED     = "\033[31m"
_GREEN   = "\033[32m"
_YELLOW  = "\033[33m"
_BLUE    = "\033[34m"
_MAGENTA = "\033[35m"
_CYAN    = "\033[36m"
_WHITE   = "\033[37m"

# Bright foreground
_BRED    = "\033[91m"
_BGREEN  = "\033[92m"
_BYELLOW = "\033[93m"
_BBLUE   = "\033[94m"
_BMAGENTA= "\033[95m"
_BCYAN   = "\033[96m"
_BWHITE  = "\033[97m"

# Backgrounds
_BG_DARK  = "\033[40m"
_BG_RED   = "\033[41m"
_BG_GREEN = "\033[42m"
_BG_BLUE  = "\033[44m"

# Cursor control
_CURSOR_UP   = "\033[F"       # move up one line
_CLEAR_LINE  = "\033[2K\r"    # clear line + carriage return
_SAVE_POS    = "\033[s"
_RESTORE_POS = "\033[u"
_HIDE_CURSOR = "\033[?25l"
_SHOW_CURSOR = "\033[?25h"

# ─────────────────────────────────────────────────────────────────────────────
# Detect whether the terminal supports ANSI (disable on Windows cmd / CI)
# ─────────────────────────────────────────────────────────────────────────────

_ANSI_OK = (
    hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    and os.environ.get("TERM", "").lower() not in ("dumb", "")
    and os.environ.get("NO_COLOR", "") == ""
)


def _c(code: str, text: str) -> str:
    """Wrap text in ANSI code if terminal supports it."""
    return f"{code}{text}{_RESET}" if _ANSI_OK else text


# ─────────────────────────────────────────────────────────────────────────────
# Custom log levels
# ─────────────────────────────────────────────────────────────────────────────

QAOA_ITER   = 15   # verbose QAOA iteration detail (below INFO)
FIDELITY    = 22   # fidelity checks / threshold decisions
DECISION    = 23   # routing decision: quantum vs classical
SHOT        = 24   # adaptive shot allocation
QOS_EVENT   = 25   # QoS compliance checks

logging.addLevelName(QAOA_ITER,  "QAOA")
logging.addLevelName(FIDELITY,   "FIDEL")
logging.addLevelName(DECISION,   "DECIS")
logging.addLevelName(SHOT,       "SHOTS")
logging.addLevelName(QOS_EVENT,  "QoS  ")

# ── Level → colour mapping ────────────────────────────────────────────────────
_LEVEL_STYLE: dict[int, tuple[str, str]] = {
    logging.DEBUG:  (_DIM + _CYAN,    "DEBUG"),
    QAOA_ITER:      (_BBLUE,          "QAOA "),
    logging.INFO:   (_BGREEN,         "INFO "),
    FIDELITY:       (_BCYAN,          "FIDEL"),
    DECISION:       (_BYELLOW,        "DECIS"),
    SHOT:           (_BMAGENTA,       "SHOTS"),
    QOS_EVENT:      (_BWHITE,         "QoS  "),
    logging.WARNING:(_YELLOW,         "WARN "),
    logging.ERROR:  (_BRED,           "ERROR"),
    logging.CRITICAL:(_BOLD + _BRED,  "CRIT "),
}

# ─────────────────────────────────────────────────────────────────────────────
# Formatter
# ─────────────────────────────────────────────────────────────────────────────

class QIoTFormatter(logging.Formatter):
    """
    Columnar, coloured log formatter.

    Format:
        HH:MM:SS.mmm  LEVEL   module_name   message
    """

    _MODULE_WIDTH = 18
    _LEVEL_WIDTH  = 5

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.") + \
             f"{int(record.msecs):03d}"

        level_num   = record.levelno
        colour, lbl = _LEVEL_STYLE.get(level_num, (_WHITE, record.levelname[:5]))

        module = record.name.split(".")[-1][:self._MODULE_WIDTH]
        msg    = record.getMessage()

        if _ANSI_OK:
            ts_str     = _c(_DIM, ts)
            level_str  = _c(colour, f"{lbl:<{self._LEVEL_WIDTH}}")
            module_str = _c(_CYAN, f"{module:<{self._MODULE_WIDTH}}")
            msg_str    = msg
        else:
            ts_str     = ts
            level_str  = f"{lbl:<{self._LEVEL_WIDTH}}"
            module_str = f"{module:<{self._MODULE_WIDTH}}"
            msg_str    = msg

        return f"{ts_str}  {level_str}  {module_str}  {msg_str}"


class QIoTPlainFormatter(logging.Formatter):
    """Plain-text formatter for the log file (no ANSI codes)."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S.") + \
             f"{int(record.msecs):03d}"
        level  = record.levelname[:5]
        module = record.name.split(".")[-1][:18]
        return f"{ts}  {level:<5}  {module:<18}  {record.getMessage()}"


# ─────────────────────────────────────────────────────────────────────────────
# Handlers
# ─────────────────────────────────────────────────────────────────────────────

class QIoTConsoleHandler(logging.StreamHandler):
    """
    Coloured streaming handler for stderr.

    Acquires a lock before writing so that in-place progress bars
    (which write directly to stdout) don't interleave.
    """

    def __init__(self):
        super().__init__(stream=sys.stderr)
        self.setFormatter(QIoTFormatter())

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.stream.write(msg + "\n")
            self.stream.flush()
        except Exception:
            self.handleError(record)


class QIoTFileHandler(logging.FileHandler):
    """Plain-text rotating-free file handler."""

    def __init__(self, path: str = "qiot_run.log"):
        super().__init__(path, mode="a", encoding="utf-8")
        self.setFormatter(QIoTPlainFormatter())

    def emit(self, record: logging.LogRecord) -> None:
        try:
            super().emit(record)
        except Exception:
            self.handleError(record)


# ─────────────────────────────────────────────────────────────────────────────
# Progress bar primitives
# ─────────────────────────────────────────────────────────────────────────────

_SPARK = " ▁▂▃▄▅▆▇█"   # 9 chars for sparkline

def _bar(frac: float, width: int = 20) -> str:
    """Render a filled ASCII progress bar."""
    filled = int(round(frac * width))
    filled = max(0, min(filled, width))
    if _ANSI_OK:
        bar = _c(_BBLUE, "█" * filled) + _c(_DIM, "░" * (width - filled))
    else:
        bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}]"


def _spark_line(values: list[float], width: int = 16) -> str:
    """Render a Unicode sparkline from a list of float values."""
    if len(values) < 2:
        return " " * width
    # Normalise to [0, 8]
    lo, hi = min(values), max(values)
    span = hi - lo if hi != lo else 1.0
    # Reverse: high energy = low sparkline position (descending is good)
    normalised = [8 - int(round(((v - lo) / span) * 8)) for v in values]
    chars = [_SPARK[n] for n in normalised[-width:]]
    return "".join(chars).ljust(width)


# ─────────────────────────────────────────────────────────────────────────────
# QAOAProgressTracker — in-place overwriting single-line display
# ─────────────────────────────────────────────────────────────────────────────

class QAOAProgressTracker:
    """
    Real-time, in-place progress display for a single QAOA optimisation run.

    Overwrites the same terminal line on each call to ``update()``,
    then emits a final summary line on ``finish()``.

    Parameters
    ----------
    algorithm : str
        Short algorithm label, e.g. 'WS-QAOA' or 'Cold-QAOA'.
    n_tasks : int
        Problem size (for display).
    max_iter : int
        Maximum optimiser iterations.
    fidelity : float
        QPU fidelity for this run.
    trial : int
        Trial index (1-based).
    """

    def __init__(
        self,
        algorithm:  str,
        n_tasks:    int,
        max_iter:   int,
        fidelity:   float = 1.0,
        trial:      int   = 1,
    ) -> None:
        self.algorithm   = algorithm
        self.n_tasks     = n_tasks
        self.max_iter    = max_iter
        self.fidelity    = fidelity
        self.trial       = trial
        self._history:   list[float] = []
        self._total_shots = 0
        self._t_start    = time.perf_counter()
        self._last_e     = float("inf")
        self._lock       = threading.Lock()
        self._active     = True

        label = _c(_BOLD + _BBLUE, f"[{algorithm}]") if _ANSI_OK else f"[{algorithm}]"
        sys.stdout.write(
            f"\n  {label}  n={n_tasks}  trial={trial}  F={fidelity:.4f}\n"
        )
        sys.stdout.flush()

    def update(
        self,
        iteration:   int,
        energy:      float,
        n_shots:     int,
        delta_param: float,
    ) -> None:
        """
        Overwrite the current line with updated QAOA iteration stats.

        Parameters
        ----------
        iteration : int
            Current iteration number (1-based).
        energy : float
            Current ⟨H⟩ estimate.
        n_shots : int
            Shots used this iteration.
        delta_param : float
            L∞ parameter update magnitude.
        """
        if not self._active:
            return
        with self._lock:
            self._history.append(energy)
            self._total_shots += n_shots
            self._last_e = energy

            frac      = iteration / max(self.max_iter, 1)
            bar       = _bar(frac, width=18)
            pct       = f"{frac * 100:5.1f}%"
            spark     = _spark_line(self._history, width=12)
            elapsed   = time.perf_counter() - self._t_start

            e_str   = f"{energy:+10.4f}"
            d_str   = f"{delta_param:.4f}"
            s_str   = f"{n_shots:>6,}"
            tot_str = f"{self._total_shots:>9,}"
            t_str   = f"{elapsed:5.1f}s"

            if _ANSI_OK:
                e_str   = _c(_BWHITE, e_str)
                d_str   = (_c(_BGREEN, d_str) if delta_param < 0.05 else _c(_BYELLOW, d_str))
                spark   = _c(_BBLUE, spark)
                tot_str = _c(_BCYAN, tot_str)

            line = (
                f"\r    iter {iteration:>4}/{self.max_iter}  "
                f"{bar} {pct}  "
                f"E={e_str}  Δ={d_str}  "
                f"shots/iter={s_str}  total={tot_str}  "
                f"spark=[{spark}]  {t_str}"
            )
            # Pad to terminal width to overwrite leftover chars
            sys.stdout.write(line[:200].ljust(200) + "\r")
            sys.stdout.flush()

    def finish(
        self,
        converged:    bool,
        total_shots:  int,
        objective:    float | None = None,
        runtime_ms:   float | None = None,
    ) -> None:
        """
        Emit a final summary line and free the live display.

        Parameters
        ----------
        converged : bool
            Whether QAOA converged within tolerance.
        total_shots : int
            Total shots used across all iterations.
        objective : float, optional
            Final decoded objective value.
        runtime_ms : float, optional
            Total wall-clock time in milliseconds.
        """
        self._active = False
        elapsed = (time.perf_counter() - self._t_start) * 1000

        if converged:
            status = _c(_BOLD + _BGREEN, "✓ CONVERGED") if _ANSI_OK else "CONVERGED"
        else:
            status = _c(_YELLOW, "○ MAX-ITER")  if _ANSI_OK else "MAX-ITER"

        n_iters = len(self._history)
        obj_str = f"  obj={objective:.4f}" if objective is not None else ""
        rt_str  = f"  wall={elapsed:.0f}ms"
        shots_str = f"{total_shots:,}"
        if _ANSI_OK:
            shots_str = _c(_BCYAN, shots_str)

        # Clear progress line, print summary
        sys.stdout.write("\r" + " " * 200 + "\r")
        sys.stdout.write(
            f"  [{self.algorithm}] {status}  "
            f"iters={n_iters}  shots={shots_str}"
            f"{obj_str}{rt_str}\n"
        )
        sys.stdout.flush()


# ─────────────────────────────────────────────────────────────────────────────
# BenchmarkProgressTracker — live results table
# ─────────────────────────────────────────────────────────────────────────────

_ALGS  = ["WS-QAOA", "Cold-QAOA", "SA", "ACO", "PSO", "Greedy"]
_ALGS_SHORT = ["WS-Q", "Cold", "SA", "ACO", "PSO", "Grd"]

class BenchmarkProgressTracker:
    """
    Prints a live, updating results table to stdout as benchmark results arrive.

    Each call to ``record()`` appends a row; ``print_table()`` can be called
    at any time to dump the current state.

    Parameters
    ----------
    task_sizes : list of int
    n_trials : int
    """

    def __init__(self, task_sizes: list[int], n_trials: int) -> None:
        self.task_sizes = task_sizes
        self.n_trials   = n_trials
        self._rows: list[dict] = []
        self._lock = threading.Lock()
        self._t_start = time.perf_counter()
        self._print_header()

    def _print_header(self) -> None:
        w = 108
        title = " Q-IoT BENCHMARK — Live Results "
        pad   = (w - len(title)) // 2
        bar   = "═" * w
        if _ANSI_OK:
            bar   = _c(_BBLUE, bar)
            title = _c(_BOLD + _BWHITE, title)
        sys.stdout.write(f"\n  ╔{bar}╗\n")
        sys.stdout.write(f"  ║{' ' * pad}{title}{' ' * (w - pad - len(title.replace(chr(27) + '[' + chr(27) + 'm', '')))}║\n")
        sys.stdout.write(f"  ╚{bar}╝\n\n")

        hdr = f"  {'Tasks':>5}  {'Scen':>6}  {'Trial':>5}  "
        hdr += "  ".join(f"{a:>9}" for a in _ALGS_SHORT)
        hdr += f"  {'Best':>9}  {'Time':>7}"
        sep = "  " + "─" * (len(hdr) - 2)
        if _ANSI_OK:
            hdr = _c(_BOLD + _BWHITE, hdr)
            sep = _c(_DIM, sep)
        sys.stdout.write(hdr + "\n")
        sys.stdout.write(sep + "\n")
        sys.stdout.flush()

    def record(
        self,
        n_tasks:   int,
        scenario:  str,
        trial:     int,
        results:   dict[str, float],   # alg_name -> quality_ratio
        elapsed_ms: float,
    ) -> None:
        """
        Record one completed (size, scenario, trial) result and print a row.

        Parameters
        ----------
        n_tasks : int
        scenario : str  — 'high', 'medium', or 'low'
        trial : int     — 1-based
        results : dict  — alg_name -> quality_ratio (lower is better)
        elapsed_ms : float
        """
        with self._lock:
            self._rows.append({
                "n": n_tasks, "scenario": scenario,
                "trial": trial, "results": results,
                "elapsed_ms": elapsed_ms,
            })
            self._print_row(n_tasks, scenario, trial, results, elapsed_ms)

    def _print_row(
        self,
        n_tasks: int,
        scenario: str,
        trial: int,
        results: dict[str, float],
        elapsed_ms: float,
    ) -> None:
        scen_col = {"high": "HIGH", "medium": "MED ", "low": "LOW "}.get(scenario, scenario[:4])
        if _ANSI_OK:
            scen_col = {
                "HIGH": _c(_BGREEN,  "HIGH"),
                "MED ": _c(_BYELLOW, "MED "),
                "LOW ": _c(_BRED,    "LOW "),
            }.get(scen_col, scen_col)

        vals = []
        min_val = min((v for v in results.values() if v is not None), default=float("inf"))
        for alg in _ALGS:
            v = results.get(alg)
            if v is None:
                vals.append(_c(_DIM, "    —    ") if _ANSI_OK else "    —    ")
            else:
                s = f"{v:8.4f}"
                if abs(v - min_val) < 1e-6:
                    s = _c(_BGREEN + _BOLD, s) if _ANSI_OK else f"*{s}*"
                vals.append(s)

        best_alg = min(results, key=lambda k: results.get(k, float("inf")))
        t_str    = f"{elapsed_ms / 1000:6.1f}s"

        row = (
            f"  {n_tasks:>5}  {scen_col:>6}  {trial:>5}  "
            + "  ".join(f"{v:>9}" for v in vals)
            + f"  {best_alg:>9}  {t_str:>7}"
        )
        sys.stdout.write(row + "\n")
        sys.stdout.flush()

    def finish(self) -> None:
        """Print a separator after the last row."""
        elapsed = time.perf_counter() - self._t_start
        total_rows = len(self._rows)
        sep = "  " + "─" * 106
        if _ANSI_OK:
            sep = _c(_DIM, sep)
        msg = f"  {total_rows} experiments completed in {elapsed:.1f}s"
        if _ANSI_OK:
            msg = _c(_DIM, msg)
        sys.stdout.write(sep + "\n")
        sys.stdout.write(msg + "\n\n")
        sys.stdout.flush()


# ─────────────────────────────────────────────────────────────────────────────
# Specialised structured event helpers
# (use these instead of log.info for richer context)
# ─────────────────────────────────────────────────────────────────────────────

def log_fidelity_check(
    logger: logging.Logger,
    fidelity: float,
    qpu_name: str,
    qaoa_depth: int,
    viable: bool,
) -> None:
    """
    Emit a FIDELITY-level log entry with a mini inline bar.

    Parameters
    ----------
    logger : logging.Logger
    fidelity : float     — F(q,t) value
    qpu_name : str       — e.g. 'IBM'
    qaoa_depth : int     — p
    viable : bool        — whether quantum execution is feasible
    """
    bar_w  = 12
    filled = int(round(fidelity * bar_w))
    bar    = ("▓" * filled + "░" * (bar_w - filled))

    if fidelity >= 0.95:
        zone = _c(_BGREEN,  "■ HIGH   ") if _ANSI_OK else "HIGH   "
    elif fidelity >= 0.85:
        zone = _c(_BYELLOW, "■ MEDIUM ") if _ANSI_OK else "MEDIUM "
    else:
        zone = _c(_BRED,    "■ LOW    ") if _ANSI_OK else "LOW    "

    viable_str = (_c(_BGREEN, "viable") if viable else _c(_BRED, "not-viable")) if _ANSI_OK else str(viable)

    logger.log(
        FIDELITY,
        "F(q,t) = %.4f  │  %s  │  p=%d  │  [%s]  %s  │  %s",
        fidelity, qpu_name, qaoa_depth, bar, zone, viable_str,
    )


def log_routing_decision(
    logger: logging.Logger,
    decision: str,
    fidelity: float,
    theta: float | None = None,
    reason: str = "",
) -> None:
    """
    Emit a DECISION-level log entry for the quantum/classical routing choice.

    Parameters
    ----------
    logger : logging.Logger
    decision : str   — 'WS-QAOA', 'Cold-QAOA', 'SA-fallback', etc.
    fidelity : float
    theta : float    — quantum advantage Θ (optional)
    reason : str     — short explanation
    """
    arrow = "→"
    if decision in ("WS-QAOA", "Cold-QAOA"):
        label = _c(_BOLD + _BBLUE, f"▶ QUANTUM  [{decision}]") if _ANSI_OK else f"QUANTUM [{decision}]"
    else:
        label = _c(_BOLD + _BYELLOW, f"◀ CLASSICAL [{decision}]") if _ANSI_OK else f"CLASSICAL [{decision}]"

    theta_str = f"  Θ={theta:.3f}" if theta is not None else ""
    logger.log(
        DECISION,
        "%s %s  F=%.4f%s%s",
        arrow, label, fidelity, theta_str,
        f"  ({reason})" if reason else "",
    )


def log_shot_allocation(
    logger: logging.Logger,
    iteration: int,
    delta_param: float,
    n_shots: int,
    total_shots: int,
) -> None:
    """
    Emit a SHOT-level log entry for adaptive shot allocation.

    Parameters
    ----------
    logger : logging.Logger
    iteration : int
    delta_param : float  — parameter update magnitude
    n_shots : int        — shots this iteration
    total_shots : int    — cumulative shots
    """
    logger.log(
        SHOT,
        "iter=%4d  Δparam=%.4f  shots=%6d  cumulative=%9d",
        iteration, delta_param, n_shots, total_shots,
    )


def log_qos_result(
    logger: logging.Logger,
    compliance: float,
    avg_latency_ms: float,
    fallback: bool,
    objective: float,
) -> None:
    """
    Emit a QOS_EVENT-level log entry after elastic allocation.

    Parameters
    ----------
    logger : logging.Logger
    compliance : float       — fraction of tasks meeting QoS budget
    avg_latency_ms : float
    fallback : bool          — whether classical fallback was triggered
    objective : float        — final assignment cost
    """
    c_str = f"{compliance * 100:.2f}%"
    if _ANSI_OK:
        c_str = (_c(_BGREEN, c_str) if compliance >= 0.99 else _c(_BYELLOW, c_str))
    fb_str = (_c(_BRED, "YES — fallback triggered") if fallback
              else _c(_BGREEN, "NO")) if _ANSI_OK else str(fallback)

    logger.log(
        QOS_EVENT,
        "compliance=%s  avg_latency=%.2fms  fallback=%s  obj=%.4f",
        c_str, avg_latency_ms, fb_str, objective,
    )


def log_qubo_build(
    logger: logging.Logger,
    n_qubits: int,
    n_z_terms: int,
    n_zz_terms: int,
    penalty: float,
) -> None:
    """Log QUBO / Ising Hamiltonian construction statistics."""
    logger.info(
        "QUBO built: %d qubits  │  %d Z-terms  │  %d ZZ-terms  │  penalty P=%.2f",
        n_qubits, n_z_terms, n_zz_terms, penalty,
    )


def log_advantage_metric(
    logger: logging.Logger,
    theta: float,
    threshold: float,
    t_classical_ms: float,
    t_quantum_ms: float,
) -> None:
    """Log the computed quantum advantage metric Θ."""
    zone = (
        (_c(_BGREEN + _BOLD, "■ QUANTUM ZONE") if theta > threshold
         else _c(_BRED, "■ CLASSICAL ZONE"))
        if _ANSI_OK
        else ("QUANTUM ZONE" if theta > threshold else "CLASSICAL ZONE")
    )
    logger.info(
        "Θ = %.3f  [threshold=%.1f]  %s  │  T_cl=%.1fms  T_qu=%.1fms",
        theta, threshold, zone, t_classical_ms, t_quantum_ms,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Final summary printer
# ─────────────────────────────────────────────────────────────────────────────

def print_final_summary(results: dict, elapsed_total_s: float = 0.0) -> None:
    """
    Print a boxed summary table to stdout after the full benchmark completes.

    Parameters
    ----------
    results : dict
        Benchmark results dict from benchmark_runner.run_benchmarks().
    elapsed_total_s : float
        Total wall-clock seconds for the full run.
    """
    W = 72
    bar   = "═" * W
    thin  = "─" * W

    def _box_line(text: str, fill: str = " ") -> None:
        inner = text[:W - 2].ljust(W - 2)
        if _ANSI_OK:
            sys.stdout.write(f"  ║ {inner} ║\n")
        else:
            sys.stdout.write(f"  | {inner} |\n")

    def _separator(ch: str = "═") -> None:
        ln = ch * W
        if _ANSI_OK:
            sys.stdout.write(f"  ╠{_c(_BBLUE, ln)}╣\n")
        else:
            sys.stdout.write(f"  +{ln}+\n")

    if _ANSI_OK:
        sys.stdout.write(f"\n  ╔{_c(_BBLUE, bar)}╗\n")
    else:
        sys.stdout.write(f"\n  +{'=' * W}+\n")

    title = "  BENCHMARK COMPLETE — Q-IoT SIMULATION SUMMARY  "
    _box_line(title.center(W - 2))
    _separator()

    h, m, s = int(elapsed_total_s // 3600), int((elapsed_total_s % 3600) // 60), int(elapsed_total_s % 60)
    _box_line(f"  Total runtime: {h:02d}:{m:02d}:{s:02d}   |   "
              f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    _separator("─")

    exps = results.get("experiments", {})

    # Header row
    hdr = f"  {'n':>4}  {'Scenario':>8}  " + "  ".join(f"{a:>9}" for a in _ALGS_SHORT)
    _box_line(hdr)
    _box_line("  " + "·" * (W - 4))

    best_wins = {a: 0 for a in _ALGS}

    for n_str in sorted(exps.keys(), key=int):
        for scn in ["high", "medium", "low"]:
            row_data = exps[n_str].get(scn, {})
            vals_num = {}
            cells    = []
            for alg in _ALGS:
                try:
                    m = row_data[alg]["mean_quality_ratio"]
                    s = row_data[alg]["std_quality_ratio"]
                    vals_num[alg] = m
                    cells.append(f"{m:5.3f}±{s:.3f}")
                except (KeyError, TypeError):
                    cells.append("   —     ")

            if vals_num:
                best_alg = min(vals_num, key=vals_num.get)
                best_wins[best_alg] += 1
                # Bold best in each row
                best_idx = _ALGS.index(best_alg)
                cells[best_idx] = f"*{cells[best_idx]}*"

            row = f"  {n_str:>4}  {scn:>8}  " + "  ".join(f"{c:>9}" for c in cells)
            _box_line(row)

    _separator("─")

    # Win count summary
    _box_line("  Algorithm wins (best quality ratio across all experiments):")
    for alg, wins in sorted(best_wins.items(), key=lambda x: -x[1]):
        bar_w = int(wins / max(sum(best_wins.values()), 1) * 30)
        bar_s = "▓" * bar_w + "░" * (30 - bar_w)
        _box_line(f"    {alg:<12} {bar_s}  {wins:>3} wins")

    _separator()

    # Shot usage note
    total_shots = 0
    for n_str, scn_data in exps.items():
        for scn, alg_data in scn_data.items():
            for alg in ["WS-QAOA", "Cold-QAOA"]:
                try:
                    total_shots += int(alg_data[alg]["mean_total_shots"])
                except (KeyError, TypeError):
                    pass

    if total_shots > 0:
        _box_line(f"  Estimated total QAOA shots: {total_shots:,}")

    _box_line(f"  Results saved to: results.json  |  Figures: figures/  |  Tables: tables/")

    if _ANSI_OK:
        sys.stdout.write(f"  ╚{_c(_BBLUE, bar)}╝\n\n")
    else:
        sys.stdout.write(f"  +{'=' * W}+\n\n")

    sys.stdout.flush()


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark phase banner
# ─────────────────────────────────────────────────────────────────────────────

def print_phase_banner(phase: str, detail: str = "") -> None:
    """
    Print a compact section banner to stdout.

    Parameters
    ----------
    phase : str
        Phase label, e.g. 'n_tasks = 30  │  MEDIUM fidelity'.
    detail : str
        Optional extra context on the same line.
    """
    ts = datetime.now().strftime("%H:%M:%S")
    text = f"── {phase}"
    if detail:
        text += f"  │  {detail}"
    text += f"  [{ts}]"
    if _ANSI_OK:
        sys.stdout.write(_c(_BOLD + _BWHITE, f"\n  {text}\n"))
    else:
        sys.stdout.write(f"\n  {text}\n")
    sys.stdout.flush()


# ─────────────────────────────────────────────────────────────────────────────
# setup_logging — installs globally, call once before any other imports
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(
    level:    int  = logging.DEBUG,
    log_file: str  = "qiot_run.log",
    verbose:  bool = True,
) -> None:
    """
    Configure the root logger with QIoT handlers, replacing any existing setup.

    Because ``logging.basicConfig`` in the other modules only fires if the
    root logger has no handlers, calling this function BEFORE importing those
    modules makes their ``basicConfig`` calls silent no-ops.

    If imported AFTER them, this function forcibly removes existing handlers
    and installs QIoT ones — so it is safe to call at any point.

    Parameters
    ----------
    level : int
        Minimum log level for console output.  Set to ``logging.WARNING``
        to suppress INFO-level chatter.  QAOA_ITER (15) shows per-iteration
        detail; INFO (20) is the default.
    log_file : str
        Path for the plain-text log file.  Pass ``""`` to disable file logging.
    verbose : bool
        If False, sets level to WARNING (overrides ``level``).
    """
    if not verbose:
        level = logging.WARNING

    root = logging.getLogger()
    root.handlers.clear()          # evict any previous basicConfig handlers
    root.setLevel(logging.DEBUG)   # root captures everything; handlers filter

    console = QIoTConsoleHandler()
    console.setLevel(level)
    root.addHandler(console)

    if log_file:
        fh = QIoTFileHandler(log_file)
        fh.setLevel(logging.DEBUG)   # always full detail in file
        root.addHandler(fh)

    # Silence noisy third-party loggers that will otherwise spam
    for noisy in ("qiskit", "qiskit_aer", "qiskit_ibm", "stevedore",
                  "urllib3", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if _ANSI_OK:
        sys.stdout.write(
            _c(_BOLD + _BBLUE, "\n  ╔══  Q-IoT Real-Time Logger active") +
            _c(_DIM,  f"  │  level={logging.getLevelName(level)}"
                      f"  │  file={log_file or 'disabled'}"
                      f"  │  ANSI=ON  ══╗\n\n")
        )
    else:
        sys.stdout.write(
            f"\n  [Q-IoT Logger] level={logging.getLevelName(level)}"
            f"  file={log_file or 'disabled'}  ANSI=OFF\n\n"
        )
    sys.stdout.flush()


def get_logger(name: str) -> logging.Logger:
    """
    Return a standard Logger for the given module name.

    Identical to ``logging.getLogger(name)`` but ensures that the root
    logger has QIoT handlers installed (calls setup_logging if needed).

    Parameters
    ----------
    name : str
        Typically ``__name__``.

    Returns
    -------
    logging.Logger
    """
    root = logging.getLogger()
    if not any(isinstance(h, QIoTConsoleHandler) for h in root.handlers):
        setup_logging()
    return logging.getLogger(name)
