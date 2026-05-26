"""
run_all.py — Main entry point for the Q-IoT simulation framework.

Usage:
    python run_all.py                  # full benchmark
    python run_all.py --quick          # quick smoke-test (small sizes, few trials)
    python run_all.py --log-level WARN # suppress INFO chatter
    python run_all.py --no-log-file    # console only, no qiot_run.log
"""

# ── Logger MUST be set up before any other Q-IoT import so that
#    logging.basicConfig calls inside those modules become no-ops.
import qiot_logger
import argparse
import time
import sys
import numpy as np

from config import TASK_SIZES, RANDOM_SEED

np.random.seed(RANDOM_SEED)


def main():
    parser = argparse.ArgumentParser(description="Q-IoT Benchmark Runner")
    parser.add_argument("--quick",       action="store_true",
                        help="Quick mode: small sizes, 2 trials each")
    parser.add_argument("--output",      default="results.json",
                        help="Path for results JSON output")
    parser.add_argument("--no-plots",    action="store_true",
                        help="Skip figure generation")
    parser.add_argument("--no-tables",   action="store_true",
                        help="Skip LaTeX table generation")
    parser.add_argument("--no-log-file", action="store_true",
                        help="Disable qiot_run.log file output")
    parser.add_argument("--log-level",   default="INFO",
                        choices=["DEBUG", "QAOA", "INFO", "WARN", "ERROR"],
                        help="Console log verbosity (QAOA shows per-iteration detail)")
    args = parser.parse_args()

    # ── Configure logger ──────────────────────────────────────────────────────
    import logging
    level_map = {
        "DEBUG": logging.DEBUG,
        "QAOA":  qiot_logger.QAOA_ITER,
        "INFO":  logging.INFO,
        "WARN":  logging.WARNING,
        "ERROR": logging.ERROR,
    }
    qiot_logger.setup_logging(
        level    = level_map[args.log_level],
        log_file = "" if args.no_log_file else "qiot_run.log",
    )

    # ── Deferred imports (after logger is live) ───────────────────────────────
    from benchmark_runner import run_benchmarks

    # ── Run benchmarks ─────────────────────────────────────────────────────────
    qiot_logger.print_phase_banner("PHASE 1/3 — Benchmarks", "running experiments")
    t_bench_start = time.perf_counter()

    results = run_benchmarks(
        task_sizes  = None,
        n_trials    = 2 if args.quick else 10,
        output_path = args.output,
        quick_mode  = args.quick,
    )
    t_bench_s = time.perf_counter() - t_bench_start

    # ── Figures ────────────────────────────────────────────────────────────────
    if not args.no_plots:
        qiot_logger.print_phase_banner("PHASE 2/3 — Figures", "generating IEEE-ready PDFs")
        try:
            import plots
            plots.generate_all_figures(results)
        except Exception as exc:
            logging.getLogger(__name__).error("Figure generation failed: %s", exc)
    else:
        qiot_logger.print_phase_banner("PHASE 2/3 — Figures", "skipped (--no-plots)")

    # ── LaTeX tables ───────────────────────────────────────────────────────────
    if not args.no_tables:
        qiot_logger.print_phase_banner("PHASE 3/3 — Tables", "generating LaTeX")
        try:
            import tables
            tables.generate_all_tables(results)
        except Exception as exc:
            logging.getLogger(__name__).error("Table generation failed: %s", exc)
    else:
        qiot_logger.print_phase_banner("PHASE 3/3 — Tables", "skipped (--no-tables)")

    # ── Final summary ──────────────────────────────────────────────────────────
    qiot_logger.print_final_summary(results, elapsed_total_s=t_bench_s)


if __name__ == "__main__":
    main()
