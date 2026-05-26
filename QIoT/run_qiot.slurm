#!/bin/bash
#SBATCH --job-name=me
#SBATCH --output=qiot_slurm_%j.log          # stdout + stderr → live log (%j = job ID)
#SBATCH --error=qiot_slurm_%j.log           # merge stderr into same file
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00                     # adjust if the full benchmark takes longer
#SBATCH --partition=cpu                     # change to your cluster's partition name
##SBATCH --gres=gpu:0                       # uncomment/edit if a GPU node is preferred

# ─── Optional: email notifications ────────────────────────────────────────────
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=your_email@example.com

# ══════════════════════════════════════════════════════════════════════════════
#  Q-IoT Simulation — SLURM batch script
#  ─────────────────────────────────────
#  • Installs all Python dependencies into a per-user venv (no sudo needed)
#  • Runs the full benchmark: benchmarks → figures → LaTeX tables
#  • All output (stdout + stderr) streams to  qiot_slurm_<jobid>.log  live,
#    so you can tail it with:
#       tail -f qiot_slurm_<jobid>.log
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail          # exit on any error, unset variable, or pipe failure

# ─── 0. Timestamp helpers ──────────────────────────────────────────────────────
log_ts() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log_ts "========================================================"
log_ts "  Job ID   : ${SLURM_JOB_ID}"
log_ts "  Job Name : ${SLURM_JOB_NAME}"
log_ts "  Node     : $(hostname)"
log_ts "  CPUs     : ${SLURM_CPUS_PER_TASK}"
log_ts "  Mem      : ${SLURM_MEM_PER_NODE:-unset} MB"
log_ts "  Start    : $(date)"
log_ts "========================================================"

# ─── 1. Load cluster modules (adjust names to your HPC module system) ──────────
# Uncomment and edit as needed for your cluster:
# module purge
# module load python/3.12
# module load gcc/12

# ─── 2. Resolve working directory ─────────────────────────────────────────────
# SLURM_SUBMIT_DIR is set automatically to the directory where sbatch was called
WORK_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
log_ts "Working directory: ${WORK_DIR}"
cd "${WORK_DIR}"

# ─── 3. Python environment setup ──────────────────────────────────────────────
VENV_DIR="${WORK_DIR}/.venv_qiot"

if [ ! -d "${VENV_DIR}" ]; then
    log_ts "Creating virtual environment at ${VENV_DIR} ..."
    python3 -m venv "${VENV_DIR}"
    log_ts "Virtual environment created."
else
    log_ts "Re-using existing virtual environment at ${VENV_DIR}"
fi

# Activate venv
source "${VENV_DIR}/bin/activate"
log_ts "Python interpreter: $(which python3)"
log_ts "Python version    : $(python3 --version)"

# ─── 4. Install / upgrade requirements ────────────────────────────────────────
log_ts "Upgrading pip ..."
pip install --quiet --upgrade pip

log_ts "Installing Q-IoT dependencies from requirements.txt ..."
pip install --quiet -r requirements.txt

log_ts "Installed packages:"
pip list | grep -E "qiskit|numpy|scipy|matplotlib|pandas"

# ─── 5. Sanity-check imports ───────────────────────────────────────────────────
log_ts "Running import sanity check ..."
python3 - <<'PYEOF'
import qiskit, qiskit_aer, qiskit_algorithms, numpy, scipy, matplotlib, pandas
print(f"  qiskit            {qiskit.__version__}")
print(f"  qiskit-aer        {qiskit_aer.__version__}")
print(f"  qiskit-algorithms {qiskit_algorithms.__version__}")
print(f"  numpy             {numpy.__version__}")
print(f"  scipy             {scipy.__version__}")
print(f"  matplotlib        {matplotlib.__version__}")
print(f"  pandas            {pandas.__version__}")
PYEOF
log_ts "Import check passed."

# ─── 6. Run simulation ─────────────────────────────────────────────────────────
log_ts "========================================================"
log_ts "  Launching Q-IoT full benchmark ..."
log_ts "========================================================"

# Python's stdout/stderr is line-buffered in a tty but fully buffered when
# piped to a file.  PYTHONUNBUFFERED=1 forces line-by-line output so that
# `tail -f qiot_slurm_<jobid>.log` shows progress in real time.
export PYTHONUNBUFFERED=1

python3 run_all.py \
    --output results.json \
    --log-level INFO

EXIT_CODE=$?

# ─── 7. Post-run summary ──────────────────────────────────────────────────────
log_ts "========================================================"
if [ ${EXIT_CODE} -eq 0 ]; then
    log_ts "  SUCCESS — simulation completed normally."
    log_ts "  Output files:"
    [ -f results.json ]    && log_ts "    results.json      $(du -sh results.json    | cut -f1)"
    [ -d figures ]         && log_ts "    figures/           $(ls figures/ | wc -l) files"
    [ -d tables ]          && log_ts "    tables/            $(ls tables/  | wc -l) files"
    [ -f qiot_run.log ]    && log_ts "    qiot_run.log       $(du -sh qiot_run.log   | cut -f1)"
else
    log_ts "  FAILED — exit code ${EXIT_CODE}"
fi
log_ts "  End: $(date)"
log_ts "========================================================"

exit ${EXIT_CODE}
