# Site configuration for the SLURM wrappers. Sourced by every job script here.
#
# Copy to cluster_env.local.sh and edit that instead -- the .local file is
# gitignored, so your account details never enter version control. Values here
# are placeholders and every one of them will need changing.
#
# If you do not use SLURM, ignore this directory entirely: each wrapper simply
# activates an environment and runs a documented `python ...` command that you
# can run directly. See docs/RUNNING.md.

# --- scheduler ---------------------------------------------------------------
# NOTE: an account and a partition cannot be applied from in here. #SBATCH
# directives are read by sbatch before the job starts, and this file is sourced
# only once the job is already running. The wrappers therefore carry no
# --account/--partition of their own, so that they stay portable; supply
# whatever your site requires at submission time:
#
#   sbatch --account=ACCT scripts/slurm/evaluate.slurm sogudiff
#
# or, to avoid repeating it, export sbatch's own environment variables in your
# submitting shell (sbatch reads these directly):
#
#   export SBATCH_ACCOUNT=ACCT SBATCH_PARTITION=PART
#
# Many sites need neither: if you have exactly one association, sbatch picks it.
# The values below are used by nothing automatically -- they are a place to
# record your site's settings alongside the rest of this configuration.
export SOGUDIFF_ACCOUNT="${SOGUDIFF_ACCOUNT:-def-youraccount}"
export SOGUDIFF_PARTITION="${SOGUDIFF_PARTITION:-}"
export SOGUDIFF_MAIL="${SOGUDIFF_MAIL:-}"          # empty disables mail

# --- environment activation --------------------------------------------------
# Replace with whatever your site needs. Two common shapes:
#
#   module load python/3.10 cuda/12.6
#   source /path/to/.venv/bin/activate
#
#   source ~/miniconda3/etc/profile.d/conda.sh && conda activate sogudiff
activate_main_env() {
    if [ -n "${SOGUDIFF_VENV:-}" ]; then
        # shellcheck disable=SC1091
        source "${SOGUDIFF_VENV}/bin/activate"
    else
        echo "Set SOGUDIFF_VENV, or edit activate_main_env in cluster_env.sh." >&2
        return 1
    fi
}

# SICNav needs its own interpreter -- see requirements/sicnav.txt.
activate_sicnav_env() {
    if [ -n "${SOGUDIFF_SICNAV_PYTHON:-}" ]; then
        export PYTHON="${SOGUDIFF_SICNAV_PYTHON}"
    else
        echo "Set SOGUDIFF_SICNAV_PYTHON to the SICNav environment's python." >&2
        return 1
    fi
}

# --- acados ------------------------------------------------------------------
# Required by the diffusion policy's projection layer and by the MPPI baseline.
setup_acados() {
    export ACADOS_SOURCE_DIR="${ACADOS_SOURCE_DIR:-$SOGUDIFF_ROOT/third_party/acados}"
    export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="$ACADOS_SOURCE_DIR/interfaces/acados_template:${PYTHONPATH:-}"
}

# --- thread budget -----------------------------------------------------------
# The MPPI kernels are numba-parallel and the expert generator runs many
# workers. Leaving each worker single-threaded avoids oversubscribing the
# allocation and removes a source of run-to-run timing variance.
setup_threads() {
    export NUMBA_NUM_THREADS="${MPPI_THREADS:-1}"
    export OMP_NUM_THREADS="${MPPI_THREADS:-1}"
    export MKL_NUM_THREADS="${MPPI_THREADS:-1}"
    export PYTHONUNBUFFERED=1
}
