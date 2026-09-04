#!/bin/bash
#SBATCH --job-name=sogudiff-sicnav
#SBATCH --output=logs/eval_sicnav_%j.out
#SBATCH --error=logs/eval_sicnav_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#
# SICNav is evaluated by a separate job because it needs its own interpreter:
# its CasADi/IPOPT plugin resolves only in the environment built from
# requirements/sicnav.txt. No GPU is requested -- SICNav is an MPC.
#
#   sbatch scripts/slurm/evaluate_sicnav.slurm

# sbatch runs a COPY of this script from the scheduler's spool directory, so
# $0 does not point into the repository. Try $0's directory first (direct
# execution), then the directory sbatch was submitted from.
_SLURM_COMMON=""
for _d in "$(dirname "$0")" "${SLURM_SUBMIT_DIR:-}/scripts/slurm" "${SOGUDIFF_ROOT:-}/scripts/slurm"; do
    if [ -n "$_d" ] && [ -f "$_d/_common.sh" ]; then _SLURM_COMMON="$_d/_common.sh"; break; fi
done
if [ -z "$_SLURM_COMMON" ]; then
    echo "Cannot locate scripts/slurm/_common.sh. Submit from the repository root," >&2
    echo "or export SOGUDIFF_ROOT=/path/to/sogudiff before sbatch." >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$_SLURM_COMMON"
activate_sicnav_env
setup_threads

export PYTHONPATH="$SOGUDIFF_ROOT/crowdnav_env:${PYTHONPATH:-}"

cd "$SOGUDIFF_ROOT/crowdnav_env/crowd_nav"

echo "=== evaluating SICNav  node=$(hostname) ==="
"$PYTHON" evaluate.py \
    --policy sicnav \
    --policy_config configs/policy_sicnav.config \
    --env_config    configs/env.config \
    --phase test \
    --no_video \
    --results_suffix sicnav
