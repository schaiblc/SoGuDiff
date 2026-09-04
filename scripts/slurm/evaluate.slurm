#!/bin/bash
#SBATCH --job-name=sogudiff-eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#
# Evaluate ONE policy on the evaluation scene set.
#
#   sbatch scripts/slurm/evaluate.slurm sogudiff
#   sbatch scripts/slurm/evaluate.slurm sarl models/sarl
#
# $1 = policy name (see crowd_nav/policy/policy_factory.py)
# $2 = model directory, for the baselines that need one (optional)
#
# Equivalent single-machine command, which is all this wrapper runs:
#
#   cd crowdnav_env/crowd_nav && python evaluate.py \
#       --policy <name> [--model_dir <dir>] --gpu --no_video \
#       --results_suffix <name>
#
# Scenes come from configs/env.config (test_sim = evaluation, test_size = 500)
# with fixed per-scene seeds, so no scene download is needed. To score a saved
# scene set instead, set SOGUDIFF_EVAL_SCENES and this adds --npz_hard.

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
activate_main_env
setup_acados
setup_threads

POLICY="${1:?usage: evaluate.slurm <policy> [model_dir]}"
MODEL_DIR="${2:-}"
SCENES="${SOGUDIFF_EVAL_SCENES:-}"

cd "$SOGUDIFF_ROOT/crowdnav_env/crowd_nav"

EXTRA=()
[ -n "$MODEL_DIR" ] && EXTRA=(--model_dir "$MODEL_DIR")
# Optional: score a saved .npz scene set rather than the procedural benchmark.
[ -n "$SCENES" ] && EXTRA+=(--npz_hard --npz_dir "$SCENES"
                            --npz_num_eval "${SOGUDIFF_EVAL_N:-500}")
# The proposed method is evaluated with per-axis guidance and weight
# normalization; these flags are ignored by every other policy.
case "$POLICY" in
    sogudiff) EXTRA+=(--infer_mode per_axis --w_normalize) ;;
esac

echo "=== evaluating ${POLICY}  node=$(hostname)  scenes=${SCENES:-procedural} ==="
python evaluate.py \
    --policy "$POLICY" \
    "${EXTRA[@]}" \
    --phase test \
    --gpu \
    --no_video \
    --results_suffix "$POLICY"

echo "=== done -> crowdnav_env/crowd_nav/results_${POLICY}/ ==="
