#!/bin/bash
#SBATCH --job-name=sogudiff-expert
#SBATCH --output=logs/expert_%j.out
#SBATCH --error=logs/expert_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#
# Generate expert demonstrations from a directory of scenes. CPU-only and
# embarrassingly parallel; the full dataset is a large job, so this script
# resumes rather than restarting.
#
# Equivalent single-machine command:
#
#   python sogudiff/generate_expert_trajectories.py \
#       --in_dir data/scenes/tartanground \
#       --out_dir data/expert/<source> \
#       --workers 64 --style_threads 1 --n_modes 4
#
# Resume model: a scene is complete only once its atomic <scene>.done marker
# is written, after every style and mode for that scene has been saved. Both
# the generator's skip logic and the completion check below read those markers
# rather than testing for output files, so a job killed mid-scene regenerates
# that scene in full instead of leaving it half-written.
#
#   sbatch scripts/slurm/generate_expert_data.slurm
#   sbatch scripts/slurm/generate_expert_data.slurm --force_redo   # start over
#
# To split across jobs, give each its own --offset/--limit AND its own
# --out_dir, then merge the directories; separate output directories keep
# their .done markers from colliding.

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
setup_threads

IN_DIR="${SOGUDIFF_SCENE_DIR:-$SOGUDIFF_ROOT/data/scenes/tartanground}"
OUT_DIR="${SOGUDIFF_EXPERT_OUT:-$SOGUDIFF_ROOT/data/expert/$(basename "${SOGUDIFF_SCENE_DIR:-tartanground}")}"
WORKERS="${SLURM_CPUS_PER_TASK:-16}"
# Empty by default, because OUT_DIR is per source and filenames cannot collide.
# Set SOGUDIFF_EXPERT_PREFIX only if you point several sources at ONE OUT_DIR:
# every scene directory numbers its scenes from zero, so without a distinct
# prefix they would overwrite each other and skip each other's work.
PREFIX="${SOGUDIFF_EXPERT_PREFIX:-}"

echo "=== expert generation  node=$(hostname)  in=${IN_DIR}  out=${OUT_DIR}  prefix=${PREFIX} ==="
python sogudiff/generate_expert_trajectories.py \
    --in_dir  "$IN_DIR" \
    --out_dir "$OUT_DIR" \
    --out_prefix "$PREFIX" \
    --workers "$WORKERS" \
    --style_threads 1 \
    --n_modes 4 \
    "$@"

N_IN=$(find "$IN_DIR" -maxdepth 1 -name 'sample_*.npz' | wc -l)
# Count only this source's markers, so a shared OUT_DIR still reports honestly.
N_DONE=$(find "$OUT_DIR" -maxdepth 1 -name "${PREFIX}*.done" | wc -l)   # this source only
echo "=== scenes complete: ${N_DONE} / ${N_IN} ==="
if [ "$N_DONE" -lt "$N_IN" ]; then
    echo "Incomplete. Re-submit this script to resume from the .done markers."
fi
