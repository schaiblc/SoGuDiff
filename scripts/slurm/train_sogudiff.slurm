#!/bin/bash
#SBATCH --job-name=sogudiff-train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#
# Train SoGuDiff. Equivalent single-machine command:
#
#   python sogudiff/train.py \
#       --dataset_dir data/expert_trajectories \
#       --exp_name sogudiff --cfg_mode union --use_map \
#       --batch_size 128 --lr 2e-4 --num_steps 500000 \
#       --num_diffusion_timesteps 100 --horizon 32 --token_dim 128
#
# A full run is ~500k steps. Checkpoints land in checkpoints/<exp_name>/ and
# training resumes from the newest one, so re-submitting continues rather than
# restarting.
#
# Submit:  sbatch scripts/slurm/train_sogudiff.slurm
# Account, partition and environment activation come from cluster_env.sh.

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

DATASET="${SOGUDIFF_DATASET:-data/expert_trajectories}"
EXP_NAME="${SOGUDIFF_EXP_NAME:-sogudiff}"

echo "=== training  node=$(hostname)  dataset=${DATASET}  exp=${EXP_NAME} ==="
python -c "import torch;print(f'torch {torch.__version__} cuda={torch.cuda.is_available()}')"

python sogudiff/train.py \
    --dataset_dir "$DATASET" \
    --exp_name    "$EXP_NAME" \
    --cfg_mode    union \
    --use_map \
    --batch_size  128 \
    --lr          2e-4 \
    --num_steps   500000 \
    --num_diffusion_timesteps 100 \
    --horizon     32 \
    --token_dim   128
