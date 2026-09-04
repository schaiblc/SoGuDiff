#!/usr/bin/env bash
#
# Reproduce the paper's comparison table on one machine, no scheduler needed.
# Runs every method over the same scene set, one after another, and prints the
# summary line from each.
#
#   scripts/run_baseline_table.sh                     # all methods, 500 scenes
#   scripts/run_baseline_table.sh --only ORCA,SFM     # a subset
#
# Scenes are generated procedurally from configs/env.config (test_sim =
# evaluation, test_size = 500) with fixed per-scene seeds, so no scene download
# is needed and every method sees identical scenes.
#
# SICNav is skipped unless SOGUDIFF_SICNAV_PYTHON points at its interpreter,
# because it needs the separate environment from requirements/sicnav.txt.
#
# Expect hours for the full 500-scene sweep. Rendering is disabled throughout;
# pass --visualize to evaluate.py directly if you want video.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SOGUDIFF_ROOT="$ROOT"

ONLY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --only)   ONLY="$2"; shift 2 ;;
        -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

# label : policy : model_dir (empty if none)
METHODS=(
    "ORCA:orca_unicycle:"
    "SFM:sfm_unicycle:"
    "CADRL:cadrl:models/cadrl"
    "LSTM-RL:lstm_rl:models/lstm_rl"
    "SARL:sarl:models/sarl"
    "RGL:rgl:models/rgl"
    "DSRNN:dsrnn:"
    "NAVISTAR:navistar:"
    "HEIGHT:height:"
    "SoGuDiff:sogudiff:"
)

cd "$ROOT/crowdnav_env/crowd_nav"

for entry in "${METHODS[@]}"; do
    IFS=':' read -r LABEL POLICY MDIR <<< "$entry"
    if [ -n "$ONLY" ] && [[ ",$ONLY," != *",$LABEL,"* ]]; then continue; fi

    ARGS=(--policy "$POLICY" --phase test --no_video --results_suffix "$LABEL")
    [ -n "$MDIR" ] && ARGS+=(--model_dir "$MDIR")
    [ "$POLICY" = "sogudiff" ] && ARGS+=(--infer_mode per_axis --w_normalize)
    command -v nvidia-smi >/dev/null 2>&1 && ARGS+=(--gpu)

    echo "=============== $LABEL ==============="
    if python evaluate.py "${ARGS[@]}"; then
        # evaluate.py names the summary after the POLICY, not the label:
        # results_<LABEL>/<policy>_eval.txt.
        SUMMARY="results_${LABEL}/${POLICY}_eval.txt"
        if [ -f "$SUMMARY" ]; then
            grep -E 'Success Rate|Total Collision Rate|Average Time to Goal' "$SUMMARY"
        else
            echo "(no summary at $SUMMARY)"
        fi
    else
        echo "!!! $LABEL failed -- continuing"
    fi
done

if [ -n "${SOGUDIFF_SICNAV_PYTHON:-}" ]; then
    echo "=============== SICNAV ==============="
    PYTHONPATH="$ROOT/crowdnav_env:${PYTHONPATH:-}" \
    "$SOGUDIFF_SICNAV_PYTHON" evaluate.py \
        --policy sicnav --policy_config configs/policy_sicnav.config \
        --phase test --no_video --results_suffix SICNAV || echo "!!! SICNAV failed"
else
    echo "Skipping SICNav: set SOGUDIFF_SICNAV_PYTHON to its interpreter to include it."
fi

echo
echo "Per-method results are in crowdnav_env/crowd_nav/results_<LABEL>/."
