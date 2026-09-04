#!/usr/bin/env python3
"""
evaluate_axis_sweep.py
=========================
Full-scene-set evaluation of ONE style axis swept over several values, with
the other three axes held at 0.

evaluate_styles.py only ever visits ±1 on each axis, so it shows that an
axis has an effect but not whether that effect is MONOTONE in the style scalar.
This script answers that: sweep e.g. prox over {-1, -0.5, +0.5, +1} and read
the corresponding metric down the summary table.

Run it four times, once per axis, to cover the whole style space:

  for ax in prox pass yield group; do
      python evaluate_axis_sweep.py --axis $ax ... --results_suffix SWEEP_$ax
  done

Relation to the other sweep script: style_sweep_figure.py is the
PAPER-FIGURE tool — one hand-picked scene, N stochastic rollouts per value,
rendered as a trajectory-distribution PNG over {-1, 0, +1}. This script is the
EVALUATION tool — every scene in the eval set, one rollout per value per scene,
aggregate metrics + synchronized panel video, with the value list free.

Neither evaluate_styles.py nor any of its outputs are modified; the rollout,
metric and rendering code are imported from it via style_variant_eval.py.

Usage
-----
  python evaluate_axis_sweep.py \\
      --policy_config configs/policy.config \\
      --env_config    configs/env.config \\
      --gpu --phase test --infer_mode joint \\
      --npz_hard --npz_dir ../../data/scenes/eval_500 --npz_num_eval 500 \\
      --axis prox --values -1,-0.5,0.5,1 \\
      --results_suffix SWEEP_prox_100 \\
      --video_file sweep_prox.mp4

Four values render as a 2x2 panel grid (override with --grid_rows/--grid_cols).
Add --no_video for a metrics-only run.
"""

import argparse
import logging
import re
import sys

from style_variant_eval import (
    add_common_args, run_variant_evaluation, STYLE_AXES,
)

# The axis whose response each swept metric should be read on. Printed into
# the summary so the relevant column is obvious without cross-referencing
# evaluate_styles.py's docstring.
_AXIS_METRICS = {
    'prox':  'MinClear (expect increasing), PSIRate (expect decreasing)',
    'pass':  'PassBias (expect increasing toward +1 = right-hand traffic)',
    'yield': 'TTCInfR and CutoffR (expect decreasing)',
    'group': 'GrpSplit (expect decreasing)',
}


def parse_values(s):
    vals = [float(v) for v in s.split(',') if v.strip() != '']
    if len(vals) < 2:
        raise argparse.ArgumentTypeError(
            f'--values needs at least 2 sweep points, got "{s}"')
    return vals


# A bare number, or a comma-joined list of them: "-1", "0.5", "-1,-0.5,0.5,1".
_VALUES_TOKEN = re.compile(r'^[+-]?(\d+\.?\d*|\.\d+)(,[+-]?(\d+\.?\d*|\.\d+))*$')


def normalize_values_argv(argv):
    """Let --values accept sweep points that start with a minus sign.

    Any sweep worth running includes negative style values, but argparse
    classifies a token beginning with '-' as an OPTION unless it looks like a
    plain negative number, so the obvious

        --values -1,-0.5,0.5,1

    dies with "argument --values: expected one argument" — the comma makes it
    fail argparse's negative-number test. Rather than force the reader to
    remember the '=' form, collect every numeric-looking token after --values
    and re-emit them as a single "--values=..." before argparse sees argv.
    This makes all three of these equivalent:

        --values -1,-0.5,0.5,1
        --values=-1,-0.5,0.5,1
        --values -1 -0.5 0.5 1

    Non-numeric tokens are left alone, so a following flag still parses.
    """
    out, i = [], 0
    while i < len(argv):
        if argv[i] == '--values':
            toks, j = [], i + 1
            while j < len(argv) and _VALUES_TOKEN.match(argv[j]):
                toks.append(argv[j])
                j += 1
            if toks:
                out.append('--values=' + ','.join(toks))
                i = j
                continue
        out.append(argv[i])
        i += 1
    return out


def main():
    p = argparse.ArgumentParser(
        description='Single-axis style sweep over the full scene set')
    add_common_args(p)
    p.add_argument('--axis', type=str, required=True, choices=STYLE_AXES,
                   help='Which style axis to sweep; the other three stay at 0')
    p.add_argument('--values', type=parse_values, default=[-1.0, -0.5, 0.5, 1.0],
                   help='Sweep points, e.g. "-1,-0.5,0.5,1" (comma-separated or '
                        'space-separated; leading minus signs are fine)')
    p.add_argument('--no_projection', action='store_true', default=False,
                   help='Run every sweep point with the feasibility-projection '
                        'layer OFF (default: ON, matching the style sweep)')
    p.add_argument('--results_suffix', type=str, default=None,
                   help='Default: SWEEP_<axis>')
    p.add_argument('--video_file', type=str, default=None,
                   help='Default: sweep_<axis>.mp4')
    args = p.parse_args(normalize_values_argv(sys.argv[1:]))

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    axis_idx = STYLE_AXES.index(args.axis)
    use_proj = not args.no_projection

    variants = []
    for v in args.values:
        vec = [0.0] * len(STYLE_AXES)
        vec[axis_idx] = float(v)
        variants.append((f'{args.axis}={v:+g}', vec, use_proj))

    suffix     = args.results_suffix or f'SWEEP_{args.axis}'
    video_file = args.video_file or f'sweep_{args.axis}.mp4'

    run_variant_evaluation(
        args, variants,
        results_dir=f'results_{suffix}',
        video_file=video_file,
        run_title=f'Single-Axis Style Sweep Summary — axis={args.axis}',
        extra_summary_lines=[
            f'Swept axis: {args.axis} (index {axis_idx} of {STYLE_AXES}); '
            f'other axes held at 0',
            f'Values: {", ".join(f"{v:+g}" for v in args.values)}',
            f'Read this sweep on: {_AXIS_METRICS[args.axis]}',
            f'Projection layer: {"ON" if use_proj else "OFF"} (all variants)',
            f'CFG mode: {args.infer_mode or "from policy.config"}',
        ],
    )


if __name__ == '__main__':
    main()
