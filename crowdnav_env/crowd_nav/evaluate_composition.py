#!/usr/bin/env python3
"""
evaluate_composition.py
===========================
Full-scene-set evaluation of COMPOSED style vectors.

evaluate_styles.py runs a fixed set of 10 single-axis variants (each axis
at ±1, plus the neutral / no-projection ablation). This script runs an
ARBITRARY list of full 4-axis style vectors — e.g. "cautious AND yielding" =
[1,0,1,0] — over the same scenes, with the same metrics and the same
synchronized multi-panel video, so composed styles can be compared to each
other and to the neutral row on equal footing.

Relation to the other composition script: style_composition_figure.py
is the PAPER-FIGURE tool — one hand-picked scene, N stochastic rollouts per
composition, rendered as a trajectory-distribution PNG. This script is the
EVALUATION tool — every scene in the eval set, one rollout per composition per
scene, aggregate metrics + video. The --composition flag syntax is deliberately
identical between the two.

The rollout, metric and rendering code are shared with evaluate_styles.py
through style_variant_eval.py, so composed and single-axis styles are scored by
exactly the same instrument.

Usage
-----
  python evaluate_composition.py \\
      --policy_config configs/policy.config \\
      --env_config    configs/env.config \\
      --gpu --phase test --infer_mode joint \\
      --npz_hard --npz_dir ../../data/scenes/eval_500 --npz_num_eval 500 \\
      --composition "Neutral:0,0,0,0" \\
      --composition "Cautious & Yielding:1,0,1,0" \\
      --composition "Assertive & Efficient:-1,0,-1,0" \\
      --composition "Social Passing:0,1,0,1" \\
      --composition "Group Aware + Wide:1,0,0,1" \\
      --composition "All On:1,1,1,1" \\
      --results_suffix STYLECOMP_100 \\
      --video_file composition.mp4

Six compositions render as a 2x3 panel grid (override with --grid_rows/--grid_cols).
Add --no_video for a metrics-only run.
"""

import argparse
import logging

from style_variant_eval import (
    add_common_args, parse_style_vector, run_variant_evaluation, STYLE_AXES,
)

# Used only when no --composition flag is passed. Mirrors
# style_composition_figure.DEFAULT_COMPOSITIONS, plus two more so
# the default run fills a 2x3 grid.
DEFAULT_COMPOSITIONS = [
    ('Neutral',               [0.0, 0.0, 0.0, 0.0]),
    ('Cautious & Yielding',   [1.0, 0.0, 1.0, 0.0]),
    ('Assertive & Efficient', [-1.0, 0.0, -1.0, 0.0]),
    ('Social Passing',        [0.0, 1.0, 0.0, 1.0]),
    ('Group Aware + Wide',    [1.0, 0.0, 0.0, 1.0]),
    ('All Axes On',           [1.0, 1.0, 1.0, 1.0]),
]


def parse_composition_arg(s):
    """'Label:prox,pass,yield,group' -> (label, [floats]).

    Same syntax as style_composition_figure.py so specs can be
    copy-pasted between the figure tool and this evaluation.
    """
    if ':' not in s:
        raise argparse.ArgumentTypeError(
            f'composition must be "Label:{",".join(STYLE_AXES)}", got "{s}"')
    label, _, vec_str = s.partition(':')
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError(f'empty label in "{s}"')
    try:
        vec = parse_style_vector(vec_str)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))
    return label, vec


def main():
    p = argparse.ArgumentParser(
        description='Composed-style evaluation over the full scene set')
    add_common_args(p)
    p.add_argument('--composition', type=parse_composition_arg, action='append',
                   default=None, metavar='LABEL:prox,pass,yield,group',
                   help='One composed style vector, e.g. "Cautious & Yielding:1,0,1,0". '
                        'Repeat for each composition. If omitted, a 6-composition '
                        'default set is used.')
    p.add_argument('--no_projection', action='store_true', default=False,
                   help='Run every composition with the feasibility-projection '
                        'layer OFF (default: ON, matching the style sweep)')
    p.add_argument('--results_suffix', type=str, default='STYLECOMP')
    p.add_argument('--video_file', type=str, default='composition.mp4')
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    comps = args.composition if args.composition else DEFAULT_COMPOSITIONS
    if args.composition is None:
        logging.info('No --composition given; using the %d default compositions',
                     len(DEFAULT_COMPOSITIONS))

    use_proj = not args.no_projection
    variants = [(label, vec, use_proj) for label, vec in comps]

    run_variant_evaluation(
        args, variants,
        results_dir=f'results_{args.results_suffix}',
        video_file=args.video_file,
        run_title='Composed-Style Evaluation Summary',
        extra_summary_lines=[
            f'Compositions: {len(variants)}',
            f'Projection layer: {"ON" if use_proj else "OFF"} (all variants)',
            f'CFG mode: {args.infer_mode or "from policy.config"}',
        ],
    )


if __name__ == '__main__':
    main()
