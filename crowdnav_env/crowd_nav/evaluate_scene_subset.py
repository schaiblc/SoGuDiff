#!/usr/bin/env python3
"""
evaluate_scene_subset.py
========================
Render ONE variant (default: neutral, projection ON) over a scene subset, so a
specific failure mode can be watched frame-by-frame instead of inferred from
aggregate rates.

Written for diagnosing MPPI timeouts at a low search budget, where the
timeouts concentrated entirely in two threat families (head-on approach to a
group, and stationary groups) while no other scene timed out at all. Collect
the scenes of interest into their own directory and point --npz_dir at it to
get one panel per scene of just that failure.

Relation to the other entry points: evaluate_styles.py runs the fixed
10-variant grid, evaluate_axis_sweep.py sweeps one axis over >=2 values.
Neither can render a SINGLE variant (--values requires >= 2 sweep points), and
neither restricts the scene set. This script does both. Like AXISSWEEP it
imports the rollout/metric/render code from style_variant_eval.py, so
evaluate_styles.py and its outputs are NOT modified.

MPPI budget override: build_env_and_policy() reads the policy config off disk,
and apply_ckpt_overrides() only touches ckpt_path/norm_file. Rather than reach
into the policy after configure() (which is what actually builds the sampler,
so a later assignment would silently evaluate the wrong budget), this writes a
COPY of the policy config with [mppi_expert] mppi_budget/mppi_seed set and
points --policy_config at the copy. The copy is kept next to the results as the
audit trail for which budget produced the video.

Usage
-----
  python evaluate_scene_subset.py \\
      --policy_config configs/policy.config \\
      --env_config    configs/env.config \\
      --policy        mppi_expert --phase test \\
      --npz_hard --npz_dir <dir of timeout scenes> --npz_num_eval 42 \\
      --mppi_budget 0.001 --mppi_seed 0 \\
      --results_suffix MPPI_TIMEOUTS_b0.001 \\
      --video_file timeouts_b0.001.mp4

  # a different single variant (e.g. the yield axis at +1):
  --style 0,0,1,0 --variant_label 'yield=+1'
  # projection OFF (the noproj failure mode):
  --no_projection
"""

import argparse
import configparser
import logging
import os
import shutil
import sys

from style_variant_eval import (add_common_args, run_variant_evaluation,
                                parse_style_vector)


def main():
    p = argparse.ArgumentParser(
        description='Single-variant evaluation over a scene subset '
                    '(failure-mode diagnosis)')
    add_common_args(p)
    p.add_argument('--style', type=str, default='0,0,0,0',
                   help="Style vector 'prox,pass,yield,group' (default neutral)")
    p.add_argument('--variant_label', type=str, default=None,
                   help='Panel label (default: derived from --style)')
    p.add_argument('--no_projection', action='store_true', default=False,
                   help='Run with the feasibility-projection layer OFF')
    # Same semantics as evaluate_styles.py's flags of these names.
    p.add_argument('--mppi_budget', type=float, default=None,
                   help='Override [mppi_expert] mppi_budget')
    p.add_argument('--mppi_seed', type=int, default=None,
                   help='Override [mppi_expert] mppi_seed')
    p.add_argument('--results_suffix', type=str, default='TIMEOUTS')
    p.add_argument('--video_file', type=str, default=None,
                   help='Default: <results_suffix>.mp4')
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    style = parse_style_vector(args.style)
    use_proj = not args.no_projection
    label = args.variant_label or (
        'neutral' if not any(style) else
        ','.join(f'{v:+g}' for v in style))
    label += '' if use_proj else ' (noproj)'

    results_dir = f'results_{args.results_suffix}'
    os.makedirs(results_dir, exist_ok=True)

    # ---- budget override via a config COPY (see module docstring) ----------
    if args.mppi_budget is not None or args.mppi_seed is not None:
        cfg = configparser.RawConfigParser()
        cfg.read(args.policy_config)
        if not cfg.has_section('mppi_expert'):
            raise ValueError('--mppi_budget/--mppi_seed need an [mppi_expert] '
                             'section in the policy config')
        for key, val in (('mppi_budget', args.mppi_budget),
                         ('mppi_seed', args.mppi_seed)):
            if val is not None:
                cfg.set('mppi_expert', key, str(val))
                logging.info('[override] mppi_expert.%s = %s', key, val)
        copy_path = os.path.join(results_dir, 'policy_used.config')
        with open(copy_path, 'w') as fh:
            cfg.write(fh)
        args.policy_config = copy_path
        logging.info('[config] evaluating with %s', copy_path)

    # One variant -> one panel.
    args.grid_rows, args.grid_cols = 1, 1

    run_variant_evaluation(
        args, [(label, style, use_proj)],
        results_dir=results_dir,
        video_file=args.video_file or f'{args.results_suffix}.mp4',
        run_title=f'Single-Variant Failure Diagnosis — {label}',
        extra_summary_lines=[
            f'Variant: {label}  style={style}  projection='
            f'{"ON" if use_proj else "OFF"}',
            f'Scene set: {args.npz_dir} (first {args.npz_num_eval})',
            f'MPPI budget: {args.mppi_budget if args.mppi_budget is not None else "from policy.config"}',
        ],
    )


if __name__ == '__main__':
    main()
