#!/usr/bin/env python3
"""
style_variant_eval.py
=====================
Shared engine for full-scene-set evaluations of an ARBITRARY list of style
variants, over the same scenes and with the same metrics as
evaluate_styles.py.

evaluate_styles.py hardcodes its 10 variants (each axis at ±1, plus the
neutral / no-projection ablation). This module takes the variant list as a
parameter instead, so the same machinery can drive:

  * evaluate_composition.py — N user-specified full 4-axis style vectors
    (composed styles, e.g. "cautious AND yielding"), and
  * evaluate_axis_sweep.py   — one axis swept over N values
    (e.g. prox at -1, -0.5, +0.5, +1).

Both produce, per scene, a synchronized multi-panel MP4 and one metrics row
per (scene, variant) — identical in format to the style sweep's outputs.

Nothing in evaluate_styles.py is modified. Its pieces are IMPORTED:

  run_one_episode()                 — the rollout (incl. the RNG-anchor
                                      restore that guarantees every variant
                                      sees the same scene)
  compute_episode_social_metrics()  — the six social metrics
  build_scene_video()               — the panel renderer

build_scene_video() reads its panel count, grid shape and per-panel labels
from MODULE-LEVEL globals in evaluate_styles (STYLE_VARIANTS, GRID_ROWS,
GRID_COLS, _ROBOT_COLORS) rather than from arguments. `variant_render_context`
below swaps those globals for the duration of one render and restores them
afterwards. That is deliberate: the alternative is a ~230-line copy of the
matplotlib code, which would drift from the style sweep's rendering the first
time either is touched — and the whole point is that these videos look
exactly like the style-sweep videos. The swap is process-local and restored in
a finally block; nothing on disk or in evaluate_styles.py changes.

One cosmetic consequence: the video suptitle stays the renderer's generic
"Style Sweep — Scene N". The identifying information (variant label, style
vector, projection on/off) is in the per-panel titles, which ARE driven by the
injected variant list.
"""

import contextlib
import csv
import glob
import logging
import math
import os
import subprocess

import numpy as np
import torch
import gym
import configparser

from crowd_nav.policy.policy_factory import policy_factory
from crowd_nav.paths import ffmpeg_binary
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.info import *  # ReachGoal, Collision, Timeout, WallCollision

import evaluate_styles as S
from evaluate_styles import (
    run_one_episode,
    build_scene_video,
    _outcome_label,
    _D_PER, _D_SOC, _D_INT, _TAU_R,
)

STYLE_AXES = ['prox', 'pass', 'yield', 'group']

_SOCIAL_KEYS = ['mean_min_clearance_m', 'psi_rate', 'passing_side_bias',
                'ttc_infraction_rate', 'path_cutoff_rate', 'group_split_rate']

# Extra panel colors beyond evaluate_styles._ROBOT_COLORS' 10, in case a
# caller asks for more variants than the style sweep ever had.
_EXTRA_COLORS = ['teal', 'crimson', 'olive', 'slateblue', 'sienna', 'magenta']


# ---------------------------------------------------------------------------
# Variant list plumbing
# ---------------------------------------------------------------------------

def parse_style_vector(s):
    """'1,0,-0.5,0' -> [1.0, 0.0, -0.5, 0.0], validated against STYLE_AXES."""
    vals = [float(v) for v in s.split(',') if v.strip() != '']
    if len(vals) != len(STYLE_AXES):
        raise ValueError(
            f'style vector must have {len(STYLE_AXES)} components '
            f'({",".join(STYLE_AXES)}), got {len(vals)} in "{s}"')
    return vals


def assert_axes_match_policy():
    """Fail loudly if this module's axis ordering has drifted from the policy's.

    Every style vector here is positional, so a reordering in
    sogudiff.py would silently mislabel every result rather than
    error. Mirrors the same assert in style_sweep_figure.py.
    """
    from crowd_nav.policy.sogudiff import STYLE_AXES as _POLICY_AXES
    assert list(_POLICY_AXES) == STYLE_AXES, (
        f'STYLE_AXES drift: sogudiff.py has {list(_POLICY_AXES)}, '
        f'style_variant_eval.py assumes {STYLE_AXES}')


def auto_grid(n, rows=None, cols=None):
    """Pick a panel grid for n variants (as close to square as possible)."""
    if rows and cols:
        if rows * cols < n:
            raise ValueError(f'grid {rows}x{cols} too small for {n} variants')
        return rows, cols
    if cols:
        return math.ceil(n / cols), cols
    if rows:
        return rows, math.ceil(n / rows)
    c = math.ceil(math.sqrt(n))
    return math.ceil(n / c), c


@contextlib.contextmanager
def variant_render_context(variants, rows, cols):
    """Swap evaluate_styles' render globals for this variant list."""
    saved = (S.STYLE_VARIANTS, S.GRID_ROWS, S.GRID_COLS, S._ROBOT_COLORS)
    try:
        S.STYLE_VARIANTS = variants
        S.GRID_ROWS, S.GRID_COLS = rows, cols
        if len(variants) > len(S._ROBOT_COLORS):
            S._ROBOT_COLORS = list(S._ROBOT_COLORS) + _EXTRA_COLORS
        yield
    finally:
        S.STYLE_VARIANTS, S.GRID_ROWS, S.GRID_COLS, S._ROBOT_COLORS = saved


# ---------------------------------------------------------------------------
# CLI plumbing shared by both entry-point scripts
# ---------------------------------------------------------------------------

def add_common_args(parser):
    parser.add_argument('--env_config',    type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--policy',        type=str,
                        default='sogudiff')
    parser.add_argument('--gpu',    action='store_true', default=False)
    parser.add_argument('--phase',  type=str, default='test')
    parser.add_argument('--num_tests', type=int, default=50,
                        help='Scene count when no scenario flag is given')
    parser.add_argument('--fps',    type=int, default=10)
    parser.add_argument('--no_video', action='store_true', default=False,
                        help='Skip rendering; metrics only (much faster)')
    parser.add_argument('--grid_rows', type=int, default=None,
                        help='Panel grid rows (default: auto, near-square)')
    parser.add_argument('--grid_cols', type=int, default=None)
    # Scenario selection — same flags/semantics as evaluate_styles.py
    parser.add_argument('--square',   action='store_true', default=False)
    parser.add_argument('--circle',   action='store_true', default=False)
    parser.add_argument('--npz_hard', action='store_true', default=False)
    parser.add_argument('--npz_dir',  type=str,
                        default='../../data/scenes/eval_500')
    parser.add_argument('--npz_num_eval', type=int, default=50)
    parser.add_argument('--map_eval', action='store_true', default=False)
    parser.add_argument('--map_eval_num', type=int, default=50)
    # SoGuDiff guidance overrides — same as evaluate_styles.py
    parser.add_argument('--infer_mode', type=str, default=None,
                        choices=['joint', 'per_axis'],
                        help='CFG mode (must match how the checkpoint was trained)')
    parser.add_argument('--cfg_w_style', type=str, default=None,
                        help="Per-axis guidance weights 'a,b,c,d' or one scalar")
    parser.add_argument('--w_normalize', action='store_true', default=False,
                        help='per_axis: hold sum of active w_i constant')
    # Checkpoint selection belongs to the JOB, not to a shared config file.
    # Several evaluations run concurrently against the same policy.config, so
    # editing ckpt_path between launches is a race: whichever value happens to
    # be on disk when a job calls configure() is the model it silently
    # evaluates, and nothing in the results says which one that was.
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='Override ckpt_path from policy.config. Use this '
                             'instead of editing the shared config, so parallel '
                             "jobs cannot pick up each other's model.")
    parser.add_argument('--norm_file', type=str, default=None,
                        help='Override norm_file from policy.config. Normally '
                             'left alone: all three models share one norm file, '
                             'and a mismatch fails silently into wrong units.')
    return parser


def apply_ckpt_overrides(policy_config, args):
    """Apply --ckpt_path / --norm_file into the parser BEFORE policy.configure().

    Done at the parser level rather than by setting attributes on the policy
    afterwards, because configure() is what actually loads the weights: a later
    assignment would leave the originally-configured checkpoint loaded while the
    logs claimed otherwise -- a worse failure than the one being fixed.
    """
    sec = args.policy
    if not policy_config.has_section(sec):
        raise ValueError(f'policy config has no section [{sec}]')
    for key, val in (('ckpt_path', args.ckpt_path), ('norm_file', args.norm_file)):
        if val:
            policy_config.set(sec, key, val)
            logging.info('[override] %s.%s = %s', sec, key, val)
    # Always log what is actually being loaded, override or not. This line is
    # the audit trail tying a results directory to a specific model.
    logging.info(
        '[model] ckpt_path=%s  norm_file=%s',
        policy_config.get(sec, 'ckpt_path') if policy_config.has_option(sec, 'ckpt_path') else '?',
        policy_config.get(sec, 'norm_file') if policy_config.has_option(sec, 'norm_file') else '?')
    return policy_config


def build_env_and_policy(args):
    """Instantiate policy + env exactly as evaluate_styles.main() does."""
    device = torch.device('cuda:0' if torch.cuda.is_available() and args.gpu else 'cpu')
    logging.info('Device: %s', device)

    policy = policy_factory[args.policy]()
    policy_config = configparser.RawConfigParser()
    policy_config.read(args.policy_config)
    apply_ckpt_overrides(policy_config, args)
    policy.configure(policy_config)

    if args.infer_mode is not None and hasattr(policy, 'set_infer_mode'):
        policy.set_infer_mode(args.infer_mode)   # validates vs trained_cfg_mode
    if args.w_normalize and hasattr(policy, 'cfg_w_normalize'):
        policy.cfg_w_normalize = True
    if args.cfg_w_style is not None and hasattr(policy, 'cfg_w_style'):
        vals = [float(v) for v in args.cfg_w_style.split(',') if v.strip()]
        n_axes = len(getattr(policy, 'style_vector', [0, 0, 0, 0]))
        policy.cfg_w_style = vals * n_axes if len(vals) == 1 else vals
    if hasattr(policy, 'cfg_infer_mode'):
        logging.info('Diffusion guidance: mode=%s w_normalize=%s w_style=%s',
                     policy.cfg_infer_mode,
                     getattr(policy, 'cfg_w_normalize', False),
                     getattr(policy, 'cfg_w_style', None))

    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    sim = env.unwrapped

    if args.square:
        sim.test_sim = 'square_crossing'
    if args.circle:
        sim.test_sim = 'circle_crossing'
    if args.npz_hard:
        sim.test_sim = 'npz_hard'
    if args.map_eval:
        sim.test_sim = 'map_eval'

    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot)

    policy.set_phase(args.phase)
    policy.set_device(device)
    policy.set_env(env)
    robot.print_info()
    return env, robot, policy, sim


def build_scene_list(args):
    """Scene list, using evaluate_styles.py's npz filter verbatim.

    The filter is "difficulty == 'hard' OR threat_type present", which accepts
    scene sets whatever threat vocabulary they use. evaluate.py applies the
    same rule, so a given directory yields the same scenes from either entry
    point -- that is what makes these runs scene-for-scene comparable to the
    style sweep.
    """
    if args.npz_hard:
        files, threats = [], []
        for f in sorted(glob.glob(os.path.join(args.npz_dir, '*.npz'))):
            try:
                d = np.load(f, allow_pickle=True)
                difficulty = str(d['difficulty']) if 'difficulty' in d else None
                threat     = str(d['threat_type']) if 'threat_type' in d else None
                if difficulty == 'hard' or threat is not None:
                    files.append(f)
                    threats.append(threat or 'unknown')
            except Exception as e:
                logging.warning('Could not load %s: %s', f, e)
        if not files:
            raise ValueError(f'No hard npz files found in {args.npz_dir}')
        files, threats = files[:args.npz_num_eval], threats[:args.npz_num_eval]
        logging.info('NPZ hard: %d scenes', len(files))
        return files, threats
    n = args.map_eval_num if args.map_eval else args.num_tests
    logging.info('%s: %d scenes', 'Map-eval' if args.map_eval else 'Standard', n)
    return [None] * n, ['generated'] * n


# ---------------------------------------------------------------------------
# The evaluation itself
# ---------------------------------------------------------------------------

def run_variant_evaluation(args, variants, results_dir, video_file, run_title,
                           extra_summary_lines=()):
    """
    Run `variants` over every scene, writing videos + CSV + summary.

    variants : list of (label, style_vec, use_projection) — the same triple
               shape evaluate_styles.STYLE_VARIANTS uses.
    """
    assert_axes_match_policy()

    env, robot, policy, sim = build_env_and_policy(args)
    scene_files, scene_threats = build_scene_list(args)
    num_scenes = len(scene_files)
    rows, cols = auto_grid(len(variants), args.grid_rows, args.grid_cols)
    logging.info('%d variants in a %dx%d panel grid over %d scenes',
                 len(variants), rows, cols, num_scenes)
    for label, vec, proj in variants:
        logging.info('  variant: %-28s style=%s proj=%s', label, vec, proj)

    os.makedirs(results_dir, exist_ok=True)
    tmp_dir = os.path.join(results_dir, 'tmp_scenes')
    os.makedirs(tmp_dir, exist_ok=True)
    base_video, ext = os.path.splitext(video_file)

    metrics = {label: {k: [] for k in
                       ['successes', 'timeouts', 'collisions', 'wall_colls',
                        'ep_times', 'path_lens', 'min_dists', 'avg_min_dists']
                       + _SOCIAL_KEYS}
               for label, _, _ in variants}
    csv_rows = []
    all_scene_videos = []

    for scene_idx in range(num_scenes):
        npz_path = scene_files[scene_idx]
        logging.info('Scene %d / %d  (npz=%s)', scene_idx + 1, num_scenes,
                     os.path.basename(npz_path) if npz_path else 'generated')

        # Snapshot the RNG anchor before any variant touches the env;
        # run_one_episode restores it per variant so all panels share a scene.
        scene_testoffset = sim.testoffset
        scene_counter    = sim.case_counter[args.phase]

        episodes = []
        for v_idx, (label, style_vec, use_proj) in enumerate(variants):
            logging.info('  Variant %d/%d: %s  style=%s  proj=%s',
                         v_idx + 1, len(variants), label, style_vec, use_proj)
            ep = run_one_episode(
                env, robot, policy,
                style_vec=style_vec,
                use_proj=use_proj,
                phase=args.phase,
                scene_idx=scene_idx,
                npz_path=npz_path,
                scene_testoffset=scene_testoffset,
                scene_counter=scene_counter,
            )
            episodes.append(ep)

            m    = metrics[label]
            info = ep['info']
            m['successes'].append(1 if isinstance(info, ReachGoal) else 0)
            m['timeouts'].append(1   if isinstance(info, Timeout)  else 0)
            m['collisions'].append(1 if isinstance(info, Collision) else 0)
            try:
                m['wall_colls'].append(1 if isinstance(info, WallCollision) else 0)
            except NameError:
                m['wall_colls'].append(0)
            if isinstance(info, ReachGoal):
                m['ep_times'].append(ep['global_time'])
                m['path_lens'].append(ep['pathlength'])
            m['min_dists'].append(ep['minobsdist'])
            m['avg_min_dists'].append(
                np.mean(ep['avgobsdist']) if ep['avgobsdist'] else float('nan'))
            sm = ep['social_metrics']
            for k in _SOCIAL_KEYS:
                m[k].append(sm[k])

            row = {
                'scene':      scene_idx + 1,
                'threat':     scene_threats[scene_idx],
                'variant':    label,
                'projection': use_proj,
                'style':      ','.join(f'{v:+.2f}' for v in style_vec),
                'outcome':    _outcome_label(info),
                'time':       f'{ep["global_time"]:.3f}',
                'path_len':   f'{ep["pathlength"]:.3f}',
                'min_dist':   f'{ep["minobsdist"]:.3f}',
            }
            row.update({k: f'{sm[k]:.4f}' for k in _SOCIAL_KEYS})
            csv_rows.append(row)

            logging.info('    -> %s  t=%.2fs  path=%.2f  minD=%.3f',
                         _outcome_label(info), ep['global_time'],
                         ep['pathlength'], ep['minobsdist'])

        if not args.no_video:
            scene_video = os.path.join(tmp_dir, f'scene_{scene_idx + 1:04d}{ext}')
            with variant_render_context(variants, rows, cols):
                build_scene_video(episodes, scene_idx, scene_video, fps=args.fps)
            all_scene_videos.append(scene_video)

    # ── Concatenate per-scene videos ─────────────────────────────────────────
    if all_scene_videos:
        combined = os.path.join(results_dir, os.path.basename(video_file))
        list_file = os.path.join(tmp_dir, 'video_list.txt')
        with open(list_file, 'w') as fh:
            for vid in all_scene_videos:
                fh.write(f"file '{os.path.abspath(vid)}'\n")
        subprocess.run([ffmpeg_binary(), '-y', '-f', 'concat', '-safe', '0',
                        '-i', list_file, '-c', 'copy', combined], check=False)
        logging.info('Combined video: %s', combined)

    # ── Per-scene CSV ────────────────────────────────────────────────────────
    csv_path = os.path.join(results_dir, 'per_scene_metrics.csv')
    fieldnames = ['scene', 'threat', 'variant', 'projection', 'style',
                  'outcome', 'time', 'path_len', 'min_dist'] + _SOCIAL_KEYS
    with open(csv_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)
    logging.info('Per-scene CSV: %s', csv_path)

    # ── Summary (same layout as the style sweep's summary.txt) ───────────────
    summary_path = os.path.join(results_dir, 'summary.txt')
    with open(summary_path, 'w') as fh:
        fh.write('=' * 60 + '\n')
        fh.write(f'{run_title}\n')
        fh.write(f'Scenes evaluated: {num_scenes}\n')
        fh.write(f'Scene source: {args.npz_dir if args.npz_hard else "generated"}\n')
        # PROVENANCE -- record what produced these numbers, in the numbers' own
        # file.  Without this a results directory cannot be attributed after the
        # fact: the checkpoint came from a shared policy.config that other jobs
        # edit, and whether guidance was magnitude-matched (--w_normalize) is
        # invisible in the metrics yet changes per-axis totals by 3-4x on
        # multi-axis style vectors.
        fh.write(f'Checkpoint: {getattr(args, "ckpt_path", None) or "(from policy.config)"}\n')
        fh.write(f'Norm file: {getattr(args, "norm_file", None) or "(from policy.config)"}\n')
        fh.write(f'Inference mode: {getattr(args, "infer_mode", None) or "(from policy.config)"}\n')
        fh.write(f'cfg_w_style: {getattr(args, "cfg_w_style", None) or "(from policy.config)"}\n')
        fh.write(f'w_normalize: {bool(getattr(args, "w_normalize", False))}\n')
        for line in extra_summary_lines:
            fh.write(line + '\n')
        fh.write('=' * 60 + '\n\n')

        fh.write('Variants (style axes: ' + ', '.join(STYLE_AXES) + ')\n')
        for label, vec, proj in variants:
            fh.write(f"  {label:<28} [{','.join(f'{v:+.2f}' for v in vec)}]  "
                     f"proj={'ON' if proj else 'OFF'}\n")
        fh.write('\n')

        header = (f"{'Variant':<28} {'SR%':>6} {'TO%':>6} {'CR%':>6} "
                  f"{'WallCR%':>8} {'AvgT':>7} {'AvgPL':>7} {'AvgMinD':>8}\n")
        fh.write(header)
        fh.write('-' * len(header) + '\n')
        for label, _, _ in variants:
            m = metrics[label]
            def _m(k): return np.mean(m[k]) if m[k] else float('nan')
            fh.write(f"{label:<28} {100*_m('successes'):6.1f} {100*_m('timeouts'):6.1f} "
                     f"{100*_m('collisions'):6.1f} {100*_m('wall_colls'):8.1f} "
                     f"{_m('ep_times'):7.3f} {_m('path_lens'):7.3f} "
                     f"{_m('min_dists'):8.3f}\n")

        fh.write('\nColumn key:\n')
        fh.write('  SR%    = Success Rate\n')
        fh.write('  TO%    = Timeout Rate\n')
        fh.write('  CR%    = Agent Collision Rate\n')
        fh.write('  WallCR%= Static Map Collision Rate\n')
        fh.write('  AvgT   = Average Time to Goal (successful episodes)\n')
        fh.write('  AvgPL  = Average Path Length to Goal (successful episodes)\n')
        fh.write('  AvgMinD= Average Minimum Distance to Obstacles (all episodes)\n')

        fh.write('\n\nSocial Style Metrics (Hall-proxemics thresholds — comparable across variants)\n')
        fh.write(f'  D_per = 4 ft = {_D_PER:.4f} m (personal space)\n')
        fh.write(f'  D_soc = 12 ft = {_D_SOC:.4f} m (social space)\n')
        fh.write(f'  D_int = 1.5 ft = {_D_INT:.4f} m (intimate space)\n')
        fh.write(f'  tau_r = {_TAU_R:.2f} s (human reaction time)\n\n')
        fh.write('Metric -> axis it responds to:\n')
        fh.write('  MinClear (higher) / PSIRate (lower)   <- prox+\n')
        fh.write('  PassBias  -> +1 right-hand traffic, -1 left-hand  <- pass\n')
        fh.write('  TTCInfR / CutoffR (lower)             <- yield+\n')
        fh.write('  GrpSplit (lower)                      <- group+\n\n')

        soc_header = (f"{'Variant':<28} {'MinClear':>9} {'PSIRate':>8} {'PassBias':>9} "
                      f"{'TTCInfR':>8} {'CutoffR':>8} {'GrpSplit':>9}\n")
        fh.write(soc_header)
        fh.write('-' * len(soc_header) + '\n')
        for label, _, _ in variants:
            m = metrics[label]
            def _a(k): return np.nanmean(m[k]) if m[k] else float('nan')
            fh.write(f"{label:<28} {_a('mean_min_clearance_m'):9.3f} "
                     f"{_a('psi_rate'):8.4f} {_a('passing_side_bias'):+9.3f} "
                     f"{_a('ttc_infraction_rate'):8.4f} {_a('path_cutoff_rate'):8.4f} "
                     f"{_a('group_split_rate'):9.4f}\n")

        # 95% bootstrap CIs + nonzero support, same helper the style sweep uses.
        # Separate block so the headline table stays readable; 'support' is the
        # number that tells you whether an axis is measurable at all on this
        # scene set.
        fh.write('\n\n95% bootstrap CIs over scenes (10k resamples) '
                 'and nonzero-support counts\n')
        ci_header = (f"{'Variant':<28} {'MinClear':>20} {'PassBias':>20} "
                     f"{'TTCInfR':>20} {'GrpSplit':>20}\n")
        fh.write(ci_header)
        fh.write('-' * len(ci_header) + '\n')
        for label, _, _ in variants:
            m = metrics[label]
            cells = []
            for k in ('mean_min_clearance_m', 'passing_side_bias',
                      'ttc_infraction_rate', 'group_split_rate'):
                lo, hi, n_nz, n = S._bootstrap_ci(m[k])
                cells.append(f"[{lo:+.3f},{hi:+.3f}] {n_nz}/{n}")
            fh.write(f"{label:<28} " + ' '.join(f"{c:>20}" for c in cells) + '\n')

    logging.info('Summary: %s', summary_path)
    with open(summary_path) as fh:
        print(fh.read())
    return metrics
