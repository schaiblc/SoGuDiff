#!/usr/bin/env python3
"""
evaluate_baselines.py

Run the UNSTYLED baseline planners (CADRL, LSTM-RL, SARL, RGL, ORCA, SFM) on
the SAME style-probe scenes that evaluate_styles.py evaluates, and report
the SAME metric set, so the numbers drop straight into the style table as the
"no style knob at all" reference row.

Why this script exists instead of `evaluate.py --npz_hard`
---------------------------------------------------------------
evaluate.py is the general single-policy entry point and is the right tool for
the comparison table. It is the wrong tool here for one reason:

  It only computes MinD and PSI. The style claim also rests on
  passing_side_bias, ttc_infraction_rate, path_cutoff_rate and
  group_split_rate, and a baseline row with four blanks in it cannot support
  "the styled deltas are significant relative to unstyled". This script imports
  compute_episode_social_metrics from evaluate_styles, so there is exactly one
  implementation of those metrics.

Scene identity with the style run
---------------------------------
Scenes are the sorted .npz list, and the env RNG anchor (sim.testoffset,
sim.case_counter[phase]) is snapshotted per scene and restored before every
baseline's reset() — the same protocol run_one_episode() uses across style
variants. So baseline k on scene i and style variant v on scene i see a
bit-identical episode setup.

Baselines are run in one process, scene-major (outer loop = scene, inner loop
= baseline), for that reason: it is the arrangement that makes the shared RNG
anchor trivially correct.

ORCA / SFM are instantiated as `orca_unicycle` / `sfm_unicycle`, NOT as the
holonomic `orca` / `sfm`. Two independent reasons:
  * crowd_sim.reset() calls robot.policy._reset_unicycle_state() unconditionally
    and raises NotImplementedError otherwise; only the *_unicycle variants
    implement it.
  * every other agent in this comparison (RL baselines and the diffusion
    policy) is unicycle-constrained, so a holonomic ORCA would not be a
    like-for-like baseline.

Caveat worth stating in the paper, not fixable here: baselines have no
occupancy-map input (no set_static_map hook), so on the 11 of 100 scenes with
has_map == 1 they navigate blind to walls. Per-scene rows carry `has_map`, and
the summary reports the map-free subset separately.

Video
-----
One synchronized panel per baseline per scene, concatenated into a single MP4 —
the same renderer and layout the style sweep uses (build_scene_video, driven via
style_variant_eval.variant_render_context). Panels hold their last frame while
longer episodes finish, so the baselines can be compared tick-for-tick on the
same scene. The trajectory overlays (sampled / selected / projected) stay empty
because these planners publish no predicted_traj. Use --no_video for a
numbers-only run.

Usage
-----
  python evaluate_baselines.py \
      --env_config    configs/env.config \
      --policy_config configs/policy.config \
      --npz_hard --npz_dir ../../data/scenes/eval_500 --npz_num_eval 500 \
      --results_suffix BASELINES_STYLESCENES \
      --video_file baselines.mp4 \
      --gpu
"""

import argparse
import configparser
import copy
import csv
import glob
import logging
import os
import subprocess

import gym
import numpy as np
import torch

from crowd_nav.policy.policy_factory import policy_factory
from crowd_nav.paths import ffmpeg_binary
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.info import *  # ReachGoal, Collision, Timeout, WallCollision

# Single source of truth for the social metrics — same code the style sweep runs.
from evaluate_styles import (
    compute_episode_social_metrics, build_scene_video,
    _D_PER, _D_SOC, _D_INT, _TAU_R,
)
# Panel-grid sizing + the global-swap that lets build_scene_video render an
# arbitrary panel list (see style_variant_eval.variant_render_context).
from style_variant_eval import auto_grid, variant_render_context

_SOCIAL_KEYS = ['mean_min_clearance_m', 'psi_rate', 'passing_side_bias',
                'ttc_infraction_rate', 'path_cutoff_rate', 'group_split_rate']

# label -> (policy_factory key, model_dir or None)
# Labels are the row names used in the paper's comparison table.
BASELINES = {
    'CADRL':    ('cadrl',                'models/cadrl'),
    'LSTM-RL':  ('lstm_rl',              'models/lstm_rl'),
    'SARL':     ('sarl',                 'models/sarl'),
    'RGL':      ('rgl',                  'models/rgl'),
    'ORCA':     ('orca_unicycle',        None),
    'SFM':      ('sfm_unicycle',         None),

    # The four below have no model_dir. Each is an evaluation-only adapter
    # (crowd_nav/policy/*.py) that reads its checkpoint path from a named
    # section of the policy config rather than from a directory layout.
    #
    # DSRNN, NaviSTAR and HEIGHT are published as holonomic policies, and the
    # authors' recipes did not converge when retrained unicycle-native. The
    # reported rows therefore use each holonomic checkpoint with its (vx, vy)
    # output projected onto the shared unicycle envelope at evaluation time --
    # the same projection ORCAUnicycle and SFMUnicycle use. kinematics is
    # 'unicycle' for all four, making them like-for-like with the diffusion
    # policy. The unprojected holonomic arms are registered separately as
    # *_holonomic and reported as the cost of that conversion.
    'DSRNN':    ('dsrnn',  None),
    'NAVISTAR': ('navistar',        None),
    'HEIGHT':   ('height',          None),

    # SICNav-np: a CasADi/IPOPT MPC with no learned checkpoint. Runs only in
    # the dedicated SICNav environment (see requirements/sicnav.txt).
    'SICNAV':   ('sicnav',               None),
}

# label -> policy_config file, for labels whose section is not in the default
# configs/policy.config. SICNav is the only one: policy_sicnav.config carries
# the sicnav_root and horizon settings behind the paper's SICNav row, which
# differ from the defaults in policy.config.
_POLICY_CONFIG_OVERRIDE = {
    'SICNAV':   'configs/policy_sicnav.config',
}


def _outcome_label(info):
    if isinstance(info, ReachGoal):
        return 'success'
    if isinstance(info, Collision):
        return 'collision'
    if isinstance(info, Timeout):
        return 'timeout'
    return 'other'


def _reset_policy_state(policy):
    """Zero per-episode velocity memory so each run starts cold (mirrors STYLES)."""
    if hasattr(policy, '_reset_unicycle_state'):
        policy._reset_unicycle_state()
    else:
        for attr in ('prev_v', 'prev_omega', '_v0_from_last_step', '_w0_from_last_step'):
            if hasattr(policy, attr):
                setattr(policy, attr, 0.0)


def _config_as_dict(path):
    cp = configparser.RawConfigParser()
    cp.read(path)
    return {s: dict(cp.items(s)) for s in cp.sections()}


def build_policy(label, env, device, args):
    """Instantiate + configure one baseline, loading RL weights where needed."""
    key, model_dir = BASELINES[label]
    if args.model_dir_override and label in args.model_dir_override:
        model_dir = args.model_dir_override[label]

    policy_config_file = (os.path.join(model_dir, 'policy.config') if model_dir
                          else _POLICY_CONFIG_OVERRIDE.get(label, args.policy_config))

    policy = policy_factory[key]()
    cp = configparser.RawConfigParser()
    cp.read(policy_config_file)
    policy.configure(cp)

    if policy.trainable:
        if model_dir is None:
            raise ValueError(f'{label}: trainable policy needs a model_dir')
        weights = os.path.join(model_dir, 'resumed_rl_model.pth')
        if not os.path.exists(weights):
            weights = os.path.join(model_dir, 'il_model.pth' if args.il else 'rl_model.pth')
        logging.info('  %s: loading %s', label, weights)
        policy.get_model().load_state_dict(torch.load(weights, map_location='cpu'))

    policy.set_phase(args.phase)
    policy.set_device(device)
    policy.set_env(env)

    # The model_dir env.config must agree with the one the env was built from,
    # otherwise the baseline is being scored under different time_limit /
    # radius / v_pref than the styled policy and the comparison is void.
    if model_dir:
        mine  = _config_as_dict(os.path.join(model_dir, 'env.config'))
        theirs = _config_as_dict(args.env_config)
        if mine != theirs:
            logging.warning('%s: %s/env.config DIFFERS from %s — comparison may be '
                            'confounded. Diff them before trusting these numbers.',
                            label, model_dir, args.env_config)

    logging.info('  %s: policy=%s kinematics=%s trainable=%s config=%s',
                 label, key, policy.kinematics, policy.trainable, policy_config_file)
    return policy


def main():
    p = argparse.ArgumentParser('Unstyled baselines on the style-probe scenes')
    p.add_argument('--env_config', type=str, default='configs/env.config')
    p.add_argument('--policy_config', type=str, default='configs/policy.config',
                   help='Config for baselines without a model_dir (ORCA, SFM). '
                        'Baselines given a --model_dir read that directory\'s '
                        'own policy.config instead, which pins the architecture '
                        'their checkpoint was trained under.')
    p.add_argument('--baselines', type=str, default=','.join(BASELINES),
                   help='Comma-separated subset of: ' + ','.join(BASELINES))
    p.add_argument('--model_dir', action='append', default=[], metavar='LABEL=DIR',
                   help='Override a baseline weights dir, e.g. --model_dir SARL=output_SARL_v2')
    p.add_argument('--il', action='store_true', help='Use il_model.pth instead of rl_model.pth.')
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--phase', type=str, default='test')
    p.add_argument('--npz_hard', action='store_true',
                   help='Evaluate on the pre-generated npz scenes (the style-probe set).')
    p.add_argument('--npz_dir', type=str, default='../../data/scenes/eval_500')
    p.add_argument('--npz_num_eval', type=int, default=100)
    p.add_argument('--num_tests', type=int, default=50,
                   help='Scene count when --npz_hard is not given.')
    p.add_argument('--square', action='store_true')
    p.add_argument('--circle', action='store_true')
    p.add_argument('--results_suffix', type=str, default='BASELINES')
    p.add_argument('--video_file', type=str, default='baselines.mp4',
                   help='Combined output video (one synchronized panel per baseline, '
                        'same layout as the style sweep videos)')
    p.add_argument('--fps', type=int, default=10)
    p.add_argument('--no_video', action='store_true',
                   help='Skip rendering; metrics only (much faster)')
    p.add_argument('--grid_rows', type=int, default=None,
                   help='Panel grid rows (default: auto, near-square)')
    p.add_argument('--grid_cols', type=int, default=None)
    args = p.parse_args()

    args.model_dir_override = {}
    for spec in args.model_dir:
        label, _, d = spec.partition('=')
        args.model_dir_override[label] = d

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    labels = [l.strip() for l in args.baselines.split(',') if l.strip()]
    unknown = [l for l in labels if l not in BASELINES]
    if unknown:
        raise ValueError(f'Unknown baseline(s) {unknown}; known: {list(BASELINES)}')

    device = torch.device('cuda:0' if torch.cuda.is_available() and args.gpu else 'cpu')
    logging.info('Device: %s', device)

    # ── Env ──────────────────────────────────────────────────────────────────
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

    robot = Robot(env_config, 'robot')
    env.set_robot(robot)

    # ── Policies ─────────────────────────────────────────────────────────────
    logging.info('Building %d baselines: %s', len(labels), labels)
    policies = {l: build_policy(l, env, device, args) for l in labels}

    # ── Scene list — SAME filter as evaluate_styles.py ───────────────────
    scene_threats = []
    scene_has_map = []
    if args.npz_hard:
        scene_files = []
        for f in sorted(glob.glob(os.path.join(args.npz_dir, '*.npz'))):
            try:
                d = np.load(f, allow_pickle=True)
                difficulty = str(d['difficulty']) if 'difficulty' in d else None
                threat     = str(d['threat_type']) if 'threat_type' in d else None
                if difficulty == 'hard' or threat is not None:
                    scene_files.append(f)
                    scene_threats.append(threat or 'unknown')
                    scene_has_map.append(float(d['has_map']) if 'has_map' in d else 0.0)
            except Exception as e:
                logging.warning('Could not load %s: %s', f, e)
        if not scene_files:
            raise ValueError(f'No npz scenes found in {args.npz_dir}')
        scene_files    = scene_files[:args.npz_num_eval]
        scene_threats  = scene_threats[:args.npz_num_eval]
        scene_has_map  = scene_has_map[:args.npz_num_eval]
        logging.info('NPZ scenes: %d from %s', len(scene_files), args.npz_dir)
    else:
        scene_files   = [None] * args.num_tests
        scene_threats = ['generated'] * args.num_tests
        scene_has_map = [0.0] * args.num_tests
    num_scenes = len(scene_files)

    results_dir = f'results_{args.results_suffix}' if args.results_suffix else 'results_baselines'
    os.makedirs(results_dir, exist_ok=True)
    tmp_dir = os.path.join(results_dir, 'tmp_scenes')
    if not args.no_video:
        os.makedirs(tmp_dir, exist_ok=True)
    _, video_ext = os.path.splitext(args.video_file)

    # Panel descriptors for build_scene_video, which expects the style sweep's
    # (label, style_vec, use_projection) triple. These baselines have no style
    # vector and no feasibility-projection layer, so the vector is empty (the
    # panel subtitle renders as "[]") and use_projection is False — both
    # literally true here, rather than faking a [0,0,0,0] neutral style that
    # would imply a style input these planners do not have.
    panel_variants = [(label, [], False) for label in labels]
    grid_rows, grid_cols = auto_grid(len(labels), args.grid_rows, args.grid_cols)
    if not args.no_video:
        logging.info('Rendering %d panels in a %dx%d grid',
                     len(labels), grid_rows, grid_cols)
    all_scene_videos = []

    # ep_times/path_lens hold ONLY successful episodes (the style sweep's
    # convention, so AvgT/AvgPL are directly comparable). time_all/path_all are
    # per-scene and NaN on failure, which is what the subset breakdowns need.
    metrics = {l: {k: [] for k in
                   ['successes', 'timeouts', 'collisions', 'wall_colls',
                    'ep_times', 'path_lens', 'time_all', 'path_all',
                    'min_dists', 'avg_min_dists', 'has_map'] + _SOCIAL_KEYS}
               for l in labels}
    csv_rows = []

    # ── Scene loop (outer) × baseline loop (inner) ───────────────────────────
    for i in range(num_scenes):
        npz_path = scene_files[i]
        logging.info('Scene %d / %d  (%s)', i + 1, num_scenes,
                     os.path.basename(npz_path) if npz_path else 'generated')

        # Snapshot the RNG anchor before any baseline touches the env, then
        # restore it per baseline — reset() bumps testoffset every call, so
        # without this each baseline would draw a different scene.
        scene_testoffset = sim.testoffset
        scene_counter    = sim.case_counter[args.phase]
        episodes = []

        for label in labels:
            policy = policies[label]
            robot.set_policy(policy)      # also syncs robot.kinematics
            env.set_robot(robot)
            _reset_policy_state(policy)

            sim.testoffset = scene_testoffset
            sim.case_counter[args.phase] = scene_counter
            if npz_path is not None:
                env.load_npz_scenario(npz_path)

            ob = env.reset(phase=args.phase, test_case=i)
            done = False
            while not done:
                ob, _, done, info = env.step(robot.act(ob))

            sm = compute_episode_social_metrics(
                copy.deepcopy(sim.states), sim.humans, sim.time_step)

            if not args.no_video:
                # Same fields run_one_episode() returns — that is the contract
                # build_scene_video reads. Baselines set no policy.predicted_traj,
                # so sim.predicted_trajs is a list of None and the sampled/
                # selected/projected trajectory overlays simply stay empty.
                episodes.append({
                    'states':           copy.deepcopy(sim.states),
                    'predicted_trajs':  copy.deepcopy(sim.predicted_trajs),
                    'info':             info,
                    'global_time':      sim.global_time,
                    'pathlength':       sim.pathlength,
                    'human_radii':      [h.radius for h in sim.humans],
                    'n_humans':         len(sim.humans),
                    'robot_radius':     robot.radius,
                    'robot_kinematics': robot.kinematics,
                    'goal_pos':         tuple(robot.get_goal_position()),
                    'time_step':        float(sim.time_step),
                    'occ_map':    copy.deepcopy(getattr(sim, '_npz_occ_map', None)),
                    'has_map':    float(getattr(sim, '_npz_has_map', 0.0)),
                    'map_extent': float(getattr(sim, '_npz_map_extent', 10.0)),
                })

            m = metrics[label]
            m['successes'].append(1 if isinstance(info, ReachGoal) else 0)
            m['timeouts'].append(1 if isinstance(info, Timeout) else 0)
            m['collisions'].append(1 if isinstance(info, Collision) else 0)
            m['wall_colls'].append(1 if isinstance(info, WallCollision) else 0)
            if isinstance(info, ReachGoal):
                m['ep_times'].append(sim.global_time)
                m['path_lens'].append(sim.pathlength)
            m['time_all'].append(sim.global_time if isinstance(info, ReachGoal) else np.nan)
            m['path_all'].append(sim.pathlength if isinstance(info, ReachGoal) else np.nan)
            m['min_dists'].append(sim.minobsdist)
            m['avg_min_dists'].append(np.mean(sim.avgobsdist) if sim.avgobsdist else float('nan'))
            m['has_map'].append(scene_has_map[i])
            for k in _SOCIAL_KEYS:
                m[k].append(sm[k])

            row = {
                'scene': i + 1,
                'variant': label,
                'threat': scene_threats[i],
                'has_map': scene_has_map[i],
                'outcome': _outcome_label(info),
                'time': f'{sim.global_time:.3f}',
                'path_len': f'{sim.pathlength:.3f}',
                'min_dist': f'{sim.minobsdist:.3f}',
            }
            row.update({k: f'{sm[k]:.4f}' for k in _SOCIAL_KEYS})
            csv_rows.append(row)

            logging.info('  %-8s → %-9s t=%.2fs path=%.2f minD=%.3f',
                         label, _outcome_label(info), sim.global_time,
                         sim.pathlength, sim.minobsdist)

        # ── Synchronized panel video for this scene ──────────────────────────
        if not args.no_video:
            scene_video = os.path.join(tmp_dir, f'scene_{i + 1:04d}{video_ext}')
            with variant_render_context(panel_variants, grid_rows, grid_cols):
                build_scene_video(episodes, i, scene_video, fps=args.fps)
            all_scene_videos.append(scene_video)

    # ── Concatenate per-scene videos ─────────────────────────────────────────
    if all_scene_videos:
        combined = os.path.join(results_dir, os.path.basename(args.video_file))
        list_file = os.path.join(tmp_dir, 'video_list.txt')
        with open(list_file, 'w') as fh:
            for vid in all_scene_videos:
                fh.write(f"file '{os.path.abspath(vid)}'\n")
        subprocess.run([ffmpeg_binary(), '-y', '-f', 'concat', '-safe', '0',
                        '-i', list_file, '-c', 'copy', combined], check=False)
        logging.info('Combined video: %s', combined)

    # ── Per-scene CSV ────────────────────────────────────────────────────────
    csv_path = os.path.join(results_dir, 'per_scene_metrics.csv')
    fieldnames = ['scene', 'variant', 'threat', 'has_map', 'outcome',
                  'time', 'path_len', 'min_dist'] + _SOCIAL_KEYS
    with open(csv_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)
    logging.info('Per-scene CSV: %s', csv_path)

    # ── Summary (same layout as the style sweep's summary.txt) ───────────────
    summary_path = os.path.join(results_dir, 'summary.txt')
    with open(summary_path, 'w') as fh:
        fh.write('=' * 60 + '\n')
        fh.write('Unstyled Baseline Evaluation Summary\n')
        fh.write(f'Scenes evaluated: {num_scenes}\n')
        fh.write(f'Scene source: {args.npz_dir if args.npz_hard else "generated"}\n')
        fh.write('Metric definitions identical to evaluate_styles.py\n')
        fh.write('=' * 60 + '\n\n')

        header = (f"{'Baseline':<20} {'SR%':>6} {'TO%':>6} {'CR%':>6} "
                  f"{'WallCR%':>8} {'AvgT':>7} {'AvgPL':>7} {'AvgMinD':>8}\n")
        fh.write(header)
        fh.write('-' * len(header) + '\n')
        for label in labels:
            m = metrics[label]
            def _m(k): return np.mean(m[k]) if m[k] else float('nan')
            fh.write(f"{label:<20} {100*_m('successes'):6.1f} {100*_m('timeouts'):6.1f} "
                     f"{100*_m('collisions'):6.1f} {100*_m('wall_colls'):8.1f} "
                     f"{_m('ep_times'):7.3f} {_m('path_lens'):7.3f} {_m('min_dists'):8.3f}\n")

        fh.write('\nColumn key:\n')
        fh.write('  SR%    = Success Rate\n')
        fh.write('  TO%    = Timeout Rate\n')
        fh.write('  CR%    = Agent Collision Rate\n')
        fh.write('  WallCR%= Static Map Collision Rate\n')
        fh.write('  AvgT   = Average Time to Goal (successful episodes)\n')
        fh.write('  AvgPL  = Average Path Length to Goal (successful episodes)\n')
        fh.write('  AvgMinD= Average Minimum Distance to Obstacles (all episodes)\n')

        fh.write('\n\nSocial Style Metrics (Hall-proxemics thresholds — comparable to the style sweep)\n')
        fh.write(f'  D_per = 4 ft = {_D_PER:.4f} m (personal space)\n')
        fh.write(f'  D_soc = 12 ft = {_D_SOC:.4f} m (social space)\n')
        fh.write(f'  D_int = 1.5 ft = {_D_INT:.4f} m (intimate space)\n')
        fh.write(f'  tau_r = {_TAU_R:.2f} s (human reaction time)\n')
        fh.write('These baselines expose no style knob, so this row is the '
                 '"unstyled" reference the styled deltas are measured against.\n\n')

        soc_header = (f"{'Baseline':<20} {'MinClear':>9} {'PSIRate':>8} {'PassBias':>9} "
                      f"{'TTCInfR':>8} {'CutoffR':>8} {'GrpSplit':>9}\n")
        fh.write(soc_header)
        fh.write('-' * len(soc_header) + '\n')
        for label in labels:
            m = metrics[label]
            def _a(k): return np.nanmean(m[k]) if m[k] else float('nan')
            fh.write(f"{label:<20} {_a('mean_min_clearance_m'):9.3f} {_a('psi_rate'):8.4f} "
                     f"{_a('passing_side_bias'):+9.3f} {_a('ttc_infraction_rate'):8.4f} "
                     f"{_a('path_cutoff_rate'):8.4f} {_a('group_split_rate'):9.4f}\n")

        # Baselines get no occupancy map, so map scenes penalize them for a
        # missing input rather than for their navigation behavior. Report the
        # map-free subset so the comparison can be made on equal information.
        mapped = [j for j in range(num_scenes) if scene_has_map[j] > 0.5]
        if mapped:
            keep = [j for j in range(num_scenes) if scene_has_map[j] <= 0.5]
            fh.write(f'\n\nMap-free subset ({len(keep)} of {num_scenes} scenes; the '
                     f'{len(mapped)} scenes with an occupancy map are excluded because '
                     f'these baselines have no map input and would be scored on a '
                     f'missing observation rather than on their navigation)\n')
            fh.write(header)
            fh.write('-' * len(header) + '\n')
            for label in labels:
                m = metrics[label]
                sub = lambda k: np.array([m[k][j] for j in keep], dtype=np.float64)
                fh.write(f"{label:<20} {100*np.mean(sub('successes')):6.1f} "
                         f"{100*np.mean(sub('timeouts')):6.1f} "
                         f"{100*np.mean(sub('collisions')):6.1f} "
                         f"{100*np.mean(sub('wall_colls')):8.1f} "
                         f"{np.nanmean(sub('time_all')):7.3f} "
                         f"{np.nanmean(sub('path_all')):7.3f} "
                         f"{np.mean(sub('min_dists')):8.3f}\n")

        # Per-threat-type breakdown: which encounter categories the unstyled
        # baselines already handle, and which the style axes are meant to move.
        if args.npz_hard:
            fh.write('\n\n--- Per Threat Type (Success Rate %) ---\n')
            uniq = sorted(set(scene_threats[:num_scenes]))
            fh.write(f"{'Baseline':<20}" + ''.join(f'{t[:13]:>15}' for t in uniq) + '\n')
            for label in labels:
                m = metrics[label]
                cells = []
                for t in uniq:
                    idxs = [j for j, tt in enumerate(scene_threats[:num_scenes]) if tt == t]
                    cells.append(f"{100*np.mean([m['successes'][j] for j in idxs]):15.1f}")
                fh.write(f'{label:<20}' + ''.join(cells) + '\n')

    logging.info('Summary: %s', summary_path)
    with open(summary_path) as fh:
        print(fh.read())


if __name__ == '__main__':
    main()
