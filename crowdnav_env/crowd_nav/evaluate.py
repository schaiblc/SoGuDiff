"""Score a single policy over the evaluation scene set.

This is the primary entry point for the paper's numbers. It runs one policy
across every scene and writes success/collision/timeout rates, time and path
length to goal, clearance, and the social metrics shared with the style sweep.

Examples
--------
Smoke test -- no weights or downloaded scenes required, because scenes are
generated procedurally when ``--npz_hard`` is absent::

    python evaluate.py --policy orca_unicycle --phase test --no_video

The proposed method on the 500-scene evaluation set::

    python evaluate.py \
        --policy sogudiff \
        --infer_mode per_axis --w_normalize --gpu \
        --npz_hard --npz_dir ../../data/scenes/eval_500 --npz_num_eval 500 \
        --results_suffix PAPER --no_video

A learned baseline, whose architecture is pinned by its model directory::

    python evaluate.py --policy sarl --model_dir models/sarl \
        --gpu --npz_hard --npz_dir ../../data/scenes/eval_500 --npz_num_eval 500

Results are written to ``results_<results_suffix>/``. Rendering dominates wall
time on large runs, so pass ``--no_video`` for numbers-only evaluation.

Pass ``--visualize`` with a ``--video_file`` to render the run; omit
``--no_video``. Rendering dominates wall time, so keep the scene count small.
"""

import logging
import argparse
import configparser
import os
import torch
import numpy as np
import gym
import glob
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory
from crowd_nav.paths import ffmpeg_binary
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.utils.info import *

# Hall personal-space distance (same threshold used for PSI rate in
# evaluate_styles.py), converted from feet to the simulator's meters.
_FT_TO_M = 0.3048
_D_PER   = 4.0 * _FT_TO_M  # ≈ 1.2192 m


def _compute_mind_and_psi(states):
    """
    MinD and PSI rate for one episode, computed identically to
    evaluate_styles.py: raw center-to-center robot-human distance (no
    radius subtraction, humans only — no static-map obstacles), minimized
    over humans at each timestep.

      MinD = (1/T) * sum_t min_j || p^r_t - p^j_t ||
      PSI  = (1/T) * sum_t  1[ min_j || p^r_t - p^j_t || < D_per ]
    """
    T = len(states)
    M = len(states[0][1]) if T > 0 else 0
    if T == 0 or M == 0:
        return float('inf'), 0.0

    rob_xy = np.array([s[0].position for s in states], dtype=np.float64)
    hum_xy = np.array(
        [[s[1][j].position for j in range(M)] for s in states], dtype=np.float64
    )
    dist = np.linalg.norm(rob_xy[:, None, :] - hum_xy, axis=-1)
    min_dist_per_t = dist.min(axis=1)
    mind = float(min_dist_per_t.mean())
    psi  = float((min_dist_per_t < _D_PER).mean())
    return mind, psi


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--policy', type=str, default='orca')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=False, action='store_true')
    parser.add_argument('--video_file', type=str, default=None)
    parser.add_argument('--traj', default=False, action='store_true')
    parser.add_argument("--results_suffix", type=str, default="",
                        help="Optional suffix for results folder, e.g. 'stageB'")
    # ── NPZ hard evaluation ───────────────────────────────────────────────────
    parser.add_argument('--npz_hard', default=False, action='store_true',
                        help='Evaluate on pre-generated hard npz scenarios')
    parser.add_argument('--npz_dir', type=str, default='../../data/scenes/eval_500',
                        help='Directory of saved .npz scenes (see '
                             'assets/MANIFEST.md). Used only with --npz_hard.')
    parser.add_argument('--npz_num_eval', type=int, default=50,
                        help='Number of scenes to evaluate on. Only applies '
                             'with --npz_hard; without it the episode count '
                             'comes from env.config test_size.')
    # ── Map-eval (structured layout) evaluation ───────────────────────────────
    parser.add_argument('--map_eval', default=False, action='store_true',
                        help='Generate structured map layouts (corridor, room, etc.) '
                             'for evaluation. Only meaningful when use_map=true in policy config.')
    parser.add_argument('--map_eval_num', type=int, default=50,
                        help='Number of map_eval scenarios to evaluate')
    # ── SoGuDiff guidance overrides (ignored by non-diffusion policies) ───────
    parser.add_argument('--infer_mode', type=str, default=None,
                        choices=['joint', 'per_axis'],
                        help='Override the diffusion policy CFG mode. Only valid '
                             'if the checkpoint was trained for it (joint/per_axis/union).')
    parser.add_argument('--style_vector', type=str, default=None,
                        help="Fixed style for the diffusion policy, e.g. '0,0,0,0'. "
                             'Default: whatever policy.config specifies (neutral).')
    parser.add_argument('--cfg_w_style', type=str, default=None,
                        help="Per-axis guidance weights 'a,b,c,d' or one scalar.")
    parser.add_argument('--w_normalize', action='store_true',
                        help='per_axis: hold Σ_active w_i constant.')
    parser.add_argument('--no_video', action='store_true',
                        help='Numbers-only: skip rendering, which dominates wall '
                             'time on a 500-scene run. Metrics are identical '
                             'either way.')
    args = parser.parse_args()

    # Rendering names its per-episode clips after --video_file, so a rendering
    # run needs one. Default it rather than failing deep in the episode loop
    # with a TypeError from os.path.splitext(None).
    if args.video_file is None and not args.no_video:
        args.video_file = f'{args.policy}.mp4'

    if args.model_dir is not None:
        env_config_file = os.path.join(args.model_dir, os.path.basename(args.env_config))
        policy_config_file = os.path.join(args.model_dir, os.path.basename(args.policy_config))
        if args.il:
            model_weights = os.path.join(args.model_dir, 'il_model.pth')
        else:
            if os.path.exists(os.path.join(args.model_dir, 'resumed_rl_model.pth')):
                model_weights = os.path.join(args.model_dir, 'resumed_rl_model.pth')
            else:
                model_weights = os.path.join(args.model_dir, 'rl_model.pth')
    else:
        env_config_file = args.env_config
        policy_config_file = args.policy_config

    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)

    policy = policy_factory[args.policy]()
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_file)
    policy.configure(policy_config)

    # ── Apply SoGuDiff guidance overrides (no-ops for baselines) ─────────────
    # These set instance attributes the patched diffusion policy reads at sample
    # time, so we can pick joint vs per_axis and the style/weights without a
    # separate config file. Guarded by hasattr so ORCA/SFM/RL policies ignore it.
    if args.infer_mode is not None and hasattr(policy, 'set_infer_mode'):
        policy.set_infer_mode(args.infer_mode)   # validates against trained_cfg_mode
    if args.w_normalize and hasattr(policy, 'cfg_w_normalize'):
        policy.cfg_w_normalize = True
    if args.style_vector is not None and hasattr(policy, 'style_vector'):
        sv = [float(v) for v in args.style_vector.split(',') if v.strip()]
        policy.style_vector = np.array(sv, dtype=np.float32)
    if args.cfg_w_style is not None and hasattr(policy, 'cfg_w_style'):
        vals = [float(v) for v in args.cfg_w_style.split(',') if v.strip()]
        n = len(getattr(policy, 'style_vector', [0, 0, 0, 0]))
        policy.cfg_w_style = vals * n if len(vals) == 1 else vals
    if hasattr(policy, 'cfg_infer_mode'):
        logging.info('Diffusion guidance: mode=%s w_normalize=%s style=%s',
                     policy.cfg_infer_mode,
                     getattr(policy, 'cfg_w_normalize', False),
                     getattr(policy, 'style_vector', None))

    if policy.trainable:
        if args.model_dir is None:
            parser.error('Trainable policy must be specified with a model weights directory')
        # map_location='cpu' so a checkpoint saved on GPU still loads on a
        # CPU-only machine; set_device below moves it to the target device.
        policy.get_model().load_state_dict(
            torch.load(model_weights, map_location='cpu'))

    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    # Use env.unwrapped to reach the underlying CrowdSim directly — gym wrappers
    # do not forward __setattr__ to the wrapped env, so env.test_sim = '...'
    # would only set the attribute on the wrapper, not on the CrowdSim itself.
    _sim = env.unwrapped
    if args.square:
        _sim.test_sim = 'square_crossing'
    if args.circle:
        _sim.test_sim = 'circle_crossing'
    if args.npz_hard:
        _sim.test_sim = 'npz_hard'
    if args.map_eval:
        _sim.test_sim = 'map_eval'
    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot)
    explorer = Explorer(env, robot, device, gamma=0.9)

    policy.set_phase(args.phase)
    policy.set_device(device)
    if isinstance(robot.policy, ORCA):
        if robot.visible:
            robot.policy.safety_space = 0
        else:
            robot.policy.safety_space = 0
        logging.info('ORCA agent buffer: %f', robot.policy.safety_space)

    policy.set_env(env)
    robot.print_info()

    if args.visualize or args.no_video:

        # ── Build scenario list ───────────────────────────────────────────────
        if args.npz_hard:
            # Load hard-only npz files, sorted for full determinism across policies
            all_files = sorted(glob.glob(os.path.join(args.npz_dir, '*.npz')))
            hard_files = []
            for f in all_files:
                try:
                    d = np.load(f, allow_pickle=True)
                    difficulty = str(d['difficulty']) if 'difficulty' in d else None
                    threat = str(d['threat_type']) if 'threat_type' in d else None
                    # Accept a scene labelled 'hard', or any scene that
                    # carries a threat_type at all. Scene generators use
                    # different threat vocabularies (head_on, cutoff,
                    # sneak_up, group_head_on, corridor_head_on,
                    # stationary_group, ...), so matching a fixed list here
                    # would silently reject whole scene sets. This is the same
                    # rule evaluate_styles.py applies, so both entry points
                    # select the same scenes from a given directory.
                    is_hard = (difficulty == 'hard' or threat is not None)
                    if is_hard:
                        hard_files.append(f)
                except Exception as e:
                    logging.warning('Could not load %s: %s', f, e)

            if len(hard_files) == 0:
                raise ValueError(
                    f'No usable .npz scenes found in {args.npz_dir}. Scenes must carry a threat_type or difficulty field; see assets/MANIFEST.md.')

            hard_files = hard_files[:args.npz_num_eval]
            num_tests = len(hard_files)
            logging.info('NPZ hard evaluation: %d scenarios from %s', num_tests, args.npz_dir)
        elif args.map_eval:
            # map_eval: scenes are generated on-the-fly from a seeded RNG
            hard_files = None
            num_tests  = args.map_eval_num
            logging.info('Map-eval evaluation: %d generated scenarios', num_tests)
        else:
            # Procedural scenes: the episode count is env.config's test_size
            # (500 for the paper's benchmark), so editing the config for a
            # quick pass actually shortens the run.
            hard_files = None
            num_tests = env.case_size[args.phase]
            logging.info('Procedural evaluation: %d scenarios (%s test_size)',
                         num_tests, os.path.basename(env_config_file))

        # ── Metrics ──────────────────────────────────────────────────────────
        episode_times = []
        path_lengths = []
        successes = []
        timeouts = []
        collisions = []           # agent-agent collisions (Collision)
        static_map_collisions = [] # robot-static-map collisions (WallCollision)
        min_dists = []
        avg_min_dists = []
        mind_vals = []            # MinD (matches style-sweep definition)
        psi_rates = []            # PSI rate (matches style-sweep definition)
        threat_types = []         # map_type string for map_eval; threat label otherwise
        rendered_files = []
        base_policy = None

        for i in range(num_tests):
            logging.info('Starting test case %d / %d', i + 1, num_tests)

            # ── Reset: npz_hard / map_eval / standard ────────────────────────
            if args.npz_hard:
                npz_path = hard_files[i]
                env.load_npz_scenario(npz_path)
                # test_case=i gives a fixed per-scenario seed — same across policies
                ob = env.reset(phase='test', test_case=i)

                # Record threat type for per-category breakdown
                try:
                    d = np.load(npz_path, allow_pickle=True)
                    threat_types.append(str(d.get('threat_type', 'unknown')))
                except Exception:
                    threat_types.append('unknown')
            elif args.map_eval:
                # Scene is generated inside reset() via the seeded RNG;
                # test_case=i ensures each scenario is repeatable.
                ob = env.reset(phase='test', test_case=i)
                # Record the map type generated for this episode
                map_type = getattr(env, '_npz_map_eval_type', 'unknown')
                threat_types.append(map_type)
            else:
                ob = env.reset(phase='test')
                threat_types.append('standard')

            done = False
            last_pos = np.array(robot.get_position())

            while not done:
                action = robot.act(ob)
                ob, _, done, info = env.step(action)
                current_pos = np.array(robot.get_position())
                logging.debug('Speed: %.2f',
                              np.linalg.norm(current_pos - last_pos) / robot.time_step)
                last_pos = current_pos

            # ── Render (skipped in numbers-only mode) ─────────────────────────
            if args.no_video:
                base_policy = os.path.splitext(args.video_file)[0] if args.video_file else args.policy
                ext = '.mp4'
            elif args.traj:
                env.render(mode='human')
                base, ext = os.path.splitext(args.video_file)
                base_policy = base
                env.render(mode='traj', output_file=f'{base}_{i+1}{ext}',
                           basepolicy=base, testnum=i)
            else:
                env.render(mode='human')
                base, ext = os.path.splitext(args.video_file)
                base_policy = base
                env.render(mode='video', output_file=f'{base}_{i+1}{ext}',
                           basepolicy=base, testnum=i)
                rendered_files.append(f'{base}_{i+1}{ext}')

            # ── Record metrics ───────────────────────────────────────────────
            if isinstance(info, ReachGoal):
                episode_times.append(env.global_time)
                path_lengths.append(env.pathlength)

            successes.append(1 if isinstance(info, ReachGoal) else 0)
            timeouts.append(1 if isinstance(info, Timeout) else 0)
            collisions.append(1 if isinstance(info, Collision) else 0)
            static_map_collisions.append(1 if isinstance(info, WallCollision) else 0)
            min_dists.append(env.minobsdist)
            avg_min_dists.append(np.mean(env.avgobsdist) if env.avgobsdist else float('inf'))
            mind_val, psi_val = _compute_mind_and_psi(env.states)
            mind_vals.append(mind_val)
            psi_rates.append(psi_val)

            logging.info('Test %d: %.2fs | %s | path=%.2f minD=%.3f MinD=%.3f psiRate=%.3f',
                         i + 1, env.global_time, info,
                         env.pathlength, env.minobsdist, mind_val, psi_val)

        # ── Aggregate and save results ────────────────────────────────────────
        avg_time                  = np.mean(episode_times) if episode_times else float('nan')
        avg_path_length           = np.mean(path_lengths)  if path_lengths  else float('nan')
        success_rate              = 100 * np.mean(successes)
        timeout_rate              = 100 * np.mean(timeouts)
        collision_rate            = 100 * np.mean(collisions)
        static_map_collision_rate = 100 * np.mean(static_map_collisions)
        avg_min_dist              = np.mean(min_dists)
        avg_avg_min_dist          = np.mean(avg_min_dists)
        avg_mind                  = np.mean(mind_vals) if mind_vals else float('nan')
        avg_psi_rate              = np.mean(psi_rates) if psi_rates else float('nan')

        results_dir = f'results_{args.results_suffix}' if args.results_suffix else 'results'
        os.makedirs(results_dir, exist_ok=True)
        eval_file = os.path.join(results_dir, f'{base_policy}_eval.txt')

        with open(eval_file, 'w') as f:
            f.write('========== Evaluation Results ==========\n')
            mode_label = 'NPZ Hard' if args.npz_hard else 'Standard'
            f.write(f'Evaluation mode: {mode_label}\n')
            if args.npz_hard:
                f.write(f'NPZ directory: {args.npz_dir}\n')
            f.write(f'Number of test episodes: {num_tests}\n')
            if args.map_eval:
                f.write('Map types evaluated: corridor, room, doorway, pillars, '
                        'narrow_passage, L_corner, single_wall, open\n')
            f.write(f'Success Rate: {success_rate:.2f} %\n')
            f.write(f'Timeout Rate: {timeout_rate:.2f} %\n')
            f.write(f'Agent Collision Rate: {collision_rate:.2f} %\n')
            f.write(f'Static Map Collision Rate: {static_map_collision_rate:.2f} %\n')
            f.write(f'Total Collision Rate: {collision_rate + static_map_collision_rate:.2f} %\n')
            f.write(f'Average Time to Goal: {avg_time:.3f}\n')
            f.write(f'Average Path Length to Goal: {avg_path_length:.3f}\n')
            f.write(f'Average Minimum Distance to Obstacles: {avg_min_dist:.3f}\n')
            f.write(f'Average Per-Episode Avg Min Distance: {avg_avg_min_dist:.3f}\n')
            f.write(f'Average MinD (robot-human center distance, matches style sweep): {avg_mind:.3f}\n')
            f.write(f'Average PSI Rate (D_per={_D_PER:.4f} m, matches style sweep): {avg_psi_rate:.4f}\n')
            f.write(f'Policy: {base_policy}\n')

            # Per-category breakdown (threat type for npz_hard, map type for map_eval)
            if args.npz_hard or args.map_eval:
                label = 'Map Type' if args.map_eval else 'Threat Type'
                f.write(f'\n--- Per {label} ---\n')
            if args.npz_hard or args.map_eval:
                unique_threats = sorted(set(threat_types))
                for tt in unique_threats:
                    idxs = [j for j, t in enumerate(threat_types) if t == tt]
                    tt_sr     = 100 * np.mean([successes[j] for j in idxs])
                    tt_col    = 100 * np.mean([collisions[j] for j in idxs])
                    tt_mapcol = 100 * np.mean([static_map_collisions[j] for j in idxs])
                    tt_min    = np.mean([min_dists[j] for j in idxs])
                    tt_mind   = np.mean([mind_vals[j] for j in idxs])
                    tt_psi    = np.mean([psi_rates[j] for j in idxs])
                    f.write(f'  {tt:10s}: n={len(idxs):4d}  '
                            f'SR={tt_sr:.1f}%  '
                            f'AgentCR={tt_col:.1f}%  '
                            f'MapCR={tt_mapcol:.1f}%  '
                            f'minD={tt_min:.3f}  '
                            f'MinD={tt_mind:.3f}  '
                            f'psiRate={tt_psi:.3f}\n')

        logging.info('Results saved to %s', eval_file)

        # ── Combine videos ────────────────────────────────────────────────────
        if rendered_files:
            import subprocess
            list_file = 'video_list.txt'
            with open(list_file, 'w') as f:
                for vid in rendered_files:
                    f.write(f"file '{vid}'\n")
            combined = os.path.join(results_dir, f'{base_policy}_VIDEOS{ext}')
            # -y: ffmpeg was prompting "Overwrite? [y/N]" on stdin in this
            # non-interactive job and getting EOF -> "N", silently leaving the
            # PREVIOUS run's combined video in place while the code below
            # still deleted this run's per-episode clips and logged success
            # (, baselines/height eval after the RNN-state fix: numbers
            # saved correctly, video silently stayed on the pre-fix footage).
            result = subprocess.run([ffmpeg_binary(), '-y', '-f', 'concat', '-safe', '0',
                                     '-i', list_file, '-c', 'copy', combined])
            if result.returncode == 0:
                for f in rendered_files:
                    os.remove(f)
                logging.info('Combined video saved as %s', combined)
            else:
                logging.error('ffmpeg concat failed (rc=%d) — keeping per-episode '
                              'clips instead of deleting them', result.returncode)
            os.remove(list_file)

    else:
        explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True)


if __name__ == '__main__':
    main()