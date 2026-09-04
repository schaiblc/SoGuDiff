#!/usr/bin/env python3
"""
style_sweep_figure.py
==================================
Paper-figure tool: sweep ONE style axis (prox | pass | yield | group) over
{-1, 0, +1} on a SINGLE hand-picked scene, run N stochastic rollouts per
value, and render every resulting robot trajectory overlaid on the static
scene as one tightly-cropped, square image.

Rollouts come from `run_one_episode()` in evaluate_styles.py, so the figure
shows exactly what the style-sweep evaluation scores.

Why repeated rollouts differ
-----------------------------
The diffusion policy draws fresh Gaussian noise for every call to
predict() (no seeding anywhere in the stack), so calling run_one_episode()
N times with the same scene + same style vector naturally yields N
different sampled trajectories — that's exactly the "distribution" this
script visualizes.

What gets saved
----------------
  <out_dir>/<out_file>.png   the paper figure
  <out_dir>/<out_file>.npz   the raw swept trajectories + static scene
                              info, so the figure can be re-rendered without
                              re-running the policy (see replot_from_npz()
                              below / --replot).

Usage
-----
  python style_sweep_figure.py \\
      --policy_config configs/policy.config \\
      --env_config    configs/env.config \\
      --gpu \\
      --npz_path      /path/to/one_scene.npz \\
      --axis          pass \\
      --n_per_value   10 \\
      --out_dir       results_style_sweep_figs \\
      --out_file      pass_sweep_C1

Re-plot only (no GPU / policy needed) from a previously saved trajectory
distribution:
  python style_sweep_figure.py --replot results_style_sweep_figs/pass_sweep_C1.npz
"""

# These figure scripts live one directory below the evaluation code they share
# rollout and metric functions with, so make crowd_nav/ importable regardless of
# where the script is launched from.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import configparser
import logging
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import matplotlib.patheffects as patheffects
from matplotlib import animation


def _pad_to(lst, target_len):
    """Return a list of length target_len, repeating the last element —
    matches evaluate_styles.py's _pad_to() so shorter prediction
    histories hold their last frame instead of running out early."""
    if not lst:
        return [None] * target_len
    return list(lst) + [lst[-1]] * max(0, target_len - len(lst))


def render_run_video(ep, out_path, fps=10, robot_color='gold', title=None):
    """Render a single episode (one rollout) as an MP4, reusing the exact
    per-panel visual language of build_scene_video() in
    evaluate_styles.py — robot + trail, pedestrians, heading arrow,
    goal, AND the diffusion policy's per-step prediction (candidate
    samples / selected / projected), plus a legend and an outcome banner
    on the final frame — just for one episode instead of a 6-panel grid."""
    from evaluate_styles import _outcome_label, _OUTCOME_COLORS

    states = ep['states']
    n_frames = len(states)
    if n_frames == 0:
        return
    n_humans = ep['n_humans']
    robot_radius = ep['robot_radius']
    human_radii = ep['human_radii']
    kinematics = ep['robot_kinematics']
    time_step = ep['time_step']
    goal_pos = ep['goal_pos']
    occ_map = ep['occ_map']
    has_map = ep['has_map']
    map_extent = ep['map_extent']
    predicted_trajs = ep.get('predicted_trajs') or []
    padded_trajs = _pad_to(predicted_trajs, n_frames)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect('equal', adjustable='box')

    if occ_map is not None and has_map > 0.5:
        half = map_extent / 2.0
        ax.imshow(occ_map, extent=[-half, half, -half, half], origin='lower',
                  cmap='gray_r', alpha=0.45, zorder=0, interpolation='nearest')

    # ---- bounds: robot/human trajectory over the whole episode AND every
    #      per-step predicted sample, so the background candidates are
    #      never cropped out of frame ----
    all_x = [0.0, float(goal_pos[0])]
    all_y = [0.0, float(goal_pos[1])]
    for st in states:
        all_x.append(st[0].position[0]); all_y.append(st[0].position[1])
        for j in range(n_humans):
            all_x.append(st[1][j].position[0]); all_y.append(st[1][j].position[1])
    for td in predicted_trajs:
        if not isinstance(td, dict):
            continue
        for key in ('selected_sample', 'projection'):
            arr = td.get(key)
            if arr is not None:
                arr = np.asarray(arr)
                all_x.extend(arr[:, 0].tolist()); all_y.extend(arr[:, 1].tolist())
        for arr in (td.get('all_samples') or []):
            arr = np.asarray(arr)
            all_x.extend(arr[:, 0].tolist()); all_y.extend(arr[:, 1].tolist())
    if occ_map is not None and has_map > 0.5:
        half = map_extent / 2.0
        all_x.extend([-half, half]); all_y.extend([-half, half])
    margin = 0.8
    xmin, xmax = min(all_x) - margin, max(all_x) + margin
    ymin, ymax = min(all_y) - margin, max(all_y) + margin
    r = max(xmax - xmin, ymax - ymin)
    xc, yc = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    ax.set_xlim(xc - r / 2.0, xc + r / 2.0)
    ax.set_ylim(yc - r / 2.0, yc + r / 2.0)

    ax.set_xlabel('x (m)', fontsize=11)
    ax.set_ylabel('y (m)', fontsize=11)
    ax.tick_params(labelsize=9)
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold')

    goal_dot = mlines.Line2D([goal_pos[0]], [goal_pos[1]], color='red', marker='*',
                             linestyle='None', markersize=13, label='Goal', zorder=6)
    ax.add_artist(goal_dot)

    cmap_human = plt.cm.get_cmap('hsv', 10)
    x_off = y_off = 0.11
    s0 = states[0]
    robot_circle = plt.Circle(s0[0].position, robot_radius, fill=True,
                              facecolor=robot_color, edgecolor='black',
                              linewidth=1.6, zorder=7)
    ax.add_artist(robot_circle)
    robot_trail, = ax.plot([], [], '--', color=robot_color, lw=1.5, alpha=0.7, zorder=1)

    h_circles, h_labels = [], []
    for j in range(n_humans):
        hp = s0[1][j].position
        rad = human_radii[j] if j < len(human_radii) else 0.3
        hc = plt.Circle(hp, rad, fill=False, edgecolor=cmap_human(j % 10),
                        linewidth=2.0, zorder=5)
        ax.add_artist(hc)
        lbl = ax.text(hp[0] - x_off, hp[1] - y_off, str(j), color='black',
                     fontsize=9, fontweight='bold', zorder=6)
        h_circles.append(hc); h_labels.append(lbl)

    time_txt = ax.text(0.02, 0.98, 'T: 0.00 s', transform=ax.transAxes,
                       fontsize=9, va='top', zorder=8)
    arrows = []

    # ---- diffusion-policy prediction artists (candidates / selected / projected) ----
    K = 0
    for td in predicted_trajs:
        if isinstance(td, dict) and td.get('all_samples'):
            K = len(td['all_samples'])
            break
    bg_lines = [ax.plot([], [], '-', color='orange', lw=1.0, alpha=0.5, zorder=1.5)[0]
               for _ in range(max(K - 1, 0))]
    sel_line, = ax.plot([], [], '-', color='purple', lw=2.0, alpha=0.85,
                        zorder=2, label='Selected sample')
    proj_line, = ax.plot([], [], '-', color='royalblue', lw=2.5, alpha=0.9,
                         zorder=3, label='Projected sample')

    legend_handles = [robot_circle, goal_dot, sel_line, proj_line]
    legend_labels = ['Robot', 'Goal', 'Selected', 'Projected']
    if bg_lines:
        cand_handle = mlines.Line2D([], [], color='orange', lw=1.0, alpha=0.6,
                                    label='Candidate samples')
        legend_handles.append(cand_handle); legend_labels.append('Candidate samples')
    ax.legend(handles=legend_handles, labels=legend_labels, fontsize=8,
             loc='upper right', framealpha=0.85)

    # ---- outcome banner, shown only on the final frame ----
    outcome = _outcome_label(ep['info'])
    out_col = _OUTCOME_COLORS[outcome]
    out_str = f'SUCCESS  {ep["global_time"]:.2f}s' if outcome == 'success' else outcome.upper()
    out_text = ax.text(xc, yc, out_str, fontsize=13, fontweight='bold', color=out_col,
                       ha='center', va='center',
                       bbox=dict(facecolor='white', alpha=0.8, edgecolor=out_col, boxstyle='round'),
                       visible=False, zorder=10)

    def _update(frame):
        st = states[frame]
        td = padded_trajs[frame] if frame < len(padded_trajs) else None
        r_pos = st[0].position
        robot_circle.center = r_pos
        trail_x = [states[k][0].position[0] for k in range(frame + 1)]
        trail_y = [states[k][0].position[1] for k in range(frame + 1)]
        robot_trail.set_data(trail_x, trail_y)
        for j in range(n_humans):
            hp = st[1][j].position
            h_circles[j].center = hp
            h_labels[j].set_position((hp[0] - x_off, hp[1] - y_off))
        for arr in arrows:
            arr.remove()
        arrows.clear()
        if kinematics == 'unicycle':
            th = st[0].theta
            arr = patches.FancyArrowPatch(
                r_pos, (r_pos[0] + robot_radius * np.cos(th), r_pos[1] + robot_radius * np.sin(th)),
                color='red', arrowstyle=patches.ArrowStyle('->', head_length=4, head_width=2), zorder=8)
            ax.add_artist(arr)
            arrows.append(arr)
        time_txt.set_text(f'T: {frame * time_step:.2f} s')

        for bl in bg_lines:
            bl.set_data([], [])
        sel_line.set_data([], [])
        proj_line.set_data([], [])
        if isinstance(td, dict):
            allsmp = td.get('all_samples') or []
            for i, bl in enumerate(bg_lines):
                if i + 1 < len(allsmp):
                    s = np.asarray(allsmp[i + 1])
                    bl.set_data(s[:, 0], s[:, 1])
            sel = td.get('selected_sample')
            if sel is not None:
                sel = np.asarray(sel)
                sel_line.set_data(sel[:, 0], sel[:, 1])
            proj = td.get('projection')
            if proj is not None:
                proj = np.asarray(proj)
                proj_line.set_data(proj[:, 0], proj[:, 1])

        out_text.set_visible(frame == n_frames - 1)
        return []

    anim = animation.FuncAnimation(fig, _update, frames=n_frames,
                                   interval=1000 // fps, blit=False)
    writer = animation.FFMpegWriter(fps=fps, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    anim.save(out_path, writer=writer)
    plt.close(fig)
    logging.info('Saved run video: %s', out_path)

# ---------------------------------------------------------------------------
# Style-axis conventions. Declared locally (NOT imported at module level)
# so `--replot` works with numpy + matplotlib only, no torch/diffusers/gym
# needed. Must match STYLE_AXES in sogudiff.py exactly — this
# is asserted in collect_sweep() the one time this script actually touches
# the policy, so drift is caught rather than silently ignored.
# ---------------------------------------------------------------------------
STYLE_AXES = ["prox", "pass", "yield", "group"]

AXIS_TITLES = {
    "prox":  "Proximity Preference",
    "pass":  "Passing-Side Preference",
    "yield": "Yielding Preference",
    "group": "Group Deference",
}

# One perceptually-distinct sequential colormap per axis. Within a colormap,
# darker = -1, medium = 0, lighter = +1 (see _style_color()).
AXIS_CMAPS = {
    "prox":  "Purples",
    "pass":  "Blues",
    "yield": "Greens",
    "group": "Oranges",
}

# What each swept value means in plain language, per axis — this is what
# ends up in the legend now that there is no separate plot title. -1/0/+1
# ordering matches STYLE_VALUES below.
AXIS_VALUE_LABELS = {
    "prox":  {-1.0: "Close Proximity Tolerance", 0.0: "Neutral",
             1.0: "Clearance Preference"},
    "pass":  {-1.0: "Left-Side Passing Convention", 0.0: "Neutral",
             1.0: "Right-Side Passing Convention"},
    "yield": {-1.0: "Assertive (Low Yielding)", 0.0: "Neutral",
             1.0: "Early Deference (High Yielding)"},
    "group": {-1.0: "Group-Agnostic", 0.0: "Neutral",
             1.0: "Group-Preserving"},
}

STYLE_VALUES = [-1.0, 0.0, 1.0]


def _points_per_data_unit(ax, fig):
    """Font points per 1 data-unit (meter) for the axes' CURRENT limits, so
    text/marker sizes can be specified in meters and still look right
    regardless of how zoomed-in the scene's tight bounding box is."""
    p0 = ax.transData.transform((0.0, 0.0))
    p1 = ax.transData.transform((1.0, 0.0))
    px_per_unit = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    return px_per_unit * 72.0 / fig.dpi


def _style_color(cmap_name, value):
    """value=-1 -> darkest, value=0 -> medium, value=+1 -> lightest, all
    drawn from the same sequential colormap so a legend reader immediately
    sees "one family of color, three shades". The band is kept away from
    both ends of the colormap: too close to 0 and the "lightest" shade is
    nearly invisible against a white background; too close to 1 and the
    "darkest" shade loses the family's hue entirely."""
    cmap = plt.get_cmap(cmap_name)
    frac = (1.0 - value) / 2.0          # -1 -> 1.0 (dark), 0 -> 0.5, +1 -> 0.0 (light)
    frac = 0.35 + frac * 0.60            # lightest=0.35, medium=0.65, darkest=0.95
    return cmap(frac)


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def build_env_and_policy(args):
    import torch
    import gym
    from crowd_nav.policy.policy_factory import policy_factory
    from crowd_sim.envs.utils.robot import Robot

    device = torch.device('cuda:0' if torch.cuda.is_available() and args.gpu else 'cpu')
    logging.info('Device: %s', device)

    policy = policy_factory[args.policy]()
    policy_config = configparser.RawConfigParser()
    policy_config.read(args.policy_config)
    policy.configure(policy_config)

    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    sim = env.unwrapped
    sim.test_sim = 'npz_hard'   # required so reset() honors the pre-loaded npz scenario

    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot)

    policy.set_phase(args.phase)
    policy.set_device(device)
    policy.set_env(env)
    robot.print_info()

    return env, robot, policy, sim


def collect_sweep(args):
    """Run stochastic rollouts for each of {-1, 0, +1} on the swept axis,
    holding the other 3 axes at 0, KEEPING ONLY rollouts that reach the
    goal (ReachGoal) — collisions/timeouts are discarded and re-rolled so
    every saved/plotted trajectory in the figure is a successful run.
    Returns:
      trajs_by_value: {value: [ (T,2) array, ... n_per_value arrays ]}
      scene:          dict of static scene info for plotting
    """
    from evaluate_styles import run_one_episode
    from crowd_sim.envs.utils.info import ReachGoal
    from crowd_nav.policy.sogudiff import STYLE_AXES as _POLICY_STYLE_AXES
    assert list(_POLICY_STYLE_AXES) == STYLE_AXES, (
        f'STYLE_AXES drift: sogudiff.py has {_POLICY_STYLE_AXES}, '
        f'this script assumes {STYLE_AXES}')

    env, robot, policy, sim = build_env_and_policy(args)
    axis_idx = STYLE_AXES.index(args.axis)
    video_cmap_name = AXIS_CMAPS[args.axis]

    # Anchor the RNG once so every rollout below (which explicitly reloads
    # the same npz file every time via run_one_episode(npz_path=...)) sees
    # an identical scene — only the policy's sampled noise differs.
    scene_testoffset = sim.testoffset
    scene_counter = sim.case_counter[args.phase]

    trajs_by_value = {v: [] for v in STYLE_VALUES}
    first_ep = None
    max_attempts = max(args.max_attempts_per_value, args.n_per_value * 10)

    for v in STYLE_VALUES:
        style_vec = [0.0, 0.0, 0.0, 0.0]
        style_vec[axis_idx] = v
        n_success = 0
        n_attempts = 0
        while n_success < args.n_per_value and n_attempts < max_attempts:
            n_attempts += 1
            ep = run_one_episode(
                env, robot, policy,
                style_vec=style_vec,
                use_proj=not args.no_projection,
                phase=args.phase,
                scene_idx=0,
                npz_path=args.npz_path,
                scene_testoffset=scene_testoffset,
                scene_counter=scene_counter,
            )
            if not isinstance(ep['info'], ReachGoal):
                outcome = type(ep['info']).__name__
                logging.info('axis=%s  value=%+.0f  attempt %d: did NOT reach '
                            'goal (%s) — discarding, retrying',
                            args.axis, v, n_attempts, outcome)
                if args.save_run_videos or args.save_failure_videos:
                    video_path = os.path.join(
                        args.tmp_video_dir,
                        f'{args.axis}_{v:+.0f}_attempt{n_attempts:02d}_FAILED_{outcome}.mp4')
                    render_run_video(ep, video_path, fps=args.run_video_fps,
                                     robot_color=_style_color(video_cmap_name, v),
                                     title=f'{args.axis} = {v:+.0f}  (FAILED: {outcome})')
                continue
            n_success += 1
            logging.info('axis=%s  value=%+.0f  kept success %d/%d  (attempt %d)',
                         args.axis, v, n_success, args.n_per_value, n_attempts)
            traj = np.array([s[0].position for s in ep['states']], dtype=np.float32)
            trajs_by_value[v].append(traj)
            if args.save_run_videos:
                video_path = os.path.join(
                    args.tmp_video_dir, f'{args.axis}_{v:+.0f}_run{n_success:02d}.mp4')
                render_run_video(ep, video_path, fps=args.run_video_fps,
                                 robot_color=_style_color(video_cmap_name, v),
                                 title=f'{args.axis} = {v:+.0f}')
            if first_ep is None:
                first_ep = ep

        if n_success < args.n_per_value:
            logging.warning('axis=%s  value=%+.0f: only %d/%d successful rollouts '
                            'after %d attempts (max_attempts_per_value=%d)',
                            args.axis, v, n_success, args.n_per_value,
                            n_attempts, max_attempts)

    s0 = first_ep['states'][0]
    h0 = s0[1]
    n_humans = first_ep['n_humans']
    scene = dict(
        axis=args.axis,
        goal=np.asarray(first_ep['goal_pos'], dtype=np.float32),
        has_map=float(first_ep['has_map']),
        map_extent=float(first_ep['map_extent']),
        occupancy_map=(np.asarray(first_ep['occ_map'], dtype=np.float32)
                       if first_ep['occ_map'] is not None else np.zeros((1, 1), dtype=np.float32)),
        human_positions0=np.array([h0[j].position for j in range(n_humans)], dtype=np.float32),
        human_velocities0=np.array([[h0[j].vx, h0[j].vy] for j in range(n_humans)], dtype=np.float32),
        human_radii=np.array(first_ep['human_radii'], dtype=np.float32),
        robot_radius=float(first_ep['robot_radius']),
        robot_theta0=float(getattr(s0[0], 'theta', 0.0)),
        robot_kinematics=str(first_ep['robot_kinematics']),
        n_per_value=args.n_per_value,
        style_values=np.array(STYLE_VALUES, dtype=np.float32),
    )
    return trajs_by_value, scene


# ---------------------------------------------------------------------------
# Save / load the raw trajectory distribution
# ---------------------------------------------------------------------------

def save_sweep(trajs_by_value, scene, out_path):
    save_dict = dict(scene)
    for v in STYLE_VALUES:
        save_dict[f'trajectories_{v:+.0f}'] = np.array(trajs_by_value[v], dtype=object)
    np.savez(out_path, **save_dict)
    logging.info('Saved trajectory distribution: %s', out_path)


def load_sweep(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    scene = {k: d[k] for k in d.files if not k.startswith('trajectories_')}
    trajs_by_value = {
        v: list(d[f'trajectories_{v:+.0f}']) for v in STYLE_VALUES
    }
    return trajs_by_value, scene


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _snap_final_point_to_goal(traj, goal, close_enough=0.5):
    """Append `goal` as one extra final point IF the trajectory's last
    recorded position is already within `close_enough` meters of it — the
    episode terminates as soon as the robot enters the goal-reached radius,
    so every successful rollout is cut one control step short of actually
    touching the goal, which reads as a visible gap in the plotted line.
    Trajectories that end far from the goal (collision/timeout) are left
    untouched rather than fabricating a connection to the goal marker."""
    if len(traj) == 0:
        return traj
    last = traj[-1]
    if np.hypot(goal[0] - last[0], goal[1] - last[1]) <= close_enough:
        return np.vstack([traj, np.asarray(goal, dtype=traj.dtype)])
    return traj


def plot_sweep(trajs_by_value, scene, out_path, fig_size=8.0, dpi=300,
               margin=0.8, legend_loc='upper right', snap_to_goal=True):
    axis = str(scene['axis'])
    cmap_name = AXIS_CMAPS[axis]
    friendly = AXIS_TITLES[axis]

    goal = np.asarray(scene['goal'], dtype=float)
    has_map = float(scene['has_map']) > 0.5
    occ_map = scene['occupancy_map'] if has_map else None
    map_extent = float(scene['map_extent'])
    human_pos0 = np.asarray(scene['human_positions0'], dtype=float).reshape(-1, 2)
    human_vel0 = np.asarray(scene['human_velocities0'], dtype=float).reshape(-1, 2)
    human_radii = np.asarray(scene['human_radii'], dtype=float).reshape(-1)
    robot_radius = float(scene['robot_radius'])
    robot_theta0 = float(scene.get('robot_theta0', 0.0))
    robot_kinematics = str(scene.get('robot_kinematics', 'unicycle'))

    fig = plt.figure(figsize=(fig_size, fig_size))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')

    # ---- static map underlay ----
    if occ_map is not None:
        half = map_extent / 2.0
        ax.imshow(occ_map, extent=[-half, half, -half, half], origin='lower',
                  cmap='gray_r', alpha=0.45, zorder=0, interpolation='nearest')

    # ---- tight square bounds: robot start, goal, humans, trajectories,
    #      and (only) occupied wall cells (not the whole empty map extent) ----
    all_x = [0.0, float(goal[0])]
    all_y = [0.0, float(goal[1])]
    for px, py in human_pos0:
        all_x.append(float(px)); all_y.append(float(py))
    for v in STYLE_VALUES:
        for traj in trajs_by_value[v]:
            all_x.extend(traj[:, 0].tolist())
            all_y.extend(traj[:, 1].tolist())
    if occ_map is not None:
        H, W = occ_map.shape
        res = map_extent / W
        rows, cols = np.where(occ_map > 0.5)
        if len(rows):
            wx = (cols - W / 2 + 0.5) * res
            wy = (rows - H / 2 + 0.5) * res
            all_x.extend(wx.tolist()); all_y.extend(wy.tolist())

    xmin, xmax = min(all_x) - margin, max(all_x) + margin
    ymin, ymax = min(all_y) - margin, max(all_y) + margin
    xr, yr = xmax - xmin, ymax - ymin
    r = max(xr, yr)
    xc, yc = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    ax.set_xlim(xc - r / 2.0, xc + r / 2.0)
    ax.set_ylim(yc - r / 2.0, yc + r / 2.0)

    # ---- trajectories, drawn before markers so markers sit on top ----
    for v in STYLE_VALUES:
        color = _style_color(cmap_name, v)
        for traj in trajs_by_value[v]:
            if snap_to_goal:
                traj = _snap_final_point_to_goal(traj, goal)
            ax.plot(traj[:, 0], traj[:, 1], '-', color=color,
                    linewidth=1.6, alpha=0.85, zorder=2)

    # ---- pedestrians: outlined circle (STYLES-eval hsv cycle, unfilled —
    #      exactly as in build_scene_video), a large centered index number,
    #      and a thick heading arrow flush against the circle's edge ----
    cmap_human = plt.cm.get_cmap('hsv', 10)
    ped_arrow_style = patches.ArrowStyle('->', head_length=5, head_width=3)
    ppu = _points_per_data_unit(ax, fig)
    for j in range(human_pos0.shape[0]):
        px, py = human_pos0[j]
        vx, vy = human_vel0[j]
        rad = human_radii[j] if j < len(human_radii) else 0.3
        h_color = cmap_human(j % 10)
        circ = plt.Circle((px, py), rad, fill=False, edgecolor=h_color,
                          linewidth=2.2, zorder=5)
        ax.add_artist(circ)

        # Number sized to fill most of the circle's interior, regardless of
        # scene scale, with a white halo so it stays legible over both the
        # gray-scale map and colored trajectory lines running through it.
        num_fontsize = np.clip(rad * ppu * 1.55, 7, 24)
        txt = ax.text(px, py, str(j), color=h_color, fontsize=num_fontsize,
                      fontweight='bold', ha='center', va='center', zorder=6)
        txt.set_path_effects([patheffects.withStroke(linewidth=num_fontsize * 0.18,
                                                       foreground='white')])

        speed = float(np.hypot(vx, vy))
        if speed > 0.05:
            ux, uy = vx / speed, vy / speed
            arrow_len = 0.55
            start = (px + rad * ux, py + rad * uy)
            end = (start[0] + arrow_len * ux, start[1] + arrow_len * uy)
            arrow = patches.FancyArrowPatch(start, end,
                                            color=h_color, linewidth=2.6,
                                            arrowstyle=ped_arrow_style,
                                            shrinkA=0, shrinkB=0, zorder=6)
            ax.add_patch(arrow)

    # ---- robot: colored circle at the start position + heading arrow
    #      (same convention as the STYLES test-eval framework) ----
    robot_color = 'gold'
    robot_circle = plt.Circle((0.0, 0.0), robot_radius, fill=True,
                              facecolor=robot_color, edgecolor='black',
                              linewidth=2.2, zorder=7)
    ax.add_artist(robot_circle)
    if robot_kinematics == 'unicycle':
        robot_arrow_style = patches.ArrowStyle('->', head_length=4, head_width=2)
        ax.add_artist(patches.FancyArrowPatch(
            (0.0, 0.0),
            (robot_radius * np.cos(robot_theta0), robot_radius * np.sin(robot_theta0)),
            color='red', arrowstyle=robot_arrow_style, linewidth=1.6, zorder=8))

    # ---- goal marker (markersize=11 matches build_scene_video's goal_dot
    #      in evaluate_styles.py) ----
    ax.plot(goal[0], goal[1], marker='*', color='red', markersize=14, zorder=7)

    # ---- axis labels (no title — the swept axis is now explained in the
    #      legend, per-value, below) ----
    ax.set_xlabel('x (m)', fontsize=14)
    ax.set_ylabel('y (m)', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)

    # ---- legend ----
    legend_handles = [
        mlines.Line2D([], [], color='red', marker='*', linestyle='None',
                      markersize=12, label='Goal'),
        mlines.Line2D([], [], marker='o', linestyle='None',
                      markerfacecolor=robot_color, markeredgecolor='black',
                      markeredgewidth=2.2, markersize=8, label='Robot'),
        mlines.Line2D([], [], marker='o', linestyle='None',
                      markerfacecolor='none', markeredgecolor='gray',
                      markeredgewidth=2, markersize=8, label='Pedestrian'),
    ]
    value_labels = AXIS_VALUE_LABELS[axis]
    for v in STYLE_VALUES:
        color = _style_color(cmap_name, v)
        desc = value_labels[v]
        legend_handles.append(
            mlines.Line2D([], [], color=color, lw=3,
                          label=f'{v:+.0f}: {desc}'))
    ax.legend(handles=legend_handles, loc=legend_loc, fontsize=12,
             framealpha=0.9, title=friendly, title_fontsize=14)

    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    logging.info('Saved figure: %s', out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Sweep one style axis over {-1,0,+1} on a single scene '
                    'and plot the resulting trajectory distributions for a paper figure.')
    parser.add_argument('--env_config',    type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--policy',        type=str, default='sogudiff')
    parser.add_argument('--gpu',           action='store_true', default=False)
    parser.add_argument('--phase',         type=str, default='test')

    parser.add_argument('--npz_path',      type=str,
                        help='Path to the single scene .npz file to sweep (required unless --replot)')
    parser.add_argument('--axis',          type=str, choices=STYLE_AXES,
                        help='Which style axis to sweep: prox | pass | yield | group '
                             '(required unless --replot)')
    parser.add_argument('--n_per_value',   type=int, default=10,
                        help='Number of SUCCESSFUL (goal-reaching) rollouts to keep per '
                             'style value (10 -> 30 total). Collisions/timeouts are '
                             'discarded and re-rolled, not saved.')
    parser.add_argument('--max_attempts_per_value', type=int, default=100,
                        help='Safety cap on total rollout attempts per style value '
                             '(successes + discarded failures) before giving up early')
    parser.add_argument('--no_projection', action='store_true', default=False,
                        help='Disable the feasibility-projection layer (default: projection ON)')

    parser.add_argument('--out_dir',       type=str, default='results_style_sweep_figs')
    parser.add_argument('--out_file',      type=str, default=None,
                        help='Basename (no extension) for the .png/.npz outputs; '
                             'defaults to "<axis>_sweep_<scene_basename>"')
    parser.add_argument('--fig_size',      type=float, default=8.0, help='Square figure side, inches')
    parser.add_argument('--dpi',           type=int, default=300)
    parser.add_argument('--margin',        type=float, default=0.8,
                        help='Meters of padding around the tight scene bounding box')
    parser.add_argument('--legend_loc',    type=str, default='upper right')
    parser.add_argument('--no_snap_to_goal', action='store_true', default=False,
                        help='Do not append the goal as a final trajectory point '
                             '(by default, successful rollouts that end within 0.5m '
                             'of the goal are snapped to it, since the episode '
                             'terminates one control step before actually touching it)')

    parser.add_argument('--save_run_videos', action='store_true', default=False,
                        help='Save an MP4 of each individual SUCCESSFUL rollout as it '
                             'completes, so you can watch how each run went while '
                             'iterating. Off by default (adds render time per rollout).')
    parser.add_argument('--save_failure_videos', action='store_true', default=False,
                        help='Save an MP4 of each discarded FAILED rollout (collision/'
                             'timeout) too — these are otherwise silently dropped and '
                             're-rolled, so this is the way to see what went wrong.')
    parser.add_argument('--run_video_fps', type=int, default=10)
    parser.add_argument('--tmp_video_dir', type=str, default=None,
                        help='Where per-run videos go (default: "<out_dir>/tmp_run_videos")')

    parser.add_argument('--replot',        type=str, default=None,
                        help='Re-render the figure ONLY from a previously saved '
                             '<out_file>.npz (no policy/env/GPU needed)')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    os.makedirs(args.out_dir, exist_ok=True)
    if args.tmp_video_dir is None:
        args.tmp_video_dir = os.path.join(args.out_dir, 'tmp_run_videos')
    if args.save_run_videos or args.save_failure_videos:
        os.makedirs(args.tmp_video_dir, exist_ok=True)

    if args.replot:
        trajs_by_value, scene = load_sweep(args.replot)
        base = os.path.splitext(os.path.basename(args.replot))[0]
        png_path = os.path.join(args.out_dir, base + '.png')
        plot_sweep(trajs_by_value, scene, png_path, fig_size=args.fig_size,
                  dpi=args.dpi, margin=args.margin, legend_loc=args.legend_loc,
                  snap_to_goal=not args.no_snap_to_goal)
        return

    if not args.npz_path or not args.axis:
        parser.error('--npz_path and --axis are required unless --replot is given')

    scene_base = os.path.splitext(os.path.basename(args.npz_path))[0]
    out_base = args.out_file or f'{args.axis}_sweep_{scene_base}'
    npz_out_path = os.path.join(args.out_dir, out_base + '.npz')
    png_out_path = os.path.join(args.out_dir, out_base + '.png')

    trajs_by_value, scene = collect_sweep(args)
    save_sweep(trajs_by_value, scene, npz_out_path)
    plot_sweep(trajs_by_value, scene, png_out_path, fig_size=args.fig_size,
              dpi=args.dpi, margin=args.margin, legend_loc=args.legend_loc,
              snap_to_goal=not args.no_snap_to_goal)


if __name__ == '__main__':
    main()
