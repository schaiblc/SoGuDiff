#!/usr/bin/env python3
"""
style_composition_step_figure.py
========================================
Paper-figure tool: on a SINGLE scene, at a SINGLE instant of time (the
initial planning step, t=0 — no rollout), visualize the diffusion policy's
raw PREDICTION for each of several style-vector COMPOSITIONS:

  - up to N_CANDIDATES raw diffusion samples (the policy's full candidate
    pool before scoring/selection) — thin, dotted, lightest shade
  - the SELECTED sample (highest-scored candidate, before projection)     — thicker, solid, medium shade
  - the PROJECTED sample (after the feasibility-projection layer, if      — thickest, solid, darkest
    enabled and it succeeded)                                              shade

Each line is a single FLAT color (no gradient along its own length) — the
"lighter -> darker" progression happens ACROSS the three tiers, not within
any one trajectory.

Each composition gets its own color family (sequential colormap), all
overlaid on one static scene (map / pedestrians / goal / robot), so the
figure shows how differently-composed styles shift the policy's PLANNING
distribution at a single glance — no simulation, no goal-reaching, no
stochastic rollout ensemble like style_composition_figure.py.

Per-episode state is reset through `_reset_policy_state` from
evaluate_styles.py, so each composition starts from the same cold state the
evaluation uses.

Why this needs only ONE robot.act(ob) call per composition
-------------------------------------------------------------
robot.act(ob) -> policy.predict(state) already produces, in ONE call:
  policy.predicted_traj = {
      'all_samples':     list of K (T,2) WORLD-frame diffusion candidates
                         (index 0 == selected_sample),
      'selected_sample': (T,2) WORLD-frame, highest-scored candidate,
      'projection':      (T,2) WORLD-frame feasible correction, or None,
      'used_projection': bool,
  }
No env.step() / rollout is needed — sim.predicted_trajs (the per-episode
list) is only populated inside step(), but the single most-recent-step
dict lives on the policy object right after act() returns.

Compositions are defined at RUN TIME via repeated --composition flags,
exactly like style_composition_figure.py:
  --composition "Cautious & Yielding:1,0,1,0"
Each is "Label:prox,pass,yield,group". If none given, DEFAULT_COMPOSITIONS
below is used.

What gets saved
----------------
  <out_dir>/<out_file>.png   the paper figure
  <out_dir>/<out_file>.npz   the raw per-composition candidates/selected/
                              projection + static scene info, so the figure
                              can be re-rendered without re-running the
                              policy (--replot).

Usage
-----
  python style_composition_step_figure.py \\
      --policy_config configs/policy.config \\
      --env_config    configs/env.config \\
      --gpu \\
      --npz_path      /path/to/one_scene.npz \\
      --composition "Neutral:0,0,0,0" \\
      --composition "Cautious & Yielding:1,0,1,0" \\
      --composition "Assertive & Efficient:-1,0,-1,0" \\
      --composition "Social Passing:0,1,0,1" \\
      --n_candidates  10 \\
      --out_dir       results_style_prediction_figs \\
      --out_file      prediction_C1

Re-plot only (no GPU / policy needed):
  python style_composition_step_figure.py --replot results_style_prediction_figs/prediction_C1.npz
"""

# These figure scripts live one directory below the evaluation code they share
# rollout and metric functions with, so make crowd_nav/ importable regardless of
# where the script is launched from.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import configparser
import json
import logging
import os
import struct
import zlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import matplotlib.patheffects as patheffects

STYLE_AXES = ["prox", "pass", "yield", "group"]

DEFAULT_COMPOSITIONS = [
    ("Neutral",                [0.0, 0.0, 0.0, 0.0]),
    ("Cautious & Yielding",    [1.0, 0.0, 1.0, 0.0]),
    ("Assertive & Efficient",  [-1.0, 0.0, -1.0, 0.0]),
    ("Social Passing",        [0.0, 1.0, 0.0, 1.0]),
]

# One sequential colormap ("color family") per composition, cycled by index.
COMPOSITION_CMAPS = ['Blues', 'Oranges', 'Greens', 'Purples', 'Reds', 'Greys']

# Each tier is a single FLAT shade (no gradient within one line) — the
# "gradient" is only ACROSS tiers: lightest (candidates) -> medium
# (selected) -> darkest (projected), all from the same per-composition
# colormap family.
CANDIDATE_FRAC = 0.30     # lightest shade: raw diffusion candidates
SELECTED_FRAC = 0.60      # medium shade, solid: the scored/selected sample
PROJECTED_FRAC = 0.88     # darkest shade: the projected sample


def _sanitize_label(label):
    return ''.join(c if c.isalnum() else '_' for c in label).strip('_') or 'composition'


def parse_composition_arg(s):
    if ':' not in s:
        raise argparse.ArgumentTypeError(
            f'--composition must be "Label:prox,pass,yield,group", got: {s!r}')
    label, vec_str = s.split(':', 1)
    label = label.strip()
    parts = [p.strip() for p in vec_str.split(',')]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f'--composition vector must have exactly 4 comma-separated values '
            f'(prox,pass,yield,group), got {len(parts)}: {s!r}')
    try:
        vec = [float(p) for p in parts]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f'--composition vector must be numeric: {s!r}') from e
    return label, vec


def _points_per_data_unit(ax, fig):
    p0 = ax.transData.transform((0.0, 0.0))
    p1 = ax.transData.transform((1.0, 0.0))
    px_per_unit = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    return px_per_unit * 72.0 / fig.dpi


# =============================================================================
# Optional WIDE real-map background (same feature/rationale as in
# style_composition_figure.py — see that file for the full
# explanation). Duplicated here since these are standalone scripts.
# =============================================================================
def _read_png_gray8_builtin(path):
    with open(path, 'rb') as f:
        data = f.read()
    assert data[:8] == b'\x89PNG\r\n\x1a\n', 'not a PNG'
    pos = 8
    width = height = bitdepth = colortype = interlace = None
    idat = bytearray()
    while pos < len(data):
        length = struct.unpack('>I', data[pos:pos + 4])[0]
        ctype = data[pos + 4:pos + 8].decode('ascii')
        chunk = data[pos + 8:pos + 8 + length]
        pos += 8 + length + 4
        if ctype == 'IHDR':
            width, height, bitdepth, colortype, comp, filt, interlace = \
                struct.unpack('>IIBBBBB', chunk)
        elif ctype == 'IDAT':
            idat += chunk
        elif ctype == 'IEND':
            break
    assert bitdepth == 8 and colortype == 0 and interlace == 0, \
        'only 8-bit grayscale non-interlaced PNGs are supported by the built-in reader'
    raw = zlib.decompress(bytes(idat))
    stride = width
    img = np.zeros((height, width), dtype=np.int16)
    prev = np.zeros(stride, dtype=np.int16)
    off = 0
    for r in range(height):
        filt_type = raw[off]; off += 1
        line = np.frombuffer(raw[off:off + stride], dtype=np.uint8).astype(np.int16)
        off += stride
        if filt_type == 1:
            for i in range(1, stride):
                line[i] = (line[i] + line[i - 1]) % 256
        elif filt_type == 2:
            line = (line + prev) % 256
        elif filt_type == 3:
            for i in range(stride):
                a = line[i - 1] if i >= 1 else 0
                b = prev[i]
                line[i] = (line[i] + (a + b) // 2) % 256
        elif filt_type == 4:
            for i in range(stride):
                a = int(line[i - 1]) if i >= 1 else 0
                b = int(prev[i])
                c = int(prev[i - 1]) if i >= 1 else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                line[i] = (line[i] + pr) % 256
        img[r] = line
        prev = line
    return img.astype(np.uint8)


def _read_occupancy_png(path):
    try:
        from PIL import Image
        return np.array(Image.open(path).convert('L'), dtype=np.uint8)
    except ImportError:
        return _read_png_gray8_builtin(path)


def _resample_wide_background(source_png, robot_x, robot_y, heading_deg,
                              xlim, ylim, res=0.2, subsamples=3):
    meta_path = source_png.replace('_occupancy.png', '_metadata.json')
    meta = json.load(open(meta_path))
    mpp_x, mpp_y = meta['metres_per_pixel']['x'], meta['metres_per_pixel']['y']
    wmin_x, wmin_y = meta['min'][0], meta['min'][1]
    free_val = meta['convention']['free']
    raw_gray = _read_occupancy_png(source_png)
    H_raw, W_raw = raw_gray.shape
    raw_free = (raw_gray == free_val)

    theta = np.deg2rad(heading_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    nx = int(np.ceil((xlim[1] - xlim[0]) / res))
    ny = int(np.ceil((ylim[1] - ylim[0]) / res))
    xs = xlim[0] + (np.arange(nx) + 0.5) * res
    ys = ylim[0] + (np.arange(ny) + 0.5) * res
    half_cell = res / 2.0
    offsets = np.linspace(-half_cell, half_cell, subsamples)

    occ = np.zeros((ny, nx), dtype=np.float32)
    for r, ly in enumerate(ys):
        for c, lx in enumerate(xs):
            obstacle = False
            for dly in offsets:
                for dlx in offsets:
                    slx, sly = lx + dlx, ly + dly
                    wx = robot_x + cos_t * slx - sin_t * sly
                    wy = robot_y + sin_t * slx + cos_t * sly
                    col = (wx - wmin_x) / mpp_x
                    row = (H_raw - 1) - (wy - wmin_y) / mpp_y
                    ci, ri = int(round(col)), int(round(row))
                    if ci < 0 or ri < 0 or ci >= W_raw or ri >= H_raw:
                        obstacle = True
                        break
                    if not raw_free[ri, ci]:
                        obstacle = True
                        break
                if obstacle:
                    break
            occ[r, c] = 1.0 if obstacle else 0.0
    return occ, (xlim[0], xlim[1], ylim[0], ylim[1])


# ---------------------------------------------------------------------------
# Single-step prediction collection
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
    sim.test_sim = 'npz_hard'

    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot)

    policy.set_phase(args.phase)
    policy.set_device(device)
    policy.set_env(env)
    robot.print_info()

    return env, robot, policy, sim


def collect_predictions(args, compositions):
    """One robot.act(ob) call per composition (t=0, no rollout). Returns:
      preds_by_label: {label: {'candidates': (n,T,2), 'selected': (T,2),
                               'projection': (T,2) or None}}
      scene:          dict of static scene info for plotting
    """
    from evaluate_styles import _reset_policy_state
    from crowd_nav.policy.sogudiff import STYLE_AXES as _POLICY_STYLE_AXES
    assert list(_POLICY_STYLE_AXES) == STYLE_AXES, (
        f'STYLE_AXES drift: sogudiff.py has {_POLICY_STYLE_AXES}, '
        f'this script assumes {STYLE_AXES}')

    env, robot, policy, sim = build_env_and_policy(args)
    rng = np.random.default_rng(args.seed)

    scene_testoffset = sim.testoffset
    scene_counter = sim.case_counter[args.phase]

    preds_by_label = {}
    scene = None

    for label, style_vec in compositions:
        policy.style_vector = np.array(style_vec, dtype=np.float32)
        policy.use_projection = not args.no_projection
        _reset_policy_state(policy)

        sim.testoffset = scene_testoffset
        sim.case_counter[args.phase] = scene_counter
        env.load_npz_scenario(args.npz_path)
        ob = env.reset(phase=args.phase, test_case=0)

        _ = robot.act(ob)
        pred = policy.predicted_traj
        if pred is None:
            raise RuntimeError(f'composition {label!r}: policy.predicted_traj is None '
                              f'after act() — policy did not populate a prediction')

        all_samples = pred['all_samples']          # list of K (T,2), index 0 == selected
        selected = np.asarray(pred['selected_sample'], dtype=np.float32)
        projection = (np.asarray(pred['projection'], dtype=np.float32)
                     if pred.get('projection') is not None else None)

        pool = all_samples[1:] if len(all_samples) > 1 else []
        n_take = min(args.n_candidates, len(pool))
        if n_take < len(pool):
            idx = rng.choice(len(pool), size=n_take, replace=False)
        else:
            idx = np.arange(len(pool))
        candidates = np.asarray([pool[i] for i in idx], dtype=np.float32) \
            if n_take > 0 else np.zeros((0,) + selected.shape, dtype=np.float32)

        logging.info('composition=%r: %d candidates (of %d available), selected %s, '
                    'projection %s', label, n_take, len(pool),
                    selected.shape, 'available' if projection is not None else 'none')

        preds_by_label[label] = dict(candidates=candidates, selected=selected,
                                     projection=projection)

        if scene is None:
            n_humans = len(sim.humans)
            scene = dict(
                composition_labels=np.array([l for l, _ in compositions]),
                composition_vectors=np.array([v for _, v in compositions], dtype=np.float32),
                goal=np.asarray(robot.get_goal_position(), dtype=np.float32),
                has_map=float(getattr(sim, '_npz_has_map', 0.0)),
                map_extent=float(getattr(sim, '_npz_map_extent', 10.0)),
                occupancy_map=(np.asarray(sim._npz_occ_map, dtype=np.float32)
                              if getattr(sim, '_npz_occ_map', None) is not None
                              else np.zeros((1, 1), dtype=np.float32)),
                human_positions0=np.array([[h.px, h.py] for h in sim.humans], dtype=np.float32),
                human_velocities0=np.array([[h.vx, h.vy] for h in sim.humans], dtype=np.float32),
                human_radii=np.array([h.radius for h in sim.humans], dtype=np.float32),
                robot_radius=float(robot.radius),
                robot_pos0=np.array([robot.px, robot.py], dtype=np.float32),
                robot_theta0=float(robot.theta),
                robot_kinematics=str(robot.kinematics),
            )

            try:
                raw_npz = np.load(args.npz_path, allow_pickle=True)
                if all(k in raw_npz.files for k in ('source_png', 'robot_world_xy', 'robot_heading_deg')):
                    source_png = str(raw_npz['source_png'])
                    source_png_dir = os.path.dirname(os.path.abspath(args.npz_path))
                    candidates_paths = [
                        os.path.join(source_png_dir, source_png),
                        os.path.join('data/occupancy_maps/interiorgs',
                                    source_png),
                    ]
                    resolved = next((c for c in candidates_paths if os.path.isfile(c)), None)
                    if resolved is not None:
                        scene['source_png'] = np.array(resolved)
                        scene['robot_world_xy'] = np.asarray(raw_npz['robot_world_xy'], dtype=np.float32)
                        scene['robot_heading_deg'] = np.float32(raw_npz['robot_heading_deg'])
                        logging.info('Found map provenance -> wide background enabled (%s)', resolved)
            except Exception as e:
                logging.warning('Could not check for wide-map provenance: %s', e)

    return preds_by_label, scene


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_predictions(preds_by_label, scene, out_path):
    save_dict = dict(scene)
    labels = list(scene['composition_labels'])
    for i, label in enumerate(labels):
        p = preds_by_label[label]
        save_dict[f'candidates_{i}'] = p['candidates']
        save_dict[f'selected_{i}'] = p['selected']
        save_dict[f'projection_{i}'] = p['projection'] if p['projection'] is not None \
            else np.zeros((0, 2), dtype=np.float32)
        save_dict[f'has_projection_{i}'] = np.bool_(p['projection'] is not None)
    np.savez(out_path, **save_dict)
    logging.info('Saved prediction data: %s', out_path)


def load_predictions(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    scene = {k: d[k] for k in d.files
            if not (k.startswith('candidates_') or k.startswith('selected_')
                    or k.startswith('projection_') or k.startswith('has_projection_'))}
    labels = list(scene['composition_labels'])
    preds_by_label = {}
    for i, label in enumerate(labels):
        has_proj = bool(d[f'has_projection_{i}'])
        preds_by_label[label] = dict(
            candidates=d[f'candidates_{i}'],
            selected=d[f'selected_{i}'],
            projection=d[f'projection_{i}'] if has_proj else None,
        )
    return preds_by_label, scene


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_predictions(preds_by_label, scene, out_path, fig_size=8.0, dpi=300,
                    margin=0.8, legend_loc='upper right'):
    labels = list(scene['composition_labels'])
    vectors = np.asarray(scene['composition_vectors'], dtype=float).reshape(len(labels), 4)

    goal = np.asarray(scene['goal'], dtype=float)
    has_map = float(scene['has_map']) > 0.5
    occ_map = scene['occupancy_map'] if has_map else None
    map_extent = float(scene['map_extent'])
    human_pos0 = np.asarray(scene['human_positions0'], dtype=float).reshape(-1, 2)
    human_vel0 = np.asarray(scene['human_velocities0'], dtype=float).reshape(-1, 2)
    human_radii = np.asarray(scene['human_radii'], dtype=float).reshape(-1)
    robot_radius = float(scene['robot_radius'])
    robot_pos0 = np.asarray(scene.get('robot_pos0', [0.0, 0.0]), dtype=float)
    robot_theta0 = float(scene.get('robot_theta0', 0.0))
    robot_kinematics = str(scene.get('robot_kinematics', 'unicycle'))

    fig = plt.figure(figsize=(fig_size, fig_size))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')

    # ---- bounds: robot, goal, humans, ALL predicted trajectories, and (if a
    #      static map is present) the full map extent ----
    all_x = [float(robot_pos0[0]), float(goal[0])]
    all_y = [float(robot_pos0[1]), float(goal[1])]
    for px, py in human_pos0:
        all_x.append(float(px)); all_y.append(float(py))
    for label in labels:
        p = preds_by_label[label]
        arrs = [p['candidates'], p['selected']]
        if p['projection'] is not None:
            arrs.append(p['projection'])
        for arr in arrs:
            arr = np.asarray(arr)
            if arr.size:
                all_x.extend(arr[..., 0].reshape(-1).tolist())
                all_y.extend(arr[..., 1].reshape(-1).tolist())
    if occ_map is not None:
        half = map_extent / 2.0
        all_x.extend([-half, half])
        all_y.extend([-half, half])

    xmin, xmax = min(all_x) - margin, max(all_x) + margin
    ymin, ymax = min(all_y) - margin, max(all_y) + margin
    r = max(xmax - xmin, ymax - ymin)
    xc, yc = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    xlim = (xc - r / 2.0, xc + r / 2.0)
    ylim = (yc - r / 2.0, yc + r / 2.0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # ---- map underlay (authoritative 10m x 10m + optional wide real-map
    #      background, same convention as style_composition_figure.py) ----
    has_wide_bg = all(k in scene for k in ('source_png', 'robot_world_xy', 'robot_heading_deg'))
    if has_wide_bg:
        try:
            wide_occ, wide_extent = _resample_wide_background(
                str(scene['source_png']),
                float(np.asarray(scene['robot_world_xy']).reshape(-1)[0]),
                float(np.asarray(scene['robot_world_xy']).reshape(-1)[1]),
                float(scene['robot_heading_deg']),
                xlim, ylim)
            ax.imshow(wide_occ, extent=wide_extent, origin='lower',
                      cmap='gray_r', alpha=0.35, zorder=-1, interpolation='nearest')
        except Exception as e:
            logging.warning('Could not build wide background: %s', e)
            has_wide_bg = False
    if occ_map is not None:
        half = map_extent / 2.0
        ax.imshow(occ_map, extent=[-half, half, -half, half], origin='lower',
                  cmap='gray_r', alpha=0.45, zorder=0, interpolation='nearest')

    # ---- per-composition prediction triads ----
    for i, label in enumerate(labels):
        cmap_name = COMPOSITION_CMAPS[i % len(COMPOSITION_CMAPS)]
        cmap = plt.get_cmap(cmap_name)
        p = preds_by_label[label]

        for cand in p['candidates']:
            ax.plot(cand[:, 0], cand[:, 1], linestyle=':', linewidth=1.2,
                    color=cmap(CANDIDATE_FRAC), alpha=0.8, zorder=2)

        sel = p['selected']
        ax.plot(sel[:, 0], sel[:, 1], linestyle='-', linewidth=2.2,
                color=cmap(SELECTED_FRAC), alpha=0.95, zorder=3)

        proj = p['projection']
        if proj is not None and len(proj):
            ax.plot(proj[:, 0], proj[:, 1], linestyle='-', linewidth=3.2,
                    color=cmap(PROJECTED_FRAC), alpha=1.0, zorder=4)

    # ---- pedestrians ----
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
            ax.add_patch(patches.FancyArrowPatch(
                start, end, color=h_color, linewidth=2.6, arrowstyle=ped_arrow_style,
                shrinkA=0, shrinkB=0, zorder=6))

    # ---- robot ----
    robot_color = 'gold'
    ax.add_artist(plt.Circle(robot_pos0, robot_radius, fill=True,
                             facecolor=robot_color, edgecolor='black',
                             linewidth=2.2, zorder=7))
    if robot_kinematics == 'unicycle':
        robot_arrow_style = patches.ArrowStyle('->', head_length=4, head_width=2)
        ax.add_artist(patches.FancyArrowPatch(
            tuple(robot_pos0),
            (robot_pos0[0] + robot_radius * np.cos(robot_theta0),
            robot_pos0[1] + robot_radius * np.sin(robot_theta0)),
            color='red', arrowstyle=robot_arrow_style, linewidth=1.6, zorder=8))

    ax.plot(goal[0], goal[1], marker='*', color='red', markersize=11, zorder=7)

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
    legend_handles += [
        mlines.Line2D([], [], color='gray', linestyle=':', linewidth=1.2,
                      label='Candidate samples'),
        mlines.Line2D([], [], color='gray', linestyle='-', linewidth=2.2,
                      label='Selected sample'),
        mlines.Line2D([], [], color='dimgray', linestyle='-', linewidth=3.2,
                      label='Projected sample'),
    ]
    for i, label in enumerate(labels):
        cmap = plt.get_cmap(COMPOSITION_CMAPS[i % len(COMPOSITION_CMAPS)])
        p, pa, y, g = vectors[i]
        vec_str = f'prox{p:+.0f} pass{pa:+.0f} yield{y:+.0f} group{g:+.0f}'
        legend_handles.append(
            mlines.Line2D([], [], color=cmap(SELECTED_FRAC), lw=3,
                          label=f'{label}  [{vec_str}]'))
    ax.legend(handles=legend_handles, loc=legend_loc, fontsize=10,
             framealpha=0.9, title='Prediction at t=0', title_fontsize=12)

    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    logging.info('Saved figure: %s', out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Visualize the diffusion policy\'s single-step prediction '
                    '(candidates / selected / projected) for several style '
                    'compositions on one scene, at t=0 (no rollout).')
    parser.add_argument('--env_config',    type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--policy',        type=str, default='sogudiff')
    parser.add_argument('--gpu',           action='store_true', default=False)
    parser.add_argument('--phase',         type=str, default='test')

    parser.add_argument('--npz_path',      type=str,
                        help='Path to the single scene .npz file (required unless --replot)')
    parser.add_argument('--composition',   type=parse_composition_arg, action='append',
                        default=None, metavar='LABEL:prox,pass,yield,group',
                        help='One style-vector composition, e.g. "Cautious & Yielding:1,0,1,0". '
                             'Repeat for multiple compositions (typically 4).')
    parser.add_argument('--n_candidates',  type=int, default=10,
                        help='Number of raw diffusion candidates to draw per composition '
                             '(randomly chosen if more are available; all of them if fewer)')
    parser.add_argument('--seed',          type=int, default=None,
                        help='Seed for choosing WHICH candidates to draw when more than '
                             '--n_candidates are available (does not affect the diffusion '
                             'sampling itself, which is always stochastic)')
    parser.add_argument('--no_projection', action='store_true', default=False,
                        help='Disable the feasibility-projection layer (default: projection ON)')

    parser.add_argument('--out_dir',       type=str, default='results_style_prediction_figs')
    parser.add_argument('--out_file',      type=str, default=None,
                        help='Basename (no extension) for the .png/.npz outputs; '
                             'defaults to "prediction_<scene_basename>"')
    parser.add_argument('--fig_size',      type=float, default=8.0, help='Square figure side, inches')
    parser.add_argument('--dpi',           type=int, default=300)
    parser.add_argument('--margin',        type=float, default=0.8,
                        help='Meters of padding around the tight scene bounding box')
    parser.add_argument('--legend_loc',    type=str, default='upper right')

    parser.add_argument('--replot',        type=str, default=None,
                        help='Re-render the figure ONLY from a previously saved '
                             '<out_file>.npz (no policy/env/GPU needed)')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    os.makedirs(args.out_dir, exist_ok=True)

    if args.replot:
        preds_by_label, scene = load_predictions(args.replot)
        base = os.path.splitext(os.path.basename(args.replot))[0]
        png_path = os.path.join(args.out_dir, base + '.png')
        plot_predictions(preds_by_label, scene, png_path, fig_size=args.fig_size,
                        dpi=args.dpi, margin=args.margin, legend_loc=args.legend_loc)
        return

    if not args.npz_path:
        parser.error('--npz_path is required unless --replot is given')

    compositions = args.composition if args.composition else DEFAULT_COMPOSITIONS
    logging.info('Using %d compositions:', len(compositions))
    for label, vec in compositions:
        logging.info('  %r -> %s', label, vec)

    scene_base = os.path.splitext(os.path.basename(args.npz_path))[0]
    out_base = args.out_file or f'prediction_{scene_base}'
    npz_out_path = os.path.join(args.out_dir, out_base + '.npz')
    png_out_path = os.path.join(args.out_dir, out_base + '.png')

    preds_by_label, scene = collect_predictions(args, compositions)
    save_predictions(preds_by_label, scene, npz_out_path)
    plot_predictions(preds_by_label, scene, png_out_path, fig_size=args.fig_size,
                    dpi=args.dpi, margin=args.margin, legend_loc=args.legend_loc)


if __name__ == '__main__':
    main()
