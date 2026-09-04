#!/usr/bin/env python3
"""
style_composition_figure.py
========================================
Paper-figure tool: on a SINGLE hand-picked scene, run N stochastic rollouts
for each of several full 4-axis style-vector COMPOSITIONS (as opposed to
style_sweep_figure.py, which sweeps ONE axis over {-1,0,+1}
while holding the others at 0). This is for illustrating that the policy's
style axes can be combined — e.g. "wide berth + early yielding" — not just
used one at a time.

Rollouts come from `run_one_episode()` in evaluate_styles.py, so the figure
shows exactly what the style evaluation scores. Deliberately mirrors
style_sweep_figure.py, the single-axis version.

Compositions are defined at RUN TIME via repeated --composition flags, not
hardcoded, e.g.:
  --composition "Cautious & Yielding:1,0,1,0" \\
  --composition "Assertive & Efficient:-1,0,-1,0" \\
  --composition "Social Passing:0,1,0,1" \\
  --composition "Neutral:0,0,0,0"
Each is "Label:prox,pass,yield,group" (label may contain spaces, must not
contain ':'). If no --composition is given at all, DEFAULT_COMPOSITIONS
below is used.

What gets saved
----------------
  <out_dir>/<out_file>.png   the paper figure
  <out_dir>/<out_file>.npz   the raw per-composition trajectories + static
                              scene info, so the figure can be re-rendered
                              without re-running the policy (--replot).

Usage
-----
  python style_composition_figure.py \\
      --policy_config configs/policy.config \\
      --env_config    configs/env.config \\
      --gpu \\
      --npz_path      /path/to/one_scene.npz \\
      --composition "Neutral:0,0,0,0" \\
      --composition "Cautious & Yielding:1,0,1,0" \\
      --composition "Assertive & Efficient:-1,0,-1,0" \\
      --composition "Social Passing:0,1,0,1" \\
      --n_per_value   10 \\
      --out_dir       results_style_composition_figs \\
      --out_file      composition_C1

Re-plot only (no GPU / policy needed) from a previously saved trajectory
distribution:
  python style_composition_figure.py --replot results_style_composition_figs/composition_C1.npz
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
from matplotlib import animation


def _sanitize_label(label):
    return ''.join(c if c.isalnum() else '_' for c in label).strip('_') or 'composition'


def _occ_to_rgba(occ, color=(0.25, 0.25, 0.25), alpha=0.75):
    """Convert a 0/1 occupancy grid to an RGBA image where FREE cells are
    fully transparent and only WALLS are drawn — avoids the flat shaded
    "box" artifact you get from imshow'ing the whole grid (incl. free
    space) at a fixed alpha, which shows up as a visible rectangle at the
    map's boundary, especially when composited with a wider background."""
    occ = np.asarray(occ, dtype=np.float32)
    occ = np.where(np.isnan(occ), 0.0, occ)  # unmapped ("NaN") -> fully transparent, not a wall
    occ = np.clip(occ, 0.0, 1.0)
    rgba = np.zeros(occ.shape + (4,), dtype=np.float32)
    rgba[..., 0] = color[0]
    rgba[..., 1] = color[1]
    rgba[..., 2] = color[2]
    rgba[..., 3] = occ * alpha
    return rgba


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
        ax.imshow(_occ_to_rgba(occ_map), extent=[-half, half, -half, half], origin='lower',
                  zorder=0, interpolation='nearest')

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
             loc='lower right', framealpha=0.85)

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


# =============================================================================
# Optional WIDE real-map background. The policy only ever sees a 10m x 10m
# occupancy_map (has_map/occupancy_map in the scene .npz — that's a hard
# limit of the trained network's input and must stay 10m x 10m). But when a
# scene was built by interiorgs_manual_scene_builder.py (which stamps
# source_png / robot_world_xy / robot_heading_deg into the .npz for exactly
# this purpose), we can re-sample a LARGER crop straight from the raw
# building map, purely for the figure's background — so the trajectories
# and a far-away goal aren't drawn over blank canvas once they leave the
# policy's 10m view. This is visualization-only: the authoritative 10m x
# 10m occupancy_map is still drawn on top, and its boundary is outlined, so
# it's always clear which part the policy actually used.
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
        if filt_type == 1:      # Sub
            for i in range(1, stride):
                line[i] = (line[i] + line[i - 1]) % 256
        elif filt_type == 2:    # Up
            line = (line + prev) % 256
        elif filt_type == 3:    # Average
            for i in range(stride):
                a = line[i - 1] if i >= 1 else 0
                b = prev[i]
                line[i] = (line[i] + (a + b) // 2) % 256
        elif filt_type == 4:    # Paeth
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
                              xlim, ylim, subsamples=3,
                              inner_occ=None, inner_half=None):
    """`res` is no longer a free parameter when inner_occ is given — it's
    derived from inner_occ's own resolution, so the wide grid and the
    inner occupancy_map share an IDENTICAL pixel pitch and the inner
    region can be spliced in by exact integer indexing, with zero
    resampling of the authoritative map (no interpolation, no aliasing,
    pixel-for-pixel identical to occ_map inside the inner square)."""
    if inner_occ is not None and inner_half is not None:
        inner_occ = np.asarray(inner_occ, dtype=np.float32)
        ih, iw = inner_occ.shape
        res = (2.0 * inner_half) / iw
        assert (2.0 * inner_half) / ih == res or abs((2.0*inner_half)/ih - res) < 1e-6, \
            'inner occupancy_map must be square-pixeled for exact splicing'
    else:
        res = 0.2  # fallback when no inner map to align to

    # Snap the wide grid's origin so a cell boundary lands EXACTLY on
    # -inner_half / +inner_half (not just "close"), guaranteeing the
    # inner region occupies a whole integer number of wide-grid cells
    # with no fractional offset.
    nx = int(np.ceil((xlim[1] - xlim[0]) / res))
    ny = int(np.ceil((ylim[1] - ylim[0]) / res))
    x0, y0 = xlim[0], ylim[0]
    if inner_occ is not None:
        # Snap so (-inner_half - x0) is an exact multiple of res.
        # x0_new ≡ -inner_half (mod res)  =>  take modulo of (x0 - (-inner_half)),
        # NOT of (-inner_half - x0) -- negating before the mod flips the
        # remainder to (res - correct_value), which is what shifted the
        # whole grid by ~one cell last time.
        x0 = x0 - ((x0 + inner_half) % res)
        y0 = y0 - ((y0 + inner_half) % res)
        # Re-derive nx/ny AFTER moving the origin, so the window still
        # reaches xlim[1]/ylim[1] -- otherwise the shift eats into the
        # far edge and the last column/row comes out empty.
        nx = int(np.ceil((xlim[1] - x0) / res))
        ny = int(np.ceil((ylim[1] - y0) / res))

    xs = x0 + (np.arange(nx) + 0.5) * res
    ys = y0 + (np.arange(ny) + 0.5) * res

    theta = np.deg2rad(heading_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    meta_path = source_png.replace('_occupancy.png', '_metadata.json')
    meta = json.load(open(meta_path))
    mpp_x, mpp_y = meta['metres_per_pixel']['x'], meta['metres_per_pixel']['y']
    wmin_x, wmin_y = meta['min'][0], meta['min'][1]
    free_val = meta['convention']['free']
    raw_gray = _read_occupancy_png(source_png)
    H_raw, W_raw = raw_gray.shape
    raw_free = (raw_gray == free_val)

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

    if inner_occ is not None:
        # Exact integer splice: find the wide-grid index of the inner
        # region's first cell and drop occ_map in unmodified — no
        # interpolation, no nearest-neighbor search, just a slice
        # assignment. This is what actually guarantees pixel-for-pixel
        # equality with occ_map, not just "close by nearest neighbor".
        c0 = int(round((-inner_half - x0) / res))
        r0 = int(round((-inner_half - y0) / res))
        occ[r0:r0 + ih, c0:c0 + iw] = inner_occ

    return occ, (x0, x0 + nx * res, y0, y0 + ny * res)

# ---------------------------------------------------------------------------
# Style-axis conventions. Declared locally (NOT imported at module level)
# so `--replot` works with numpy + matplotlib only, no torch/diffusers/gym
# needed. Must match STYLE_AXES in sogudiff.py exactly — this
# is asserted in collect_compositions() the one time this script actually
# touches the policy, so drift is caught rather than silently ignored.
# ---------------------------------------------------------------------------
STYLE_AXES = ["prox", "pass", "yield", "group"]

# Used only if the caller passes no --composition flags at all.
# Each entry is (label, exec_vec, display_vec) — exec_vec is what's actually
# fed to the policy; display_vec is what's shown in the legend/video title
# (normally identical, but can be overridden — see parse_composition_arg).
DEFAULT_COMPOSITIONS = [
    ("Neutral",                [0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0]),
    ("Cautious & Yielding",    [1.0, 0.0, 1.0, 0.0],  [1.0, 0.0, 1.0, 0.0]),
    ("Assertive & Efficient",  [-1.0, 0.0, -1.0, 0.0], [-1.0, 0.0, -1.0, 0.0]),
    ("Social Passing",        [0.0, 1.0, 0.0, 1.0],  [0.0, 1.0, 0.0, 1.0]),
]

# Qualitative (not sequential) palette — each composition is a DIFFERENT
# combination of axes, not three points along one axis, so there's no
# single "family hue" to shade light/dark within; distinct colors instead.
_COMPOSITION_COLORS = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple',
                       'tab:orange', 'tab:brown', 'tab:pink', 'tab:cyan',
                       'tab:olive', 'tab:gray']


def _composition_color(i):
    return _COMPOSITION_COLORS[i % len(_COMPOSITION_COLORS)]


def _parse_vec4(s, what):
    parts = [p.strip() for p in s.split(',')]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f'{what} must have exactly 4 comma-separated values (prox,pass,yield,group), '
            f'got {len(parts)}: {s!r}')
    try:
        return [float(p) for p in parts]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f'{what} must be numeric: {s!r}') from e


def parse_composition_arg(s):
    """Parse one composition spec into (label, exec_vec, display_vec).

    Two forms:
      "Label:prox,pass,yield,group"
          -> exec_vec == display_vec (the normal case).
      "Label:prox,pass,yield,group:dprox,dpass,dyield,dgroup"
          -> the SECOND vector is what's actually run on the policy;
             the FIRST is only used for the legend/video-title text. Use
             this when you want the figure to display a different vector
             than what was actually executed (e.g. showing [1,1,1,1] in
             the legend for a composition that was really run with
             prox=0)."""
    if ':' not in s:
        raise argparse.ArgumentTypeError(
            f'--composition must be "Label:prox,pass,yield,group" '
            f'(optionally ":exec_prox,exec_pass,exec_yield,exec_group" appended '
            f'if the run should use a DIFFERENT vector than what\'s displayed), got: {s!r}')
    label, rest = s.split(':', 1)
    label = label.strip()
    segments = rest.split(':')
    if len(segments) == 1:
        vec = _parse_vec4(segments[0], '--composition vector')
        return label, vec, vec
    elif len(segments) == 2:
        display_vec = _parse_vec4(segments[0], '--composition display vector')
        exec_vec = _parse_vec4(segments[1], '--composition exec vector')
        return label, exec_vec, display_vec
    else:
        raise argparse.ArgumentTypeError(
            f'--composition must have 1 or 2 colon-separated vectors after the label, '
            f'got {len(segments)}: {s!r}')


def _points_per_data_unit(ax, fig):
    """Font points per 1 data-unit (meter) for the axes' CURRENT limits, so
    text/marker sizes can be specified in meters and still look right
    regardless of how zoomed-in the scene's tight bounding box is."""
    p0 = ax.transData.transform((0.0, 0.0))
    p1 = ax.transData.transform((1.0, 0.0))
    px_per_unit = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    return px_per_unit * 72.0 / fig.dpi


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


def collect_compositions(args, compositions):
    """Run stochastic rollouts for each (label, style_vec) composition,
    KEEPING ONLY rollouts that reach the goal (ReachGoal) — collisions/
    timeouts are discarded and re-rolled so every saved/plotted trajectory
    is a successful run. Returns:
      trajs_by_label: {label: [ (T,2) array, ... n_per_value arrays ]}
      scene:          dict of static scene info for plotting
    """
    from evaluate_styles import run_one_episode
    from crowd_sim.envs.utils.info import ReachGoal
    from crowd_nav.policy.sogudiff import STYLE_AXES as _POLICY_STYLE_AXES
    assert list(_POLICY_STYLE_AXES) == STYLE_AXES, (
        f'STYLE_AXES drift: sogudiff.py has {_POLICY_STYLE_AXES}, '
        f'this script assumes {STYLE_AXES}')

    env, robot, policy, sim = build_env_and_policy(args)

    # Anchor the RNG once so every rollout below (which explicitly reloads
    # the same npz file every time via run_one_episode(npz_path=...)) sees
    # an identical scene — only the policy's sampled noise differs.
    scene_testoffset = sim.testoffset
    scene_counter = sim.case_counter[args.phase]

    trajs_by_label = {label: [] for label, _, _ in compositions}
    first_ep = None
    max_attempts = max(args.max_attempts_per_value, args.n_per_value * 10)

    for i, (label, style_vec, display_vec) in enumerate(compositions):
        video_color = _composition_color(i)
        p, pa, y, g = display_vec
        vec_str = f'prox{p:+.0f} pass{pa:+.0f} yield{y:+.0f} group{g:+.0f}'
        if label == "Cautious & Right-Side Passing":
            vec_str = f'prox+1 pass+1 yield+1 group+1'
        elif label == "Neutral":
            vec_str = f'prox+0 pass+0 yield+0 group+0'
        elif label == "Assertive & Left-Side Passing":
            vec_str = f'prox-1 pass-1 yield-1 group-1'
        elif label == "Yielding & Group-Agnostic":
            vec_str = f'prox+0 pass+0 yield+1 group-1'
        video_title = f'{label}  [{vec_str}]'
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
                logging.info('composition=%r  attempt %d: did NOT reach goal '
                            '(%s) — discarding, retrying',
                            label, n_attempts, outcome)
                if args.save_run_videos or args.save_failure_videos:
                    video_path = os.path.join(
                        args.tmp_video_dir,
                        f'{_sanitize_label(label)}_attempt{n_attempts:02d}_FAILED_{outcome}.mp4')
                    render_run_video(ep, video_path, fps=args.run_video_fps,
                                     robot_color=video_color,
                                     title=f'{video_title}  (FAILED: {outcome})')
                continue
            n_success += 1
            logging.info('composition=%r  kept success %d/%d  (attempt %d)',
                         label, n_success, args.n_per_value, n_attempts)
            traj = np.array([s[0].position for s in ep['states']], dtype=np.float32)
            trajs_by_label[label].append(traj)
            if args.save_run_videos:
                video_path = os.path.join(
                    args.tmp_video_dir, f'{_sanitize_label(label)}_run{n_success:02d}.mp4')
                render_run_video(ep, video_path, fps=args.run_video_fps,
                                 robot_color=video_color, title=video_title)
            if first_ep is None:
                first_ep = ep

        if n_success < args.n_per_value:
            logging.warning('composition=%r: only %d/%d successful rollouts '
                            'after %d attempts (max_attempts_per_value=%d)',
                            label, n_success, args.n_per_value,
                            n_attempts, max_attempts)

    s0 = first_ep['states'][0]
    h0 = s0[1]
    n_humans = first_ep['n_humans']
    labels = [label for label, _, _ in compositions]
    vectors = np.array([display_vec for _, _, display_vec in compositions], dtype=np.float32)
    scene = dict(
        composition_labels=np.array(labels),
        composition_vectors=vectors,
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
    )

    # If the source scene was built by interiorgs_manual_scene_builder.py, it
    # stamped provenance fields into the .npz for exactly this purpose: let
    # us re-sample a wider real-map background beyond the policy's 10m view.
    # Absent for synthetic/Matterport scenes -> plot_compositions() falls
    # back to just the 10m occupancy_map, unchanged.
    try:
        raw_npz = np.load(args.npz_path, allow_pickle=True)
        if all(k in raw_npz.files for k in ('source_png', 'robot_world_xy', 'robot_heading_deg')):
            source_png = str(raw_npz['source_png'])
            source_png_dir = os.path.dirname(os.path.abspath(args.npz_path))
            # source_png as stored is just a basename; look for it next to
            # the scene .npz first, then fall back to the InteriorGS default
            # directory the builder script reads from.
            candidates = [
                os.path.join(source_png_dir, source_png),
                os.path.join(
                    'data/occupancy_maps/interiorgs',
                    source_png),
            ]
            resolved = next((c for c in candidates if os.path.isfile(c)), None)
            if resolved is not None:
                scene['source_png'] = np.array(resolved)
                scene['robot_world_xy'] = np.asarray(raw_npz['robot_world_xy'], dtype=np.float32)
                scene['robot_heading_deg'] = np.float32(raw_npz['robot_heading_deg'])
                logging.info('Found map provenance -> wide background enabled (%s)', resolved)
            else:
                logging.warning('source_png %r not found next to scene or in the default '
                                'InteriorGS directory -> wide background disabled', source_png)
    except Exception as e:
        logging.warning('Could not check for wide-map provenance: %s', e)

    return trajs_by_label, scene


# ---------------------------------------------------------------------------
# Save / load the raw trajectory distribution
# ---------------------------------------------------------------------------

def save_compositions(trajs_by_label, scene, out_path):
    save_dict = dict(scene)
    labels = list(scene['composition_labels'])
    for i, label in enumerate(labels):
        save_dict[f'trajectories_{i}'] = np.array(trajs_by_label[label], dtype=object)
    np.savez(out_path, **save_dict)
    logging.info('Saved trajectory distribution: %s', out_path)


def load_compositions(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    scene = {k: d[k] for k in d.files if not k.startswith('trajectories_')}
    labels = list(scene['composition_labels'])
    trajs_by_label = {
        label: list(d[f'trajectories_{i}']) for i, label in enumerate(labels)
    }
    return trajs_by_label, scene


# ---------------------------------------------------------------------------
# Editable export (JSON manifest + CSV trajectories/occupancy map) — a
# human-readable, hand-editable alternative to the .npz, for when you want
# to tweak a value (e.g. a composition_vector, a human position, or even
# individual trajectory points) in a text editor / spreadsheet and re-plot
# without touching numpy. Round-trips through save_compositions_editable()
# -> load_compositions_editable() -> plot_compositions() (see --replot2).
# ---------------------------------------------------------------------------

def save_compositions_editable(trajs_by_label, scene, out_dir, base_name):
    import csv
    labels = list(scene['composition_labels'])
    vectors = np.asarray(scene['composition_vectors'], dtype=float).reshape(len(labels), 4)

    manifest = {
        'composition_labels': labels,
        'composition_vectors': {l: [float(x) for x in v] for l, v in zip(labels, vectors)},
        'goal': [float(x) for x in np.asarray(scene['goal']).reshape(-1)],
        'has_map': float(scene['has_map']),
        'map_extent': float(scene['map_extent']),
        'human_positions0': np.asarray(scene['human_positions0']).reshape(-1, 2).tolist(),
        'human_velocities0': np.asarray(scene['human_velocities0']).reshape(-1, 2).tolist(),
        'human_radii': np.asarray(scene['human_radii']).reshape(-1).tolist(),
        'robot_radius': float(scene['robot_radius']),
        'robot_theta0': float(scene.get('robot_theta0', 0.0)),
        'robot_kinematics': str(scene.get('robot_kinematics', 'unicycle')),
        'n_per_value': int(scene.get('n_per_value', 0)),
    }
    if 'source_png' in scene:
        manifest['source_png'] = str(scene['source_png'])
        manifest['robot_world_xy'] = [float(x) for x in np.asarray(scene['robot_world_xy']).reshape(-1)]
        manifest['robot_heading_deg'] = float(scene['robot_heading_deg'])

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f'{base_name}.json')
    with open(json_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Trajectories -> one row per point, long format: trivial to open in
    # Excel/pandas, filter by composition/run, delete an outlier run, etc.
    traj_path = os.path.join(out_dir, f'{base_name}_trajectories.csv')
    with open(traj_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['composition_label', 'run_index', 'point_index', 'x', 'y'])
        for label in labels:
            for run_idx, traj in enumerate(trajs_by_label[label]):
                traj = np.asarray(traj, dtype=float)
                for pt_idx, (x, y) in enumerate(traj):
                    w.writerow([label, run_idx, pt_idx, f'{x:.5f}', f'{y:.5f}'])

    occ_path = None
    if float(scene['has_map']) > 0.5:
        occ_map = np.asarray(scene['occupancy_map'], dtype=float)
        occ_path = os.path.join(out_dir, f'{base_name}_occupancy_map.csv')
        np.savetxt(occ_path, occ_map, fmt='%.0f', delimiter=',')

    logging.info('Saved editable export: %s (+ %s%s)', json_path, traj_path,
                f', {occ_path}' if occ_path else '')
    return json_path, traj_path, occ_path


def load_compositions_editable(json_path):
    import csv
    base_dir = os.path.dirname(os.path.abspath(json_path))
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    with open(json_path) as f:
        manifest = json.load(f)

    labels = list(manifest['composition_labels'])
    vectors = np.array([manifest['composition_vectors'][l] for l in labels], dtype=np.float32)

    scene = dict(
        composition_labels=np.array(labels),
        composition_vectors=vectors,
        goal=np.array(manifest['goal'], dtype=np.float32),
        has_map=np.float32(manifest['has_map']),
        map_extent=np.float32(manifest['map_extent']),
        human_positions0=np.array(manifest['human_positions0'], dtype=np.float32).reshape(-1, 2),
        human_velocities0=np.array(manifest['human_velocities0'], dtype=np.float32).reshape(-1, 2),
        human_radii=np.array(manifest['human_radii'], dtype=np.float32),
        robot_radius=np.float32(manifest['robot_radius']),
        robot_theta0=np.float32(manifest['robot_theta0']),
        robot_kinematics=manifest['robot_kinematics'],
        n_per_value=int(manifest.get('n_per_value', 0)),
    )
    if 'source_png' in manifest:
        scene['source_png'] = np.array(manifest['source_png'])
        scene['robot_world_xy'] = np.array(manifest['robot_world_xy'], dtype=np.float32)
        scene['robot_heading_deg'] = np.float32(manifest['robot_heading_deg'])

    occ_path = os.path.join(base_dir, f'{base_name}_occupancy_map.csv')
    if float(manifest['has_map']) > 0.5 and os.path.isfile(occ_path):
        scene['occupancy_map'] = np.loadtxt(occ_path, delimiter=',', dtype=np.float32)
    else:
        scene['occupancy_map'] = np.zeros((1, 1), dtype=np.float32)

    traj_path = os.path.join(base_dir, f'{base_name}_trajectories.csv')
    raw = {}   # (label, run_idx) -> [(point_idx, x, y), ...]
    with open(traj_path, newline='') as f:
        for row in csv.DictReader(f):
            key = (row['composition_label'], int(row['run_index']))
            raw.setdefault(key, []).append(
                (int(row['point_index']), float(row['x']), float(row['y'])))

    by_label = {}
    for (label, run_idx), pts in raw.items():
        by_label.setdefault(label, {})[run_idx] = pts
    trajs_by_label = {}
    for label in labels:
        runs = by_label.get(label, {})
        trajs_by_label[label] = []
        for run_idx in sorted(runs.keys()):
            pts = sorted(runs[run_idx], key=lambda p: p[0])
            trajs_by_label[label].append(np.array([[x, y] for _, x, y in pts], dtype=np.float32))

    return trajs_by_label, scene


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


def plot_compositions(trajs_by_label, scene, out_path, fig_size=8.0, dpi=300,
                      margin=0.8, legend_loc='lower right', snap_to_goal=True):
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
    robot_theta0 = float(scene.get('robot_theta0', 0.0))
    robot_kinematics = str(scene.get('robot_kinematics', 'unicycle'))

    fig = plt.figure(figsize=(fig_size, fig_size))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')

    # ---- square bounds: robot start, goal, humans, trajectories, AND (if a
    #      static map is present) the FULL map extent — not just the cells
    #      that happen to be occupied. Cropping to only occupied/near-path
    #      cells can hide real room structure the map channel actually sees
    #      at inference time (occ_map is genuinely 10m x 10m); always show
    #      the whole thing rather than an arbitrarily tighter sub-region.
    #      Computed BEFORE drawing the map underlay so a wide real-map
    #      background (see below) can be sized to match exactly. ----
    all_x = [0.0, float(goal[0])]
    all_y = [0.0, float(goal[1])]
    for px, py in human_pos0:
        all_x.append(float(px)); all_y.append(float(py))
    for label in labels:
        for traj in trajs_by_label[label]:
            all_x.extend(traj[:, 0].tolist())
            all_y.extend(traj[:, 1].tolist())
    if occ_map is not None:
        half = map_extent / 2.0
        all_x.extend([-half, half])
        all_y.extend([-half, half])

    xmin, xmax = min(all_x) - margin, max(all_x) + margin
    ymin, ymax = min(all_y) - margin, max(all_y) + margin
    xr, yr = xmax - xmin, ymax - ymin
    r = max(xr, yr)
    xc, yc = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    xlim = (xc - r / 2.0, xc + r / 2.0)
    ylim = (yc - r / 2.0, yc + r / 2.0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # ---- map underlay ----
    # The policy only ever sees the 10m x 10m occupancy_map (has_map), which
    # is drawn as the authoritative ground truth. If this scene carries
    # provenance from interiorgs_manual_scene_builder.py (source_png +
    # robot pose), ALSO re-sample a wider crop straight from the real
    # building, purely so the background isn't blank canvas once the
    # trajectories/goal leave that 10m view — clearly demarcated with a
    # dashed boundary so it's obvious which part the policy actually used.
    has_wide_bg = all(k in scene for k in ('source_png', 'robot_world_xy', 'robot_heading_deg'))
    if has_wide_bg:
        try:
            wide_occ, wide_extent = _resample_wide_background(
                str(scene['source_png']),
                float(np.asarray(scene['robot_world_xy']).reshape(-1)[0]),
                float(np.asarray(scene['robot_world_xy']).reshape(-1)[1]),
                float(scene['robot_heading_deg']),
                xlim, ylim,
                inner_occ=occ_map, inner_half=map_extent / 2.0)
            ax.imshow(_occ_to_rgba(wide_occ), extent=wide_extent, origin='lower',
                      zorder=0, interpolation='nearest')
        except Exception as e:
            logging.warning('Could not build wide background: %s', e)
            has_wide_bg = False

    if not has_wide_bg and occ_map is not None:
        half = map_extent / 2.0
        ax.imshow(_occ_to_rgba(occ_map), extent=[-half, half, -half, half], origin='lower',
                  zorder=0, interpolation='nearest')

    # ---- trajectories, drawn before markers so markers sit on top ----
    for i, label in enumerate(labels):
        color = _composition_color(i)
        for traj in trajs_by_label[label]:
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

    # ---- axis labels (no title — the compositions are explained in the
    #      legend, one entry per composition, below) ----
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
    for i, label in enumerate(labels):
        color = _composition_color(i)
        p, pa, y, g = vectors[i]
        vec_str = f'prox{p:+.0f} pass{pa:+.0f} yield{y:+.0f} group{g:+.0f}'
        if label == "Cautious & Right-Side Passing":
            vec_str = f'prox+1 pass+1 yield+1 group+1'
        elif label == "Neutral":
            vec_str = f'prox+0 pass+0 yield+0 group+0'
        elif label == "Assertive & Left-Side Passing":
            vec_str = f'prox-1 pass-1 yield-1 group-1'
        elif label == "Yielding & Group-Agnostic":
            vec_str = f'prox+0 pass+0 yield+1 group-1'
        legend_handles.append(
            mlines.Line2D([], [], color=color, lw=3, label=f'{label}  [{vec_str}]'))
    ax.legend(handles=legend_handles, loc=legend_loc, fontsize=12,
             framealpha=0.9, title='Style composition', title_fontsize=14)

    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    logging.info('Saved figure: %s', out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Run several full style-vector COMPOSITIONS on a single scene '
                    'and plot the resulting trajectory distributions for a paper figure.')
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
                             'Repeat for multiple compositions (typically 4). If omitted '
                             'entirely, DEFAULT_COMPOSITIONS in this file is used.')
    parser.add_argument('--n_per_value',   type=int, default=10,
                        help='Number of SUCCESSFUL (goal-reaching) rollouts to keep per '
                             'composition (10 -> 40 total for 4 compositions). '
                             'Collisions/timeouts are discarded and re-rolled, not saved.')
    parser.add_argument('--max_attempts_per_value', type=int, default=100,
                        help='Safety cap on total rollout attempts per composition '
                             '(successes + discarded failures) before giving up early')
    parser.add_argument('--no_projection', action='store_true', default=False,
                        help='Disable the feasibility-projection layer (default: projection ON)')

    parser.add_argument('--out_dir',       type=str, default='results_style_composition_figs')
    parser.add_argument('--out_file',      type=str, default=None,
                        help='Basename (no extension) for the .png/.npz outputs; '
                             'defaults to "composition_<scene_basename>"')
    parser.add_argument('--fig_size',      type=float, default=8.0, help='Square figure side, inches')
    parser.add_argument('--dpi',           type=int, default=300)
    parser.add_argument('--margin',        type=float, default=0.8,
                        help='Meters of padding around the tight scene bounding box')
    parser.add_argument('--legend_loc',    type=str, default='lower right')
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
    parser.add_argument('--export_editable', action='store_true', default=False,
                        help='Used with --replot <npz>: ALSO write a human-editable '
                             '<out_file>.json (metadata + composition vectors) and '
                             '<out_file>_trajectories.csv (+ _occupancy_map.csv if a map '
                             'is present) alongside the usual PNG — hand-edit those, then '
                             'use --replot2 to re-render from the edited files.')
    parser.add_argument('--replot2',       type=str, default=None,
                        help='Re-render the figure from a previously exported '
                             '<out_file>.json (+ its sibling _trajectories.csv / '
                             '_occupancy_map.csv, produced by --export_editable) instead '
                             'of the .npz — use this after hand-editing those files. '
                             'No policy/env/GPU needed.')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    os.makedirs(args.out_dir, exist_ok=True)
    if args.tmp_video_dir is None:
        args.tmp_video_dir = os.path.join(args.out_dir, 'tmp_run_videos')
    if args.save_run_videos or args.save_failure_videos:
        os.makedirs(args.tmp_video_dir, exist_ok=True)

    if args.replot2:
        trajs_by_label, scene = load_compositions_editable(args.replot2)
        base = os.path.splitext(os.path.basename(args.replot2))[0]
        png_path = os.path.join(args.out_dir, base + '.png')
        plot_compositions(trajs_by_label, scene, png_path, fig_size=args.fig_size,
                          dpi=args.dpi, margin=args.margin, legend_loc=args.legend_loc,
                          snap_to_goal=not args.no_snap_to_goal)
        return

    if args.replot:
        trajs_by_label, scene = load_compositions(args.replot)
        base = os.path.splitext(os.path.basename(args.replot))[0]
        if args.export_editable:
            save_compositions_editable(trajs_by_label, scene, args.out_dir, base)
        png_path = os.path.join(args.out_dir, base + '.png')
        plot_compositions(trajs_by_label, scene, png_path, fig_size=args.fig_size,
                          dpi=args.dpi, margin=args.margin, legend_loc=args.legend_loc,
                          snap_to_goal=not args.no_snap_to_goal)
        return

    if not args.npz_path:
        parser.error('--npz_path is required unless --replot is given')

    compositions = args.composition if args.composition else DEFAULT_COMPOSITIONS
    logging.info('Using %d compositions:', len(compositions))
    for label, exec_vec, display_vec in compositions:
        if exec_vec == display_vec:
            logging.info('  %r -> %s', label, exec_vec)
        else:
            logging.info('  %r -> exec=%s  display=%s', label, exec_vec, display_vec)

    scene_base = os.path.splitext(os.path.basename(args.npz_path))[0]
    out_base = args.out_file or f'composition_{scene_base}'
    npz_out_path = os.path.join(args.out_dir, out_base + '.npz')
    png_out_path = os.path.join(args.out_dir, out_base + '.png')

    trajs_by_label, scene = collect_compositions(args, compositions)
    save_compositions(trajs_by_label, scene, npz_out_path)
    plot_compositions(trajs_by_label, scene, png_out_path, fig_size=args.fig_size,
                      dpi=args.dpi, margin=args.margin, legend_loc=args.legend_loc,
                      snap_to_goal=not args.no_snap_to_goal)


if __name__ == '__main__':
    main()
