#!/usr/bin/env python3
"""
evaluate_styles.py

Style-sweep evaluation for SoGuDiff.
Runs 6 policy variants on every test scene and produces a single 6-panel
video per scene where all panels are time-synchronized:

  Panel 0 — style=[0,0,0,0]  + projection ON   (neutral baseline)
  Panel 1 — style=[1,0,0,0]  + projection ON   (proxemic conservative)
  Panel 2 — style=[0,1,0,0]  + projection ON   (pass-side bias)
  Panel 3 — style=[0,0,1,0]  + projection ON   (yielding)
  Panel 4 — style=[0,0,0,1]  + projection ON   (group deference)
  Panel 5 — style=[0,0,0,0]  + projection OFF  (neutral, no feasibility layer)

All 6 panels advance frame-by-frame in lockstep.  Shorter episodes hold
their last frame until the longest episode in that scene finishes.  All
per-scene videos are concatenated into one combined output file.

A CSV summary of per-variant, per-scene metrics is written to the results dir.

Usage
-----
  python evaluate_styles.py \\
      --policy_config  path/to/policy.config \\
      --env_config     path/to/env.config \\
      --policy         sogudiff \\
      --video_file     style_sweep.mp4 \\
      [--num_tests 50] \\
      [--npz_hard --npz_dir DIR] [--map_eval] [--square] [--circle] \\
      [--fps 10] [--results_suffix TAG]
"""

import logging
import argparse
import configparser
import copy
import os
import subprocess
import glob
import csv
import json

import numpy as np
import torch
import gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
from matplotlib import animation

from crowd_nav.policy.policy_factory import policy_factory
from crowd_nav.paths import ffmpeg_binary


def _bootstrap_ci(values, n_boot=10000, seed=0):
    """(lo, hi, n_nonzero, n) -- percentile bootstrap CI of the mean.

    Resamples SCENES, which is the unit of independence here: the same scene is
    replayed under every style variant, so scene difficulty is a shared random
    effect and the per-scene values are what vary.

    n_nonzero is reported alongside because it is the diagnostic that matters
    for the rare-event rates (group split, TTC infraction). A metric that is
    zero in most scenes has almost no support, and a tight-looking CI around a
    near-zero mean says "nothing happened here", not "this effect is precise".
    """
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)],
                   dtype=float)
    n = len(v)
    n_nz = int((v != 0).sum())
    if n == 0:
        return float('nan'), float('nan'), 0, 0
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), n_nz, n
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.utils.info import *  # ReachGoal, Collision, Timeout, WallCollision …

# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------
# Each entry: (short_label, style_vector, use_projection)
# Axis ordering matches STYLE_AXES in sogudiff.py:
#   [prox, pass, yield, group]
STYLE_VARIANTS = [
    ("neutral+proj",   [0.0, 0.0, 0.0, 0.0], True),
    ("prox+proj",      [1.0, 0.0, 0.0, 0.0], True),
    ("pass+proj",      [0.0, 1.0, 0.0, 0.0], True),
    ("yield+proj",     [0.0, 0.0, 1.0, 0.0], True),
    ("group+proj",     [0.0, 0.0, 0.0, 1.0], True),
    ("neutral-noproj", [0.0, 0.0, 0.0, 0.0], False),
    ("prox-proj",      [-1.0, 0.0, 0.0, 0.0], True),
    ("pass-proj",      [0.0, -1.0, 0.0, 0.0], True),
    ("yield-proj",     [0.0, 0.0, -1.0, 0.0], True),
    ("group-proj",     [0.0, 0.0, 0.0, -1.0], True),
]
N_VARIANTS = len(STYLE_VARIANTS)  # 10

# Video layout: 2 rows × 3 cols
GRID_ROWS = 2
GRID_COLS = 5

_ROBOT_COLORS = ['gold', 'tomato', 'salmon',
                 'limegreen', 'darkgreen',
                 'deepskyblue', 'navy',
                 'mediumpurple', 'indigo',
                 'darkorange']

# Predicted/selected/projected trajectories share ONE color scheme across
# all style panels (only the past/trail trajectory is per-style colored via
# _ROBOT_COLORS), so style-vs-style comparisons aren't muddied by color noise.
_PROJ_COLOR   = 'royalblue'

_OUTCOME_COLORS = {
    'success':   'green',
    'collision': 'red',
    'timeout':   'darkorange',
    'other':     'gray',
}




# ---------------------------------------------------------------------------
# Social metric computation for style comparison
#
# All distances are computed in meters (the simulator's native unit) but the
# *thresholds* below are derived from Hall's proxemic distances, which are
# specified in feet and converted here. All metrics use NEUTRAL weights (no
# style modulation) so numbers are directly comparable across variants.
#
# prox axis  → metrics: mean_min_clearance_m (MinD), psi_rate
# pass axis  → metrics: passing_side_bias (signed, not a penalty)
# yield axis → metrics: ttc_infraction_rate, path_cutoff_rate
# group axis → metrics: group_split_rate
#
# Every per-human/per-pair heading used below (u_hat) is the human's
# INSTANTANEOUS per-timestep velocity direction (hum_vel[t, j]), not an
# episode-averaged heading — required for PSB/PCR to be evaluated at the
# correct instant. sim.states[t] stores (position, velocity) as a
# self-consistent snapshot for both robot and humans (confirmed against
# crowd_sim.py's step()/Agent.step()), so no off-by-one correction is needed.
# ---------------------------------------------------------------------------

_FT_TO_M = 0.3048

_D_PER = 4.0  * _FT_TO_M   # personal space   (Hall) ≈ 1.2192 m
_D_SOC = 12.0 * _FT_TO_M   # social space      (Hall) ≈ 3.6576 m
_D_INT = 1.5  * _FT_TO_M   # intimate space    (Hall) ≈ 0.4572 m

# Group MEMBERSHIP -- mirrors the EXPERT COST's own definition of a group, so
# the metric scores the same object the demonstrations were generated against.
#
# social_cost_torch.py gates a pair two ways:
#     pair   = (sep0 < D_SOC) & (dv <= DV_MAX)      <- structural, style-free
#     member = (sep0 < ladder(s_group))             <- style-DEPENDENT radius
# where ladder() gives 0.50 m at s=-1, 1.22 m at s=0 and 3.66 m at s=+1.
#
# The metric must use the STRUCTURAL gate, at the most inclusive radius, for
# every variant. A group is ground truth about the scene, not a function of the
# robot's style: if group+ recognizes a formation and detours around it, that
# detour has to register as "did not split a group". Scoring membership at a
# tighter radius would simply not see the formation group+ was avoiding, and
# would under-credit exactly the variant the metric exists to measure.
#
# The co-motion gate matters for the same reason common sense says it does: two
# people walking in OPPOSITE directions who happen to pass close by are not a
# group, and splitting them is meaningless. Distance alone cannot tell those
# apart from a walking pair.
_D_GROUP = _D_SOC          # 3.66 m -- the widest radius any style recognizes
_DV_MAX = 0.67             # m/s, = V_BAR/2, the co-motion gate from the cost

_TAU_R = 1.5               # human reaction time (s)

_MOVING_EPS = 1e-3   # human speed below this treated as "not moving" (no heading)

# Passing-Side Bias (PSB) gating — restrict to head-on / overtaking
# encounters where "which side did we pass on" is unambiguous. A crossing
# encounter (robot heading roughly perpendicular to the human's) doesn't
# have a clean pass-side answer the way head-on/overtaking do, and a
# near-stationary human has too noisy a heading to classify a side against
# at all. Lumping those cases into passing_side_bias dilutes it with
# encounters the pass style axis was never meant to describe.
_PASS_MIN_SPEED = 0.1              # m/s; below this, a heading isn't well-defined
_PASS_HEADING_TOL_DEG = 30.0       # max deviation from parallel/anti-parallel
_PASS_HEADING_COS_THRESH = np.cos(np.radians(_PASS_HEADING_TOL_DEG))


def compute_episode_social_metrics(states, humans_data, time_step):
    """
    Compute per-episode social metrics for one episode.

    Parameters
    ----------
    states      : list of [robot_full_state, [human_full_states...]]
                  as stored in sim.states
    humans_data : list of human objects (for radius; use sim.humans at
                  episode end — radii don't change mid-episode)
    time_step   : float (unused directly — all metrics here are averages
                  over timesteps rather than time integrals, per spec)

    Returns
    -------
    dict of scalar metrics
    """
    T = len(states)
    M = len(states[0][1]) if T > 0 else 0

    if T == 0 or M == 0:
        return _zero_metrics()

    # ── Extract trajectories (instantaneous position + velocity) ──────────
    rob_xy  = np.array([s[0].position for s in states], dtype=np.float64)  # (T, 2)
    rob_vx  = np.array([s[0].vx       for s in states], dtype=np.float64)  # (T,)
    rob_vy  = np.array([s[0].vy       for s in states], dtype=np.float64)  # (T,)
    rob_vel = np.stack([rob_vx, rob_vy], axis=-1)                          # (T, 2)

    hum_xy = np.array(
        [[s[1][j].position for j in range(M)] for s in states], dtype=np.float64
    )  # (T, M, 2)
    hum_vx = np.array([[s[1][j].vx for j in range(M)] for s in states], dtype=np.float64)
    hum_vy = np.array([[s[1][j].vy for j in range(M)] for s in states], dtype=np.float64)
    hum_vel = np.stack([hum_vx, hum_vy], axis=-1)  # (T, M, 2)   instantaneous, per timestep

    # p^r_t - p^j_t
    delta = rob_xy[:, None, :] - hum_xy            # (T, M, 2)
    dist  = np.linalg.norm(delta, axis=-1)          # (T, M)

    # ── 1a. Average minimum clearance (MinD) ──────────────────────────────
    min_dist_per_t = dist.min(axis=1)               # (T,)
    mean_min_clearance_m = float(min_dist_per_t.mean())

    # ── 1b. Personal Space Intrusion (PSI) rate ───────────────────────────
    psi_rate = float((min_dist_per_t < _D_PER).mean())

    # ── 2. Passing-Side Bias (PSB) ─────────────────────────────────────────
    # For each human j: find the instant t_j* of closest approach to the
    # robot over the whole episode, then (if that approach happened in the
    # social zone, both agents were moving meaningfully fast, and the
    # robot's heading was parallel or anti-parallel to the human's — an
    # oncoming or overtaking encounter, not a crossing one) determine which
    # side of human j's heading the robot passed on.
    s_list = []
    for j in range(M):
        d_j = dist[:, j]
        t_star = int(np.argmin(d_j))
        d_star = float(d_j[t_star])
        v_star = hum_vel[t_star, j]
        speed_star = float(np.linalg.norm(v_star))
        r_star = rob_vel[t_star]
        rob_speed_star = float(np.linalg.norm(r_star))
        if (d_star < _D_SOC
                and speed_star > _PASS_MIN_SPEED
                and rob_speed_star > _PASS_MIN_SPEED):
            u_hat = v_star / speed_star
            r_hat = r_star / rob_speed_star
            cos_theta = float(np.dot(u_hat, r_hat))
            if abs(cos_theta) < _PASS_HEADING_COS_THRESH:
                continue  # crossing encounter — no clean pass side, skip
            l_hat = np.array([-u_hat[1], u_hat[0]])          # 90° CCW of heading
            rel   = rob_xy[t_star] - hum_xy[t_star, j]        # p^r - p^j at t*
            s_j   = float(np.sign(np.dot(rel, l_hat)))
            if s_j != 0.0:
                s_list.append(s_j)
    # NaN, not 0.0, when no qualifying encounter occurred. 0.0 is
    # indistinguishable from "passed left and right equally often", so episodes
    # with no clean pass were silently pulling every variant's mean toward zero
    # and shrinking the very effect the pass axis is meant to show. nanmean and
    # the bootstrap's isfinite filter both skip NaN correctly.
    passing_side_bias = float(np.mean(s_list)) if s_list else float('nan')

    # ── 3. TTC Infraction Rate ─────────────────────────────────────────────
    # Constant-velocity extrapolation of the time at which the robot-human
    # separation would first reach D_int, restricted to closing pairs.
    rel_pos = -delta                          # p^j - p^r          (T, M, 2)
    rel_vel = hum_vel - rob_vel[:, None, :]   # v^j - v^r          (T, M, 2)
    closing = (rel_vel * rel_pos).sum(axis=-1) < 0.0             # (T, M) bool

    a = (rel_vel ** 2).sum(axis=-1)                               # (T, M)
    b = 2.0 * (rel_pos * rel_vel).sum(axis=-1)                    # (T, M)
    c = (rel_pos ** 2).sum(axis=-1) - _D_INT ** 2                 # (T, M)

    ttc = np.full((T, M), np.inf, dtype=np.float64)

    # Already inside D_int while closing → immediate infraction (TTC = 0).
    already_intimate = c <= 0.0
    ttc[already_intimate & closing] = 0.0

    # Otherwise solve ||rel_pos + rel_vel*tau|| = D_int for the smallest
    # non-negative root (a > 0 is guaranteed here since closing requires
    # rel_vel != 0).
    solvable = closing & (~already_intimate) & (a > 1e-9)
    a_safe = a.clip(min=1e-9)
    disc = b ** 2 - 4.0 * a_safe * c
    has_root = solvable & (disc >= 0.0)
    sqrt_disc = np.sqrt(np.clip(disc, 0.0, None))
    tau1 = np.where((-b - sqrt_disc) >= 0.0, (-b - sqrt_disc) / (2.0 * a_safe), np.inf)
    tau2 = np.where((-b + sqrt_disc) >= 0.0, (-b + sqrt_disc) / (2.0 * a_safe), np.inf)
    tau_min = np.minimum(tau1, tau2)
    ttc[has_root] = tau_min[has_root]

    min_ttc_per_t = ttc.min(axis=1)   # (T,)
    ttc_infraction_rate = float((min_ttc_per_t < _TAU_R).mean())

    # ── 4. Path Cutoff Rate (PCR) ──────────────────────────────────────────
    # Fraction of timesteps where the robot sits inside a moving human's
    # reaction-horizon corridor: laterally within D_int of their path AND
    # longitudinally within the distance that human could cover in tau_r.
    hum_spd = np.linalg.norm(hum_vel, axis=-1)                    # (T, M)
    moving_mask = hum_spd > _MOVING_EPS
    hum_spd_safe = hum_spd.clip(min=1e-9)
    u_hat_t = hum_vel / hum_spd_safe[..., None]                   # (T, M, 2)

    s_par = (delta * u_hat_t).sum(axis=-1)                        # (T, M)
    perp_vec = delta - s_par[..., None] * u_hat_t
    d_perp = np.linalg.norm(perp_vec, axis=-1)                    # (T, M)

    cutoff_cond = (
        moving_mask
        & (d_perp < _D_INT)
        & (s_par >= 0.0)
        & (s_par <= hum_spd * _TAU_R)
    )                                                              # (T, M)
    path_cutoff_rate = float(cutoff_cond.any(axis=1).mean())

    # ── 5. Group Split Rate (GSR) ──────────────────────────────────────────
    # NaN until a group is actually detected. Same reasoning as
    # passing_side_bias: a scene containing no group offers no opportunity to
    # split one, and scoring it 0.0 would dilute the metric with episodes that
    # could never have contributed. 0.0 is reserved for "a group was present and
    # the robot did not cut through it".
    group_split_rate = float('nan')
    if M >= 2:
        # seg[t, i, j] = p^j_t - p^i_t
        seg = hum_xy[:, None, :, :] - hum_xy[:, :, None, :]       # (T, M, M, 2)
        hh_dist = np.linalg.norm(seg, axis=-1)                    # (T, M, M)

        # A pair is a group when it is BOTH close AND co-moving for most of the
        # episode -- the same two conditions the expert cost applies. The cost
        # tests them once at t=0; here they are required for >50% of the episode,
        # which is the same idea made robust to humans manoeuvring mid-episode.
        dv = np.linalg.norm(hum_vel[:, None, :, :] - hum_vel[:, :, None, :],
                            axis=-1)                              # (T, M, M)
        together = (hh_dist < _D_GROUP) & (dv <= _DV_MAX)         # (T, M, M)
        G = together.mean(axis=0)                                 # (M, M)
        triu = np.triu(np.ones((M, M), dtype=bool), k=1)
        pair_mask = (G > 0.5) & triu                               # (M, M)

        if pair_mask.any():
            group_split_rate = 0.0   # a group exists: 0.0 now means 'did not split'
            seg_sq = (seg ** 2).sum(axis=-1).clip(min=1e-9)        # (T, M, M)
            # robrel[t, i] = p^r_t - p^i_t, broadcast over j
            robrel_i = rob_xy[:, None, :] - hum_xy                 # (T, M, 2)
            robrel = robrel_i[:, :, None, :]                       # (T, M, 1, 2) -> (T, M, M, 2)

            u = (robrel * seg).sum(axis=-1) / seg_sq               # (T, M, M) unclipped
            u_c = u.clip(0.0, 1.0)
            d_perp_seg = np.linalg.norm(robrel - u_c[..., None] * seg, axis=-1)  # (T, M, M)

            infraction = (
                pair_mask[None, :, :]
                & (d_perp_seg < _D_PER)
                & (u >= 0.0) & (u <= 1.0)
            )  # (T, M, M)
            group_split_rate = float(infraction.any(axis=(1, 2)).mean())

    return {
        # prox axis — MinD higher is better, PSI lower is better; prox+ should move both that way
        'mean_min_clearance_m':  mean_min_clearance_m,
        'psi_rate':              psi_rate,
        # pass axis — signed; pass+ (RHT) → toward +1, pass- (LHT) → toward -1
        'passing_side_bias':     passing_side_bias,
        # yield axis — lower is better; yield+ should reduce both
        'ttc_infraction_rate':   ttc_infraction_rate,
        'path_cutoff_rate':      path_cutoff_rate,
        # group axis — lower is better; group+ should reduce
        'group_split_rate':      group_split_rate,
    }


def _zero_metrics():
    return {
        'mean_min_clearance_m':  float('inf'),
        'psi_rate':              0.0,
        'passing_side_bias':     float('nan'),   # no humans -> no pass to score
        'ttc_infraction_rate':   0.0,
        'path_cutoff_rate':      0.0,
        'group_split_rate':      float('nan'),  # no humans -> no group to split
    }



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_policy_state(policy):
    """Zero out per-episode velocity memory so each run starts cold."""
    if hasattr(policy, '_reset_unicycle_state'):
        policy._reset_unicycle_state()
    else:
        for attr, val in [('prev_v', 0.0), ('prev_omega', 0.0),
                          ('_v0_from_last_step', 0.0),
                          ('_w0_from_last_step', 0.0)]:
            if hasattr(policy, attr):
                setattr(policy, attr, val)


def _outcome_label(info):
    if isinstance(info, ReachGoal):
        return 'success'
    if isinstance(info, Collision):
        return 'collision'
    if isinstance(info, Timeout):
        return 'timeout'
    return 'other'


def run_one_episode(env, robot, policy, style_vec, use_proj,
                    phase, scene_idx, npz_path=None,
                    scene_testoffset=None, scene_counter=None):
    """
    Run one episode for the given (style_vec, use_proj) variant on scene_idx.

    scene_testoffset / scene_counter: if provided, the env's internal RNG
    anchor (sim.testoffset and sim.case_counter[phase]) is restored to these
    values before reset().  This guarantees all 6 variants see the same scene
    regardless of how many reset() calls have accumulated beforehand.

    Returns a dict of everything needed for rendering + metrics.
    """
    policy.style_vector    = np.array(style_vec, dtype=np.float32)
    policy.use_projection  = use_proj
    _reset_policy_state(policy)

    sim = env.unwrapped

    # Restore the RNG anchor so this variant gets the same scene as the first
    # variant for scene_idx.  testoffset increments on every reset(), so
    # without restoring it each subsequent variant would receive a different
    # random seed and thus a different scene.
    if scene_testoffset is not None:
        sim.testoffset = scene_testoffset
    if scene_counter is not None:
        sim.case_counter[phase] = scene_counter

    if npz_path is not None:
        env.load_npz_scenario(npz_path)

    ob = env.reset(phase=phase, test_case=scene_idx)

    done = False
    while not done:
        action = robot.act(ob)
        ob, _, done, info = env.step(action)

    # Compute fixed-scale social metrics for style comparison
    social_metrics = compute_episode_social_metrics(
        copy.deepcopy(sim.states),
        sim.humans,
        sim.time_step,
    )

    return {
        'states':           copy.deepcopy(sim.states),
        'predicted_trajs':  copy.deepcopy(sim.predicted_trajs),
        'info':             info,
        'global_time':      sim.global_time,
        'pathlength':       sim.pathlength,
        'minobsdist':       sim.minobsdist,
        'avgobsdist':       list(sim.avgobsdist),
        'human_radii':      [h.radius for h in sim.humans],
        'n_humans':         len(sim.humans),
        'robot_radius':     robot.radius,
        'robot_kinematics': robot.kinematics,
        'goal_pos':         tuple(robot.get_goal_position()),
        'time_step':        float(sim.time_step),
        'occ_map':  copy.deepcopy(getattr(sim, '_npz_occ_map', None)),
        'has_map':  float(getattr(sim, '_npz_has_map', 0.0)),
        'map_extent': float(getattr(sim, '_npz_map_extent', 10.0)),
        'social_metrics': social_metrics,   # ← new
    }


def _pad_to(lst, target_len):
    """Return a list of length target_len, repeating the last element."""
    if not lst:
        return [None] * target_len
    return list(lst) + [lst[-1]] * max(0, target_len - len(lst))


# ---------------------------------------------------------------------------
# Multi-panel per-scene video
# ---------------------------------------------------------------------------

def build_scene_video(episodes, scene_idx, output_path, fps=10):
    """
    Build one MP4 with GRID_ROWS × GRID_COLS panels from `episodes` list.
    All panels are synchronized: shorter episodes hold their last frame.
    """
    max_frames = max(len(ep['states']) for ep in episodes)

    padded_states = [_pad_to(ep['states'],         max_frames) for ep in episodes]
    padded_trajs  = [_pad_to(ep['predicted_trajs'], max_frames) for ep in episodes]

    cmap_human = plt.cm.get_cmap('hsv', 10)
    x_off = y_off = 0.11
    arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

    fig, axes = plt.subplots(GRID_ROWS, GRID_COLS,
                              figsize=(6 * GRID_COLS, 6 * GRID_ROWS))
    axes = np.array(axes).flatten()
    fig.suptitle(f'Style Sweep — Scene {scene_idx + 1}', fontsize=16,
                 fontweight='bold', y=0.995)
    # Leave headroom at the top for the suptitle (rect) and widen the gap
    # between rows (hspace) so row-2 panel titles don't overlap row-1's
    # x-axis labels/ticks.
    fig.tight_layout(pad=2.0, rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(hspace=0.4)

    panels = []  # per-panel mutable render state

    for p, (ep, (label, style_vec, use_proj)) in enumerate(
            zip(episodes, STYLE_VARIANTS)):

        ax          = axes[p]
        r_color     = _ROBOT_COLORS[p]

        ax.set_xlim(-7.5, 7.5)
        ax.set_ylim(-7.5, 7.5)
        ax.tick_params(labelsize=9)
        ax.set_xlabel('x (m)', fontsize=9)
        ax.set_ylabel('y (m)', fontsize=9)

        style_str = "[" + ",".join(f"{v:+.0f}" for v in style_vec) + "]"
        proj_str  = "proj=ON" if use_proj else "proj=OFF"
        ax.set_title(f'{label}\n{style_str}  {proj_str}', fontsize=9,
                     fontweight='bold')

        # Static map underlay
        occ = ep['occ_map']
        if occ is not None and ep['has_map'] > 0.5:
            half = ep['map_extent'] / 2.0
            ax.imshow(occ, extent=[-half, half, -half, half],
                      origin='lower', cmap='gray_r', alpha=0.35, zorder=0,
                      interpolation='nearest')

        # Goal marker
        gx, gy = ep['goal_pos']
        goal_dot = mlines.Line2D([gx], [gy], color='red', marker='*',
                                  linestyle='None', markersize=11, label='Goal',
                                  zorder=5)
        ax.add_artist(goal_dot)

        # Initial positions
        s0 = padded_states[p][0]
        r_pos0 = s0[0].position
        h_pos0 = [s0[1][j].position for j in range(ep['n_humans'])]

        robot_circle = plt.Circle(r_pos0, ep['robot_radius'],
                                   fill=True, color=r_color, zorder=4)
        ax.add_artist(robot_circle)

        robot_trail, = ax.plot([], [], '--', color=r_color, lw=1.5,
                                alpha=0.7, zorder=1)

        h_circles = [
            plt.Circle(h_pos0[j], ep['human_radii'][j],
                       fill=False, color=cmap_human(j), zorder=3)
            for j in range(ep['n_humans'])
        ]
        h_labels = [
            ax.text(h_pos0[j][0] - x_off, h_pos0[j][1] - y_off, str(j),
                    color='black', fontsize=8, zorder=6)
            for j in range(ep['n_humans'])
        ]
        for hc in h_circles:
            ax.add_artist(hc)

        # Initial heading arrow
        arrows = []
        if ep['robot_kinematics'] == 'unicycle':
            th0 = s0[0].theta
            r0  = ep['robot_radius']
            arr = patches.FancyArrowPatch(
                r_pos0,
                (r_pos0[0] + r0 * np.cos(th0),
                 r_pos0[1] + r0 * np.sin(th0)),
                color='red', arrowstyle=arrow_style, zorder=4
            )
            ax.add_artist(arr)
            arrows.append(arr)

        # Trajectory lines
        K = 0
        for td in ep['predicted_trajs']:
            if isinstance(td, dict) and td.get('all_samples'):
                K = len(td['all_samples'])
                break
        bg_lines = [
            ax.plot([], [], '-', color='orange', lw=1.2, alpha=0.55, zorder=1)[0]
            for _ in range(max(K - 1, 0))
        ]
        sel_line,  = ax.plot([], [], '-', color='purple',  lw=2,   alpha=0.8,
                              zorder=2, label='Selected')
        proj_line, = ax.plot([], [], '-', color=_PROJ_COLOR, lw=2.5, alpha=0.9,
                              zorder=3, label='Projected')

        time_txt = ax.text(-7.0, 6.5, 'T: 0.00 s', fontsize=8, zorder=7)

        # Outcome annotation (shown on final frame)
        outcome = _outcome_label(ep['info'])
        if outcome == 'success':
            out_str = f'SUCCESS  {ep["global_time"]:.2f}s'
        else:
            out_str = outcome.upper()
        out_col  = _OUTCOME_COLORS[outcome]
        out_text = ax.text(0, 0, out_str, fontsize=11, fontweight='bold',
                            color=out_col, ha='center', va='center',
                            bbox=dict(facecolor='white', alpha=0.75,
                                      edgecolor=out_col, boxstyle='round'),
                            visible=False, zorder=10)

        ax.legend(handles=[robot_circle, goal_dot, sel_line, proj_line],
                  labels=['Robot', 'Goal', 'Selected', 'Projected'],
                  fontsize=7, loc='upper right')

        panels.append({
            'ax':           ax,
            'robot_circle': robot_circle,
            'robot_trail':  robot_trail,
            'h_circles':    h_circles,
            'h_labels':     h_labels,
            'arrows':       arrows,
            'bg_lines':     bg_lines,
            'sel_line':     sel_line,
            'proj_line':    proj_line,
            'time_txt':     time_txt,
            'out_text':     out_text,
            'ep_len':       len(ep['states']),
            'kinematics':   ep['robot_kinematics'],
            'robot_radius': ep['robot_radius'],
            'n_humans':     ep['n_humans'],
            'time_step':    ep['time_step'],
        })

    def _update(frame_num):
        for p, pd in enumerate(panels):
            # Hold last frame for episodes that ended earlier
            f  = min(frame_num, pd['ep_len'] - 1)
            st = padded_states[p][f]
            td = padded_trajs[p][f]

            ax = pd['ax']

            # Robot
            r_pos = st[0].position
            pd['robot_circle'].center = r_pos

            trail_x = [padded_states[p][k][0].position[0] for k in range(f + 1)]
            trail_y = [padded_states[p][k][0].position[1] for k in range(f + 1)]
            pd['robot_trail'].set_data(trail_x, trail_y)

            # Humans
            for j, hc in enumerate(pd['h_circles']):
                hp = st[1][j].position
                hc.center = hp
                pd['h_labels'][j].set_position((hp[0] - x_off, hp[1] - y_off))

            # Heading arrow
            for arr in pd['arrows']:
                arr.remove()
            pd['arrows'].clear()
            if pd['kinematics'] == 'unicycle':
                th = st[0].theta
                r0 = pd['robot_radius']
                arr = patches.FancyArrowPatch(
                    r_pos,
                    (r_pos[0] + r0 * np.cos(th),
                     r_pos[1] + r0 * np.sin(th)),
                    color='red',
                    arrowstyle=patches.ArrowStyle("->", head_length=4, head_width=2),
                    zorder=4,
                )
                ax.add_artist(arr)
                pd['arrows'].append(arr)

            # Time label
            pd['time_txt'].set_text(f'T: {f * pd["time_step"]:.2f} s')

            # Trajectory lines
            for bl in pd['bg_lines']:
                bl.set_data([], [])
            pd['sel_line'].set_data([], [])
            pd['proj_line'].set_data([], [])

            if isinstance(td, dict):
                sel    = td.get('selected_sample')
                proj   = td.get('projection')
                allsmp = td.get('all_samples') or []
                for i, bl in enumerate(pd['bg_lines']):
                    if i + 1 < len(allsmp):
                        s = np.asarray(allsmp[i + 1])
                        bl.set_data(s[:, 0], s[:, 1])
                if sel is not None:
                    sel = np.asarray(sel)
                    pd['sel_line'].set_data(sel[:, 0], sel[:, 1])
                if proj is not None:
                    proj = np.asarray(proj)
                    pd['proj_line'].set_data(proj[:, 0], proj[:, 1])

            # Show outcome label on the final synchronized frame
            if frame_num >= max_frames - 1:
                pd['out_text'].set_visible(True)

        return []   # blit=False — no need to return artist list

    anim = animation.FuncAnimation(
        fig, _update, frames=max_frames, interval=1000 // fps, blit=False
    )
    writer = animation.FFMpegWriter(
        fps=fps,
        metadata={'title': f'style_sweep_scene{scene_idx + 1}'},
        extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'],
    )
    anim.save(output_path, writer=writer)
    plt.close(fig)
    logging.info('  Saved scene video: %s', output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Style-sweep evaluation (6 variants, synchronized video)')
    parser.add_argument('--env_config',    type=str,
                        default='configs/env.config')
    parser.add_argument('--policy_config', type=str,
                        default='configs/policy.config')
    parser.add_argument('--policy',        type=str,
                        default='sogudiff')
    parser.add_argument('--gpu',           action='store_true', default=False)
    parser.add_argument('--phase',         type=str, default='test')
    parser.add_argument('--num_tests',     type=int, default=50,
                        help='Number of test scenes to evaluate')
    parser.add_argument('--video_file',    type=str,
                        default='style_sweep.mp4')
    parser.add_argument('--fps',           type=int, default=10)
    parser.add_argument('--results_suffix', type=str, default='',
                        help='Optional suffix for the results directory')
    # Scenario types (same as evaluate.py)
    parser.add_argument('--square',    action='store_true', default=False)
    parser.add_argument('--circle',    action='store_true', default=False)
    parser.add_argument('--npz_hard',  action='store_true', default=False)
    parser.add_argument('--npz_dir',   type=str,
                        default='../../data/scenes/eval_500')
    parser.add_argument('--npz_num_eval', type=int, default=50)
    parser.add_argument('--map_eval',  action='store_true', default=False)
    parser.add_argument('--map_eval_num', type=int, default=50)
    # ── SoGuDiff guidance overrides ──────────────────────────────────────────
    parser.add_argument('--infer_mode', type=str, default=None,
                        choices=['joint', 'per_axis'],
                        help='CFG mode for the diffusion policy (joint/per_axis). '
                             'Must match how the checkpoint was trained.')
    parser.add_argument('--cfg_w_style', type=str, default=None,
                        help="Per-axis guidance weights 'a,b,c,d' or one scalar.")
    parser.add_argument('--w_normalize', action='store_true',
                        help='per_axis: hold Σ_active w_i constant.')
    parser.add_argument('--no_video', action='store_true', default=False,
                        help='Metrics-only run: skip per-scene rendering and the '
                             'ffmpeg concat. Much faster and far smaller on disk; '
                             'summary.txt and per_scene_metrics.csv are identical. '
                             'Same flag name as the composition/sweep evaluators.')
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
    # Same race argument as ckpt_path above, for the MPPI frontier sweep:
    # mppi_budget lives in the shared [mppi_expert] block, and an array of
    # budget points must not depend on which value happens to be on disk when
    # each task calls configure(). Overrides are applied pre-configure() and
    # echoed to run_meta.json so every results dir records what actually ran.
    parser.add_argument('--mppi_budget', type=float, default=None,
                        help='Override [mppi_expert] mppi_budget — THE swept '
                             'frontier knob (fraction of the expert search).')
    parser.add_argument('--mppi_seed', type=int, default=None,
                        help='Override [mppi_expert] mppi_seed so each budget '
                             'point can be repeated under different search '
                             'noise (error bars).')
    # Parallelizm across scenes: the STYLE eval is 10 variants x scenes x ~32
    # ticks of MPPI per budget point, so wall time is bounded by sharding the
    # scene list over array tasks and merging per_scene_metrics.csv offline.
    parser.add_argument('--scene_shard', type=str, default=None,
                        help="Split the scene list across parallel jobs: "
                             "'j/N' runs every N-th scene starting at index j "
                             "(original numbering preserved; merging all N "
                             "shards reproduces the full scene set).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s, %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() and args.gpu
                          else 'cpu')
    logging.info('Device: %s', device)

    # ── Build and configure the policy ONCE ──────────────────────────────
    policy = policy_factory[args.policy]()
    policy_config = configparser.RawConfigParser()
    policy_config.read(args.policy_config)
    # Applied BEFORE configure(), which is what actually loads the weights.
    # Setting these on the policy afterwards would leave the config's checkpoint
    # loaded while the log claimed otherwise -- worse than no override at all.
    _sec = args.policy
    if not policy_config.has_section(_sec):
        raise ValueError(f'policy config has no section [{_sec}]')
    # (key, value, target section): ckpt/norm belong to the running policy's
    # own section; the MPPI knobs always target [mppi_expert] regardless of
    # which policy is being evaluated.
    _overrides = [('ckpt_path', args.ckpt_path, _sec),
                  ('norm_file', args.norm_file, _sec)]
    if args.mppi_budget is not None or args.mppi_seed is not None:
        if not policy_config.has_section('mppi_expert'):
            raise ValueError('--mppi_budget/--mppi_seed need an [mppi_expert] '
                             'section in the policy config')
        _overrides += [('mppi_budget', args.mppi_budget, 'mppi_expert'),
                       ('mppi_seed',  args.mppi_seed,  'mppi_expert')]
    for _key, _val, _tgt in _overrides:
        if _val is not None:
            policy_config.set(_tgt, _key, str(_val))
            logging.info('[override] %s.%s = %s', _tgt, _key, _val)
    # Audit trail tying this results directory to a specific model.
    logging.info(
        '[model] ckpt_path=%s  norm_file=%s',
        policy_config.get(_sec, 'ckpt_path') if policy_config.has_option(_sec, 'ckpt_path') else '?',
        policy_config.get(_sec, 'norm_file') if policy_config.has_option(_sec, 'norm_file') else '?')
    policy.configure(policy_config)

    # ── Apply guidance overrides (mode / weights) ────────────────────────────
    if args.infer_mode is not None and hasattr(policy, 'set_infer_mode'):
        policy.set_infer_mode(args.infer_mode)   # validates against trained_cfg_mode
    if args.w_normalize and hasattr(policy, 'cfg_w_normalize'):
        policy.cfg_w_normalize = True
    if args.cfg_w_style is not None and hasattr(policy, 'cfg_w_style'):
        vals = [float(v) for v in args.cfg_w_style.split(',') if v.strip()]
        _n_axes = len(getattr(policy, 'style_vector', [0, 0, 0, 0]))
        policy.cfg_w_style = vals * _n_axes if len(vals) == 1 else vals
    if hasattr(policy, 'cfg_infer_mode'):
        logging.info('Diffusion guidance: mode=%s w_normalize=%s',
                     policy.cfg_infer_mode, getattr(policy, 'cfg_w_normalize', False))

    # ── Build env and robot ───────────────────────────────────────────────
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

    # ── Build the scene list ──────────────────────────────────────────────
    if args.npz_hard:
        all_files = sorted(glob.glob(os.path.join(args.npz_dir, '*.npz')))
        hard_files = []
        for f in all_files:
            try:
                d = np.load(f, allow_pickle=True)
                difficulty = str(d['difficulty']) if 'difficulty' in d else None
                threat     = str(d['threat_type']) if 'threat_type' in d else None
                # Accept diverseset_v4-style "hard" labels AND any
                # hand-crafted scene that carries a threat_type at all
                # (e.g. the style-probe scenes in scenegen/ use
                # their own threat_type vocabulary: head_on, cutoff,
                # sneak_up, group_head_on, composite, ...).
                if (difficulty == 'hard' or threat is not None):
                    hard_files.append(f)
            except Exception as e:
                logging.warning('Could not load %s: %s', f, e)
        if not hard_files:
            raise ValueError(f'No hard npz files found in {args.npz_dir}')
        hard_files = hard_files[:args.npz_num_eval]
        scene_files = hard_files
        num_scenes  = len(hard_files)
        logging.info('NPZ hard: %d scenes', num_scenes)
    elif args.map_eval:
        scene_files = [None] * args.map_eval_num
        num_scenes  = args.map_eval_num
        logging.info('Map-eval: %d scenes', num_scenes)
    else:
        scene_files = [None] * args.num_tests
        num_scenes  = args.num_tests
        logging.info('Standard: %d scenes', num_scenes)

    # ── Optional scene sharding (parallel array tasks) ────────────────────
    # 'j/N' keeps every N-th scene of the deterministic (sorted, truncated)
    # scene list starting at index j. Original numbering is preserved —
    # scene_idx stays the position in the FULL list — so a merged sweep has
    # the same scene ids as an unsharded run, each shard sees a
    # representative stride of the difficulty mix, and identical shard specs
    # across budgets guarantee every frontier point evaluates the same scenes.
    # Within-scene variant identity is untouched: the per-scene RNG-anchor
    # snapshot in run_one_episode() still synchronizes all 10 variants.
    _shard_idx = _shard_cnt = None
    if args.scene_shard:
        try:
            _si, _sn = args.scene_shard.split('/')
            _shard_idx, _shard_cnt = int(_si), int(_sn)
        except ValueError:
            raise ValueError(f"--scene_shard must look like '3/8', "
                             f"got {args.scene_shard!r}") from None
        if _shard_cnt < 1 or not (0 <= _shard_idx < _shard_cnt):
            raise ValueError(f'shard index must be in [0, {_shard_cnt}), '
                             f"got {args.scene_shard!r}")
        scene_entries = [(i, f) for i, f in enumerate(scene_files)
                         if i % _shard_cnt == _shard_idx]
        logging.info('Shard %d/%d: %d of %d scenes',
                     _shard_idx, _shard_cnt, len(scene_entries), num_scenes)
    else:
        scene_entries = list(enumerate(scene_files))

    # ── Results directory ─────────────────────────────────────────────────
    tag         = f'_styles_{args.results_suffix}' if args.results_suffix \
                  else '_styles'
    if _shard_cnt is not None:
        # Per-shard results dir: parallel tasks must never share one
        # per_scene_metrics.csv (last writer wins). The aggregator globs the
        # __shardIofN marker and merges.
        tag += f'__shard{_shard_idx}of{_shard_cnt}'
    base_video, ext = os.path.splitext(args.video_file)
    results_dir = f'results{tag}'
    os.makedirs(results_dir, exist_ok=True)
    tmp_dir = os.path.join(results_dir, 'tmp_scenes')
    os.makedirs(tmp_dir, exist_ok=True)

    # ── Per-variant metric accumulators ──────────────────────────────────
    metrics = {label: {
        'successes':   [], 'timeouts':      [], 'collisions': [],
        'wall_colls':  [], 'ep_times':      [], 'path_lens':  [],
        'min_dists':   [], 'avg_min_dists': [],
        # Social metrics for cross-variant comparison
        'mean_min_clearance_m':  [],
        'psi_rate':              [],
        'passing_side_bias':     [],
        'ttc_infraction_rate':   [],
        'path_cutoff_rate':      [],
        'group_split_rate':      [],
    } for label, _, _ in STYLE_VARIANTS}

    all_scene_videos = []
    csv_rows = []   # one row per (scene, variant)

    # ── Scene loop ────────────────────────────────────────────────────────
    for scene_idx, npz_entry in scene_entries:
        npz_path = npz_entry if npz_entry else None
        logging.info('Scene %d / %d  (npz=%s)',
                     scene_idx + 1, len(scene_entries),
                     os.path.basename(npz_path) if npz_path else 'generated')

        # Snapshot the RNG anchor BEFORE any variant touches the env.
        # sim.testoffset increments on every reset() call regardless of
        # test_case, so without restoring it each variant would receive a
        # different seed and thus a completely different scene.
        scene_testoffset = sim.testoffset
        scene_counter    = sim.case_counter[args.phase]

        episodes = []

        for v_idx, (label, style_vec, use_proj) in enumerate(STYLE_VARIANTS):
            logging.info('  Variant %d/%d: %s  style=%s  proj=%s',
                         v_idx + 1, N_VARIANTS, label, style_vec, use_proj)
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

            # Accumulate metrics
            m = metrics[label]
            info = ep['info']
            m['successes'].append(1 if isinstance(info, ReachGoal)      else 0)
            m['timeouts'].append(1   if isinstance(info, Timeout)        else 0)
            m['collisions'].append(1 if isinstance(info, Collision)      else 0)
            try:
                m['wall_colls'].append(1 if isinstance(info, WallCollision) else 0)
            except NameError:
                m['wall_colls'].append(0)
            if isinstance(info, ReachGoal):
                m['ep_times'].append(ep['global_time'])
                m['path_lens'].append(ep['pathlength'])
            m['min_dists'].append(ep['minobsdist'])
            m['avg_min_dists'].append(
                np.mean(ep['avgobsdist']) if ep['avgobsdist'] else float('nan')
            )
            # Social metrics
            sm = ep['social_metrics']
            for key in ['mean_min_clearance_m', 'psi_rate',
                        'passing_side_bias', 'ttc_infraction_rate',
                        'path_cutoff_rate', 'group_split_rate']:
                m[key].append(sm[key])

            csv_rows.append({
                'scene':      scene_idx + 1,
                'variant':    label,
                'projection': use_proj,
                'style':      str(style_vec),
                'outcome':    _outcome_label(info),
                'time':       f'{ep["global_time"]:.3f}',
                'path_len':   f'{ep["pathlength"]:.3f}',
                'min_dist':   f'{ep["minobsdist"]:.3f}',
                # Social metrics
                'mean_min_clearance_m':  f'{sm["mean_min_clearance_m"]:.3f}',
                'psi_rate':              f'{sm["psi_rate"]:.4f}',
                'passing_side_bias':     f'{sm["passing_side_bias"]:+.3f}',
                'ttc_infraction_rate':   f'{sm["ttc_infraction_rate"]:.4f}',
                'path_cutoff_rate':      f'{sm["path_cutoff_rate"]:.4f}',
                'group_split_rate':      f'{sm["group_split_rate"]:.4f}',
            })

            logging.info('    → %s  t=%.2fs  path=%.2f  minD=%.3f',
                         _outcome_label(info), ep['global_time'],
                         ep['pathlength'], ep['minobsdist'])

        # ── Build synchronized 6-panel video for this scene ──────────────
        # Rendering dominates wall clock on a metrics-only run (one animation
        # per scene, then an ffmpeg concat) and the output is large. --no_video
        # skips both; the metrics and per_scene_metrics.csv are unaffected.
        if not args.no_video:
            scene_video = os.path.join(tmp_dir, f'scene_{scene_idx + 1:04d}{ext}')
            build_scene_video(episodes, scene_idx, scene_video, fps=args.fps)
            all_scene_videos.append(scene_video)

    # ── Concatenate all scene videos ─────────────────────────────────────
    combined_video = os.path.join(results_dir, os.path.basename(args.video_file))
    if all_scene_videos and not args.no_video:
        list_file = os.path.join(tmp_dir, 'video_list.txt')
        with open(list_file, 'w') as fh:
            for vid in all_scene_videos:
                fh.write(f"file '{os.path.abspath(vid)}'\n")
        subprocess.run(
            [ffmpeg_binary(), '-y', '-f', 'concat', '-safe', '0',
             '-i', list_file, '-c', 'copy', combined_video],
            check=False,
        )
        logging.info('Combined video: %s', combined_video)

    # ── Write per-scene CSV ───────────────────────────────────────────────
    csv_path = os.path.join(results_dir, 'per_scene_metrics.csv')
    fieldnames = ['scene', 'variant', 'projection', 'style',
                  'outcome', 'time', 'path_len', 'min_dist',
                  'mean_min_clearance_m', 'psi_rate',
                  'passing_side_bias', 'ttc_infraction_rate',
                  'path_cutoff_rate', 'group_split_rate']
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    logging.info('Per-scene CSV: %s', csv_path)

    # ── Write aggregate summary ───────────────────────────────────────────
    summary_path = os.path.join(results_dir, 'summary.txt')
    with open(summary_path, 'w') as fh:
        fh.write('=' * 60 + '\n')
        fh.write('Style-Sweep Evaluation Summary\n')
        # Every percentage below is over THIS process's episodes. Under
        # --scene_shard that is the shard's scene count, not the full set —
        # reporting num_scenes here made a 32-scene shard's summary claim
        # "Scenes evaluated: 500" while its SR% was computed over 32.
        fh.write(f'Scenes evaluated: {len(scene_entries)}'
                 + (f'  (shard {_shard_idx}/{_shard_cnt} of the '
                    f'{num_scenes}-scene set; merge all shards for full-set '
                    f'numbers)' if _shard_cnt is not None else '') + '\n')
        fh.write('=' * 60 + '\n\n')

        header = (f"{'Variant':<20} {'SR%':>6} {'TO%':>6} {'CR%':>6} "
                  f"{'WallCR%':>8} {'AvgT':>7} {'AvgPL':>7} "
                  f"{'AvgMinD':>8}\n")
        fh.write(header)
        fh.write('-' * len(header) + '\n')

        for label, _, _ in STYLE_VARIANTS:
            m = metrics[label]
            sr   = 100 * np.mean(m['successes'])   if m['successes']   else float('nan')
            to   = 100 * np.mean(m['timeouts'])    if m['timeouts']    else float('nan')
            cr   = 100 * np.mean(m['collisions'])  if m['collisions']  else float('nan')
            wcr  = 100 * np.mean(m['wall_colls'])  if m['wall_colls']  else float('nan')
            avgt = np.mean(m['ep_times'])           if m['ep_times']    else float('nan')
            avpl = np.mean(m['path_lens'])          if m['path_lens']   else float('nan')
            avmd = np.mean(m['min_dists'])          if m['min_dists']   else float('nan')
            fh.write(f"{label:<20} {sr:6.1f} {to:6.1f} {cr:6.1f} "
                     f"{wcr:8.1f} {avgt:7.3f} {avpl:7.3f} {avmd:8.3f}\n")

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
        fh.write('Expected direction per axis:\n')
        fh.write('  prox+  → higher mean_min_clearance_m, lower psi_rate\n')
        fh.write('  prox-  → lower mean_min_clearance_m, higher psi_rate\n')
        fh.write('  pass+  → passing_side_bias toward +1 (right-hand traffic)\n')
        fh.write('  pass-  → passing_side_bias toward -1 (left-hand traffic)\n')
        fh.write('  yield+ → lower ttc_infraction_rate, lower path_cutoff_rate\n')
        fh.write('  yield- → higher ttc_infraction_rate, higher path_cutoff_rate\n')
        fh.write('  group+ → lower group_split_rate\n')
        fh.write('  group- → higher group_split_rate\n\n')

        soc_header = (
            f"{'Variant':<20} "
            f"{'MinClear':>9} {'PSIRate':>8} "
            f"{'PassBias':>9} "
            f"{'TTCInfR':>8} {'CutoffR':>8} "
            f"{'GrpSplit':>9}\n"
        )
        fh.write(soc_header)
        fh.write('-' * len(soc_header) + '\n')

        for label, _, _ in STYLE_VARIANTS:
            m = metrics[label]
            def _avg(k): return np.nanmean(m[k]) if m[k] else float('nan')
            fh.write(
                f"{label:<20} "
                f"{_avg('mean_min_clearance_m'):9.3f} "
                f"{_avg('psi_rate'):8.4f} "
                f"{_avg('passing_side_bias'):+9.3f} "
                f"{_avg('ttc_infraction_rate'):8.4f} "
                f"{_avg('path_cutoff_rate'):8.4f} "
                f"{_avg('group_split_rate'):9.4f}\n"
            )

        # ── 95% bootstrap CIs, and the SUPPORT count ─────────────────────────
        # Kept in a separate block rather than inlined into the table above, so
        # the headline table stays readable and this stays available for
        # deciding which differences are real.
        #
        # 'support' = scenes where the metric is nonzero. It is the number that
        # explains an unmeasurable axis: a rate that fires in 9 of 100 scenes
        # cannot resolve a style effect no matter how many scenes are added,
        # because the scenes themselves do not admit the behavior.
        fh.write('\n\n95% bootstrap CIs over scenes (10k resamples) '
                 'and nonzero-support counts\n')
        ci_header = (f"{'Variant':<20} {'MinClear':>20} {'PassBias':>20} "
                     f"{'TTCInfR':>20} {'GrpSplit':>20}\n")
        fh.write(ci_header)
        fh.write('-' * len(ci_header) + '\n')
        for label, _, _ in STYLE_VARIANTS:
            m = metrics[label]
            cells = []
            for k in ('mean_min_clearance_m', 'passing_side_bias',
                      'ttc_infraction_rate', 'group_split_rate'):
                lo, hi, n_nz, n = _bootstrap_ci(m[k])
                cells.append(f"[{lo:+.3f},{hi:+.3f}] {n_nz}/{n}")
            fh.write(f"{label:<20} " + ' '.join(f"{c:>20}" for c in cells) + '\n')

    logging.info('Summary: %s', summary_path)

    # Print to console as well
    with open(summary_path) as fh:
        print(fh.read())

    # ── Machine-readable run record (sweep aggregation input) ────────────
    # Metrics alone don't say WHERE on the compute frontier this run sits:
    # the MPPI budget actually used (post-override), its rollout
    # decomposition, the per-tick latency distribution and search-failure
    # counts live only on the policy object. Persist them beside the CSV so
    # a frontier can be assembled by globbing results dirs, not scraping
    # slurm logs. For diffusion runs this instead records the sample budget
    # (num_samples x num_inference_steps) and guidance configuration, which
    # is that policy's coordinate on the same figure.
    def _jsonable(v):
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, float) and not np.isfinite(v):
            return str(v)
        return v

    meta = {
        'policy':       args.policy,
        'args':         {k: _jsonable(v) for k, v in vars(args).items()},
        'scene_shard':  ({'index': _shard_idx, 'count': _shard_cnt}
                         if _shard_cnt is not None else None),
        'scenes_run':   len(scene_entries),
        'scene_files':  [os.path.basename(f) if f else None
                         for _, f in scene_entries],
    }
    for attr in ('mppi_budget', 'mppi_seed', 'mppi_samples', 'mppi_iters',
                 'mppi_restarts', 'mppi_seed_pool',
                 'num_samples', 'num_inference_steps',
                 'trained_cfg_mode', 'cfg_infer_mode',
                 'cfg_w_style', 'cfg_w_normalize'):
        if hasattr(policy, attr):
            v = getattr(policy, attr)
            meta[attr] = (v.tolist() if isinstance(v, np.ndarray) else _jsonable(v))
    if hasattr(policy, 'timing_summary'):
        meta['timing'] = {k: _jsonable(v)
                          for k, v in policy.timing_summary(skip=10).items()}
    meta_path = os.path.join(results_dir, 'run_meta.json')
    with open(meta_path, 'w') as fh:
        json.dump(meta, fh, indent=2, default=str)
    logging.info('Run metadata: %s', meta_path)

    # Raw per-tick latency arrays: medians/percentiles don't merge from
    # per-shard summary stats, so keep the samples for exact pooled values.
    if hasattr(policy, 'timing'):
        np.savez(os.path.join(results_dir, 'timing_raw.npz'),
                 timing=np.asarray(list(policy.timing), dtype=np.float64),
                 timing_sampler=(np.asarray(list(policy.timing_sampler),
                                            dtype=np.float64)
                                 if hasattr(policy, 'timing_sampler')
                                 else np.array([])),
                 timing_ok=(np.asarray(list(policy._timing_ok), dtype=bool)
                            if hasattr(policy, '_timing_ok')
                            else np.array([])),
                 n_search_fail_geo=int(getattr(policy, 'n_search_fail_geo', -1)),
                 n_search_fail_cand=int(getattr(policy, 'n_search_fail_cand', -1)))


if __name__ == '__main__':
    main()