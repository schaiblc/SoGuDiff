"""SoGuDiff: style-conditioned diffusion planning with a feasibility layer.

The policy denoises a horizon of waypoints conditioned on three things: the
surrounding crowd, a static occupancy map, and a four-dimensional style vector
over proxemic distance, passing side, yielding and group deference.
Classifier-free guidance can be applied jointly over the whole style vector or
per axis (``cfg_infer_mode``), the latter being what the paper reports.

Per control step:

1. **Sample.** ``num_samples`` trajectories are drawn with DDIM over
   ``num_inference_steps``, guided by the configured style vector.
2. **Score.** Each candidate is ranked by a weighted sum of collision risk,
   smoothness, goal progress and control effort -- the ``*_coef`` keys in the
   policy config.
3. **Project.** The winner is projected onto the feasible, collision-free set
   by an acados SQP solve over the robot's unicycle dynamics, so the executed
   trajectory respects speed, turn-rate and acceleration limits rather than
   merely approximating them.
4. **Fall back if needed.** A controlled stop is issued when the projection
   fails, leaves significant slack violation, or deviates from the sampled
   reference beyond ``proj_max_dev_thresh``.

Warm-state conditioning (``_v0_from_last_step`` / ``_w0_from_last_step``) is
taken from the *projected* trajectory rather than the raw diffusion output, so
the next step is conditioned on what the robot actually executed.

Sections marked ``-- PROJECTION --`` implement steps 3 and 4. The projection
layer is the only part that needs acados; see docs/INSTALL.md. Setting
``use_projection = false`` in the config gives the unprojected ablation.

Configuration lives under ``[sogudiff]`` in
``configs/policy.config``, which documents each key.
"""

from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY, ActionRot
import torch
import numpy as np
import math
from diffusers import DDPMScheduler, DDIMScheduler
from diffusers import UNet1DModel
import torch.nn as nn
from datetime import datetime
import torch.nn.functional as F

import sys
import os
import time
import contextlib

from crowd_nav.paths import repo_root, resolve_path

# Repository root, for the vendored diffusers_unet_1d_condition fork.
sys.path.append(repo_root())

from diffusers_unet_1d_condition import UNet1DConditionModel

# ── PROJECTION ─────────────────────────────────────────────────────────
# This directory, for the sibling projection_solver / unicycle model modules.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from projection_solver import (
    build_projection_solver,
    pack_obstacle_params,
    solve_projection,
    kinematic_warm_start_from_xy,
)
# ───────────────────────────────────────────────────────────────────────

# ── SOCIAL STYLE ───────────────────────────────────────────────────────
# Canonical axis ordering must match the training script exactly.
STYLE_AXES    = ["prox", "pass", "yield", "group"]
N_STYLE_AXES  = len(STYLE_AXES)
# ───────────────────────────────────────────────────────────────────────


# =====================================================================
# (Token embedder, helper losses — unchanged)
# =====================================================================

class MapEncoder(nn.Module):
    """
    Encodes a [B, 1, H, W] ego-centered occupancy map into a sequence of
    spatially-aware tokens with 2D positional embeddings.

    Default: 50x50 input → 7x7 feature map → 49 tokens of dim `token_dim`.
    """
    def __init__(self, token_dim, in_size=50, base_channels=32):
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        # 3 stride-2 stages: 50 -> 25 -> 13 -> 7 (with padding=1)
        self.cnn = nn.Sequential(
            nn.Conv2d(1,  c1, kernel_size=3, stride=2, padding=1),  # 50 -> 25
            nn.GroupNorm(8, c1),
            nn.SiLU(),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),  # 25 -> 13
            nn.GroupNorm(8, c2),
            nn.SiLU(),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),  # 13 -> 7
            nn.GroupNorm(8, c3),
            nn.SiLU(),
        )
        # Compute output spatial size with a dummy pass
        with torch.no_grad():
            dummy = torch.zeros(1, 1, in_size, in_size)
            feat = self.cnn(dummy)
            _, _, self.h_out, self.w_out = feat.shape
        self.n_tokens = self.h_out * self.w_out

        # Project CNN channels to token_dim
        self.proj = nn.Linear(c3, token_dim)

        # 2D positional embedding (learned, one per cell)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.n_tokens, token_dim) * 0.02
        )
        # Distinct type embedding for map tokens
        self.type_embed = nn.Parameter(torch.randn(1, 1, token_dim) * 0.02)

        # Learned "null map" tokens for when no map is provided
        self.null_map_tokens = nn.Parameter(
            torch.randn(1, self.n_tokens, token_dim) * 0.02
        )

        self.norm = nn.LayerNorm(token_dim)

    def forward(self, occ_map, has_map):
        """
        occ_map: [B, 1, H, W]   binary
        has_map: [B]            1.0 if map present, 0.0 otherwise
        returns: [B, n_tokens, token_dim]
        """
        B = occ_map.shape[0]
        feat = self.cnn(occ_map)                  # [B, C, h, w]
        feat = feat.flatten(2).transpose(1, 2)    # [B, h*w, C]
        tokens = self.proj(feat)                  # [B, n_tokens, token_dim]
        tokens = tokens + self.pos_embed + self.type_embed

        # Replace with null tokens where has_map == 0
        gate = has_map.view(B, 1, 1)              # [B, 1, 1]
        tokens = gate * tokens + (1.0 - gate) * self.null_map_tokens.expand(B, -1, -1)

        return self.norm(tokens)


# =============================================================================
# StyleTokenEmbedder — one MLP per social-style axis.
# Matches the training script exactly so checkpoint weights load correctly.
# =============================================================================
class StyleTokenEmbedder(nn.Module):
    def __init__(self, axis_names, token_dim):
        super().__init__()
        self.axis_names = list(axis_names)
        self.n_axes = len(self.axis_names)
        self.token_dim = token_dim
        hidden_dim = token_dim
        self.axis_mlps = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, token_dim),
            )
            for name in self.axis_names
        })
        self.type_embed  = nn.Parameter(torch.randn(self.n_axes, token_dim) * 0.02)
        self.null_tokens = nn.Parameter(torch.randn(self.n_axes, token_dim) * 0.02)
        self.final_norm  = nn.LayerNorm(token_dim)

    def forward(self, style_values, style_drop_mask):
        """
        style_values:    [B, n_axes]  floats in ~[-1, 1]
        style_drop_mask: [B, n_axes]  1.0 = replace with null token, 0.0 = use real
        returns:         [B, n_axes, token_dim]
        """
        B = style_values.shape[0]
        out = []
        for i, name in enumerate(self.axis_names):
            scalar = style_values[:, i:i + 1]
            tok    = self.axis_mlps[name](scalar) + self.type_embed[i]
            null   = self.null_tokens[i].unsqueeze(0).expand(B, -1)
            drop   = style_drop_mask[:, i:i + 1]
            tok    = drop * null + (1.0 - drop) * tok
            out.append(tok.unsqueeze(1))
        out = torch.cat(out, dim=1)   # [B, n_axes, D]
        return self.final_norm(out)


# =============================================================================
# SceneTokenEmbedder — updated to match the training script:
#   • Per-section null tokens (start, goal, obs) instead of a monolithic null_token.
#   • Accepts style_tokens from StyleTokenEmbedder and appends them.
#   • Accepts scene_drop (per-sample {0,1} flag) for CFG unconditional path.
#   • Returns (tokens, valid.float()) so callers can see the validity mask.
# =============================================================================
class SceneTokenEmbedder(nn.Module):
    def __init__(self, start_dim, goal_dim, obs_dim, token_dim, k_max,
                 n_self_attn_layers=2, map_size=50, use_map=True):
        super().__init__()
        hidden_dim = token_dim
        self.use_map = use_map
        self.k_max   = k_max

        def make_mlp(input_dim):
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, token_dim),
            )

        self.start_proj = make_mlp(start_dim)
        self.goal_proj  = make_mlp(goal_dim)
        self.obs_proj   = make_mlp(obs_dim)
        self.type_embed = nn.Parameter(torch.randn(3, token_dim) * 0.02)

        if use_map:
            self.map_encoder = MapEncoder(token_dim=token_dim, in_size=map_size)
            n_map_tokens = self.map_encoder.n_tokens
        else:
            self.map_encoder = None
            n_map_tokens = 0
        self.n_map_tokens = n_map_tokens

        # Per-section null tokens for CFG scene dropout.
        self.null_start_token = nn.Parameter(torch.randn(1, 1,     token_dim) * 0.02)
        self.null_goal_token  = nn.Parameter(torch.randn(1, 1,     token_dim) * 0.02)
        self.null_obs_tokens  = nn.Parameter(torch.randn(1, k_max, token_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=4,
            dim_feedforward=token_dim * 4,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.scene_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_self_attn_layers
        )
        self.final_norm = nn.LayerNorm(token_dim)

    def forward(self, start, goal, obstacles, obs_mask=None,
                occ_map=None, has_map=None,
                style_tokens=None,   # [B, n_axes, D] from StyleTokenEmbedder
                scene_drop=None,     # [B] float in {0., 1.} — 1.0 = drop all scene
                ):
        """
        Returns (tokens [B, N_tok, D], valid [B, N_tok] float).
        Token layout: [ start | goal | obstacles | (map) | (style) ]
        """
        B      = start.shape[0]
        device = start.device

        if scene_drop is None:
            scene_drop = torch.zeros(B, device=device)
        sd = scene_drop.view(B, 1, 1)

        start_tok = sd * self.null_start_token + (1.0 - sd) * self.start_proj(start).unsqueeze(1)
        start_tok = start_tok + self.type_embed[0]

        goal_tok  = sd * self.null_goal_token  + (1.0 - sd) * self.goal_proj(goal).unsqueeze(1)
        goal_tok  = goal_tok  + self.type_embed[1]

        obs_tok   = sd * self.null_obs_tokens  + (1.0 - sd) * self.obs_proj(obstacles)
        obs_tok   = obs_tok   + self.type_embed[2]

        tokens = torch.cat([start_tok, goal_tok, obs_tok], dim=1)   # [B, 2+K, D]

        # Validity mask — when scene is dropped, null obs tokens are still valid.
        scene_drop_bool = scene_drop.bool().unsqueeze(1)   # [B, 1]
        if obs_mask is not None:
            obs_valid = obs_mask.bool() | scene_drop_bool  # [B, k_max]
        else:
            obs_valid = torch.ones(B, self.k_max, device=device, dtype=torch.bool)
        valid = torch.cat(
            [torch.ones(B, 2, device=device, dtype=torch.bool), obs_valid], dim=1
        )

        if self.use_map:
            assert occ_map is not None and has_map is not None, \
                "use_map=True requires occ_map and has_map"
            effective_has_map = has_map * (1.0 - scene_drop)
            map_tokens = self.map_encoder(occ_map, effective_has_map)
            tokens = torch.cat([tokens, map_tokens], dim=1)
            valid  = torch.cat(
                [valid, torch.ones(B, self.n_map_tokens, device=device, dtype=torch.bool)],
                dim=1,
            )

        if style_tokens is not None:
            n_style = style_tokens.shape[1]
            tokens = torch.cat([tokens, style_tokens], dim=1)
            valid  = torch.cat(
                [valid, torch.ones(B, n_style, device=device, dtype=torch.bool)],
                dim=1,
            )

        key_padding_mask = ~valid
        tokens = self.scene_encoder(tokens, src_key_padding_mask=key_padding_mask)
        return self.final_norm(tokens), valid.float()


def build_attn_mask(obs_mask, n_map_tokens=0, n_style_tokens=0):
    """
    obs_mask:       [B, K]
    n_map_tokens:   int — map tokens after obstacles
    n_style_tokens: int — style tokens after map tokens
    """
    B, K = obs_mask.shape
    device = obs_mask.device
    base = torch.ones(B, 2, device=device)   # start + goal always valid
    pieces = [base, obs_mask]
    if n_map_tokens > 0:
        pieces.append(torch.ones(B, n_map_tokens, device=device))
    if n_style_tokens > 0:
        pieces.append(torch.ones(B, n_style_tokens, device=device))
    return torch.cat(pieces, dim=1)


def sinusoidal_pos_emb(T, dim, device):
    pos = torch.arange(T, device=device)
    div = torch.exp(
        torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim)
    )
    emb = torch.zeros(T, dim, device=device)
    emb[:, 0::2] = torch.sin(pos[:, None] * div)
    emb[:, 1::2] = torch.cos(pos[:, None] * div)
    return emb


def move_scheduler_to_device(scheduler, device):
    for name, value in scheduler.__dict__.items():
        if torch.is_tensor(value):
            scheduler.__dict__[name] = value.to(device)
    return scheduler


def control_effort_penalty(x_trajs_denorm):
    pos = x_trajs_denorm[:, :, 0:2, :]
    vel = pos[:, :, :, 1:] - pos[:, :, :, :-1]
    forward_effort = vel[:, :, 0, :].pow(2)
    lateral_effort = vel[:, :, 1, :].pow(2)
    effort = forward_effort + 3.0 * lateral_effort
    return effort.mean(dim=2)


def collision_cost(x_trajs, obstacles, obs_mask, dt=0.05, safety_radius=0.5):
    B, K, D, T = x_trajs.shape
    traj_xy = x_trajs[:, :, 0:2, :].permute(0, 1, 3, 2)
    obs_pos = obstacles[:, :, 0:2].unsqueeze(2)
    obs_vel = obstacles[:, :, 2:4].unsqueeze(2)
    t_vec   = torch.arange(T, device=x_trajs.device).float() * dt
    t_vec   = t_vec.view(1, 1, T, 1)
    obs_future = obs_pos + obs_vel * t_vec
    traj_xy    = traj_xy.unsqueeze(2)
    obs_future = obs_future.unsqueeze(1)
    dist       = torch.norm(traj_xy - obs_future, dim=-1)
    penetration = F.relu(safety_radius - dist)
    penalty     = penetration ** 2
    obs_mask_exp = obs_mask.unsqueeze(1).unsqueeze(-1)
    cost = (penalty * obs_mask_exp).sum(dim=[2, 3])
    return cost


def smoothness_penalty(x_trajs_denorm):
    pos     = x_trajs_denorm[:, :, 0:2, :]
    vel     = pos[:, :, :, 1:] - pos[:, :, :, :-1]
    acc     = vel[:, :, :, 1:] - vel[:, :, :, :-1]
    acc_mag = acc.pow(2).sum(dim=2)
    return acc_mag.pow(1.5).mean(dim=2)


def goal_progress_reward(x_trajs_denorm, goal, goal_mean, goal_std):
    device = x_trajs_denorm.device
    goal_mean_t = torch.as_tensor(goal_mean[0:2], dtype=torch.float32, device=device)
    goal_std_t  = torch.as_tensor(goal_std[0:2],  dtype=torch.float32, device=device)
    goal_xy = (goal[:, 0:2] * goal_std_t + goal_mean_t)
    goal_xy = goal_xy.unsqueeze(1).unsqueeze(3)
    traj_xy       = x_trajs_denorm[:, :, 0:2, :]
    dist_to_goal  = torch.norm(traj_xy - goal_xy, dim=2)
    progress      = dist_to_goal[:, :, 0] - dist_to_goal[:, :, -1]
    step_progress = (dist_to_goal[:, :, :-1] - dist_to_goal[:, :, 1:]).mean(dim=2)
    return progress + 0.5 * step_progress


def score_trajectories(x_trajs, obstacles, obs_mask, goal, norm,
                       collision_coef, smooth_pen_coef, goal_reward_coef, control_effort_coef,
                       safety_radius=0.5, dt=0.05,
                       static_obs_xy=None):
    """
    Score K diffusion trajectory samples.

    Parameters
    ----------
    x_trajs       : (B, K, traj_dim, T) normalized samples from the diffusion
    obstacles     : (B, k_max, 4)  normalized dynamic human states (px,py,vx,vy)
    obs_mask      : (B, k_max)     1 = valid human slot, 0 = padding
    static_obs_xy : (B, M, 2) physical ego-frame positions of static map cells,
                    or None.  Zero velocity is assumed — positions stay fixed
                    across the horizon.  Penalized with the same collision_coef.

    Returns
    -------
    rewards  : (B, K)
    best_idx : (B,)
    """
    traj_std  = torch.as_tensor(norm["traj_std"], dtype=torch.float32, device=x_trajs.device)
    traj_mean = torch.as_tensor(norm["traj_mean"], dtype=torch.float32, device=x_trajs.device)
    obs_std   = torch.as_tensor(norm["obs_std"],  dtype=torch.float32, device=x_trajs.device)
    obs_mean  = torch.as_tensor(norm["obs_mean"], dtype=torch.float32, device=x_trajs.device)
    obs_denorm = obstacles * obs_std + obs_mean

    x_denorm = x_trajs * traj_std.view(1, 1, -1, 1) + traj_mean.view(1, 1, -1, 1)

    # ── Dynamic human collision cost ────────────────────────────────────
    C_dyn = collision_cost(x_denorm, obs_denorm, obs_mask, dt=dt, safety_radius=safety_radius)

    # ── Static map cell collision cost ──────────────────────────────────
    # static_obs_xy is in physical ego-frame meters; zero velocity means
    # obs_future = obs_pos for every horizon node — collision_cost handles
    # this naturally when obs_vel = 0.
    C_static = torch.zeros_like(C_dyn)
    if static_obs_xy is not None and static_obs_xy.shape[1] > 0:
        B_s, M_s, _ = static_obs_xy.shape
        zeros_vel    = torch.zeros(B_s, M_s, 2,
                                   device=static_obs_xy.device,
                                   dtype=static_obs_xy.dtype)
        static_obs_4d = torch.cat([static_obs_xy, zeros_vel], dim=-1)   # (B, M, 4)
        static_mask   = torch.ones(B_s, M_s, device=static_obs_xy.device)
        C_static = collision_cost(x_denorm, static_obs_4d, static_mask,
                                  dt=dt, safety_radius=safety_radius)

    smooth_pen = smoothness_penalty(x_denorm)
    goal_rew   = goal_progress_reward(x_denorm, goal, norm["goal_mean"], norm["goal_std"])
    effort_pen = control_effort_penalty(x_denorm)

    rewards = (
        - collision_coef      * (C_dyn + C_static)   # same weight for both types
        - smooth_pen_coef     * smooth_pen
        - control_effort_coef * effort_pen
        + goal_reward_coef    * goal_rew
    )

    # A trajectory is "safe" only if it avoids BOTH dynamic agents AND walls.
    EPS = 1e-6
    is_safe  = (C_dyn < EPS) & (C_static < EPS)
    has_safe = is_safe.any(dim=1)
    rewards_masked = rewards.masked_fill(~is_safe, float('-inf'))
    best_idx_safe = rewards_masked.argmax(dim=1)
    best_idx_all  = rewards.argmax(dim=1)
    best_idx = torch.where(has_safe, best_idx_safe, best_idx_all)

    return rewards, best_idx


# =====================================================================
# Policy
# =====================================================================

class SoGuDiff(Policy):
    def __init__(self):
        super().__init__()
        self.model = None
        self.token_embedder = None
        self.diffusion = None
        self.norm = None
        self.device = None
        self.horizon = None
        self.k_max = None
        self.start_dim = None
        self.goal_dim = None
        self.kinematics = None
        self.enforce_lims = False
        self.max_vel = float('inf')
        self.max_wrot = float('inf')
        self.max_accel = float('inf')
        self.max_w_accel = float('inf')
        self.multiagent_training = None
        self.predicted_traj = None
        self.name = "SoGuDiff"
        self.prev_v = 0.0
        self.prev_omega = 0.0

        # ── TIMING ───────────────────────────────────────────────────────
        # Per-call CANDIDATE-GENERATION latency in seconds: everything from
        # receiving the state to having one committed candidate trajectory,
        # EXCLUDING the shared projection back-end (both policies pay that
        # identically). This is the x-axis of the latency/fidelity frontier.
        #
        # The MPPI baseline records the SAME quantity under the same
        # definition, which is what makes the two numbers comparable.
        # GPU work is synchronized before the clock stops — without that we
        # would be timing CUDA kernel LAUNCHES, not execution.
        #
        # NOTE: torch.compile(reduce-overhead) + cudnn.benchmark make the
        # first several calls dramatically slower (compilation, autotuning,
        # CUDA-graph capture). numba's JIT does the same on the MPPI side.
        # Discard warmup calls before reporting a mean — timing_summary()
        # takes a `skip` argument for exactly this.
        self.timing = []

        # Per-call PROJECTION-LAYER latency in seconds: the WHOLE
        # _project_trajectory() call, not just solver.solve(). The
        # `solve=` field of the [proj] print is the QP time alone and
        # excludes obstacle gathering, per-node parameter packing, warm
        # starting, the solver.set() loop, solution extraction and slack
        # inspection — all of which a real robot pays too. Quote
        # timing_proj, not solve_ms, as the cost of the projection layer.
        self.timing_proj = []

        # Per-call END-TO-END latency in seconds: state in → action out,
        # with visualizer-only work subtracted (see self._viz_s). This is
        # the number that answers "how long between receiving the
        # conditions and having a control action". It is NOT
        # timing + timing_proj: it also covers the action extraction,
        # enforce_lims clipping and the projection pre/post-processing.
        self.timing_total = []

        # Scratch accumulator (seconds) for work done ONLY for the
        # visualizer/logger inside one predict() call — pulling all K
        # candidates back to the CPU, converting them to world frame and
        # populating self.predicted_traj. None of it exists on a robot, so
        # it is measured and subtracted from timing_total rather than
        # silently inflating the latency claim.
        self._viz_s = 0.0
        # ─────────────────────────────────────────────────────────────────

        # ── PROJECTION ──────────────────────────────────────────────────
        self.use_projection      = True   # set False to disable layer entirely
        self.proj_solver         = None
        self.proj_model          = None
        self.proj_constraint     = None
        self.proj_dt             = 0.05   # diffusion sample period
        self.proj_N              = None   # set in configure() = horizon - 1
        self.proj_Tf             = None   # set in configure() = (horizon-1) * dt
        self.proj_max_dev_thresh = 1.5    # [m] fallback if deviation exceeds this
        # Acceptance gate on obstacle-constraint violation, expressed as a
        # physical penetration depth in meters. NOT the raw acados slack —
        # that lives in squared-distance units (since h = dist^2 - r^2 >= -xi)
        # and is not directly comparable to a meter-scale clearance
        # tolerance. See _project_trajectory for the conversion.
        self.proj_max_penetration_thresh = 0.1  # [m] fallback if penetration exceeds this
        self.proj_solver_mode    = "RTI"  # "RTI" or "SQP"
        # SQP-mode budget (ignored in RTI, which is one iteration by design).
        # These bound the projection layer's worst case: the solver returns
        # acados status 2 when it exhausts max_iter without reaching tol, so
        # max_iter sets the latency tail and tol sets how often that happens.
        # Both are logged per solve (see the `iters=` field of the [proj]
        # line) so the accepted-but-unconverged steps are visible rather than
        # hidden behind an "OK".
        self.proj_sqp_max_iter   = 5
        self.proj_sqp_tol        = 1e-3
        # Rolling counts for the per-episode timing summary.
        self.proj_iters          = []   # SQP iterations per solve
        self.proj_maxiter_hits   = 0    # solves that returned acados status 2
        self.proj_solves         = 0    # solves attempted (status 2 or not)
        # Obstacle budget split:
        #   k_max    → dynamic agents only.  Set in configure() from config.
        #   k_static → dedicated static map cell slots.  NEVER shared with
        #              dynamic agents, so dense walls always get MPC coverage
        #              regardless of crowd size.
        #   proj_n_obs = k_max + k_static  → total n_obs_max passed to acados.
        self.k_static   = 20    # overridden in configure()
        self.proj_n_obs = None  # set in configure() = k_max + k_static
        # Clearance philosophy:
        #   The projection uses the SAME safety_radius the diffusion's
        #   collision_cost / scoring uses. This is a SINGLE GLOBAL
        #   "robot-center to obstacle-center" distance — it bundles the
        #   robot radius, obstacle radius, and any margin into one number.
        #   We do NOT add r_robot + r_human separately, since safety_radius
        #   is already meant to encode that combined clearance.
        #   self.safety_radius is populated in configure() from
        #   [sogudiff] safety_radius (× 1.3).
        # Warm-start policy: ALWAYS from the current diffusion sample.
        # We do NOT cache the previous solve, because the diffusion may
        # produce a different homotopy class each tick (different side
        # of an obstacle, different timing through gates), making the
        # previous solve a misleading anchor for SQP linearization.
        # The diffusion path itself is by construction in the right
        # homotopy class — that's the strongest warm start we have.
        # ────────────────────────────────────────────────────────────────

        self.use_map           = False
        self.map_size          = 50
        self.map_extent        = 10
        self._cur_occ_world_xy = None
        self._cur_has_map      = 0.0          # float
        self._cur_map_extent   = float(self.map_extent)  # valid default until set_static_map()

        # ── SOCIAL STYLE ─────────────────────────────────────────────────
        # style_vector: numpy array of shape (N_STYLE_AXES,), values in [-1,1].
        # Set in configure() from config or left as neutral zeros.
        # Kept fixed for the lifetime of this policy instance — no per-step changes.
        self.style_vector  = np.zeros(N_STYLE_AXES, dtype=np.float32)
        self.n_style_axes  = N_STYLE_AXES
        self.style_embedder = None   # StyleTokenEmbedder, built in configure()

        # Compositional CFG weights + inference MODE.
        #   cfg_w_scene : how hard to push scene conditioning (goal, obstacles, map)
        #   cfg_w_style : per-axis style guidance strength (list, one per axis)
        #   cfg_infer_mode : "joint"  -> 3 passes, one all-on style delta
        #                    "per_axis"-> 2+n passes, one delta per axis
        # These MUST correspond to how the checkpoint was trained (see the
        # trainer's resolve_infer_mode): a "joint"/"per_axis" checkpoint forces
        # its own mode; a "union" checkpoint supports either.  Set in configure()
        # from policy.config; fall back to legacy cfg_weight if keys are absent.
        self.cfg_w_scene    = 1.0
        self.cfg_w_style    = [1.0] * N_STYLE_AXES   # per-axis
        self.cfg_infer_mode = "joint"                # matches trainer default
        self.cfg_w_normalize = False
        # ─────────────────────────────────────────────────────────────────

        # ── AMP ──────────────────────────────────────────────────────────
        # Optional bf16 autocast around the UNet forward pass in the
        # denoising loop. Off by default; set use_amp = true under
        # [sogudiff] to enable. Kept as a flag
        # (rather than always-on) so it can be A/B compared for timing
        # impact without touching the eager/fp32 numerics otherwise.
        self.use_amp = False
        self.amp_dtype = torch.bfloat16
        # ─────────────────────────────────────────────────────────────────

    def _resolve_infer_mode(self, requested):
        """Validate a requested inference mode against the checkpoint's training.

        Mirrors the trainer's resolve_infer_mode: a union checkpoint may run
        either scheme; a joint/per_axis checkpoint may run ONLY its own, because
        the other mode's masks were never trained and querying them subtracts an
        unfitted quantity (the bug the structured dropout fixed). Use this both
        at configure() time and whenever a driver overrides cfg_infer_mode.
        """
        requested = (requested or 'auto').strip().lower()
        tm = getattr(self, 'trained_cfg_mode', 'union')
        if tm == 'union':
            return 'joint' if requested in ('auto', 'joint') else 'per_axis'
        if requested in ('auto', tm):
            return tm
        raise ValueError(
            f"cfg_infer_mode={requested!r} needs a union or {requested!r} "
            f"checkpoint, but trained_cfg_mode={tm!r}. This checkpoint never "
            f"trained the {requested!r} masks.")

    def set_infer_mode(self, mode):
        """Driver-facing setter that runs the same validation as configure()."""
        self.cfg_infer_mode = self._resolve_infer_mode(mode)
        return self.cfg_infer_mode

    def configure(self, config):
        import torch
        import numpy as np

        self.start_dim = config.getint('sogudiff', 'start_dim')
        self.goal_dim = config.getint('sogudiff', 'goal_dim')
        self.token_dim = config.getint('sogudiff', 'token_dim')
        self.horizon = config.getint('sogudiff', 'horizon')
        self.k_max = config.getint('sogudiff', 'k_max')
        self.kinematics = config.get('action_space', 'kinematics')
        self.enforce_lims = config.getboolean('action_space', 'enforce_lims')
        self.max_vel = config.getfloat('action_space', 'max_vel')
        self.max_wrot = config.getfloat('action_space', 'max_wrot')
        self.max_accel = config.getfloat('action_space', 'max_accel')
        self.max_w_accel = config.getfloat('action_space', 'max_w_accel')
        self.multiagent_training = config.getboolean('sogudiff', 'multiagent_training')
        self.num_diffusion_timesteps = config.getint('sogudiff', 'num_diffusion_timesteps')
        self.num_inference_steps = config.getint('sogudiff', 'num_inference_steps')
        self.ddim_inference = config.getboolean('sogudiff', 'ddim_inference')
        self.cfg_weight = config.getfloat('sogudiff', 'cfg_weight')

        # Compositional CFG weights + inference mode (optional; fall back to
        # cfg_weight).  cfg_w_scene is the scene-guidance strength.  cfg_w_style
        # is per-axis: accepts either one scalar (broadcast to all axes) or a
        # comma-separated list of N_STYLE_AXES values.
        _sec = 'sogudiff'
        self.cfg_w_scene = (config.getfloat(_sec, 'cfg_w_scene')
                            if config.has_option(_sec, 'cfg_w_scene')
                            else self.cfg_weight)

        if config.has_option(_sec, 'cfg_w_style'):
            _raw = config.get(_sec, 'cfg_w_style').strip()
            _vals = [float(v) for v in _raw.split(',') if v.strip() != '']
        else:
            _vals = [self.cfg_weight]
        if len(_vals) == 1:
            self.cfg_w_style = [_vals[0]] * self.n_style_axes
        elif len(_vals) == self.n_style_axes:
            self.cfg_w_style = _vals
        else:
            raise ValueError(
                f"cfg_w_style must be 1 or {self.n_style_axes} values, got {len(_vals)}")

        # What the CHECKPOINT was trained for.  Weights carry no such metadata,
        # so the operator must declare it in policy.config next to the model
        # path: trained_cfg_mode = joint | per_axis | union.  Defaults to
        # 'union' (permissive) only if unset, with a warning — set it explicitly.
        self.trained_cfg_mode = (config.get(_sec, 'trained_cfg_mode').strip().lower()
                                 if config.has_option(_sec, 'trained_cfg_mode')
                                 else 'union')
        if self.trained_cfg_mode not in ('joint', 'per_axis', 'union'):
            raise ValueError(f"trained_cfg_mode must be joint|per_axis|union, "
                             f"got {self.trained_cfg_mode!r}")

        # Inference mode.  "auto"/"" -> the checkpoint's own mode for
        # joint/per_axis, or 'joint' for a union checkpoint.
        req = (config.get(_sec, 'cfg_infer_mode').strip().lower()
               if config.has_option(_sec, 'cfg_infer_mode') else 'auto')
        self.cfg_infer_mode = self._resolve_infer_mode(req)

        self.cfg_w_normalize = (config.getboolean(_sec, 'cfg_w_normalize')
                                if config.has_option(_sec, 'cfg_w_normalize')
                                else False)
        self.num_samples = config.getint('sogudiff', 'num_samples')
        self.collision_coef = config.getfloat('sogudiff', 'collision_coef')
        self.smooth_pen_coef = config.getfloat('sogudiff', 'smooth_pen_coef')
        self.goal_reward_coef = config.getfloat('sogudiff', 'goal_reward_coef')
        self.safety_radius = config.getfloat('sogudiff', 'safety_radius') * 1.3
        self.control_effort_coef = config.getfloat('sogudiff', 'control_effort_coef')

        self.use_map = (config.getboolean('sogudiff', 'use_map')
                if config.has_option('sogudiff', 'use_map')
                else False)
        self.map_size = (config.getint('sogudiff', 'map_size')
                        if config.has_option('sogudiff', 'map_size')
                        else 50)
        self.map_extent = (config.getfloat('sogudiff', 'map_extent')
                        if config.has_option('sogudiff', 'map_extent')
                        else 10)

        # ── AMP ──────────────────────────────────────────────────────────
        self.use_amp = (config.getboolean('sogudiff', 'use_amp')
                        if config.has_option('sogudiff', 'use_amp')
                        else False)
        amp_dtype_str = (config.get('sogudiff', 'amp_dtype')
                        if config.has_option('sogudiff', 'amp_dtype')
                        else 'bfloat16')
        _amp_dtype_map = {'bfloat16': torch.bfloat16, 'bf16': torch.bfloat16,
                          'float16': torch.float16, 'fp16': torch.float16}
        assert amp_dtype_str in _amp_dtype_map, (
            f"amp_dtype must be one of {list(_amp_dtype_map)}, got {amp_dtype_str!r}"
        )
        self.amp_dtype = _amp_dtype_map[amp_dtype_str]
        # ─────────────────────────────────────────────────────────────────

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        print("Device:", device)

        # Fixed-shape 1D convs run every tick with identical sizes — let
        # cuDNN autotune the fastest algorithm once instead of using the
        # default heuristic every call. Same math, same precision, just
        # picks a faster kernel implementation.
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        norm_file = resolve_path(config.get('sogudiff', 'norm_file'))
        self.norm = np.load(norm_file, allow_pickle=True).item()

        # Cached GPU tensor copies of self.norm for the hot (per-tick)
        # math path, so predict()/score_trajectories() don't rebuild +
        # re-transfer these from numpy on every call. self.norm itself is
        # left untouched (still numpy) since build_condition_from_state()
        # relies on plain numpy arithmetic.
        self.norm_t = {
            k: torch.as_tensor(v, dtype=torch.float32, device=self.device)
            for k, v in self.norm.items()
        }

        self.traj_dim = 2

        self.token_embedder = SceneTokenEmbedder(
            start_dim=self.start_dim,
            goal_dim=self.goal_dim,
            obs_dim=4,
            token_dim=self.token_dim,
            k_max=self.k_max,
            map_size=self.map_size,
            use_map=self.use_map,
        ).to(self.device)

        self.model = UNet1DConditionModel(
            sample_size=self.horizon,
            in_channels=self.traj_dim,
            out_channels=self.traj_dim,
            layers_per_block=3,
            block_out_channels=(64, 128, 256),
            cross_attention_dim=self.token_dim,
            down_block_types=(
                "CrossAttnDownBlock1D",
                "CrossAttnDownBlock1D",
                "CrossAttnDownBlock1D",
            ),
            up_block_types=(
                "CrossAttnUpBlock1D",
                "CrossAttnUpBlock1D",
                "CrossAttnUpBlock1D",
            ),
            mid_block_type="UNetMidBlock1DCrossAttn",
        ).to(self.device)

        if self.ddim_inference:
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=self.num_diffusion_timesteps,
                beta_schedule="squaredcos_cap_v2",
                prediction_type="epsilon",
                clip_sample=False,
            )
        else:
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.num_diffusion_timesteps,
                beta_schedule="squaredcos_cap_v2",
                prediction_type="epsilon",
                clip_sample=False,
            )
        self.noise_scheduler = move_scheduler_to_device(self.noise_scheduler, self.device)

        # ── SOCIAL STYLE ─────────────────────────────────────────────────
        # Optional: fixed style vector to use for inference.
        # Format in config: style_vector = 0.0,0.0,0.0,0.0  (prox,pass,yield,group)
        if config.has_option('sogudiff', 'style_vector'):
            raw = config.get(
                'sogudiff', 'style_vector'
            ).strip()
            if raw:
                vals = [float(v.strip()) for v in raw.split(',') if v.strip()]
            else:
                vals = []
            sv = np.zeros(self.n_style_axes, dtype=np.float32)
            n = min(len(vals), self.n_style_axes)
            if n > 0:
                sv[:n] = np.array(vals[:n], dtype=np.float32)
            self.style_vector = np.clip(sv, -1.0, 1.0)
        else:
            self.style_vector = np.zeros(self.n_style_axes, dtype=np.float32)
        print(f"Social style vector: {dict(zip(STYLE_AXES, self.style_vector))}")

        self.style_embedder = StyleTokenEmbedder(
            axis_names=STYLE_AXES,
            token_dim=self.token_dim,
        ).to(self.device)
        # ─────────────────────────────────────────────────────────────────

        ckpt_path = resolve_path(config.get('sogudiff', 'ckpt_path'))
        ckpt = torch.load(ckpt_path, map_location=self.device)

        if "ema_model_state_dict" in ckpt:
            print("Loading EMA weights for inference")
            self.model.load_state_dict(ckpt["ema_model_state_dict"])
            # Support both old key name and new key name for scene embedder.
            scene_key = ("ema_scene_embedder_state_dict"
                         if "ema_scene_embedder_state_dict" in ckpt
                         else "ema_token_embedder_state_dict")
            self.token_embedder.load_state_dict(ckpt[scene_key])
            if "ema_style_embedder_state_dict" in ckpt:
                self.style_embedder.load_state_dict(ckpt["ema_style_embedder_state_dict"])
                print("Loaded EMA style embedder weights")
            else:
                print("WARNING: no style embedder weights in checkpoint — "
                      "style_embedder will use random initialization")
        else:
            print("EMA weights not found — falling back to bare model weights")
            self.model.load_state_dict(ckpt["model_state_dict"])
            scene_key = ("scene_embedder_state_dict"
                         if "scene_embedder_state_dict" in ckpt
                         else "token_embedder_state_dict")
            self.token_embedder.load_state_dict(ckpt[scene_key])
            if "style_embedder_state_dict" in ckpt:
                self.style_embedder.load_state_dict(ckpt["style_embedder_state_dict"])
            else:
                print("WARNING: no style embedder weights in checkpoint — "
                      "style_embedder will use random initialization")

        self.model.eval()
        self.token_embedder.eval()
        self.style_embedder.eval()

        print(f"AMP autocast for UNet forward: "
              f"{'enabled (' + str(self.amp_dtype) + ')' if self.use_amp else 'disabled'}")

        # Fuse Q/K/V projection weights for attention layers (self-attn:
        # fuse all 3; cross-attn: fuse K+V since both read encoder_hidden_states).
        # Concatenates the already-trained weight matrices into one combined
        # matmul — identical outputs, fewer kernel launches. This model's
        # cross-attn blocks (CrossAttnDownBlock1D/UpBlock1D, the mid block)
        # dominate its structure, so this directly targets the per-step cost.
        # HuggingFace marks this API "experimental"; it self-guards by raising
        # ValueError for architectures it can't safely fuse (e.g. added-KV
        # projections), so a failure here is loud, not silent.
        try:
            self.model.fuse_qkv_projections()
            print("Fused QKV projections enabled for UNet attention layers")
        except Exception as e:
            print(f"fuse_qkv_projections unavailable, leaving attention unfused: {e}")

        # Compile the UNet forward pass — same graph, fused CUDA kernels.
        # predict() calls self.model with identical tensor shapes every
        # step (N_V*K, traj_dim, horizon), so this specializes once and
        # is reused for the rest of the run. Falls back to eager mode if
        # compilation isn't supported on this torch/CUDA build, so a
        # compile failure can never break inference.
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead")
            print("torch.compile enabled for UNet (mode=reduce-overhead)")
        except Exception as e:
            print(f"torch.compile unavailable, falling back to eager UNet: {e}")

        # ── PROJECTION ──────────────────────────────────────────────────
        # Optional config knobs (with defaults) — add these to your config
        # under [sogudiff] if you want non-default values.
        if config.has_option('sogudiff', 'use_projection'):
            self.use_projection = config.getboolean(
                'sogudiff', 'use_projection')
        if config.has_option('sogudiff', 'proj_max_dev_thresh'):
            self.proj_max_dev_thresh = config.getfloat(
                'sogudiff', 'proj_max_dev_thresh')
        if config.has_option('sogudiff', 'proj_max_penetration_thresh'):
            self.proj_max_penetration_thresh = config.getfloat(
                'sogudiff', 'proj_max_penetration_thresh')
        elif config.has_option('sogudiff', 'proj_max_slack_thresh'):
            # Legacy key. It used to be compared directly against the raw
            # squared-distance acados slack, which was a units bug (a
            # value like 0.5 was never actually a 0.5 m tolerance — see
            # _project_trajectory). Rather than silently reinterpreting an
            # old config value in the new units, ignore it and warn.
            print("WARNING: 'proj_max_slack_thresh' is deprecated and no "
                  "longer used (it compared raw squared-distance slack "
                  "against a meter-scale threshold, which was incorrect). "
                  "Set 'proj_max_penetration_thresh' [m] instead. Using "
                  f"default {self.proj_max_penetration_thresh} m.")
        if config.has_option('sogudiff', 'proj_solver_mode'):
            self.proj_solver_mode = config.get(
                'sogudiff', 'proj_solver_mode')

        assert self.proj_solver_mode in ("RTI", "SQP"), (
            f"proj_solver_mode must be 'RTI' or 'SQP', got "
            f"{self.proj_solver_mode!r}"
        )

        # SQP budget. Read unconditionally (not only when mode == SQP) so a
        # config that switches modes needs no other edit; the builder ignores
        # both in RTI mode.
        if config.has_option('sogudiff', 'proj_sqp_max_iter'):
            self.proj_sqp_max_iter = config.getint(
                'sogudiff', 'proj_sqp_max_iter')
        if config.has_option('sogudiff', 'proj_sqp_tol'):
            self.proj_sqp_tol = config.getfloat(
                'sogudiff', 'proj_sqp_tol')
        assert self.proj_sqp_max_iter >= 1, (
            f"proj_sqp_max_iter must be >= 1, got {self.proj_sqp_max_iter}")
        assert self.proj_sqp_tol > 0, (
            f"proj_sqp_tol must be > 0, got {self.proj_sqp_tol}")

        # Dedicated static map obstacle budget — independent of k_max so
        # dense walls are never crowded out by dynamic agents.
        if config.has_option('sogudiff', 'k_static'):
            self.k_static = config.getint('sogudiff', 'k_static')

        # Total slots for the acados solver = dynamic + static.
        self.proj_n_obs = self.k_max + self.k_static

        # Note: discomfort_dist and any analogous training-side safety
        # buffer are RL training/reward concerns and are intentionally NOT
        # used here. The projection enforces clearance purely via
        # self.safety_radius, which is the same number the diffusion's
        # scoring uses, ensuring the projector and the sampler agree on
        # what "safe" means.

        if self.use_projection:
            self._init_projection()
        # ────────────────────────────────────────────────────────────────

    def _init_projection(self):
        """Build the acados projection solver.

        Factored out of configure() so that a SUBCLASS which does not load a
        diffusion checkpoint (e.g. the online MPPI baseline in mppi_expert.py)
        can construct the byte-for-byte IDENTICAL projection solver. Both
        policies must feed their chosen candidate through the same OCP, so this
        must be the single source of truth for how that solver is built.

        Requires (all set in configure() before this is called): self.horizon,
        self.proj_dt, self.proj_n_obs, self.k_max, self.k_static,
        self.proj_solver_mode.
        """
        # Horizon convention (matches the diffusion exactly):
        #   The candidate has T = self.horizon points at times
        #     0, dt, 2*dt, ..., (T-1)*dt
        #   where INDEX 0 IS THE CURRENT ROBOT POSITION (i.e. always
        #   (0,0) in ego frame, modulo sampler noise) and indices
        #   1..T-1 are the future positions.
        #
        #   The OCP therefore uses:
        #     N  = T - 1   shooting intervals  (so T nodes total)
        #     Tf = (T - 1) * proj_dt
        #
        #   Node 0 is the initial-condition node (x0 = current state),
        #   nodes 1..N track candidate[1..T-1] as references.
        #   We deliberately do NOT track candidate[0] — it is only an
        #   approximation of (0,0), but we know the true value exactly via x0.
        self.proj_N  = self.horizon - 1
        self.proj_Tf = self.proj_N * self.proj_dt

        _sqp_cfg = (f" max_iter={self.proj_sqp_max_iter} tol={self.proj_sqp_tol:g}"
                    if self.proj_solver_mode == "SQP" else "")
        print(f"Building acados projection solver: "
              f"N={self.proj_N}, Tf={self.proj_Tf:.3f}s, "
              f"n_obs={self.proj_n_obs} (k_max={self.k_max} dynamic "
              f"+ k_static={self.k_static} static), "
              f"mode={self.proj_solver_mode}{_sqp_cfg}")
        t0 = datetime.now()
        # Unique per-process build dir: AcadosOcpSolver always (re)generates
        # and recompiles C code into a cwd-relative directory with a fixed
        # default name ("c_generated_proj"). Two processes building
        # concurrently in the same cwd -- e.g. a SLURM array of scene shards,
        # which is exactly how the MPPI budget sweep and the CFG/composition
        # sweeps are launched -- race on the same .o/.so files mid-compile
        # ("Stale file handle", missing .so at dlopen). Suffixing by job+task
        # (falling back to PID outside SLURM) gives every task its own tree.
        _uniq = '{}_{}'.format(
            os.environ.get('SLURM_ARRAY_JOB_ID', os.environ.get('SLURM_JOB_ID', 'pid')),
            os.environ.get('SLURM_ARRAY_TASK_ID', os.getpid()),
        )
        self.proj_solver, self.proj_model, self.proj_constraint = \
            build_projection_solver(
                N=self.proj_N,
                Tf=self.proj_Tf,
                n_obs_max=self.proj_n_obs,
                solver_mode=self.proj_solver_mode,
                sqp_max_iter=self.proj_sqp_max_iter,
                sqp_tol=self.proj_sqp_tol,
                code_export_dir=f"c_generated_proj_{_uniq}",
                json_file=f"acados_proj_ocp_{_uniq}.json",
            )
        print(f"  solver built in {datetime.now() - t0}")

    # -----------------------------------------------------------------
    # Latency instrumentation helpers.
    # -----------------------------------------------------------------
    def _begin_predict_timing(self):
        """Start the end-to-end predict() clock and zero the per-call
        visualizer-overhead accumulator.

        Both policies call this as the FIRST statement of predict(), so the
        end-to-end window opens before any conditioning work and the two
        numbers stay comparable. perf_counter (not datetime.now) so the
        end-to-end clock and the viz/projection sub-clocks share one
        monotonic time base.
        """
        self._viz_s = 0.0
        return time.perf_counter()

    def _log_predict_total(self, t_predict_start, tag=""):
        """Close the end-to-end window, record it, and print it.

        Reports the DEPLOYMENT latency — wall time minus the visualizer-only
        work accumulated in self._viz_s — and shows the raw wall time and the
        viz share alongside so the subtraction is auditable rather than
        implicit. Only the deployment number is appended to timing_total.
        """
        wall_ms = (time.perf_counter() - t_predict_start) * 1e3
        viz_ms  = self._viz_s * 1e3
        deploy_ms = max(0.0, wall_ms - viz_ms)
        self.timing_total.append(deploy_ms * 1e-3)
        print(f"  [predict] {tag}total: {deploy_ms:.1f} ms "
              f"(wall {wall_ms:.1f} ms − viz {viz_ms:.1f} ms)")

    def _viz_done(self, t0):
        """Charge [t0, now) to visualizer-only overhead.

        Call immediately after a block that exists solely to feed the
        renderer / logger. The accumulated total is subtracted from the
        reported end-to-end latency.
        """
        self._viz_s += time.perf_counter() - t0

    def timing_summary(self, skip=10):
        """Latency stats in ms for the three instrumented stages.

        Keys are prefixed by stage:
          cand_*  — candidate generation (self.timing): conditioning +
                    denoising (or MPPI) + scoring, GPU-synchronized.
          proj_*  — the full projection layer (self.timing_proj), i.e. all
                    of _project_trajectory, not just the QP solve.
          total_* — end-to-end state→action (self.timing_total), with
                    visualizer-only work excluded.
        The unprefixed keys (mean_ms, median_ms, ...) are kept as aliases of
        the cand_* stage so existing callers and the MPPI subclass keep
        reading the same quantity they always did.

        skip : number of leading calls to DISCARD as warmup. The first calls
               pay torch.compile / cudnn autotune / CUDA-graph capture here,
               and numba JIT on the MPPI side — including them inflates the
               mean by an order of magnitude and is not representative of
               steady-state deployment latency. Report the median/p95 too;
               they are what a real-time claim rests on.
        """
        def _stats(samples, prefix):
            t = np.asarray(samples[skip:], dtype=np.float64) * 1e3
            if t.size == 0:
                return {}
            return {
                f"{prefix}n_calls":   int(t.size),
                f"{prefix}mean_ms":   float(t.mean()),
                f"{prefix}median_ms": float(np.median(t)),
                f"{prefix}p95_ms":    float(np.percentile(t, 95)),
                f"{prefix}max_ms":    float(t.max()),
            }

        cand = _stats(self.timing, "")
        if not cand:
            return {}

        out = dict(cand)
        out["n_skipped"] = int(min(skip, len(self.timing)))
        out.update(_stats(self.timing, "cand_"))
        out.update(_stats(self.timing_proj, "proj_"))
        out.update(_stats(self.timing_total, "total_"))

        # Solver-effort context for the projection numbers. Latency per solve
        # is only interpretable next to the iteration count it bought — 30 ms
        # is alarming for one iteration and unremarkable for five.
        out["proj_solver_mode"] = self.proj_solver_mode
        if self.proj_solver_mode == "SQP":
            out["proj_sqp_max_iter"] = self.proj_sqp_max_iter
            out["proj_sqp_tol"]      = self.proj_sqp_tol
        if self.proj_solves:
            out["proj_solves"]        = self.proj_solves
            out["proj_maxiter_hits"]  = self.proj_maxiter_hits
            out["proj_maxiter_frac"]  = self.proj_maxiter_hits / self.proj_solves
        # Same warmup discard as the latency stats, so ms_per_iter divides two
        # quantities measured over the SAME calls.
        if self.proj_iters[skip:]:
            it = np.asarray(self.proj_iters[skip:], dtype=np.float64)
            out["proj_iter_mean"] = float(it.mean())
            out["proj_iter_max"]  = int(it.max())
            # Cost per SQP iteration — the number that says whether the OCP
            # itself is expensive or the solver is simply iterating a lot.
            if "proj_mean_ms" in out and it.mean() > 0:
                out["proj_ms_per_iter"] = out["proj_mean_ms"] / float(it.mean())
        return out

    def reset_timing(self):
        """Clear latency samples — call between style variants / budgets so
        each configuration reports its own latency instead of a pooled mean."""
        self.timing = []
        self.timing_proj = []
        self.timing_total = []
        self.proj_iters = []
        self.proj_maxiter_hits = 0
        self.proj_solves = 0

    def set_static_map(self, occ_world_xy, has_map, map_extent):
        """
        Called by the env at episode start.
        occ_world_xy: (N, 2) numpy array of occupied-cell centers in WORLD frame,
                    or None.
        has_map:      1.0 if scenario has a map, 0.0 otherwise.
        map_extent:   physical side length of the map (meters).
        """
        self._cur_occ_world_xy = occ_world_xy
        self._cur_has_map      = float(has_map)
        # Guard: preserve the non-zero default when the map is absent so
        # reach computations using _cur_map_extent never divide by zero.
        self._cur_map_extent   = float(map_extent) if float(map_extent) > 0 else float(self.map_extent)

    def _get_static_cells_ego(self, state):
        """
        Return ego-frame (px, py) positions of the k_static nearest occupied
        map cells in physical meters.  Identical ego-frame transform as Pool B
        in _gather_obstacles_ego_physical and _rasterize_ego_map.

        Called before trajectory scoring so sample selection is also wall-aware.

        Returns numpy (M, 2) with M <= k_static, or None when no map / no cells.
        """
        if not (self.use_map
                and self._cur_has_map > 0.5
                and self._cur_occ_world_xy is not None
                and self._cur_occ_world_xy.shape[0] > 0
                and self.k_static > 0):
            return None

        px_r  = state.self_state.px
        py_r  = state.self_state.py
        theta = state.self_state.theta
        cos_t = np.cos(theta); sin_t = np.sin(theta)

        # World → ego: subtract robot position, rotate by -theta
        pts_w = self._cur_occ_world_xy
        ddx   = pts_w[:, 0] - px_r
        ddy   = pts_w[:, 1] - py_r
        x_e   =  cos_t * ddx + sin_t * ddy
        y_e   = -sin_t * ddx + cos_t * ddy
        dists = np.sqrt(x_e ** 2 + y_e ** 2)

        # Include all cells within the map half-extent (full coverage)
        reach = self._cur_map_extent / 2.0
        sel   = dists < reach
        if not sel.any():
            return None

        x_e_s, y_e_s, d_s = x_e[sel], y_e[sel], dists[sel]
        order = np.argsort(d_s)[:self.k_static]
        return np.stack([x_e_s[order], y_e_s[order]], axis=1).astype(np.float32)
        
    def _rasterize_ego_map(self, state):
        """
        Build a (map_size, map_size) ego-centered occupancy map from the
        world-frame occupied-cell positions, using the robot's CURRENT pose.

        Returns (occ_map_np, has_map_float). When use_map is off or no map
        is loaded, returns (zeros, 0.0).
        """
        H = W = self.map_size
        if (not self.use_map
                or self._cur_occ_world_xy is None
                or self._cur_has_map < 0.5
                or self._cur_occ_world_xy.shape[0] == 0):
            return np.zeros((H, W), dtype=np.float32), 0.0

        # Robot pose in world
        px = state.self_state.px
        py = state.self_state.py
        theta = state.self_state.theta

        # Transform world points into the ego frame: subtract robot position,
        # then rotate by -theta (same convention used in build_condition_from_state).
        pts_w = self._cur_occ_world_xy           # (N, 2)
        dx = pts_w[:, 0] - px
        dy = pts_w[:, 1] - py
        cos_t = np.cos(theta); sin_t = np.sin(theta)
        x_ego =  cos_t * dx + sin_t * dy
        y_ego = -sin_t * dx + cos_t * dy

        # Map cell convention: row → y, col → x, ego-centered.
        res = self._cur_map_extent / W
        cols = np.round(x_ego / res + W / 2).astype(np.int32)
        rows = np.round(y_ego / res + H / 2).astype(np.int32)

        in_bounds = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
        rows = rows[in_bounds]; cols = cols[in_bounds]

        occ = np.zeros((H, W), dtype=np.float32)
        if rows.size > 0:
            occ[rows, cols] = 1.0
        return occ, 1.0
        
    def get_model(self):
        return None

    def _reset_unicycle_state(self, v0=0.0, omega0=0.0):
        # Both eval harnesses call this at every episode boundary (the STYLES
        # script per style-variant, the standard explorer per episode), so it
        # is the natural place to emit a LIVE, CUMULATIVE latency summary
        # without touching either eval script. We print the running
        # timing_summary() (flushed) so a SLURM job streams the frontier
        # numbers converging as it goes, instead of revealing nothing until the
        # whole eval set finishes. Timing is NOT cleared — the cumulative stats
        # over the run at a fixed budget ARE the frontier point.
        if getattr(self, "timing", None):
            s = self.timing_summary()
            if s:
                extra = ""
                if "mppi_budget" in s:            # MPPI adds budget context
                    extra = (f" | budget={s.get('mppi_budget')} "
                             f"samples={s.get('mppi_samples')} "
                             f"pool={s.get('mppi_seed_pool')} "
                             f"threads={s.get('numba_threads')}")
                    # Online-search failures: the budget sweep's degradation
                    # mechanism. A latency number for a planner that gave up on
                    # a third of its ticks means something very different from
                    # the same number with zero failures, so the two are always
                    # printed together.
                    if "mppi_search_fail" in s:
                        extra += (f" search_fail={s['mppi_search_fail']}"
                                  f"/{len(self.timing)} "
                                  f"({100.0*s['mppi_search_fail_frac']:.1f}%: "
                                  f"geo={s['mppi_search_fail_geo']} "
                                  f"cand={s['mppi_search_fail_cand']})")
                    # Failed ticks are cheap, so the all-ticks mean above is
                    # discounted by giving up. plan_* excludes them: it is the
                    # cost of actually producing a candidate, and it is the
                    # like-for-like comparison against the diffusion policy,
                    # whose candidate generation cannot fail.
                    if "plan_n_calls" in s:
                        extra += (f" | plan-only mean={s['plan_mean_ms']:.1f} "
                                  f"median={s['plan_median_ms']:.1f} "
                                  f"p95={s['plan_p95_ms']:.1f} ms "
                                  f"({s['plan_n_calls']} calls)")
                print(f"[timing:{self.name}] candidate-gen over {s['n_calls']} "
                      f"calls (skip {s['n_skipped']} warmup): "
                      f"mean={s['mean_ms']:.1f} median={s['median_ms']:.1f} "
                      f"p95={s['p95_ms']:.1f} max={s['max_ms']:.1f} ms{extra}",
                      flush=True)
                # Projection layer and end-to-end state→action, on the same
                # calls. total is NOT cand + proj (it also covers action
                # extraction and the projection pre/post-processing) and it
                # excludes visualizer-only work — it is the latency to quote.
                if "proj_n_calls" in s:
                    # Solver effort alongside the latency: mode, iterations
                    # bought per solve, cost per iteration, and how often the
                    # iteration budget ran out (SQP status 2 — accepted on the
                    # feasibility gates but not converged to tol).
                    eff = f" mode={s['proj_solver_mode']}"
                    if "proj_sqp_max_iter" in s:
                        eff += (f"(max_iter={s['proj_sqp_max_iter']} "
                                f"tol={s['proj_sqp_tol']:g})")
                    if "proj_iter_mean" in s:
                        eff += (f" iters mean={s['proj_iter_mean']:.2f} "
                                f"max={s['proj_iter_max']}")
                    if "proj_ms_per_iter" in s:
                        eff += f" ({s['proj_ms_per_iter']:.1f} ms/iter)"
                    if "proj_solves" in s:
                        eff += (f" maxiter={s['proj_maxiter_hits']}/"
                                f"{s['proj_solves']} "
                                f"({100.0*s['proj_maxiter_frac']:.1f}%)")
                    print(f"[timing:{self.name}] projection layer  "
                          f"mean={s['proj_mean_ms']:.1f} "
                          f"median={s['proj_median_ms']:.1f} "
                          f"p95={s['proj_p95_ms']:.1f} "
                          f"max={s['proj_max_ms']:.1f} ms "
                          f"({s['proj_n_calls']} calls){eff}", flush=True)
                if "total_n_calls" in s:
                    print(f"[timing:{self.name}] end-to-end (viz-excl)  "
                          f"mean={s['total_mean_ms']:.1f} "
                          f"median={s['total_median_ms']:.1f} "
                          f"p95={s['total_p95_ms']:.1f} "
                          f"max={s['total_max_ms']:.1f} ms "
                          f"({s['total_n_calls']} calls)", flush=True)

        self.prev_v = v0
        self.prev_omega = omega0
        self._v0_from_last_step = v0
        self._w0_from_last_step = omega0
        # No previous-solve cache to clear — we always warm-start from the
        # current diffusion sample (see __init__ comment).

    def build_condition_from_state(self, state):
        v0 = getattr(self, '_v0_from_last_step', 0.0)
        if self.start_dim == 2:
            w0 = getattr(self, '_w0_from_last_step', 0.0)
            start = np.array([v0, w0], dtype=np.float32)
        else:
            start = np.array([v0], dtype=np.float32)

        theta = state.self_state.theta
        R = np.array([
            [ np.cos(theta),  np.sin(theta)],
            [-np.sin(theta),  np.cos(theta)]
        ], dtype=np.float32)

        goal_world = np.array([state.self_state.gx, state.self_state.gy])
        pos_world  = np.array([state.self_state.px, state.self_state.py])
        goal_ego = R @ (goal_world - pos_world)
        goal = goal_ego.astype(np.float32)

        obs_list = []
        for h in state.human_states:
            pos_ego = R @ (np.array([h.px, h.py]) - pos_world)
            vel_ego = R @ np.array([h.vx, h.vy])
            obs_list.append([pos_ego[0], pos_ego[1], vel_ego[0], vel_ego[1]])
        obstacles = np.array(obs_list, dtype=np.float32)

        obs_padded = np.zeros((self.k_max, 4), dtype=np.float32)
        obs_mask   = np.zeros((self.k_max,), dtype=np.float32)
        n = min(len(obstacles), self.k_max)
        if n > 0:
            obs_padded[:n] = obstacles[:n]
            obs_mask[:n] = 1.0

        start = (start - self.norm["start_mean"]) / self.norm["start_std"]
        goal  = (goal  - self.norm["goal_mean"])  / self.norm["goal_std"]
        if n > 0:
            obs_padded[:n] = (obs_padded[:n] - self.norm["obs_mean"]) / self.norm["obs_std"]

        start_t = torch.from_numpy(start).unsqueeze(0).to(self.device)
        goal_t  = torch.from_numpy(goal).unsqueeze(0).to(self.device)
        obs_t   = torch.from_numpy(obs_padded).unsqueeze(0).to(self.device)
        mask_t  = torch.from_numpy(obs_mask).unsqueeze(0).to(self.device)

        return start_t, goal_t, obs_t, mask_t

    # =================================================================
    # ── PROJECTION ─ helper: gather raw (un-normalized, ego-frame)
    #                obstacle state directly from the simulator state.
    #                We need this because obstacles_t passed to the
    #                diffusion is normalized; we want PHYSICAL units
    #                for the OCP.
    # =================================================================
    def _gather_obstacles_ego_physical(self, state):
        """
        Fill TWO independent obstacle pools for the MPC projection:

          slots 0 .. k_max-1          → dynamic agents (humans)
          slots k_max .. proj_n_obs-1 → static map cells

        The two pools never compete for the same slots, so a dense crowd
        cannot crowd out wall cells and vice versa.  Each pool is sorted
        by distance and capped at its budget.  pack_obstacle_params
        propagates dynamic agents forward with their velocity and leaves
        static cells stationary (v = 0).

        Returns (obs_xy, obs_v, radii) of length <= proj_n_obs.
        """
        theta = state.self_state.theta
        cos_t = np.cos(theta); sin_t = np.sin(theta)
        R = np.array([[ cos_t,  sin_t],
                      [-sin_t,  cos_t]], dtype=np.float32)
        pos_world = np.array([state.self_state.px, state.self_state.py])

        # ── Pool A: dynamic agents → slots 0..k_max-1 ──────────────────
        dyn_items = []   # (dist, xy_ego, v_ego)

        for h in state.human_states:
            p_ego = R @ (np.array([h.px, h.py]) - pos_world)
            v_ego = R @ np.array([h.vx, h.vy])
            dyn_items.append((float(np.linalg.norm(p_ego)),
                              p_ego.astype(np.float32),
                              v_ego.astype(np.float32)))

        # Legacy static_obstacles from state attribute (point obstacles
        # without velocity — treat as dynamic-pool obstacles so they don't
        # consume static-map budget).
        extra_static = getattr(state, 'static_obstacles', None) or \
                       getattr(state, 'obstacles', None)
        if extra_static is not None:
            for s in extra_static:
                p_ego = R @ (np.array([s.px, s.py]) - pos_world)
                dyn_items.append((float(np.linalg.norm(p_ego)),
                                  p_ego.astype(np.float32),
                                  np.zeros(2, dtype=np.float32)))

        dyn_items.sort(key=lambda t: t[0])
        dyn_items = dyn_items[:self.k_max]

        # ── Pool B: static map cells → slots k_max..proj_n_obs-1 ───────
        static_items = []   # (dist, xy_ego)

        if (self.use_map
                and self._cur_occ_world_xy is not None
                and self._cur_has_map > 0.5
                and self._cur_occ_world_xy.shape[0] > 0
                and self.k_static > 0):
            px_r = state.self_state.px
            py_r = state.self_state.py
            pts_w = self._cur_occ_world_xy          # (N, 2) world frame
            ddx = pts_w[:, 0] - px_r
            ddy = pts_w[:, 1] - py_r
            x_e =  cos_t * ddx + sin_t * ddy
            y_e = -sin_t * ddx + cos_t * ddy
            dists_map = np.sqrt(x_e ** 2 + y_e ** 2)

            # Use the full map half-extent as reach so no cell that lies
            # within the loaded map is missed. _get_static_cells_ego uses
            # the same value for scoring, keeping both consistent.
            reach = self._cur_map_extent / 2.0
            sel   = dists_map < reach
            if sel.any():
                x_e_s = x_e[sel]; y_e_s = y_e[sel]; d_s = dists_map[sel]
                # k_static closest — argsort is exact (no partial sort needed
                # since the sel array is already bounded to a local region).
                order = np.argsort(d_s)[:self.k_static]
                for idx in order:
                    static_items.append((float(d_s[idx]),
                                         np.array([x_e_s[idx], y_e_s[idx]],
                                                  dtype=np.float32)))

        # ── Combine pools ────────────────────────────────────────────────
        total = len(dyn_items) + len(static_items)
        if total == 0:
            return (np.zeros((0, 2), dtype=np.float32),
                    np.zeros((0, 2), dtype=np.float32),
                    np.zeros((0,),   dtype=np.float32))

        xy_list, v_list = [], []
        for _, p, v in dyn_items:
            xy_list.append(p); v_list.append(v)
        for _, p in static_items:
            xy_list.append(p)
            v_list.append(np.zeros(2, dtype=np.float32))

        obs_xy = np.stack(xy_list, axis=0)
        obs_v  = np.stack(v_list,  axis=0)
        radii  = np.full(len(xy_list), float(self.safety_radius), dtype=np.float32)
        return obs_xy, obs_v, radii

    # =================================================================
    # ── PROJECTION ─ run the OCP and return the projected trajectory
    #                or None if the projection failed.
    # =================================================================
    def _project_trajectory(self, diffusion_xy_ego, state):
        """
        Project a candidate (px, py) trajectory onto the feasible /
        collision-free set under unicycle dynamics + dynamic obstacles.

        Indexing convention (matches the ORIGINAL diffusion code):
          The diffusion outputs T = self.horizon points at times
              0, dt, 2*dt, ..., (T-1)*dt
          where INDEX 0 is the current robot position (i.e. (0,0) in ego
          frame, modulo denoising noise) and indices 1..T-1 are future
          positions.

          The OCP has N = T-1 shooting intervals (T nodes total):
              node 0    : initial-condition node, x0 = current state
              nodes 1..N: track diffusion[1..T-1] as references

          We use the actual current state (0,0,0,v0,w0) for x0 — NOT
          diffusion[0] — because diffusion[0] is only an approximation
          of (0,0) and we know the true value exactly.

        Parameters
        ----------
        diffusion_xy_ego : (T, 2) ego-frame positions in physical units
                          (meters), as output by the diffusion model
                          after de-normalization.
        state            : simulator state.

        Returns
        -------
        proj_x_traj : (T, 5) projected state trajectory at horizon ticks
                      0, dt, ..., (T-1)*dt  (ego frame), or None on failure.
        proj_u_traj : (T-1, 2) projected inputs, or None.
        info        : dict with status, slack, deviation, timing, fallback
                      reason.
        """
        T = diffusion_xy_ego.shape[0]
        N = T - 1
        dt = self.proj_dt
        assert N == self.proj_N, (
            f"Projection horizon mismatch: diffusion gave T={T} points, "
            f"solver was built with N={self.proj_N} (expected N == T-1)"
        )

        # --- Build (N+1, 2) reference ----------------------------------
        # Index 0 must be (0,0) by the indexing convention. We *force*
        # it to (0,0) regardless of what the noisy diffusion[0] says,
        # since this stage's residual against x0 = (0,0,...) will be
        # zero anyway (x0 is a hard constraint), but we want a clean
        # cost expression and zero residual contribution from stage 0.
        ref_xy = np.zeros((T, 2), dtype=np.float64)
        ref_xy[0]  = 0.0
        ref_xy[1:] = diffusion_xy_ego[1:]

        # --- Initial state in ego frame --------------------------------
        v0 = float(getattr(self, '_v0_from_last_step', 0.0))
        w0 = float(getattr(self, '_w0_from_last_step', 0.0))
        x0 = np.array([0.0, 0.0, 0.0, v0, w0], dtype=np.float64)

        # --- Obstacles (physical units, ego frame) ---------------------
        obs_xy, obs_v, radii = self._gather_obstacles_ego_physical(state)

        # --- Per-node parameters (constant-velocity propagation) -------
        # n_obs_max must match what the solver was built with (proj_n_obs =
        # k_max + k_static). Dynamic agents (slots 0..k_max-1) are propagated
        # forward with their velocity; static cells (slots k_max..proj_n_obs-1)
        # have v=0 and stay fixed across the horizon.
        params_per_node = pack_obstacle_params(
            obs_xy, obs_v, radii,
            n_obs_max=self.proj_n_obs,
            N=N, dt=dt,
        )

        # --- Warm start ALWAYS from the diffusion path -----------------
        # See __init__ docstring for rationale.
        ws_x, ws_u = kinematic_warm_start_from_xy(
            ref_xy, dt=dt, theta0=0.0, v0=v0, omega0=w0,
        )

        # --- Solve -----------------------------------------------------
        result = solve_projection(
            self.proj_solver,
            x0=x0,
            ref_xy=ref_xy,
            params_per_node=params_per_node,
            warm_start_x=ws_x,
            warm_start_u=ws_u,
            N=N,
        )

        info = dict(
            status           = result["status"],
            max_slack        = result["max_slack"],
            penetration_depth= None,
            solve_ms         = result["solve_ms"],
            n_iter           = result.get("n_iter"),
            max_dev          = None,
            fallback_reason  = None,
        )

        self.proj_solves += 1
        if result.get("n_iter") is not None:
            self.proj_iters.append(int(result["n_iter"]))
        if result["status"] == 2:
            self.proj_maxiter_hits += 1

        # acados status: 0 = converged, 2 = iteration budget exhausted.
        #
        # Status 2 means DIFFERENT things in the two modes. In RTI it is the
        # normal return — one iteration is the design, not a failure. In SQP
        # it means proj_sqp_max_iter iterations went by without reaching
        # proj_sqp_tol, so the returned trajectory is a partially converged
        # iterate. We still accept it, because acceptance does not rest on
        # the solver's convergence flag: the penetration and max_dev gates
        # below are checked on the returned trajectory either way, and reject
        # it if it is not actually feasible. What status 2 does change is the
        # LABEL — the [proj] print says MAXITER, not OK, and the count is
        # reported per episode, so an eval whose projections are routinely
        # unconverged cannot be mistaken for one where they converge.
        if result["status"] not in (0, 2):
            info["fallback_reason"] = f"acados status {result['status']}"
            return None, None, info

        proj_x = result["x_traj"]   # shape (N+1, 5) = (T, 5)
        proj_u = result["u_traj"]   # shape (N, 2)   = (T-1, 2)
        proj_xy = proj_x[:, 0:2]

        # Deviation against the diffusion reference. Skip index 0 since
        # both projection and reference are pinned to (0,0) there.
        dev = np.linalg.norm(proj_xy[1:] - ref_xy[1:], axis=1)
        max_dev = float(dev.max()) if dev.size > 0 else 0.0
        info["max_dev"] = max_dev

        # Convert raw slack (squared-distance units) to a physical
        # penetration depth in meters before gating. All obstacle slots
        # share the same radius (self.safety_radius; see
        # _gather_obstacles_ego_physical), so one r suffices here. If
        # xi <= r^2 the constraint still permits nonzero clearance; only
        # once xi exceeds r^2 has the solver conceded full penetration
        # (dist = 0 permitted), which we clamp to r rather than let go
        # complex/negative.
        r = float(self.safety_radius)
        penetration_depth = r - math.sqrt(max(r**2 - result["max_slack"], 0.0))
        info["penetration_depth"] = penetration_depth

        if penetration_depth > self.proj_max_penetration_thresh:
            info["fallback_reason"] = (
                f"obstacle penetration {penetration_depth:.3f} m "
                f"> {self.proj_max_penetration_thresh} m "
                f"(raw slack={result['max_slack']:.4f} m^2)"
            )
            return None, None, info

        if max_dev > self.proj_max_dev_thresh:
            info["fallback_reason"] = (
                f"deviation {max_dev:.3f} m > {self.proj_max_dev_thresh} m "
                f"(likely local minimum, e.g. stop-in-front-of-obstacle)"
            )
            return None, None, info

        return proj_x, proj_u, info

    # =================================================================
    # ── PROJECTION ─ controlled-stop fallback action.
    # =================================================================
    def _emergency_brake_action(self):
        """
        Apply a controlled deceleration: command a speed that is one
        max-decel-step below the previous executed speed, omega ramps to
        zero. Returns an ActionXY or ActionRot depending on kinematics.
        """
        decel = self.max_accel if np.isfinite(self.max_accel) else 2.0
        wdecel = self.max_w_accel if np.isfinite(self.max_w_accel) else 3.0

        v_target = max(0.0, self.prev_v - decel * self.time_step)
        if self.prev_omega > 0:
            w_target = max(0.0, self.prev_omega - wdecel * self.time_step)
        else:
            w_target = min(0.0, self.prev_omega + wdecel * self.time_step)

        if self.kinematics == "holonomic":
            return ActionXY(0.0, 0.0)
        else:
            return ActionRot(v_target, w_target * self.time_step)

    # =================================================================
    # ── PROJECTION ─ record a fallback frame in predicted_traj so the
    #                 visualizer can distinguish brake events from normal
    #                 steps. Keeps diffusion candidates visible (for
    #                 debugging) but sets projection=None.
    # =================================================================
    def _record_predicted_traj_fallback(self, state, x_tensor):
        """
        Build the predicted_traj dict for a brake frame.
        x_tensor : (K, traj_dim, T) post-scoring (best at index 0).
        """
        theta_robot = state.self_state.theta
        R_world = np.array([
            [np.cos(theta_robot), -np.sin(theta_robot)],
            [np.sin(theta_robot),  np.cos(theta_robot)]
        ], dtype=np.float32)
        pos_world = np.array([state.self_state.px, state.self_state.py])

        traj_norm_np = x_tensor.cpu().numpy()
        K = traj_norm_np.shape[0]
        all_samples_world = []
        for k in range(K):
            traj = traj_norm_np[k].T * self.norm["traj_std"] + self.norm["traj_mean"]
            traj_world = np.array([pos_world + R_world @ np.array([dx, dy])
                                   for dx, dy in traj])
            all_samples_world.append(traj_world)

        self.predicted_traj = {
            'selected_sample': all_samples_world[0] if K > 0 else None,
            'projection'     : None,             # ← brake event
            'all_samples'    : all_samples_world,
            'used_projection': False,
        }

    # =================================================================
    # ── PROJECTION ─ interpolate the projected state at any time t in
    #                [0, N*proj_dt]. Linearly interpolates px, py, v,
    #                omega; SLERP-style interpolates theta to handle
    #                wrap-around. Used to support sim_timestep != proj_dt.
    # =================================================================
    def _interp_projected_state(self, proj_x_traj, t):
        """
        proj_x_traj : (N+1, 5) — state trajectory at horizon ticks 0,dt,...,N*dt
        t           : float — query time in [0, N*proj_dt]
        Returns
        -------
        px, py, theta, v, omega  (floats)
        """
        N = proj_x_traj.shape[0] - 1
        dt = self.proj_dt
        T_horizon = N * dt

        # Clamp to within the horizon.
        t = max(0.0, min(t, T_horizon - 1e-9))

        # Find bracketing indices.
        idx_lo = int(np.floor(t / dt))
        idx_hi = min(idx_lo + 1, N)
        frac   = (t - idx_lo * dt) / dt

        s_lo = proj_x_traj[idx_lo]
        s_hi = proj_x_traj[idx_hi]

        # Linear interp for px, py, v, omega
        px    = (1 - frac) * s_lo[0] + frac * s_hi[0]
        py    = (1 - frac) * s_lo[1] + frac * s_hi[1]
        v     = (1 - frac) * s_lo[3] + frac * s_hi[3]
        omega = (1 - frac) * s_lo[4] + frac * s_hi[4]

        # Heading: interpolate the unwrapped delta to handle 2π discontinuity.
        dtheta = s_hi[2] - s_lo[2]
        dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi  # wrap to [-π, π]
        theta  = s_lo[2] + frac * dtheta

        return float(px), float(py), float(theta), float(v), float(omega)

    # =================================================================
    # predict()
    # =================================================================
    def predict(self, state):
        t_predict_start = self._begin_predict_timing()

        if self.reach_destination(state):
            return ActionRot(0, 0)

        # Candidate-generation clock starts here (see self.timing in __init__):
        # conditioning + denoising + scoring/selection, i.e. all planning work
        # that is NOT the shared projection back-end.
        _t_cand0 = time.perf_counter()

        start, goal, obstacles, obs_mask = self.build_condition_from_state(state)

        # Prepare map tensors for this step (no-op when use_map=False)
        if self.use_map:
            occ_np, has_map_f = self._rasterize_ego_map(state)
            occ_map_t = torch.from_numpy(occ_np).unsqueeze(0).unsqueeze(0).to(self.device)
            has_map_t = torch.tensor([has_map_f], dtype=torch.float32, device=self.device)
        else:
            occ_map_t = None
            has_map_t = None

        K = self.num_samples

        x = torch.randn(K, self.traj_dim, self.horizon, device=self.device)

        with torch.no_grad():
            # ── Compositional CFG — variant layout matches the TRAINER exactly ──
            #
            # The scheme is selected by self.cfg_infer_mode and mirrors the
            # trainer's _cfg_variant_masks() / _combine_cfg() bit for bit:
            #
            #  cfg_infer_mode = "joint"  (3 passes):
            #    V0 scene_drop=1, all style null   -> eps_uncond
            #    V1 scene_drop=0, all style null   -> eps_scene
            #    V2 scene_drop=0, ALL style kept   -> eps_all_on
            #    eps = eps_uncond + w_scene (eps_scene - eps_uncond)
            #                     + w_style (eps_all_on - eps_scene)
            #
            #  cfg_infer_mode = "per_axis"  (2 + n_axes passes):
            #    V0 eps_uncond ; V1 eps_scene ; V2..V_{n+1} scene + only axis i
            #    eps = eps_uncond + w_scene (eps_scene - eps_uncond)
            #                     + Σ_i w_i (eps_scene+axis_i - eps_scene)
            #
            # Style variants keep scene ON: at deployment the policy always sees
            # real scene conditioning, so each style delta is measured from the
            # scene-conditional baseline, not from null.  A checkpoint can only
            # be queried in a mode whose masks it was TRAINED on -- a joint
            # checkpoint has never seen the scene+single-axis masks, so running
            # per_axis on it subtracts an unfitted quantity.  Enforce joint vs
            # per_axis vs union at the config level (see policy.config comments).
            _mode = self.cfg_infer_mode
            if _mode == "joint":
                N_V = 3
                style_drop_layout = [ (0, "all"), (1, "none"), (2, "keepall") ]
            else:  # per_axis
                N_V = 2 + self.n_style_axes
                style_drop_layout = None

            style_vals_t = torch.tensor(
                self.style_vector, dtype=torch.float32, device=self.device
            ).unsqueeze(0)                                # [1, n_axes]

            scene_drop_v = torch.zeros(N_V, device=self.device)
            scene_drop_v[0] = 1.0   # V0: full null (uncond) in both modes

            style_drop_v = torch.ones(N_V, self.n_style_axes, device=self.device)
            if _mode == "joint":
                # V0 uncond: all dropped ; V1 scene: all dropped ;
                # V2 all-on: keep every axis.
                style_drop_v[2, :] = 0.0
            else:
                # V0,V1 dropped ; V2+i keeps only axis i.
                for i in range(self.n_style_axes):
                    style_drop_v[2 + i, i] = 0.0

            # Expand scene inputs to N_V for one batched embedder call
            start_v     = start.expand(N_V, -1)
            goal_v      = goal.expand(N_V, -1)
            obstacles_v = obstacles.expand(N_V, -1, -1)
            obs_mask_v  = obs_mask.expand(N_V, -1)
            if occ_map_t is not None:
                occ_map_v = occ_map_t.expand(N_V, -1, -1, -1)
                has_map_v = has_map_t.expand(N_V)
            else:
                occ_map_v = None
                has_map_v = None
            style_vals_v = style_vals_t.expand(N_V, -1)   # [N_V, n_axes]

            # Build all N_V token sequences in a single batched embedder call
            style_toks_v = self.style_embedder(style_vals_v, style_drop_v)
            tokens_v, _  = self.token_embedder(
                start_v, goal_v, obstacles_v, obs_mask_v,
                occ_map=occ_map_v, has_map=has_map_v,
                style_tokens=style_toks_v,
                scene_drop=scene_drop_v,
            )   # [N_V, seq_len, D]

            n_map_tokens = self.token_embedder.n_map_tokens if self.use_map else 0
            attn_mask_v  = build_attn_mask(
                obs_mask_v,
                n_map_tokens=n_map_tokens,
                n_style_tokens=self.n_style_axes,
            )   # [N_V, seq_len]

            # Force obstacle slots valid where scene is dropped so the transformer
            # attends to the null tokens that occupy those positions.
            sdv = scene_drop_v.bool().unsqueeze(1)   # [N_V, 1]
            attn_mask_v = attn_mask_v.clone()
            attn_mask_v[:, 2:2 + self.k_max] = torch.where(
                sdv.expand(-1, self.k_max),
                torch.ones_like(attn_mask_v[:, 2:2 + self.k_max]),
                attn_mask_v[:, 2:2 + self.k_max],
            )

            # Expand to [N_V * K, seq_len, D] — replicate each variant across K
            # noise samples (the K parallel trajectory candidates share tokens).
            tokens_6k    = (tokens_v.unsqueeze(1)
                            .expand(-1, K, -1, -1)
                            .contiguous()
                            .reshape(N_V * K, tokens_v.shape[1], tokens_v.shape[2]))
            attn_mask_6k = (attn_mask_v.unsqueeze(1)
                            .expand(-1, K, -1)
                            .contiguous()
                            .reshape(N_V * K, attn_mask_v.shape[1]))
            # attn_mask_6k never changes across the denoising loop (it only
            # depends on the scene, not on x or t) — cast to bool once here
            # instead of every one of the 5 denoising steps.
            attn_mask_6k_bool = attn_mask_6k.bool()

        denoise_mode = "DDIM" if self.ddim_inference else "DDPM"
        time1 = time.perf_counter()

        if self.ddim_inference:
            self.noise_scheduler.set_timesteps(self.num_inference_steps, device=self.device)
            timesteps = self.noise_scheduler.timesteps
        else:
            timesteps = reversed(range(self.noise_scheduler.config.num_train_timesteps))

        for t in timesteps:
            # Single (N_V*K,)-shaped fill, built directly instead of a
            # (K,) tensor + .repeat(N_V) (same broadcasted values, one
            # allocation instead of two).
            timestep_6k = torch.full((N_V * K,), t, device=self.device, dtype=torch.long)

            with torch.no_grad():
                # Expand x across N_V variants: [N_V*K, D, T]
                x_6k = (x.unsqueeze(0)
                         .expand(N_V, -1, -1, -1)
                         .contiguous()
                         .reshape(N_V * K, self.traj_dim, self.horizon))

                # Optional bf16/fp16 autocast around the UNet forward only
                # (config flag `use_amp`). Everything before/after this
                # call — CFG combination, scheduler.step, scoring — stays
                # in fp32, so amp only affects the one op we expect to
                # actually benefit from Tensor Core throughput.
                amp_ctx = (
                    torch.autocast(device_type="cuda", dtype=self.amp_dtype)
                    if (self.use_amp and self.device.type == "cuda")
                    else contextlib.nullcontext()
                )
                with amp_ctx:
                    noise_6k = self.model(
                        sample                 = x_6k,
                        timestep               = timestep_6k,
                        encoder_hidden_states  = tokens_6k,
                        encoder_attention_mask = attn_mask_6k_bool,
                    ).sample                                          # [N_V*K, D, T]
                noise_6k = noise_6k.float()

                noise_v    = noise_6k.view(N_V, K, self.traj_dim, self.horizon)
                eps_uncond = noise_v[0]       # [K, D, T]  full null
                eps_scene  = noise_v[1]       # [K, D, T]  scene only

                # Scene guidance is identical in both modes.
                noise_pred = eps_uncond + self.cfg_w_scene * (eps_scene - eps_uncond)

                if self.cfg_infer_mode == "joint":
                    # One all-on style delta.  A single scalar weight applies;
                    # we use the mean of the per-axis weights so the same
                    # policy.config cfg_w_style vector is meaningful in both
                    # modes (all-equal weights -> that common value).
                    eps_all_on = noise_v[2]           # [K, D, T]
                    w_joint = float(np.mean(self.cfg_w_style))
                    noise_pred = noise_pred + w_joint * (eps_all_on - eps_scene)
                else:
                    # per_axis: one delta per axis, each with its own weight.
                    # Optional normalization holds Σ_i w_i constant so the
                    # 4-axis composition is not over-guided relative to the
                    # single-axis sweep (see the trainer's --cfg_w_normalize).
                    eps_axes = noise_v[2:]            # [n_axes, K, D, T]
                    w_style = list(self.cfg_w_style)
                    if self.cfg_w_normalize:
                        # Normalize over the ACTIVE axes (s_i != 0), so that the
                        # total weight carried by the axes that actually encode
                        # style equals the scalar joint mode applies to its one
                        # all-on delta.  That is the correct matching:
                        #
                        #   joint     [1,0,0,0] -> 1 delta  @ w             = w
                        #   per_axis  [1,0,0,0] -> prox     @ w             = w
                        #   per_axis  [1,1,1,1] -> 4 deltas @ w/4           = w
                        #
                        # DO NOT normalize over all four axes.  That was tried
                        # and it silently destroyed style control: for a
                        # single-axis vector it scales the one informative delta
                        # by 1/4 (prox at 1.25 instead of 5.0 with cfg_w_style=5)
                        # while spending the other 3/4 of the budget on the
                        # s_i = 0 deltas, which carry almost no signal.  Measured
                        # on the ALLARMS checkpoint over 100 scenes: prox+ mean
                        # clearance collapsed from 3.100 m (joint) to 2.045 m
                        # against a neutral baseline of 2.031 m -- i.e. no effect
                        # at all -- and group/yield went flat too, while pass
                        # merely halved (+0.370 -> +0.203).  The reasoning behind
                        # that change assumed all four deltas contribute
                        # comparable magnitude; they do not.
                        #
                        # Neutral [0,0,0,0] has no active axes, so there is
                        # nothing to normalize and the weights pass through
                        # unscaled.  Harmless: with no style declared the deltas
                        # carry no style content, and the neutral rows agree
                        # across modes (2.031 vs 2.040 m).
                        active = [j for j in range(self.n_style_axes)
                                  if abs(float(self.style_vector[j])) > 1e-6]
                        tot = sum(w_style[j] for j in active)
                        if tot > 1e-9:
                            target = float(np.mean(self.cfg_w_style))
                            scale = target / tot
                            w_style = [w * scale for w in w_style]
                    for i in range(self.n_style_axes):
                        noise_pred = noise_pred + w_style[i] * (eps_axes[i] - eps_scene)

            with torch.no_grad():
                x = self.noise_scheduler.step(noise_pred, t, x).prev_sample

        with torch.no_grad():
            # Nothing between time1 and here forces a device sync, so without
            # this we would print the time to LAUNCH the denoising kernels,
            # not to run them. Same reason as the sync before self.timing.
            if self.device is not None and self.device.type == "cuda":
                torch.cuda.synchronize()
            time2 = time.perf_counter()
            # Denoising loop ONLY: conditioning/tokenization happened before
            # time1 and scoring/selection happens after, so this is a
            # sub-interval of self.timing, not the planning cost.
            print(f"{denoise_mode} sampling ({K} trajs): "
                  f"{(time2 - time1)*1e3:.1f} ms (denoise loop only)")

            x_scored = x.unsqueeze(0)
            obs_b    = obstacles
            mask_b   = obs_mask
            goal_b   = goal

            # Gather static map cells in ego frame (physical meters) for
            # scoring.  This makes sample selection wall-aware — not just
            # the MPC projection layer.  No-map episodes return None and
            # score_trajectories degrades to dynamic-only scoring.
            static_obs_xy_t = None
            if self.use_map:
                static_xy_np = self._get_static_cells_ego(state)
                if static_xy_np is not None:
                    static_obs_xy_t = (
                        torch.from_numpy(static_xy_np)
                        .unsqueeze(0)          # (1, M, 2)
                        .to(self.device)
                    )

            rewards, best_idx = score_trajectories(
                x_scored, obs_b, mask_b, goal_b, self.norm_t,
                collision_coef      = self.collision_coef,
                smooth_pen_coef     = self.smooth_pen_coef,
                goal_reward_coef    = self.goal_reward_coef,
                control_effort_coef = self.control_effort_coef,
                safety_radius       = self.safety_radius,
                dt                  = 0.05,
                static_obs_xy       = static_obs_xy_t,
            )
            best_k = best_idx[0].item()
            print(f"Rewards: {rewards[0].cpu().numpy()}  → selected traj {best_k}")

            idx_order = [best_k] + [i for i in range(K) if i != best_k]
            x = x[idx_order]
            main_traj = x[0]

        # =================================================================
        # ── PROJECTION ─ project the selected diffusion trajectory onto
        #               the feasible, collision-free set.
        # =================================================================
        # Denormalize the selected diffusion trajectory to physical
        # (ego-frame) units. Shape is (T, 2). Convention (matches the
        # ORIGINAL diffusion code):
        #   index 0     = current position (≈ (0,0), modulo denoise noise)
        #   indices 1.. = future positions at times dt, 2*dt, ...
        diffusion_xy_ego = (main_traj.cpu().numpy().T
                            * self.norm["traj_std"]
                            + self.norm["traj_mean"])     # shape (T, 2)

        # ── TIMING ── stop the candidate-generation clock. The .cpu() above
        # already forces a sync, but we synchronize explicitly so the
        # measurement stays correct if that line ever changes: without it we
        # would be timing async kernel LAUNCHES, not GPU execution.
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize()
        self.timing.append(time.perf_counter() - _t_cand0)

        # Build all K diffusion candidates in world frame for the visualizer
        # (index 0 = the selected sample). This is the ONLY policy-specific
        # part of the back-end; the projection + action extraction that
        # follows is shared verbatim with the MPPI baseline via
        # _finish_from_candidate(), so the two policies differ only in how the
        # candidate is produced.
        #
        # ── TIMING ── VISUALIZER-ONLY. A deployed robot executes the action
        # from the SELECTED candidate alone and never needs the other K-1 in
        # world frame; this block only feeds the renderer. It is charged to
        # self._viz_s and subtracted from the reported end-to-end latency.
        _t_viz0 = time.perf_counter()

        theta_robot = state.self_state.theta
        R_world = np.array([
            [np.cos(theta_robot), -np.sin(theta_robot)],
            [np.sin(theta_robot),  np.cos(theta_robot)]
        ], dtype=np.float32)
        pos_world = np.array([state.self_state.px, state.self_state.py])

        def _ego_to_world(traj_ego):
            return np.array([pos_world + R_world @ np.array([dx, dy])
                             for dx, dy in traj_ego])

        # x has been reordered so index 0 is the selected sample.
        traj_norm_np = x.cpu().numpy()
        all_samples_world = []
        for k in range(K):
            traj = traj_norm_np[k].T * self.norm["traj_std"] + self.norm["traj_mean"]
            all_samples_world.append(_ego_to_world(traj))

        self._viz_done(_t_viz0)

        return self._finish_from_candidate(
            diffusion_xy_ego, state,
            all_samples_world=all_samples_world,
            t_predict_start=t_predict_start,
        )

    # =================================================================
    # Shared candidate → action back-end (diffusion AND MPPI baseline).
    # =================================================================
    def _finish_from_candidate(self, cand_xy_ego, state,
                               all_samples_world=None, t_predict_start=None):
        """Project a chosen ego-frame candidate onto the feasible /
        collision-free set and turn it into an action, populating
        self.predicted_traj.

        This is the SHARED back-end for both the diffusion policy and the
        online MPPI baseline. Its input is policy-agnostic — a single physical
        ego-frame candidate trajectory (T, 2), index 0 ≈ (0,0), indices 1..T-1
        at times dt, 2*dt, ... — so the two policies differ ONLY in how that
        candidate is produced (amortized denoising vs. online MPPI). Everything
        here (feasibility projection, brake fallback, predicted_traj
        bookkeeping, action extraction, enforce_lims clipping, warm-state
        caching) runs identically for both. That is what makes the comparison
        an ablation of the trajectory sampler alone.

        Parameters
        ----------
        cand_xy_ego       : (T, 2) physical ego-frame candidate positions.
        all_samples_world : optional list of (T, 2) world-frame candidate
                            arrays for the visualizer (the diffusion policy
                            passes all K samples, index 0 = selected). When
                            None (MPPI, single committed candidate) a
                            one-element list is synthesized from cand_xy_ego.
        t_predict_start   : optional time.perf_counter() reading taken at the
                            top of predict() (via _begin_predict_timing) for
                            the end-to-end latency measurement.
        """
        if t_predict_start is None:
            # Standalone call (no predict() wrapper): open the window here and
            # zero the viz accumulator so the total below stays meaningful.
            t_predict_start = self._begin_predict_timing()

        theta_robot = state.self_state.theta
        R_world = np.array([
            [np.cos(theta_robot), -np.sin(theta_robot)],
            [np.sin(theta_robot),  np.cos(theta_robot)]
        ], dtype=np.float32)
        pos_world = np.array([state.self_state.px, state.self_state.py])

        def _ego_to_world(traj_ego):
            return np.array([pos_world + R_world @ np.array([dx, dy])
                             for dx, dy in traj_ego])

        if all_samples_world is None:
            # VISUALIZER-ONLY (MPPI path): synthesize the single-candidate
            # render list. Charged to _viz_s like the diffusion policy's
            # K-candidate block so both policies' totals exclude the same
            # category of work.
            _t_viz0 = time.perf_counter()
            all_samples_world = [_ego_to_world(cand_xy_ego)]
            self._viz_done(_t_viz0)

        used_proj   = False
        proj_info   = None
        proj_x_traj = None  # (T, 5) — same length as the candidate
        proj_u_traj = None  # (T-1, 2)

        if self.use_projection:
            # ── TIMING ── the projection LAYER, end to end: obstacle
            # gathering, per-node parameter packing, warm-start construction,
            # the solver.set() loop, the QP solve itself, solution extraction
            # and slack/deviation inspection. proj_info['solve_ms'] covers
            # only solver.solve() and therefore understates what the robot
            # pays; both are printed so the gap is visible.
            _t_proj0 = time.perf_counter()
            proj_x_traj, proj_u_traj, proj_info = \
                self._project_trajectory(cand_xy_ego, state)
            proj_layer_ms = (time.perf_counter() - _t_proj0) * 1e3
            self.timing_proj.append(proj_layer_ms * 1e-3)

            if proj_x_traj is not None:
                used_proj = True
                exec_traj_ego = proj_x_traj[:, 0:2]   # (T, 2)
                # MAXITER (status 2) in SQP mode = accepted on the feasibility
                # gates but NOT converged to tol. Labelled distinctly so it is
                # not read as a clean solve. In RTI status 2 is the normal
                # single-iteration return, hence the mode-dependent label.
                _ok = (proj_info['status'] == 0
                       or self.proj_solver_mode == "RTI")
                _label = "OK     " if _ok else "MAXITER"
                _iters = proj_info.get('n_iter')
                _iters_s = f"{_iters}" if _iters is not None else "n/a"
                print(f"  [proj] {_label}  status={proj_info['status']}  "
                      f"iters={_iters_s}  "
                      f"slack={proj_info['max_slack']:.4f} m^2  "
                      f"penetration={proj_info['penetration_depth']:.3f} m  "
                      f"max_dev={proj_info['max_dev']:.3f} m  "
                      f"solve={proj_info['solve_ms']:.2f} ms  "
                      f"layer={proj_layer_ms:.2f} ms")
            else:
                print(f"  [proj] FALLBACK: {proj_info['fallback_reason']}")
                # Apply the same controlled deceleration we're about to
                # command, in self.prev_v / self.prev_omega, so that the
                # next tick's enforce_lims clipping anchors at the
                # post-brake state rather than stale pre-brake values.
                decel = self.max_accel if np.isfinite(self.max_accel) else 2.0
                wdecel = self.max_w_accel if np.isfinite(self.max_w_accel) else 3.0
                v_brake = max(0.0, self.prev_v - decel * self.time_step)
                if self.prev_omega > 0:
                    w_brake = max(0.0, self.prev_omega - wdecel * self.time_step)
                else:
                    w_brake = min(0.0, self.prev_omega + wdecel * self.time_step)
                self.prev_v, self.prev_omega = v_brake, w_brake

                # Same values for next-tick conditioning.
                self._v0_from_last_step = v_brake
                self._w0_from_last_step = w_brake

                # Record fallback frame for the visualizer: keep the
                # candidate(s) visible (so we can see WHAT was rejected) but
                # set projection=None and used_projection=False to flag the
                # brake event distinctly from a normal step.
                _t_viz0 = time.perf_counter()
                self.predicted_traj = {
                    'selected_sample': all_samples_world[0] if all_samples_world else None,
                    'projection'     : None,
                    'all_samples'    : all_samples_world,
                    'used_projection': False,
                }
                self._viz_done(_t_viz0)

                self._log_predict_total(t_predict_start, tag="BRAKE ")

                return self._emergency_brake_action()
        else:
            # No projection: fall through to action-extraction on the raw
            # candidate path. exec_traj_ego shape is (T, 2) with index 0 the
            # current position.
            exec_traj_ego = cand_xy_ego

        # =================================================================
        # Visualization data: store all candidates, the selected sample, and
        # the projected trajectory as separate world-frame entries so consumers
        # can distinguish them.
        #     'selected_sample' : (T, 2) world-frame — the candidate chosen by
        #                          the sampler, BEFORE projection.
        #     'projection'      : (T, 2) world-frame — the feasible/
        #                          collision-free projection, or None if
        #                          projection was disabled or fell back.
        #     'all_samples'     : list of (T, 2) world-frame arrays — every
        #                          candidate. Index 0 matches 'selected_sample'.
        #     'used_projection' : bool — True iff the executed action came from
        #                          the projection.
        # =================================================================
        # VISUALIZER-ONLY: the executed action is read from proj_x_traj in the
        # EGO frame below, so the world-frame conversion and this bookkeeping
        # exist purely for consumers of predicted_traj.
        _t_viz0 = time.perf_counter()

        projection_world = (_ego_to_world(proj_x_traj[:, 0:2])
                            if (used_proj and proj_x_traj is not None) else None)

        self.predicted_traj = {
            'selected_sample': all_samples_world[0] if all_samples_world else None,
            'projection'     : projection_world,
            'all_samples'    : all_samples_world,
            'used_projection': used_proj,
        }

        self._viz_done(_t_viz0)

        # =================================================================
        # Action extraction.
        #
        # General convention: simulator advances by self.time_step which
        # may be larger than self.proj_dt. We compute the action by
        # interpolating the executed trajectory at t = self.time_step
        # (= where the robot needs to be one sim-tick from now, and at
        # what velocity).
        #
        # Time axis convention:
        #   exec_traj_ego[0] is at time 0  (= current position, (0,0))
        #   exec_traj_ego[k] is at time k * proj_dt
        #
        # If projection was used we have v(t) and omega(t) directly from
        # the OCP solution and use those rather than finite-differencing.
        # =================================================================
        dt_diff   = self.proj_dt
        T_horizon = (exec_traj_ego.shape[0] - 1) * dt_diff
        t_apply   = min(self.time_step, T_horizon - 1e-9)

        if used_proj and proj_x_traj is not None:
            # Read v, omega, theta directly from the projected state
            # interpolated at t = time_step.
            _, _, theta_apply, v_apply, omega_apply = \
                self._interp_projected_state(proj_x_traj, t_apply)

            v     = float(v_apply)
            omega = float(omega_apply)

            # Cache for next-tick conditioning. By the next call, the robot
            # will have advanced by self.time_step, so the cached v0/w0
            # should be the values at t = time_step.
            self._v0_from_last_step = v
            self._w0_from_last_step = omega

            if self.kinematics == "holonomic":
                vx_ego = v * np.cos(theta_apply)
                vy_ego = v * np.sin(theta_apply)
                v_world = R_world @ np.array([vx_ego, vy_ego])
                action_world_xy = (float(v_world[0]), float(v_world[1]))
            else:
                action_world_xy = None

        else:
            # ----- Action-extraction without projection ----
            traj = exec_traj_ego  # shape (T, 2), index 0 = current pos

            steps  = self.time_step / dt_diff
            idx_lo = int(np.floor(steps))
            idx_hi = min(idx_lo + 1, traj.shape[0] - 1)
            interp = steps - idx_lo

            # Position to reach at t = time_step  (linear interp in time)
            dx = traj[idx_lo, 0] * (1 - interp) + traj[idx_hi, 0] * interp
            dy = traj[idx_lo, 1] * (1 - interp) + traj[idx_hi, 1] * interp

            # IMPORTANT: this displacement is FROM CURRENT POSITION (0,0).
            # Since traj[0] = (0,0) by convention, dx, dy is the absolute
            # ego-frame target position, which equals the displacement.
            vx = dx / self.time_step
            vy = dy / self.time_step
            v  = float(np.sqrt(vx**2 + vy**2))

            theta_desired = float(np.arctan2(dy, dx)) \
                            if (abs(dx) + abs(dy)) > 1e-6 else 0.0
            omega = theta_desired / self.time_step

            self._v0_from_last_step = v
            self._w0_from_last_step = omega

            if self.kinematics == "holonomic":
                v_world = R_world @ np.array([vx, vy])
                action_world_xy = (float(v_world[0]), float(v_world[1]))
            else:
                action_world_xy = None

        # ------ enforce_lims clipping (identical policy to original) ----
        if self.enforce_lims:
            v_clipped = np.clip(v, -self.max_vel, self.max_vel)
            omega_clipped = np.clip(omega, -self.max_wrot, self.max_wrot)
            v_clipped = np.clip(
                v_clipped,
                self.prev_v - self.max_accel * self.time_step,
                self.prev_v + self.max_accel * self.time_step,
            )
            omega_clipped = np.clip(
                omega_clipped,
                self.prev_omega - self.max_w_accel * self.time_step,
                self.prev_omega + self.max_w_accel * self.time_step,
            )
            self.prev_v, self.prev_omega = v_clipped, omega_clipped
            # Sync conditioning state to the value the robot will actually be
            # at next tick.
            self._v0_from_last_step = float(v_clipped)
            self._w0_from_last_step = float(omega_clipped)
            v_out, omega_out = v_clipped, omega_clipped
        else:
            self.prev_v, self.prev_omega = v, omega
            self._v0_from_last_step = float(v)
            self._w0_from_last_step = float(omega)
            v_out, omega_out = v, omega

        # Stop the end-to-end clock HERE — after enforce_lims, with the action
        # fully determined — so the reported number really is "conditions in →
        # control action out".
        self._log_predict_total(t_predict_start)

        if self.kinematics == "holonomic":
            return ActionXY(*action_world_xy)
        else:
            return ActionRot(v_out, omega_out * self.time_step)