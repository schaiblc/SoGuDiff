"""Evaluation adapter for the NaviSTAR baseline.

NaviSTAR is a SAC actor over a spatio-temporal graph transformer backbone
("STAR"). This module loads a trained actor checkpoint from the fork under
``baselines/navistar`` and runs its forward pass for ``predict()``, so it can
be scored by the same harness as every other baseline. Training happens in
that fork with the authors' own SAC pipeline; nothing here trains.

Differences from the authors' setting
-------------------------------------
- **Radius** 0.3 -> 0.25, with v_pref and dt unchanged at 1 m/s and 0.25 s.
  Both are plain input features rather than baked-in weight shapes, so this
  costs nothing.
- **Crowd size.** STAR's GCN reshape fixes the graph at the training-time
  ``human_num`` (see ``models/STAR.py``, ``self.human_num + 2``), while
  evaluation episodes here carry 1-10 humans. Resolved with closest-N
  selection plus bounded padding, the same compromise the DSRNN adapter
  applies to its fixed-size SRNN graph.
- **Kinematics.** NaviSTAR trains fully holonomic, with no turn-rate or
  acceleration limits, while this evaluation requires unicycle actions under
  the ``[action_space]`` limits. The raw (vx, vy) output is projected onto
  (v, omega) by ``project_velocity_to_unicycle()`` -- the same helper the
  ORCA and SFM unicycle baselines use.

That last point deserves emphasis. ORCA and SFM are analytic, so projecting
their output changes nothing they depended on. NaviSTAR is *learned*, and
never saw turn-rate saturation, acceleration limits or heading lag during
training, so the projection is a genuine distribution shift. It is the
minimal non-retraining path to a common action space, not training-time
parity. ``NaviSTAR`` applies that projection and is the reported
configuration; ``NaviSTARHolonomic`` runs unprojected as an unreported
diagnostic arm, quantifying what the conversion costs. Closing that gap
properly would mean a unicycle-native retrain inside ``baselines/navistar``,
not a change here.

Import isolation
----------------
The fork ships its own top-level ``crowd_sim`` and ``crowd_nav`` packages,
whose names collide with this repository's -- already bound in ``sys.modules``
by the time ``policy_factory`` imports this module. Nothing from the NaviSTAR
side is imported under those names:

- Its ``crowd_nav/configs/config.py`` is loaded by file path via importlib,
  which is safe because that file has no imports of its own.
- ``models.STAR`` and ``offpolicy.agent.actor`` are distinctly named top-level
  packages with no crowd_sim/crowd_nav coupling, so they import normally once
  the fork's root is on ``sys.path``. ``offpolicy/utils.py`` needs only numpy,
  torch, gym and the standard library, so no Python-RVO2 build is required --
  this adapter never instantiates NaviSTAR's ORCA-driven environment.
- The fork's ``crowd_nav/policy/star.py`` wrapper is deliberately not imported,
  since doing so would reintroduce the ``crowd_nav`` collision. Its
  ``clip_action()`` -- a holonomic norm-clip to ``v_pref`` -- is reproduced
  inline in ``_clip_action()`` below.
"""
import importlib.util
import os
import sys

import numpy as np
import torch

from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.policy.unicycle_utils import project_velocity_to_unicycle
from crowd_nav.paths import resolve_path


def _load_navistar_config_class(navistar_root):
    """Load SAN-NaviSTAR's crowd_nav/configs/config.py by file path (not
    package import) -- it has no further imports, so this is safe and avoids
    the crowd_nav package-name collision entirely."""
    path = os.path.join(navistar_root, 'crowd_nav', 'configs', 'config.py')
    spec = importlib.util.spec_from_file_location('_navistar_config_module', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Config


class _NaviSTARAdapter(Policy):
    """Shared implementation for both NaviSTAR arms. Not registered directly."""

    # policy.config section holding {navistar_root, checkpoint}.
    _section = 'navistar'
    # Training-time pad offset for unused observation slots. The checkpoint was
    # trained on variable crowds whose unused slots are always exactly
    # (15, 15) (see baselines/navistar crowd_sim.py generate_ob), so pinning
    # that value avoids a distribution shift at evaluation time.
    _pad_offset_override = 15.0

    def __init__(self):
        super().__init__()
        self.name = 'NaviSTAR'
        self.trainable = False  # loaded from a checkpoint path in policy.config, not via this repo's train.py
        self.multiagent_training = True
        self.kinematics = 'unicycle'

        # Unicycle limits -- read from policy.config's [action_space] in
        # configure(), same convention as every other baseline here.
        self.enforce_lims = False
        self.max_vel = float('inf')
        self.max_wrot = float('inf')
        self.max_accel = float('inf')
        self.max_w_accel = float('inf')

        self.prev_v = 0.0
        self.prev_omega = 0.0

        self.human_num = None
        self.visible_dis = None
        self.actor = None
        self._navistar_config = None
        self._pad_offset = None

    @property
    def _unicycle_limits_active(self):
        return self.enforce_lims and self.kinematics == 'unicycle'

    def _reset_unicycle_state(self, v0=0.0, omega0=0.0):
        self.prev_v = v0
        self.prev_omega = omega0

    def configure(self, config):
        if config.has_section('action_space'):
            self.enforce_lims = config.getboolean('action_space', 'enforce_lims', fallback=False)
            self.max_vel     = config.getfloat('action_space', 'max_vel',     fallback=float('inf'))
            self.max_wrot    = config.getfloat('action_space', 'max_wrot',    fallback=float('inf'))
            self.max_accel   = config.getfloat('action_space', 'max_accel',   fallback=float('inf'))
            self.max_w_accel = config.getfloat('action_space', 'max_w_accel', fallback=float('inf'))

        navistar_root = resolve_path(config.get(self._section, 'navistar_root'))
        checkpoint_path = resolve_path(config.get(self._section, 'checkpoint'))

        if navistar_root not in sys.path:
            sys.path.insert(0, navistar_root)

        NaviSTARConfig = _load_navistar_config_class(navistar_root)
        navistar_config = NaviSTARConfig()
        self._navistar_config = navistar_config
        self.human_num = navistar_config.sim.human_num  # 10, baked into the checkpoint's weight shapes
        self.visible_dis = navistar_config.robot.visible_dis  # 10, part of the trained observation distribution

        # Padding-slot offset for predict()'s closest-N/pad scheme -- same
        # reasoning as dsrnn.py's own pad value: bounded (not astronomically
        # large) so STAR's GCN attention (which has no visibility masking,
        # unlike its cross-attention stage) sees a plausible "far away" input
        # rather than an out-of-distribution one. square_width bounds this
        # repo's actual eval scenario extent (env.config [sim] square_width).
        square_width = config.getfloat(self._section, 'square_width_for_padding', fallback=15.0)
        self._pad_offset = (
            self._pad_offset_override if self._pad_offset_override is not None
            else 2.0 * square_width
        )

        from offpolicy.agent.actor import DiagGaussianActor
        self.actor = DiagGaussianActor(navistar_config, None)
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        self.actor.load_state_dict(state_dict)
        self.actor.eval()

    def set_device(self, device):
        self.device = device
        if self.actor is not None:
            self.actor.to(device)

    def _clip_action(self, vx, vy, v_pref):
        """Replicates SAN-NaviSTAR's crowd_nav/policy/star.py STAR.clip_action
        (holonomic branch): norm-clip to v_pref only if it's exceeded."""
        act = np.array([vx, vy], dtype=np.float64)
        act_norm = np.linalg.norm(act)
        if act_norm > v_pref:
            act = act / act_norm * v_pref
        return float(act[0]), float(act[1])

    def _predict_holonomic_velocity(self, state):
        """Shared forward pass: builds STAR's observation, runs the actor,
        and returns the raw (clipped-to-v_pref) holonomic (vx, vy) NaviSTAR
        itself would have applied natively -- factored out so the unicycle
        subclass (below) and a pure-holonomic variant can both reuse it
        without duplicating the observation-construction logic."""
        self_state = state.self_state
        humans = state.human_states

        # Closest-N with far-away padding: STAR's spatial/GCN graph has a
        # fixed-size human_num slots (baked into weight shapes at training
        # time), unlike this repo's padding-aware MultiHumanRL family, so it
        # can't take a different human count per episode. See module
        # docstring for the padding-value rationale and its caveats.
        humans_sorted = sorted(
            humans, key=lambda h: np.hypot(h.px - self_state.px, h.py - self_state.py)
        )
        closest = humans_sorted[:self.human_num]

        robot_node = np.array(
            [[self_state.px, self_state.py, self_state.radius, self_state.gx, self_state.gy,
              self_state.v_pref, self_state.theta]], dtype=np.float32)
        robot_pos = np.array([[self_state.px, self_state.py, 1.0, 0.0]], dtype=np.float32)

        # spatial_edges_transformer: human_num rows + 1 goal row, each
        # [dx, dy, is_goal, is_human] -- exact column order/one-hot values
        # taken from SAN-NaviSTAR's crowd_sim.py generate_ob(): human rows
        # are [dx, dy, 0, 1], the goal row is [dx, dy, 1, 0].
        spatial = np.full((self.human_num, 4), 0.0, dtype=np.float32)
        spatial[:, 0] = self._pad_offset
        spatial[:, 1] = self._pad_offset
        spatial[:, 3] = 1.0  # human one-hot, including padded slots
        for i, h in enumerate(closest):
            spatial[i, 0] = h.px - self_state.px
            spatial[i, 1] = h.py - self_state.py
        goal_row = np.array(
            [[self_state.gx - self_state.px, self_state.gy - self_state.py, 1.0, 0.0]], dtype=np.float32)
        spatial_edges_transformer = np.concatenate((spatial, goal_row), axis=0)  # (human_num+1, 4)

        dis = np.linalg.norm(spatial_edges_transformer, axis=-1)
        visible_masks = (dis < self.visible_dis).astype(np.float32)
        visible_masks[len(closest):self.human_num] = 0.0  # padded slots never visible
        visible_masks[-1] = 1.0  # goal always "visible"

        device = self.device
        obs = {
            'robot_node': torch.as_tensor(robot_node, dtype=torch.float32, device=device),
            'spatial_edges_transformer': torch.as_tensor(spatial_edges_transformer, dtype=torch.float32, device=device),
            'visible_masks': torch.as_tensor(visible_masks, dtype=torch.float32, device=device),
            'robot_pos': torch.as_tensor(robot_pos, dtype=torch.float32, device=device),
        }
        # actor expects a batch dim of 1
        obs = {k: v.unsqueeze(0) for k, v in obs.items()}

        with torch.no_grad():
            dist = self.actor(obs, infer=True)
            action = dist.mean  # deterministic eval, matches SAN-NaviSTAR's own test_sac.py (sample=False)

        raw_action = action.squeeze(0).cpu().numpy()
        vx_des, vy_des = self._clip_action(raw_action[0], raw_action[1], self_state.v_pref)
        self.last_state = state
        return vx_des, vy_des

    def predict(self, state):
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')

        if self.reach_destination(state):
            self.prev_v = 0.0
            self.prev_omega = 0.0
            return ActionRot(0, 0)

        vx_des, vy_des = self._predict_holonomic_velocity(state)

        dt = self.time_step
        theta_now = float(state.self_state.theta) if state.self_state.theta is not None else 0.0

        v_cmd, omega_cmd = project_velocity_to_unicycle(
            vx_des, vy_des, theta_now, self.prev_v, self.prev_omega, dt,
            self.max_vel, self.max_wrot, self.max_accel, self.max_w_accel,
            self._unicycle_limits_active,
        )

        self.prev_v = v_cmd
        self.prev_omega = omega_cmd

        return ActionRot(v_cmd, omega_cmd * dt)


class NaviSTARHolonomic(_NaviSTARAdapter):
    """Diagnostic-only variant: runs the same checkpoint
    with NO unicycle projection at all -- native holonomic (vx, vy),
    instantaneous direction changes, no turn-rate/accel limits. This is
    NOT apples-to-apples with the other baselines here (all configured to
    unicycle kinematics under the shared [action_space] limits) -- it exists
    to answer a narrower question: how much of NaviSTAR's performance gap is
    attributable to the unicycle projection specifically, versus the other
    distribution shifts (human_num padding, scenario distribution, radius)
    that apply either way. Report its numbers labeled as such, never folded
    into the main comparison table uncaveated.
    """
    def __init__(self):
        super().__init__()
        self.name = 'NaviSTARHolonomic'
        self.kinematics = 'holonomic'

    def predict(self, state):
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')

        if self.reach_destination(state):
            return ActionXY(0, 0)

        vx_des, vy_des = self._predict_holonomic_velocity(state)
        return ActionXY(vx_des, vy_des)


class NaviSTAR(_NaviSTARAdapter):
    """NaviSTAR under the shared unicycle envelope -- the main comparison-table
    row. Loads the checkpoint retrained by the fork under ``baselines/navistar``
    and projects its holonomic output onto (v, omega). See NaviSTARHolonomic
    for the unprojected A/B arm."""

    def __init__(self):
        super().__init__()
        self.name = 'NaviSTAR'

    def configure(self, config):
        super().configure(config)
        # This arm exists to apply the envelope, so enforcement is forced on
        # regardless of the config flag. NaviSTARHolonomic derives from the
        # adapter directly and so does not inherit this.
        self.enforce_lims = True
