"""Evaluation adapter for the HEIGHT baseline.

HEIGHT here is the ``selfAttn_merge_SRNN`` non-lidar ablation, retrained by the
fork under ``baselines/height`` using the authors' own PPO stack and their
non-pybullet ``CrowdSimVarNum-v0`` environment, with only the surrounding task
changed to this repository's evaluation scenario: uniform-square robot and
human spawns, 1-10 actual humans behind fixed 10-slot padded observations,
radii 0.25, v_pref 1, time step 0.25 s, 25 s time limit. This module loads the
resulting checkpoint and runs its forward pass for ``predict()``; nothing here
trains. ``baselines/height/crowd_nav/configs/config.py`` and
``crowd_sim/envs/crowd_sim_var_human.py`` record each deviation from upstream.

Observation construction
------------------------
Mirrors ``CrowdSimVarNum.generate_ob`` exactly:

- ``robot_node`` (1, 7): ``[px, py, r, gx, gy, v_pref, theta]``, the field
  order of ``agent.get_full_state_list_noV()``.
- ``temporal_edges`` (1, 2): robot velocity ``[vx, vy]``.
- ``spatial_edges`` (10, 2): human positions relative to the robot, sorted by
  distance. Unused slots are padded at (15, 15), the same value
  ``generate_ob`` itself substitutes for infinity, so padding introduces no
  distribution shift.
- ``detected_human_num`` (1,): count of real entries, floored at 1 to match the
  environment's convention of assuming one dummy human at (15, 15) when nobody
  is nearby.

``SpatialEdgeSelfAttn`` masks attention by ``detected_human_num``, so padded
slots are excluded from every attention computation rather than merely being
given a far-away position.

Kinematics
----------
The checkpoint is holonomic, emitting instantaneous (vx, vy) normalized to
``v_pref`` by the authors' ``clip_action``. As with DSRNN and NaviSTAR, two
arms are registered: ``HEIGHT`` projects that output onto this
repository's unicycle envelope through ``project_velocity_to_unicycle`` (the
same helper and limits as every other baseline) and is the reported
configuration, while ``HEIGHTHolonomic`` runs the raw holonomic output as an
unreported diagnostic arm, so the cost of the projection can be measured.

Import isolation
----------------
The fork ships its own top-level ``crowd_sim`` and ``crowd_nav`` packages,
whose names collide with this repository's -- already bound in ``sys.modules``
by the time ``policy_factory`` imports this module. So:

- its ``crowd_nav/configs/config.py`` is loaded by file path via importlib,
  which is safe because that file has no imports of its own;
- only the distinctly named ``training.*`` package (the network itself) is
  imported normally, after the fork's root is placed at ``sys.path[0]``.

``training`` must not already be bound from a different fork when this adapter
loads. That holds in practice because each baseline is evaluated in its own
process.
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


def _load_height_config_class(height_root):
    """Load baselines/height's crowd_nav/configs/config.py by file path
    (not package import) — avoids the crowd_nav package-name collision."""
    path = os.path.join(height_root, 'crowd_nav', 'configs', 'config.py')
    spec = importlib.util.spec_from_file_location('_height_config_module', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Config


class HEIGHT(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'HEIGHT'
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

        self.human_num = None       # observation slot count (10)
        self.actor_critic = None
        self._height_config = None
        self.device = None
        self.rnn_hxs = None
        self.masks = None

    @property
    def _unicycle_limits_active(self):
        return self.enforce_lims and self.kinematics == 'unicycle'

    def _reset_unicycle_state(self, v0=0.0, omega0=0.0):
        self.prev_v = v0
        self.prev_omega = omega0
        # baselines/height: this is a recurrent policy (EndRNN GRU) — training
        # (PPO's RolloutStorage) carries its hidden state across every step
        # of an episode. _predict_holonomic_velocity used to hand the
        # network a fresh zeroed rnn_hxs on EVERY call and discard whatever
        # act() returned, so eval ran it effectively memoryless — something
        # it never saw in training. Fixed to match dsrnn.py's pattern:
        # persist rnn_hxs/masks as instance state, reset only here (called
        # unconditionally on every new episode — crowd_sim.py:797,
        # explorer.py:43), carry the returned state forward every other step.
        if self.actor_critic is not None:
            device = self.device if self.device is not None else torch.device('cpu')
            self.rnn_hxs = {'rnn': torch.zeros(
                1, 1, self.actor_critic.base.human_node_rnn_size, device=device)}
            # 0.0 = "first step of a new episode" (pytorchBaselines evaluation.py
            # convention — zeroes stale recurrent contributions on step one).
            self.masks = torch.zeros(1, 1, device=device)

    def configure(self, config):
        if config.has_section('action_space'):
            self.enforce_lims = config.getboolean('action_space', 'enforce_lims', fallback=False)
            self.max_vel     = config.getfloat('action_space', 'max_vel',     fallback=float('inf'))
            self.max_wrot    = config.getfloat('action_space', 'max_wrot',    fallback=float('inf'))
            self.max_accel   = config.getfloat('action_space', 'max_accel',   fallback=float('inf'))
            self.max_w_accel = config.getfloat('action_space', 'max_w_accel', fallback=float('inf'))
        # This variant exists to apply the envelope — force enforcement on
        # regardless of the config flag's current value (same as
        # DSRNN).
        self.enforce_lims = True

        height_root = resolve_path(config.get('height', 'height_root'))
        checkpoint_path = resolve_path(config.get('height', 'checkpoint'))

        # The network reads nenv from config.training.num_processes at
        # forward time (selfAttn_srnn_merge.py:294); we override it to 1
        # after construction below (single-env evaluation).
        height_config = _load_height_config_class(height_root)
        self._height_config = height_config

        self.human_num = height_config.sim.human_num + height_config.sim.human_num_range  # fixed slot count

        if height_root not in sys.path:
            sys.path.insert(0, height_root)

        from gym.spaces.box import Box
        import gym.spaces
        obs_space = {
            'robot_node': Box(low=-np.inf, high=np.inf, shape=(1, 7), dtype=np.float32),
            'temporal_edges': Box(low=-np.inf, high=np.inf, shape=(1, 2), dtype=np.float32),
            'spatial_edges': Box(low=-np.inf, high=np.inf, shape=(self.human_num, 2), dtype=np.float32),
            'detected_human_num': Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
        }
        action_space = Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

        from training.networks.model import Policy as HeightPolicy
        self.actor_critic = HeightPolicy(obs_space, action_space,
                                         base_kwargs=height_config,
                                         base='selfAttn_merge_srnn')

        state_dict = torch.load(checkpoint_path, map_location='cpu')
        self.actor_critic.load_state_dict(state_dict)
        self.actor_critic.eval()
        # single-env batch at forward time (see comment above)
        self.actor_critic.base.nenv = 1

    def set_device(self, device):
        self.device = device
        if self.actor_critic is not None:
            self.actor_critic.to(device)
        if self._height_config is not None:
            # keep the base's create_attn_mask .cpu()/.cuda() branching in
            # sync with where the tensors actually are (same as dsrnn.py)
            self._height_config.training.cuda = (
                torch.device(device).type == 'cuda')

    def _predict_holonomic_velocity(self, state):
        """Builds CrowdSimVarNum's observation, runs the network's Gaussian
        head mean, clips to v_pref (the author's clip_action semantics), and
        returns the raw holonomic (vx, vy)."""
        self_state = state.self_state
        humans = sorted(state.human_states,
                        key=lambda h: np.hypot(h.px - self_state.px, h.py - self_state.py))
        closest = humans[:self.human_num]

        robot_node = np.array(
            [[self_state.px, self_state.py, self_state.radius, self_state.gx, self_state.gy,
              self_state.v_pref, self_state.theta]], dtype=np.float32)
        temporal_edges = np.array(
            [[self_state.vx, self_state.vy]], dtype=np.float32)

        spatial_edges = np.full((self.human_num, 2), 15.0, dtype=np.float32)
        for i, h in enumerate(closest):
            spatial_edges[i, 0] = h.px - self_state.px
            spatial_edges[i, 1] = h.py - self_state.py

        # >= 1: the env feeds one dummy human at (15, 15) when nobody is
        # detected so pack_padded_sequence works; mirror that here.
        detected_human_num = np.array([max(1, len(closest))], dtype=np.float32)

        device = self.device
        # Leading batch dim of 1 on every component: reshapeT() inside the base
        # splits inputs as [seq_len, nenv, *component_shape], and the training
        # stack always fed them batched by num_processes.
        inputs = {
            'robot_node': torch.as_tensor(robot_node, device=device).unsqueeze(0),
            'temporal_edges': torch.as_tensor(temporal_edges, device=device).unsqueeze(0),
            'spatial_edges': torch.as_tensor(spatial_edges, device=device).unsqueeze(0),
            'detected_human_num': torch.as_tensor(detected_human_num, device=device).unsqueeze(0),
        }
        # Persistent recurrent state across steps (see _reset_unicycle_state).
        # Shape mirrors RolloutStorage's per-step slice at nenv=1:
        # recurrent_hidden_states['rnn'][step] = [num_processes, 1, SRNN.human_node_rnn_size].
        # (Policy has no recurrent_hidden_state_size property for this base;
        # the size lives on the base network itself.)
        if self.rnn_hxs is None:
            self._reset_unicycle_state(self.prev_v, self.prev_omega)

        with torch.no_grad():
            _, action, _, self.rnn_hxs = self.actor_critic.act(
                inputs, self.rnn_hxs, self.masks, deterministic=True)

        self.masks = torch.ones(1, 1, device=device)

        vx_des, vy_des = float(action[0, 0]), float(action[0, 1])

        # author clip_action (holonomic branch): norm-clip to v_pref
        act_norm = np.hypot(vx_des, vy_des)
        if act_norm > self_state.v_pref:
            scale = self_state.v_pref / act_norm
            vx_des *= scale
            vy_des *= scale

        self.last_state = state
        return vx_des, vy_des

    def predict(self, state):
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')

        if self.reach_destination(state):
            self._reset_unicycle_state(0.0, 0.0)
            return self._zero_action()

        vx_des, vy_des = self._predict_holonomic_velocity(state)
        return self._act_to_action(vx_des, vy_des, state)

    # ---- kinematic hooks (subclass picks the output representation) ----
    def _zero_action(self):
        return ActionRot(0, 0)

    def _act_to_action(self, vx, vy, state):
        dt = float(getattr(self, 'time_step', None) or 0.25)
        theta_now = float(state.self_state.theta) if state.self_state.theta is not None else 0.0

        v_cmd, omega_cmd = project_velocity_to_unicycle(
            vx, vy, theta_now, self.prev_v, self.prev_omega, dt,
            self.max_vel, self.max_wrot, self.max_accel, self.max_w_accel,
            self._unicycle_limits_active,
        )
        self.prev_v = v_cmd
        self.prev_omega = omega_cmd
        return ActionRot(v_cmd, omega_cmd * dt)


class HEIGHTHolonomic(HEIGHT):
    """Diagnostic-only variant: runs the exact same pretrained checkpoint
    with NO unicycle projection at all -- native holonomic (vx, vy),
    instantaneous direction changes. This is NOT apples-to-apples with the
    other baselines here (all configured to unicycle kinematics under the
    shared [action_space] limits) -- it is the A/B arm that isolates how much
    of the gap is attributable to the unicycle projection specifically.
    Report its numbers labeled as such, never folded into the main comparison
    table uncaveated -- identical protocol to dsrnn_holonomic/navistar A/B arms.
    """
    def __init__(self):
        super().__init__()
        self.name = 'HEIGHTHolonomic'
        self.kinematics = 'holonomic'

    def _zero_action(self):
        return ActionXY(0, 0)

    def _act_to_action(self, vx, vy, state):
        return ActionXY(vx, vy)
