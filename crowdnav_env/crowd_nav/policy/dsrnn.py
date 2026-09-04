"""Evaluation adapter for the holonomic DSRNN baseline.

Loads a DSRNN checkpoint trained by the fork under ``baselines/dsrnn`` and
exposes it through this repository's policy interface, so it can be scored by
the same harness as every other baseline. Training happens in that fork, using
the authors' own PPO pipeline; nothing here trains.

The checkpoint keeps the authors' published dynamics and PPO recipe --
holonomic ``ActionXY``, clip-to-``v_pref``, entropy 0.0, lr 4e-5 with no decay,
10e6 steps -- and changes only the surrounding task to this repository's
evaluation scenario: 1..10 humans with closest-N selection and padding, radii
0.25, 25 s time limit, and fixed per-episode human goals.

Why the underlying policy stays holonomic
-----------------------------------------
DSRNN is published as a holonomic policy. Retraining it unicycle-native was
attempted four ways -- an omega accumulator, clipping curricula, and post-hoc
clamps among them -- and every variant collapsed at evaluation (5-44% success).
Each failure traced to the re-engineered action dynamics rather than to the
SRNN itself, so this adapter makes no dynamic modifications: the network emits
(vx, vy), clipped to ``v_pref`` exactly as the authors' ``clip_action`` does.

That leaves a kinematic asymmetry against the unicycle-constrained baselines.
``DSRNN`` therefore projects the policy's output onto the shared unicycle
envelope at evaluation time, which is the reported configuration.
``DSRNNHolonomic`` runs the same checkpoint unprojected; it is a diagnostic
variant for measuring what that conversion costs, is not reported, and is not
comparable to the other baselines, which all operate under the unicycle
action space.

Import isolation
----------------
The checkpoint's fork ships its own top-level ``crowd_sim`` and ``crowd_nav``
packages, whose names collide with this repository's. Its ``config.py`` is
therefore loaded by file path rather than imported, and
``pytorchBaselines.a2c_ppo_acktr.utils`` is shimmed down to ``init`` and
``AddBias`` so that pulling in the network definition does not drag in an
unused OpenAI Baselines dependency.
"""
import importlib.util
import os
import sys
import types

import numpy as np
import torch

from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.policy.unicycle_utils import project_velocity_to_unicycle
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_nav.paths import resolve_path


def _shim_pytorchbaselines_utils():
    """Stand-in pytorchBaselines.a2c_ppo_acktr.utils module (init + AddBias).
    See crowd_nav/policy/dsrnn.py for why this exists."""
    name = 'pytorchBaselines.a2c_ppo_acktr.utils'
    if name in sys.modules:
        return
    import torch.nn as nn

    def init(module, weight_init, bias_init, gain=1):
        weight_init(module.weight.data, gain=gain)
        bias_init(module.bias.data)
        return module

    class AddBias(nn.Module):
        def __init__(self, bias):
            super().__init__()
            self._bias = nn.Parameter(bias.unsqueeze(1))

        def forward(self, x):
            bias = self._bias.t().view(1, -1) if x.dim() == 2 else self._bias.t().view(1, -1, 1, 1)
            return x + bias

    shim = types.ModuleType(name)
    shim.init = init
    shim.AddBias = AddBias
    sys.modules[name] = shim


def _load_dsrnn_config_class(dsrnn_root):
    """Load the checkpoint repo's crowd_nav/configs/config.py by file path."""
    path = os.path.join(dsrnn_root, 'crowd_nav', 'configs', 'config.py')
    spec = importlib.util.spec_from_file_location('_dsrnn_config_module', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Config


class DSRNNBase(Policy):
    """Base adapter. Subclasses fix which policy.config section they read."""
    _section = None  # set by each subclass, e.g. 'dsrnn'

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.trainable = False  # checkpoint loaded per policy.config; not trainable here
        self.multiagent_training = True
        # The AUTHOR'S kinematics. robot.set_policy() syncs robot.kinematics off
        # this, so the env applies ActionXY directly (no heading integration).
        self.kinematics = 'holonomic'
        self.v_pref = None

        self.human_num = None
        self.actor_critic = None
        self.rnn_hxs = None
        self.masks = None
        self._dsrnn_config = None
        self._pad_offset = None

    def _reset_unicycle_state(self, v0=0.0, omega0=0.0):
        """Required by this fork's reset()/eval loop (called unconditionally on
        every new episode). Name kept for interface compat even though this
        policy is holonomic and tracks no (v, omega) memory: what it resets is
        the recurrent state."""
        if self.actor_critic is not None:
            device = self.device if self.device is not None else torch.device('cpu')
            cfg = self._dsrnn_config
            self.rnn_hxs = {
                'human_node_rnn': torch.zeros(1, 1, cfg.SRNN.human_node_rnn_size, device=device),
                'human_human_edge_rnn': torch.zeros(1, self.human_num + 1, cfg.SRNN.human_human_edge_rnn_size,
                                                     device=device),
            }
            # 0.0 = "first step of a new episode" (pytorchBaselines evaluation.py
            # convention — zeroes stale recurrent contributions on step one).
            self.masks = torch.zeros(1, 1, device=device)

    def configure(self, config):
        section = self._section
        if not config.has_section(section):
            raise ValueError(f'[{section}] section missing from policy.config')
        dsrnn_root = resolve_path(config.get(section, 'dsrnn_root'))
        checkpoint_path = resolve_path(config.get(section, 'checkpoint'))

        if dsrnn_root not in sys.path:
            sys.path.insert(0, dsrnn_root)
        _shim_pytorchbaselines_utils()

        DSRNNConfig = _load_dsrnn_config_class(dsrnn_root)
        dsrnn_config = DSRNNConfig()
        # Single-episode inference override — matches the other adapters and
        # CrowdNav_DSRNN's own test.py (`actor_critic.base.nenv = 1`).
        dsrnn_config.training.num_processes = 1
        self._dsrnn_config = dsrnn_config
        self.human_num = dsrnn_config.sim.human_num
        self.v_pref = dsrnn_config.robot.v_pref

        # Padding-slot offset for episodes with fewer real humans than network
        # slots. Same value the training envs used: 2 * sim.square_width (the
        # PORT training env pads exactly this way; the pristine ORIG env has no
        # padding at all because its benchmark always spawns exactly
        # sim.human_num humans — for THAT checkpoint any padding input is
        # out-of-distribution, unavoidable given the fixed-size graph; the
        # offset follows the same "far but not absurd" logic as dsrnn.py).
        square_width = getattr(dsrnn_config.sim, 'square_width',
                               config.getfloat(section, 'square_width_for_padding', fallback=15.0))
        self._pad_offset = 2.0 * square_width

        import gym
        obs_space = {
            'robot_node': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 7), dtype=np.float32),
            'temporal_edges': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 2), dtype=np.float32),
            'spatial_edges': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.human_num, 2), dtype=np.float32),
        }
        high = np.inf * np.ones([2, ])
        action_space = gym.spaces.Box(-high, high, dtype=np.float32)

        from pytorchBaselines.a2c_ppo_acktr.model import Policy as DSRNNActorCritic
        self.actor_critic = DSRNNActorCritic(obs_space, action_space, base_kwargs=dsrnn_config, base='srnn')
        self.actor_critic.base.nenv = 1
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        self.actor_critic.load_state_dict(state_dict)
        self.actor_critic.eval()

    def set_device(self, device):
        self.device = device
        if self.actor_critic is not None:
            self.actor_critic.to(device)
        if self._dsrnn_config is not None:
            # srnn_model.py branches on this flag for an internal hidden-state
            # buffer's device — keep in sync (same as dsrnn.py).
            self._dsrnn_config.training.cuda = (device.type == 'cuda')

    def predict(self, state):
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.rnn_hxs is None:
            self._reset_unicycle_state()

        if self.reach_destination(state):
            return self._zero_action()

        self_state = state.self_state
        humans = state.human_states

        # Closest-N into the fixed-size spatial-edge graph, far-away padding
        # for empty slots (identical scheme to the PORT training env).
        humans_sorted = sorted(
            humans, key=lambda h: np.hypot(h.px - self_state.px, h.py - self_state.py)
        )
        closest = humans_sorted[:self.human_num]

        robot_node = np.array(
            [[self_state.px, self_state.py, self_state.radius, self_state.gx, self_state.gy,
              self_state.v_pref, self_state.theta]], dtype=np.float32)
        temporal_edges = np.array([self_state.vx, self_state.vy], dtype=np.float32)
        spatial_edges = np.full((self.human_num, 2), self._pad_offset, dtype=np.float32)
        for i, h in enumerate(closest):
            spatial_edges[i] = [h.px - self_state.px, h.py - self_state.py]

        device = self.device
        obs = {
            'robot_node': torch.as_tensor(robot_node, dtype=torch.float32, device=device).unsqueeze(0),
            'temporal_edges': torch.as_tensor(temporal_edges, dtype=torch.float32, device=device).view(1, 1, 2),
            'spatial_edges': torch.as_tensor(spatial_edges, dtype=torch.float32, device=device).unsqueeze(0),
        }

        with torch.no_grad():
            _, action, _, self.rnn_hxs = self.actor_critic.act(
                obs, self.rnn_hxs, self.masks, deterministic=(self.phase != 'train'))

        self.masks = torch.ones(1, 1, device=device)

        raw_action = action.squeeze(0).squeeze(0).cpu().numpy()
        return self._act_to_action(float(raw_action[0]), float(raw_action[1]), state)

    def _zero_action(self):
        """Zero command in this policy's native action space."""
        return ActionXY(0, 0)

    def _clip_to_v_pref(self, vx, vy):
        """AUTHOR'S holonomic clip_action, verbatim semantics: scale down to
        v_pref only if the raw command exceeds it. No other modification."""
        act_norm = np.hypot(vx, vy)
        if act_norm > self.v_pref:
            vx = vx / act_norm * self.v_pref
            vy = vy / act_norm * self.v_pref
        return vx, vy

    def _act_to_action(self, vx, vy, state):
        vx, vy = self._clip_to_v_pref(vx, vy)
        return ActionXY(vx, vy)


class DSRNNHolonomic(DSRNNBase):
    """Attempt B — author dynamics trained on this repo's evaluation task."""
    _section = 'dsrnn_holonomic'


class DSRNN(DSRNNHolonomic):
    """Attempt B checkpoint with its holonomic (vx, vy) output projected
    POST-HOC onto this repo's shared unicycle envelope, at EVALUATION TIME
    ONLY (no retraining — the training run is untouched).

    Uses crowd_sim.envs.policy.unicycle_utils.project_velocity_to_unicycle,
    the exact projection ORCAUnicycle/SFMUnicycle emit, so the constraint set
    is the same one the unicycle baseline rows operate under: forward speed
    <= max_vel, |dv| <= max_accel*dt, |omega| <= max_wrot,
    |domega| <= max_w_accel*dt, with heading error turning the agent in place
    (speed scaled by max(0, cos(heading_err))). kinematics='unicycle' makes
    the env integrate heading from omega.

    This is the reported DSRNN configuration: retraining unicycle-native did
    not converge, so the shared action space is imposed at evaluation time
    instead.
    """
    # Main comparison-table row; reads its own config section so that
    # [dsrnn] and [dsrnn_holonomic] can be pointed at different checkpoints.
    _section = 'dsrnn'

    def __init__(self):
        super().__init__()
        self.kinematics = 'unicycle'
        # Same limit fields/defaults as ORCAUnicycle; populated in configure().
        self.enforce_lims = False
        self.max_vel = float('inf')
        self.max_wrot = float('inf')
        self.max_accel = float('inf')
        self.max_w_accel = float('inf')
        # Per-step kinodynamic state (mirrors ORCAUnicycle).
        self.prev_v = 0.0
        self.prev_omega = 0.0

    def configure(self, config):
        super().configure(config)
        if config.has_section('action_space'):
            self.enforce_lims = config.getboolean('action_space', 'enforce_lims', fallback=False)
            self.max_vel     = config.getfloat('action_space', 'max_vel',     fallback=float('inf'))
            self.max_wrot    = config.getfloat('action_space', 'max_wrot',    fallback=float('inf'))
            self.max_accel   = config.getfloat('action_space', 'max_accel',   fallback=float('inf'))
            self.max_w_accel = config.getfloat('action_space', 'max_w_accel', fallback=float('inf'))
        # This variant exists to apply the envelope — force enforcement on
        # regardless of the config flag's current value.
        self.enforce_lims = True

    def _reset_unicycle_state(self, v0=0.0, omega0=0.0):
        """Reset RNN state (parent) AND the per-step (v, omega) memory."""
        super()._reset_unicycle_state(v0, omega0)
        self.prev_v = float(v0)
        self.prev_omega = float(omega0)

    def _zero_action(self):
        return ActionRot(0, 0)

    def _act_to_action(self, vx, vy, state):
        vx, vy = self._clip_to_v_pref(vx, vy)
        # injected by the env on reset; None/absent only outside a live env
        dt = float(getattr(self, 'time_step', None) or 0.25)
        theta_now = float(state.self_state.theta) if state.self_state.theta is not None else 0.0
        v_cmd, omega_cmd = project_velocity_to_unicycle(
            vx, vy, theta_now, self.prev_v, self.prev_omega, dt,
            self.max_vel, self.max_wrot, self.max_accel, self.max_w_accel,
            self.enforce_lims)
        self.prev_v = v_cmd
        self.prev_omega = omega_cmd
        return ActionRot(v_cmd, omega_cmd * dt)
