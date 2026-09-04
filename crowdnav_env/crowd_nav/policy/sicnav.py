"""Evaluation adapter for the SICNav-np baseline.

SICNav-np is the non-privileged variant of SICNav's ``CollisionAvoidMPC``
("CAMPC"): a bilevel CasADi/IPOPT model-predictive controller with an ORCA-KKT
human model. It has no learned weights and no training step -- this adapter
constructs and configures ``CollisionAvoidMPC`` from the fork under
``baselines/sicnav`` and forwards ``predict()`` calls to it, converting state
and action formats along the way.

Because the solver reaches IPOPT through CasADi's plugin system, SICNav runs in
its own Python environment (``requirements/sicnav.txt``) rather than the main
one. The evaluation harness itself is environment-agnostic.

Variant selection
-----------------
``baselines/sicnav/sicnav/configs/policy.config`` ``[mpc_env]`` decides which
variant ``CollisionAvoidMPC`` becomes. ``hum_model='orca_casadi_kkt'`` with
``priviledged_info=false`` selects SICNav-np, which gets no privileged access
to other agents' goals or intent -- matching every other baseline here, none of
which is privileged either. Setting ``priviledged_info=true`` gives SICNav-p,
and ``hum_model='cvmm'`` gives the non-interactive MPC-CVMM baseline. Switching
variants means editing that config, not this adapter.

Config alignment
----------------
``[mpc_env]`` and ``[humans]`` in the fork's own ``policy.config`` are set to
hold the action limits and human radius common with this repository's
``[action_space]`` and ``[humans]`` blocks:

- ``pref_speed`` = ``max_speed`` = 1.0, this repository's ``max_vel``.
- ``max_rev_speed`` = 0.0; no baseline here commands reverse.
- ``max_rot_degrees`` = 179.909, i.e. ``max_wrot`` = 3.14 rad/s in degrees.
  Despite the key's name, ``mpc_env.py`` bounds the input ``om_r`` directly in
  rad/s -- see the inline comment there.
- ``max_l_acc`` = ``max_l_dcc`` = 1.5, this repository's ``max_accel``.
  ``mpc_env.py`` already bounds the per-step change in ``|v|`` by
  ``max_l_acc * time_step``, matching ``max_accel * time_step`` exactly.
- ``[humans] radius`` = 0.25.

``[mpc_env] human_v_max_assumption`` and ``[humans] max_speed`` are both 1.0,
raised from upstream's 0.5 and 1.2 to match the humans this environment
actually simulates (``env.config [humans] v_pref = 1``, and the ORCA policy in
``crowd_sim/envs/policy/orca.py`` hardcodes ``max_speed = 1``).
``human_v_max_assumption`` feeds the non-privileged ORCA-KKT model's belief
about how fast humans can move (``orca_casadi.py``'s ``v_max_unobservable``).
Leaving it at 0.5 makes the controller assume humans move at half their real
speed, and its predictions of human motion are then systematically too slow;
that alone drove a roughly 20% collision rate in evaluation.

Angular acceleration
--------------------
``max_w_accel`` has no counterpart inside SICNav's MPC. Unlike linear speed,
the rate of change of ``om_r`` is only softly penalized (``q_om_prev_dot``) and
never hard-constrained. Adding a hard constraint would mean threading an
``om_r_prev`` state through the entire bilevel ORCA-KKT formulation --
``gen_kin_model``, the cost, the terminal constraint and the human constraints
-- which is beyond a non-retraining adapter and risks destabilizing the solver.
Instead, as in the DSRNN and NaviSTAR adapters, this module tracks
``prev_omega`` itself and clips the returned omega to within
``max_w_accel * time_step`` of it. See ``predict()`` below.
"""
import configparser
import os
import sys
from types import SimpleNamespace

import numpy as np

from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionRot
from crowd_nav.paths import resolve_path


class SICNav(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'SICNav'
        self.trainable = False  # optimization-based controller, no checkpoint to load
        self.multiagent_training = True
        self.kinematics = 'unicycle'

        # Angular-acceleration limit -- read from policy.config's [action_space] in
        # configure(), same convention as every other baseline here. SICNav's MPC has
        # no internal equivalent (see module docstring), so it's enforced here instead.
        self.max_w_accel = float('inf')
        self.max_vel = float('inf')
        self.max_wrot = float('inf')
        self.max_accel = float('inf')
        self.prev_omega = 0.0
        self.prev_v = 0.0

        self._campc = None
        self._sicnav_full_state_cls = None

    def _reset_unicycle_state(self, v0=0.0, omega0=0.0):
        # The MPC tracks its own v_prev internally (X_r's v_r_prev, re-read from the
        # true state each solve), but the adapter now also tracks prev_v so it can
        # apply the SAME final acceleration clamp ORCA/SFM apply -- see predict().
        self.prev_omega = omega0
        self.prev_v = v0

    def configure(self, config):
        if config.has_section('action_space'):
            # Read the SAME [action_space] block ORCA/SFM/the RL policies read, so
            # SICNav's executed action is held to an identical feasible set.
            gf = lambda k: config.getfloat('action_space', k, fallback=float('inf'))
            self.max_w_accel = gf('max_w_accel')
            self.max_vel     = gf('max_vel')
            self.max_wrot    = gf('max_wrot')
            self.max_accel   = gf('max_accel')

        sicnav_root = resolve_path(config.get('sicnav', 'sicnav_root'))
        if sicnav_root not in sys.path:
            sys.path.insert(0, sicnav_root)

        from sicnav.policy.campc import CollisionAvoidMPC
        from crowd_sim_plus.envs.utils.state_plus import FullState as SICNavFullState
        self._sicnav_full_state_cls = SICNavFullState

        sicnav_policy_config = configparser.RawConfigParser()
        sicnav_policy_config.read(os.path.join(sicnav_root, 'sicnav', 'configs', 'policy.config'))

        self._campc = CollisionAvoidMPC()
        self._campc.configure(sicnav_policy_config)

    def set_device(self, device):
        self.device = device

    def set_env(self, env):
        super().set_env(env)
        # CollisionAvoidMPC.set_env() reads env.time_limit/time_step/config and builds its
        # dummy_human from env.config's [humans] section -- passing this repo's real env here
        # (rather than SICNav's own) is exactly how radius/time_step end up held in common
        # automatically, since env.config's [humans] radius/v_pref/time_step are this repo's own.
        self._campc.set_env(env)

    def predict(self, state):
        self_state = state.self_state
        sicnav_self_state = self._sicnav_full_state_cls(
            px=self_state.px, py=self_state.py, vx=self_state.vx, vy=self_state.vy,
            radius=self_state.radius, gx=self_state.gx, gy=self_state.gy,
            v_pref=self_state.v_pref, theta=self_state.theta, omega=self_state.omega,
        )
        proxy_state = SimpleNamespace(
            self_state=sicnav_self_state,
            human_states=state.human_states,
            static_obs=state.static_obs,
        )
        # Hand the MPC the previously EXECUTED omega before it solves.
        # Inert against the stock SICNav repo (prev_avel is vestigial there --
        # reset in reset_scenario_values but never assigned or read). Against a
        # SICNav tree carrying the omega-rate constraint (see its [mpc_env]
        # max_w_accel), this anchors that constraint to the robot's real angular
        # state so the MPC plans a feasible turn rate, instead of planning an
        # infeasible one and having it clipped below.
        self._campc.prev_avel = self.prev_omega
        self._campc.prev_lvel = self.prev_v
        sicnav_action = self._campc.predict(proxy_state)

        # ---------------------------------------------------------------
        # Final action-space enforcement, mirroring EXACTLY what
        # crowd_sim/envs/policy/unicycle_utils.py:project_velocity_to_unicycle
        # applies to ORCA_unicycle and SFM_unicycle (same order, same
        # re-clip after the acceleration clamp), so every baseline's
        # executed action is held to one identical feasible set.
        #
        # These should all be no-ops: the MPC bounds v, omega and |dv|
        # internally, and (in a SICNav tree carrying the omega-rate
        # constraint) |domega| too. They are a safety net for the one path
        # that bypasses the optimizer -- select_action's warmstart override,
        # which returns a solution from a separate CasADi function that
        # carries none of the opti bounds. That path is now also gated on
        # feasibility inside campc.py, so this should never bind; it is kept
        # because "should never" is not "cannot", and because relying on a
        # SOFT (slack-penalized) internal bound is structurally weaker than
        # the hard clamp the other baselines get.
        # ---------------------------------------------------------------
        dt = self.time_step
        v_cmd = float(sicnav_action.v)
        omega_cmd = sicnav_action.r / dt if dt else 0.0

        v_cmd = float(np.clip(v_cmd, 0.0, self.max_vel))
        omega_cmd = float(np.clip(omega_cmd, -self.max_wrot, self.max_wrot))

        v_cmd = float(np.clip(v_cmd, self.prev_v - self.max_accel * dt,
                                     self.prev_v + self.max_accel * dt))
        v_cmd = float(np.clip(v_cmd, 0.0, self.max_vel))
        omega_cmd = float(np.clip(omega_cmd, self.prev_omega - self.max_w_accel * dt,
                                             self.prev_omega + self.max_w_accel * dt))
        omega_cmd = float(np.clip(omega_cmd, -self.max_wrot, self.max_wrot))

        self.prev_v = v_cmd
        self.prev_omega = omega_cmd
        return ActionRot(v_cmd, omega_cmd * dt)
