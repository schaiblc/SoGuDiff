from crowd_sim.envs.policy.social_force import SFM
from crowd_sim.envs.utils.action import ActionRot
from crowd_sim.envs.policy.unicycle_utils import project_velocity_to_unicycle


class SFMUnicycle(SFM):
    """
    Unicycle-feasible Social Force Model baseline.

    Runs the same force-based holonomic velocity computation as SFM
    (SFM._compute_desired_velocity), then projects it onto a unicycle
    (v, omega) action using the same heading-tracking + acceleration-limit
    projection ORCAUnicycle uses (crowd_sim.envs.policy.unicycle_utils).
    Without this, SFM would operate as an unconstrained holonomic agent —
    not comparable to the unicycle-constrained RL baselines it's meant to
    sit alongside.
    """

    def __init__(self):
        super().__init__()
        self.name = 'SFM_unicycle'
        self.kinematics = 'unicycle'

        # Give the robot baseline the same planning margin ORCAUnicycle uses
        # (0.15, matching train.config's IL-teacher safety_space), instead of
        # SFM's own default of 0 — see SFM.__init__ for why that default has
        # to stay 0 (it's also what simulated humans run).
        self.safety_space = 0.15

        # Unicycle limits — populated via configure(). Permissive defaults.
        self.enforce_lims = False
        self.max_vel = float('inf')
        self.max_wrot = float('inf')
        self.max_accel = float('inf')
        self.max_w_accel = float('inf')

        # Per-step kinodynamic state, tracked like the trainable policies.
        self.prev_v = 0.0
        self.prev_omega = 0.0

        # Same heading-tracking gain default as ORCAUnicycle.
        self.k_omega = 2.0

    @property
    def _unicycle_limits_active(self):
        return self.enforce_lims and self.kinematics == 'unicycle'

    def _reset_unicycle_state(self, v0=0.0, omega0=0.0):
        """Match the API used by CADRL/MultiHumanRL/ORCAUnicycle so env.reset() works uniformly."""
        self.prev_v = v0
        self.prev_omega = omega0

    def configure(self, config, section='humans'):
        super().configure(config, section=section)
        # Read the same action_space limits the trainable policies use, so
        # SFM emits actions in the identical feasible set.
        if config.has_section('action_space'):
            self.enforce_lims = config.getboolean('action_space', 'enforce_lims', fallback=False)
            self.max_vel     = config.getfloat('action_space', 'max_vel',     fallback=float('inf'))
            self.max_wrot    = config.getfloat('action_space', 'max_wrot',    fallback=float('inf'))
            self.max_accel   = config.getfloat('action_space', 'max_accel',   fallback=float('inf'))
            self.max_w_accel = config.getfloat('action_space', 'max_w_accel', fallback=float('inf'))

    def predict(self, state):
        vx_des, vy_des = self._compute_desired_velocity(state)
        theta_now = float(state.self_state.theta) if state.self_state.theta is not None else 0.0
        dt = self.time_step

        v_cmd, omega_cmd = project_velocity_to_unicycle(
            vx_des, vy_des, theta_now, self.prev_v, self.prev_omega, dt,
            self.max_vel, self.max_wrot, self.max_accel, self.max_w_accel,
            self._unicycle_limits_active, self.k_omega,
        )

        self.prev_v = v_cmd
        self.prev_omega = omega_cmd
        self.last_state = state

        return ActionRot(v_cmd, omega_cmd * dt)
