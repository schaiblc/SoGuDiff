import numpy as np
import rvo2
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionRot
from crowd_sim.envs.policy.unicycle_utils import project_velocity_to_unicycle


class ORCAUnicycle(Policy):
    """
    Unicycle-feasible imitation teacher.

    Runs ORCA (holonomic) internally to compute a desired instantaneous
    velocity, then projects that velocity onto the unicycle action space:
    a forward speed v ∈ [0, max_vel] and angular rate ω ∈ [-max_wrot, max_wrot].

    When `enforce_lims=True`, the projection also respects per-step
    acceleration limits, so the actions produced are exactly the action
    space the RL policy will operate in.

    The output is ActionRot(v, ω*dt) — same convention as ActionRot used
    everywhere else in the codebase (the `r` field stores Δθ = ω * dt).

    This eliminates the holonomic→unicycle distribution shift between IL
    and RL: the IL teacher and the RL policy emit actions with identical
    kinematics, identical (v, omega) constraints, and the robot operates
    as unicycle throughout (so theta is body orientation in both phases,
    omega is properly defined, and prev_v/prev_omega are tracked correctly).
    """

    def __init__(self):
        super().__init__()
        self.name = 'ORCA_unicycle'
        self.trainable = False
        self.multiagent_training = None
        self.kinematics = 'unicycle'  # critical: makes robot.kinematics = 'unicycle' during IL
        # Planning-radius margin beyond the true collision radius (see
        # predict()'s `self_state.radius + 0.01 + self.safety_space`).
        # 0.15 matches the buffer train.py already gives the ORCA IL teacher
        # (train.config's [imitation_learning] safety_space) so a standalone
        # orca_unicycle baseline run is on the same footing, not a more
        # aggressive/less cautious agent than the one that shaped the RL
        # policies' imitation-learning behavior. train.py overrides this
        # explicitly for the IL-teacher instance regardless, so this default
        # only takes effect for standalone (evaluate.py) baseline runs.
        self.safety_space = 0.15
        self.neighbor_dist = 10
        self.max_neighbors = 10
        self.time_horizon = 2.0
        self.time_horizon_obst = 0.50
        self.radius = 0.3
        self.max_speed = 1
        self.sim = None

        # Unicycle limits — populated either via configure() or by the
        # caller copying from the trainable policy. Set permissive defaults.
        self.enforce_lims = False
        self.max_vel = float('inf')
        self.max_wrot = float('inf')
        self.max_accel = float('inf')
        self.max_w_accel = float('inf')

        # Per-step kinodynamic state — tracked just like the trainable policy
        self.prev_v = 0.0
        self.prev_omega = 0.0

        # Heading-tracking gain for the unicycle projection.
        # ω_command ≈ k_omega * heading_error (saturated to limits).
        # k_omega = 1/dt is "turn to align this step", but that's often
        # too aggressive when limits are tight; we use a softer default.
        self.k_omega = 2.0

    @property
    def _unicycle_limits_active(self):
        return self.enforce_lims and self.kinematics == 'unicycle'

    def _reset_unicycle_state(self, v0=0.0, omega0=0.0):
        """Match the API used by CADRL/MultiHumanRL so env.reset() works uniformly."""
        self.prev_v = v0
        self.prev_omega = omega0

    def configure(self, config):
        # Read the same action_space limits the trainable policy uses, so the
        # IL teacher emits actions in the identical feasible set the RL policy
        # will operate in. All keys are optional — if absent, limits stay at inf.
        if config.has_section('action_space'):
            self.enforce_lims = config.getboolean('action_space', 'enforce_lims', fallback=False)
            self.max_vel     = config.getfloat('action_space', 'max_vel',     fallback=float('inf'))
            self.max_wrot    = config.getfloat('action_space', 'max_wrot',    fallback=float('inf'))
            self.max_accel   = config.getfloat('action_space', 'max_accel',   fallback=float('inf'))
            self.max_w_accel = config.getfloat('action_space', 'max_w_accel', fallback=float('inf'))

    def set_phase(self, phase):
        return

    def predict(self, state):
        """
        Compute a unicycle ActionRot that approximates ORCA's desired velocity.
        """
        self_state = state.self_state

        # ── 1. Run ORCA to get desired (vx, vy) ──────────────────────────
        params = self.neighbor_dist, self.max_neighbors, self.time_horizon, self.time_horizon_obst
        if self.sim is not None and self.sim.getNumAgents() != len(state.human_states) + 1:
            del self.sim
            self.sim = None
        if self.sim is None:
            self.sim = rvo2.PyRVOSimulator(self.time_step, *params, self.radius, self.max_speed)
            self.sim.addAgent(self_state.position, *params,
                              self_state.radius + 0.01 + self.safety_space,
                              self_state.v_pref, self_state.velocity)
            for human_state in state.human_states:
                self.sim.addAgent(human_state.position, *params,
                                  human_state.radius + 0.01 + self.safety_space,
                                  self.max_speed, human_state.velocity)
        else:
            self.sim.setAgentPosition(0, self_state.position)
            self.sim.setAgentVelocity(0, self_state.velocity)
            for i, human_state in enumerate(state.human_states):
                self.sim.setAgentPosition(i + 1, human_state.position)
                self.sim.setAgentVelocity(i + 1, human_state.velocity)

        velocity = np.array((self_state.gx - self_state.px, self_state.gy - self_state.py))
        speed = np.linalg.norm(velocity)
        pref_vel = velocity / speed if speed > 1 else velocity

        self.sim.setAgentPrefVelocity(0, tuple(pref_vel))
        for i, _ in enumerate(state.human_states):
            self.sim.setAgentPrefVelocity(i + 1, (0, 0))

        self.sim.doStep()
        vx_des, vy_des = self.sim.getAgentVelocity(0)

        # ── 2. Project (vx_des, vy_des) onto unicycle (v, ω), applying
        #      velocity + (if enforce_lims) acceleration limits ───────────
        dt = self.time_step
        theta_now = float(self_state.theta) if self_state.theta is not None else 0.0

        v_cmd, omega_cmd = project_velocity_to_unicycle(
            vx_des, vy_des, theta_now, self.prev_v, self.prev_omega, dt,
            self.max_vel, self.max_wrot, self.max_accel, self.max_w_accel,
            self._unicycle_limits_active, self.k_omega,
        )

        # ── 3. Update state, return ActionRot ────────────────────────────
        self.prev_v = v_cmd
        self.prev_omega = omega_cmd
        self.last_state = state
        self.sim = None  # match base ORCA convention

        return ActionRot(v_cmd, omega_cmd * dt)