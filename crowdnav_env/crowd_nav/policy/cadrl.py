import torch
import torch.nn as nn
import numpy as np
import itertools
import logging
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.utils.state import ObservableState, FullState


def mlp(input_dim, mlp_dims, last_relu=False):
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        if i != len(mlp_dims) - 2 or last_relu:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    return net


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, mlp_dims):
        super().__init__()
        self.value_network = mlp(input_dim, mlp_dims)

    def forward(self, state):
        value = self.value_network(state)
        return value


class CADRL(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'CADRL'
        self.trainable = True
        self.multiagent_training = None
        self.kinematics = None
        self.max_vel = float('inf')
        self.max_wrot = float('inf')
        self.max_accel = float('inf')
        self.max_w_accel = float('inf')
        self.enforce_lims = False
        self.epsilon = None
        self.gamma = None
        self.sampling = None
        self.speed_samples = None
        self.rotation_samples = None
        self.query_env = None
        self.action_space = None
        self.speeds = None
        self.rotations = None
        self.action_values = None
        self.with_om = None
        self.cell_num = None
        self.cell_size = None
        self.om_channel_size = None

        # Base self_state_dim: (dg, v_pref, theta, r, vx, vy) = 6
        # When enforce_lims=True + unicycle, we append (v, omega) → 8.
        # This is set in configure() once we know both flags.
        self.self_state_dim = 6
        self.human_state_dim = 7
        self.joint_state_dim = self.self_state_dim + self.human_state_dim
        # rotate() outputs 13 base dims; 15 when enforce_lims+unicycle.
        # MultiHumanRL.input_dim() must use this, not joint_state_dim.
        self._rotated_state_dim = 13  # updated in configure()


        # Runtime unicycle state — always tracked when enforce_lims=True,
        # regardless of phase, so train and test see the same dynamics.
        self.prev_v = 0.0
        self.prev_omega = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def _unicycle_limits_active(self):
        """True when we're in unicycle mode with limits enforced."""
        return self.enforce_lims and self.kinematics == 'unicycle'

    def _reset_unicycle_state(self, v0=0.0, omega0=0.0):
        """Call at the start of every episode."""
        self.prev_v = v0
        self.prev_omega = omega0

    def configure(self, config):
        self.set_common_parameters(config)



        mlp_dims = [int(x) for x in config.get('cadrl', 'mlp_dims').split(', ')]
        self.model = ValueNetwork(self.joint_state_dim, mlp_dims)
        self.multiagent_training = config.getboolean('cadrl', 'multiagent_training')
        logging.info('Policy: CADRL without occupancy map')

    def set_common_parameters(self, config):
        self.gamma = config.getfloat('rl', 'gamma')
        self.kinematics = config.get('action_space', 'kinematics')
        self.enforce_lims = config.getboolean('action_space', 'enforce_lims')
        self.max_vel = config.getfloat('action_space', 'max_vel')
        self.max_wrot = config.getfloat('action_space', 'max_wrot')
        self.max_accel = config.getfloat('action_space', 'max_accel')
        self.max_w_accel = config.getfloat('action_space', 'max_w_accel')
        self.sampling = config.get('action_space', 'sampling')
        self.speed_samples = config.getint('action_space', 'speed_samples')
        self.rotation_samples = config.getint('action_space', 'rotation_samples')
        self.query_env = config.getboolean('action_space', 'query_env')
        self.cell_num = config.getint('om', 'cell_num')
        self.cell_size = config.getfloat('om', 'cell_size')
        self.om_channel_size = config.getint('om', 'om_channel_size')
        # Update derived dims now that kinematics and enforce_lims are known.
        # This must happen here (not only in configure()) so that subclasses
        # like LstmRL and SARL that call set_common_parameters() then
        # immediately call input_dim() to build their models get the right value.
        if self.enforce_lims and self.kinematics == 'unicycle':
            self.self_state_dim = 8
            self._rotated_state_dim = 15
        else:
            self.self_state_dim = 6
            self._rotated_state_dim = 13
        self.joint_state_dim = self.self_state_dim + self.human_state_dim

    def set_device(self, device):
        self.device = device
        self.model.to(device)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    # ------------------------------------------------------------------
    # Action space
    # ------------------------------------------------------------------

    def build_action_space(self, v_pref):
        """
        Build the *nominal* (unconstrained) action space.
        Stored as self.action_space / self.speeds / self.rotations.
        Per-step feasible subsets are computed in _feasible_action_space().
        """
        holonomic = self.kinematics == 'holonomic'
        speeds = [
            (np.exp((i + 1) / self.speed_samples) - 1) / (np.e - 1) * self.max_vel
            for i in range(self.speed_samples)
        ]
        if holonomic:
            rotations = np.linspace(0, 2 * np.pi, self.rotation_samples, endpoint=False)
        else:
            rotations = np.linspace(
                -self.max_wrot * self.time_step,
                 self.max_wrot * self.time_step,
                 self.rotation_samples
            )

        action_space = [ActionXY(0, 0) if holonomic else ActionRot(0, 0)]
        for rotation, speed in itertools.product(rotations, speeds):
            if holonomic:
                action_space.append(ActionXY(speed * np.cos(rotation), speed * np.sin(rotation)))
            else:
                action_space.append(ActionRot(speed, rotation))

        self.speeds = speeds
        self.rotations = rotations
        self.action_space = action_space

    def _feasible_action_space(self):
        """
        When unicycle limits are active, return the subset of self.action_space
        reachable from (self.prev_v, self.prev_omega) within one timestep.

        ActionRot.r is stored as Δθ = omega * dt, so omega = r / dt.
        """
        if not self._unicycle_limits_active:
            return self.action_space

        dt = self.time_step
        v_prev = self.prev_v
        w_prev = self.prev_omega

        v_lo = max(-self.max_vel,   v_prev - self.max_accel  * dt)
        v_hi = min( self.max_vel,   v_prev + self.max_accel  * dt)
        w_lo = max(-self.max_wrot,  w_prev - self.max_w_accel * dt)
        w_hi = min( self.max_wrot,  w_prev + self.max_w_accel * dt)

        feasible = []
        for action in self.action_space:
            v = action.v
            w = action.r / dt if dt > 0 else 0.0   # convert Δθ → omega
            if v_lo <= v <= v_hi and w_lo <= w <= w_hi:
                feasible.append(action)

        # Always keep the stop action (v=0, w=0) for safety — if current
        # state already satisfies limits it will already be in the list;
        # we add it explicitly in case the deceleration budget is tight.
        stop = ActionRot(0.0, 0.0)
        if not feasible:
            feasible = [stop]

        return feasible

    def _clamp_and_update(self, action):
        """
        Hard-clamp an action to unicycle limits and update prev_v/prev_omega.
        Used as a final safety layer at the end of predict().
        """
        dt = self.time_step
        v = np.clip(action.v, -self.max_vel, self.max_vel)
        w = np.clip(action.r / dt, -self.max_wrot, self.max_wrot)
        w = np.clip(w, self.prev_omega - self.max_w_accel * dt,
                       self.prev_omega + self.max_w_accel * dt)
        v = np.clip(v, self.prev_v - self.max_accel * dt,
                       self.prev_v + self.max_accel * dt)
        self.prev_v = v
        self.prev_omega = w
        return ActionRot(v, w * dt)

    # ------------------------------------------------------------------
    # Kinematics propagation
    # ------------------------------------------------------------------

    def propagate(self, state, action):
        if isinstance(state, ObservableState):
            next_px = state.px + action.vx * self.time_step
            next_py = state.py + action.vy * self.time_step
            next_state = ObservableState(next_px, next_py, action.vx, action.vy, state.radius)
        elif isinstance(state, FullState):
            if self.kinematics == 'holonomic':
                next_px = state.px + action.vx * self.time_step
                next_py = state.py + action.vy * self.time_step
                next_state = FullState(next_px, next_py, action.vx, action.vy, state.radius,
                                       state.gx, state.gy, state.v_pref, state.theta)
            else:
                next_theta = state.theta + action.r
                next_vx = action.v * np.cos(next_theta)
                next_vy = action.v * np.sin(next_theta)
                next_px = state.px + next_vx * self.time_step
                next_py = state.py + next_vy * self.time_step
                next_state = FullState(next_px, next_py, next_vx, next_vy, state.radius,
                                       state.gx, state.gy, state.v_pref, next_theta)
        else:
            raise ValueError('Type error')
        return next_state

    # ------------------------------------------------------------------
    # Policy prediction
    # ------------------------------------------------------------------

    def predict(self, state):
        """
        Select the best action from the (feasible) action space.

        When enforce_lims=True + unicycle:
          - Training: use feasible action subset so the value network learns
            that only reachable actions exist, eliminating train/test mismatch.
          - Test: same feasible subset + final hard clamp for safety.
          - prev_v / prev_omega are updated every step in both phases.
        """
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')

        if self.reach_destination(state):
            zero = ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
            if self._unicycle_limits_active:
                self.prev_v = 0.0
                self.prev_omega = 0.0
            return zero

        if self.action_space is None:
            self.build_action_space(state.self_state.v_pref)

        # --- candidate actions for this step ---
        candidate_actions = self._feasible_action_space()

        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon:
            max_action = candidate_actions[np.random.choice(len(candidate_actions))]
        else:
            self.action_values = list()
            max_min_value = float('-inf')
            max_action = None

            for action in candidate_actions:
                next_self_state = self.propagate(state.self_state, action)
                ob, reward, done, info = self.env.onestep_lookahead(action)

                # Build batch of (self_state ++ human_state) tensors
                next_states = []
                for next_human_state in ob:
                    joint = next_self_state + next_human_state
                    # Append (v, omega) dims for the *next* state, derived from
                    # the candidate action. Using current (v, omega) here would
                    # mismatch the (v, omega) that transform() encodes for the
                    # same state during training, breaking the value target.
                    if self._unicycle_limits_active:
                        v_next = float(action.v)
                        w_next = float(action.r / self.time_step) if self.time_step else 0.0
                        joint = joint + (v_next, w_next)

                    next_states.append(joint)

                batch_next_states = torch.Tensor(next_states).to(self.device)
                outputs = self.model(self.rotate(batch_next_states))
                min_output, _ = torch.min(outputs, 0)
                min_value = reward + pow(self.gamma, self.time_step * state.self_state.v_pref) * min_output.data.item()
                self.action_values.append(min_value)
                if min_value > max_min_value:
                    max_min_value = min_value
                    max_action = action

        if self.phase == 'train':
            self.last_state = self.transform(state)

        # --- apply limits and update running state ---
        if self._unicycle_limits_active:
            max_action = self._clamp_and_update(max_action)
        
        return max_action

    # ------------------------------------------------------------------
    # State transformation
    # ------------------------------------------------------------------

    def transform(self, state):
        """
        Transform joint state to rotated tensor for memory storage.
 
        CADRL's value network takes a single (self_state, human_state) pair
        and the predict loop takes the min over humans for each action. To
        keep that "worst-case human" semantics in the stored last_state
        (e.g. when training in evaluation mode with multiple humans per
        episode), we rotate every human's joint state, score them all with
        the current model, and return the row of the human that gives the
        minimum value — i.e. the human the predict-time logic would have
        attended to. With a single human this collapses to the original
        single-row behavior, so the unenforced/single-human path is
        unchanged.
        """
        assert len(state.human_states) >= 1
 
        joints = []
        for human_state in state.human_states:
            joint = state.self_state + human_state
            if self._unicycle_limits_active:
                # Derive (v, omega) from the *state itself*, not from
                # self.prev_v/omega. transform() is also called during IL
                # where the IL policy (ORCA) never writes self.prev_v/omega
                # — those would stay at 0 across the entire IL dataset,
                # creating a guaranteed train/test feature distribution
                # shift. Reading from state.self_state makes transform()
                # policy-agnostic.
                v_curr = float(np.hypot(state.self_state.vx, state.self_state.vy))
                omega_curr = float(state.self_state.omega) if state.self_state.omega is not None else 0.0
                joint = joint + (v_curr, omega_curr)
            joints.append(joint)
 
        # (num_humans, state_dim) → rotate → score → pick argmin row
        batch = torch.Tensor(joints).to(self.device)
        rotated = self.rotate(batch)  # (num_humans, rotated_dim)
        if len(state.human_states) == 1:
            return rotated.squeeze(dim=0)
        with torch.no_grad():
            values = self.model(rotated).squeeze(-1)  # (num_humans,)
        worst_idx = int(torch.argmin(values).item())
        return rotated[worst_idx]

    def rotate(self, state):
        """
        Transform to agent-centric coordinates.
        Input: (batch_size, state_length)

        Base indices (enforce_lims=False, self_state_dim=6):
          px=0, py=1, vx=2, vy=3, r=4, gx=5, gy=6, v_pref=7, theta=8,
          px1=9, py1=10, vx1=11, vy1=12, r1=13

        With enforce_lims=True (self_state_dim=8), two extra dims are appended
        at positions 14 (v) and 15 (omega) — after the human state block.
        They are passed through unchanged (already in robot frame).
        """
        batch = state.shape[0]
        dx = (state[:, 5] - state[:, 0]).reshape((batch, -1))
        dy = (state[:, 6] - state[:, 1]).reshape((batch, -1))
        rot = torch.atan2(state[:, 6] - state[:, 1], state[:, 5] - state[:, 0])

        dg      = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)
        v_pref  = state[:, 7].reshape((batch, -1))
        vx      = (state[:, 2] * torch.cos(rot) + state[:, 3] * torch.sin(rot)).reshape((batch, -1))
        vy      = (state[:, 3] * torch.cos(rot) - state[:, 2] * torch.sin(rot)).reshape((batch, -1))
        radius  = state[:, 4].reshape((batch, -1))

        if self.kinematics == 'unicycle':
            theta = (state[:, 8] - rot).reshape((batch, -1))
        else:
            theta = torch.zeros_like(v_pref)

        vx1     = (state[:, 11] * torch.cos(rot) + state[:, 12] * torch.sin(rot)).reshape((batch, -1))
        vy1     = (state[:, 12] * torch.cos(rot) - state[:, 11] * torch.sin(rot)).reshape((batch, -1))
        px1     = ((state[:, 9]  - state[:, 0]) * torch.cos(rot)
                 + (state[:, 10] - state[:, 1]) * torch.sin(rot)).reshape((batch, -1))
        py1     = ((state[:, 10] - state[:, 1]) * torch.cos(rot)
                 - (state[:, 9]  - state[:, 0]) * torch.sin(rot)).reshape((batch, -1))
        radius1 = state[:, 13].reshape((batch, -1))
        radius_sum = radius + radius1
        da = torch.norm(
            torch.cat([(state[:, 0] - state[:, 9]).reshape((batch, -1)),
                       (state[:, 1] - state[:, 10]).reshape((batch, -1))], dim=1),
            2, dim=1, keepdim=True
        )

        new_state = torch.cat(
            [dg, v_pref, theta, radius, vx, vy, px1, py1, vx1, vy1, radius1, da, radius_sum],
            dim=1
        )  # shape: (batch, 13)

        if self._unicycle_limits_active:
            v_cur     = state[:, 14].reshape((batch, -1))
            omega_cur = state[:, 15].reshape((batch, -1))
            # Splice (v, omega) into the self-state block (right after vy at idx 5),
            # so that state[:, :self.self_state_dim] == 8 true self-state dims.
            # Layout becomes: [dg, v_pref, theta, radius, vx, vy, v_cur, omega_cur,
            #                  px1, py1, vx1, vy1, radius1, da, radius_sum]
            self_block  = torch.cat([dg, v_pref, theta, radius, vx, vy, v_cur, omega_cur], dim=1)
            human_block = torch.cat([px1, py1, vx1, vy1, radius1, da, radius_sum], dim=1)
            new_state   = torch.cat([self_block, human_block], dim=1)  # (batch, 15)

        return new_state