import torch
import numpy as np
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_nav.policy.cadrl import CADRL


class MultiHumanRL(CADRL):
    """
    Base class for multi-human RL policies (SARL, LSTM-RL).

    Unicycle limit handling is inherited from CADRL:
      - _feasible_action_space() filters candidates each step
      - _clamp_and_update() applies the final hard clamp
      - prev_v / prev_omega are tracked in both train and test phases
      - (v, omega) are appended to the rotated state when enforce_lims=True

    Variable human counts (e.g. evaluation mode, 1..10 humans per episode)
    are handled by padding every transformed/predicted state to a fixed
    `max_humans` along the humans dimension, so the default DataLoader
    collator can stack them. Padding rows are prepended (not appended) so
    that LSTM-RL's distance-ordered "closest human last" convention puts
    real humans at the end of the LSTM sequence — where the final hidden
    state is most influenced.
    """
    # Hard cap on humans the network architecture is sized for.
    # Evaluation mode samples up to 10 humans per episode; the original
    # circle_crossing setup uses up to 5. 10 covers both.
    max_humans = 10

    def __init__(self):
        super().__init__()

    def _pad_humans(self, tensor):
        """
        Pad a (num_humans, feature_dim) tensor along dim 0 to self.max_humans.
        Padding rows are PREPENDED with zeros. The padded rows are deterministic
        and identical across all stored states, so the network can learn to
        treat them as a constant offset (same trick as fixed-pad token
        embeddings in NLP). Real humans always occupy the last `num_humans`
        positions, preserving LSTM-RL's "closest human last" convention.

        Truncation (num > max_humans) sorts by the `da` column (robot-human
        distance, second-to-last in the rotated layout — see rotate()) before
        keeping the last max_humans rows, so the farthest humans are always
        the ones dropped. This must not assume the caller already sorted the
        rows: LstmRL.predict() sorts state.human_states before transform(),
        but SARL/CADRL never do (attention/min-over-humans don't need it), so
        without an explicit sort here truncation would drop an arbitrary
        human rather than the farthest one for those policies.
        """
        num, feat = tensor.shape
        if num >= self.max_humans:
            order = torch.argsort(tensor[:, -2], descending=True)  # farthest first
            return tensor[order][-self.max_humans:]
        pad_rows = self.max_humans - num
        padding = torch.zeros((pad_rows, feat), dtype=tensor.dtype, device=tensor.device)
        return torch.cat([padding, tensor], dim=0)

    def predict(self, state):
        """
        Value-network policy for multiple humans.

        Differences from CADRL.predict:
          - Loops over *all* human states (not min over humans)
          - Optionally uses occupancy maps
          - Feasible action filtering and unicycle state tracking
            are identical to the single-human case (inherited logic).
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

        candidate_actions = self._feasible_action_space()

        occupancy_maps = None
        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon:
            max_action = candidate_actions[np.random.choice(len(candidate_actions))]
        else:
            self.action_values = list()
            max_value = float('-inf')
            max_action = None

            for action in candidate_actions:
                next_self_state = self.propagate(state.self_state, action)

                if self.query_env:
                    next_human_states, reward, done, info = self.env.onestep_lookahead(action)
                else:
                    next_human_states = [
                        self.propagate(human_state, ActionXY(human_state.vx, human_state.vy))
                        for human_state in state.human_states
                    ]
                    reward = self.compute_reward(next_self_state, next_human_states)

                # env.onestep_lookahead() (query_env branch) returns humans in the
                # environment's fixed insertion order, not the order state.human_states
                # was in — so order-sensitive value networks (LSTM-RL) would see a
                # different human ordering at action-selection time than the ordering
                # transform() used to build their training data. Order-insensitive
                # subclasses (SARL's attention, CADRL's single-human loop) leave this
                # a no-op.
                next_human_states = self._order_next_human_states(next_self_state, next_human_states)

                # Build (self_state ++ human_state) tensor, with optional (v, omega) suffix.
                # The appended (v, omega) describes the *next* state (i.e. the
                # candidate action being evaluated), so it stays consistent
                # with what transform() encodes for the same state at training
                # time. Using current (v, omega) here would mismatch the value
                # target and prevent the network from learning to use these
                # features productively.
                if self._unicycle_limits_active:
                    v_next = float(action.v)
                    w_next = float(action.r / self.time_step) if self.time_step else 0.0
                    extra = (v_next, w_next)
                    batch_next_states = torch.cat([
                        torch.Tensor([next_self_state + nhs + extra]).to(self.device)
                        for nhs in next_human_states
                    ], dim=0)
                else:
                    batch_next_states = torch.cat([
                        torch.Tensor([next_self_state + next_human_state]).to(self.device)
                        for next_human_state in next_human_states
                    ], dim=0)

                rotated_batch_input = self.rotate(batch_next_states).unsqueeze(0)

                if self.with_om:
                    if occupancy_maps is None:
                        occupancy_maps = self.build_occupancy_maps(next_human_states).unsqueeze(0)
                    rotated_batch_input = torch.cat(
                        [rotated_batch_input, occupancy_maps.to(self.device)], dim=2
                    )
                
                # Pad along the humans dim so the model's input shape at
                # predict time matches the shape it sees at training time
                # (where transform() pads to max_humans for collator safety).
                rotated_batch_input = self._pad_humans(
                    rotated_batch_input.squeeze(0)
                ).unsqueeze(0)

                next_state_value = self.model(rotated_batch_input).data.item()
                value = reward + pow(self.gamma, self.time_step * state.self_state.v_pref) * next_state_value
                self.action_values.append(value)
                if value > max_value:
                    max_value = value
                    max_action = action

            if max_action is None:
                raise ValueError('Value network is not well trained.')

        if self.phase == 'train':
            self.last_state = self.transform(state)

        if self._unicycle_limits_active:
            max_action = self._clamp_and_update(max_action)

        return max_action

    def _order_next_human_states(self, next_self_state, next_human_states):
        """
        Hook for subclasses whose value network is order-sensitive.

        Default: identity (no reordering). SARL's attention aggregation is
        permutation-invariant and CADRL scores one human at a time, so neither
        needs this. LstmRL overrides it to match the closest-human-last
        convention its predict() enforces on state.human_states.
        """
        return next_human_states

    def compute_reward(self, nav, humans):
        dmin = float('inf')
        collision = False
        for human in humans:
            dist = np.linalg.norm((nav.px - human.px, nav.py - human.py)) - nav.radius - human.radius
            if dist < 0:
                collision = True
                break
            if dist < dmin:
                dmin = dist

        reaching_goal = np.linalg.norm((nav.px - nav.gx, nav.py - nav.gy)) < nav.radius
        if collision:
            reward = -0.25
        elif reaching_goal:
            reward = 1
        elif dmin < 0.2:
            reward = (dmin - 0.2) * 0.5 * self.time_step
        else:
            reward = 0
        return reward

    def transform(self, state):
        if self._unicycle_limits_active:
            v_curr = float(np.hypot(state.self_state.vx, state.self_state.vy))
            omega_curr = float(state.self_state.omega) if state.self_state.omega is not None else 0.0
            extra = (v_curr, omega_curr)
            state_tensor = torch.cat([
                torch.Tensor([state.self_state + human_state + extra]).to(self.device)
                for human_state in state.human_states
            ], dim=0)
        else:
            state_tensor = torch.cat([
                torch.Tensor([state.self_state + human_state]).to(self.device)
                for human_state in state.human_states
            ], dim=0)

        if self.with_om:
            occupancy_maps = self.build_occupancy_maps(state.human_states)
            state_tensor = torch.cat(
                [self.rotate(state_tensor), occupancy_maps.to(self.device)], dim=1
            )
        else:
            state_tensor = self.rotate(state_tensor)

        # Pad to fixed max_humans so DataLoader's default collator can stack
        # entries from episodes with different numbers of humans.
        state_tensor = self._pad_humans(state_tensor)
        return state_tensor

    def input_dim(self):
        # Must use _rotated_state_dim (post-rotate width) not joint_state_dim
        # (pre-rotate width), because the LSTM/MLP receives rotated tensors.
        # _rotated_state_dim is 13 normally, 15 when enforce_lims+unicycle.
        return self._rotated_state_dim + (self.cell_num ** 2 * self.om_channel_size if self.with_om else 0)

    def build_occupancy_maps(self, human_states):
        """
        :param human_states:
        :return: tensor of shape (# human - 1, self.cell_num ** 2)
        """
        occupancy_maps = []
        for human in human_states:
            other_humans = np.concatenate([
                np.array([(other_human.px, other_human.py, other_human.vx, other_human.vy)])
                for other_human in human_states if other_human != human
            ], axis=0)
            other_px = other_humans[:, 0] - human.px
            other_py = other_humans[:, 1] - human.py
            human_velocity_angle = np.arctan2(human.vy, human.vx)
            other_human_orientation = np.arctan2(other_py, other_px)
            rotation = other_human_orientation - human_velocity_angle
            distance = np.linalg.norm([other_px, other_py], axis=0)
            other_px = np.cos(rotation) * distance
            other_py = np.sin(rotation) * distance

            other_x_index = np.floor(other_px / self.cell_size + self.cell_num / 2)
            other_y_index = np.floor(other_py / self.cell_size + self.cell_num / 2)
            other_x_index[other_x_index < 0] = float('-inf')
            other_x_index[other_x_index >= self.cell_num] = float('-inf')
            other_y_index[other_y_index < 0] = float('-inf')
            other_y_index[other_y_index >= self.cell_num] = float('-inf')
            grid_indices = self.cell_num * other_y_index + other_x_index
            occupancy_map = np.isin(range(self.cell_num ** 2), grid_indices)

            if self.om_channel_size == 1:
                occupancy_maps.append([occupancy_map.astype(int)])
            else:
                other_human_velocity_angles = np.arctan2(other_humans[:, 3], other_humans[:, 2])
                rotation = other_human_velocity_angles - human_velocity_angle
                speed = np.linalg.norm(other_humans[:, 2:4], axis=1)
                other_vx = np.cos(rotation) * speed
                other_vy = np.sin(rotation) * speed
                dm = [list() for _ in range(self.cell_num ** 2 * self.om_channel_size)]
                for i, index in np.ndenumerate(grid_indices):
                    if index in range(self.cell_num ** 2):
                        if self.om_channel_size == 2:
                            dm[2 * int(index)].append(other_vx[i])
                            dm[2 * int(index) + 1].append(other_vy[i])
                        elif self.om_channel_size == 3:
                            dm[3 * int(index)].append(1)
                            dm[3 * int(index) + 1].append(other_vx[i])
                            dm[3 * int(index) + 2].append(other_vy[i])
                        else:
                            raise NotImplementedError
                for i, cell in enumerate(dm):
                    dm[i] = sum(dm[i]) / len(dm[i]) if len(dm[i]) != 0 else 0
                occupancy_maps.append([dm])

        return torch.from_numpy(np.concatenate(occupancy_maps, axis=0)).float()