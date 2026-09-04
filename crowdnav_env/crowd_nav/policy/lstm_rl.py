import torch
import torch.nn as nn
import numpy as np
import logging
from crowd_nav.policy.cadrl import mlp
from crowd_nav.policy.multi_human_rl import MultiHumanRL


class ValueNetwork1(nn.Module):
    def __init__(self, input_dim, self_state_dim, mlp_dims, lstm_hidden_dim):
        super().__init__()
        self.self_state_dim = self_state_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.mlp = mlp(self_state_dim + lstm_hidden_dim, mlp_dims)
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, batch_first=True)

    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans, length of a joint state)
        :return:
        """
        # ── Padding-aware forward ─────────────────────────────────────────
        # MultiHumanRL.transform() pads the humans dimension up to a fixed
        # max_humans by PREPENDING all-zero rows. SARL absorbs this via
        # attention; LSTM-RL cannot — feeding the LSTM zero rows poisons the
        # final hidden state (hn becomes the LSTM's response to "nothing"
        # accumulated for several timesteps, not to the closest human).
        # Two corrections are needed:
        #   1. self_state must be read from a REAL human row, not row 0
        #      (which is padding when num_humans < max_humans). Every row
        #      carries the same self_state in its first self_state_dim
        #      columns by construction in transform(), so we read from the
        #      LAST row (always a real human after prepended padding).
        #   2. The LSTM should only consume real-human rows. We detect
        #      padding rows by the all-zero pattern and pack the variable-
        #      length sequence before feeding it to the LSTM. This way the
        #      hidden state at the end of each sample reflects exactly the
        #      real humans for that sample.
        size = state.shape
        device = state.device

        # Per-sample mask of real (non-padding) rows: True where the row has
        # any non-zero entry. Padding rows in transform() are torch.zeros,
        # while real rows always have non-zero dg or v_pref.
        real_mask = state.abs().sum(dim=2) > 0          # (batch, num_humans)
        lengths = real_mask.sum(dim=1).clamp(min=1)     # (batch,) at least 1

        # self_state from the LAST row of each sample (always real after
        # prepended padding). Falls back to row 0 if for some reason every
        # row is padding (shouldn't happen, but defensive).
        last_idx = (size[1] - 1)
        self_state = state[:, last_idx, :self.self_state_dim]

        # If every sample has the same length (e.g. always max_humans during
        # circle_crossing training), skip the packing overhead.
        if torch.all(lengths == size[1]):
            h0 = torch.zeros(1, size[0], self.lstm_hidden_dim, device=device)
            c0 = torch.zeros(1, size[0], self.lstm_hidden_dim, device=device)
            output, (hn, cn) = self.lstm(state, (h0, c0))
        else:
            # Pack so the LSTM only processes real rows per-sample. We rely
            # on prepended padding, so we feed each sample's TAIL (real rows
            # at the end) as its real sequence by extracting it explicitly.
            # PyTorch's pack expects sequences starting at index 0, so we
            # build a left-aligned tensor of real rows for each sample.
            max_len = int(lengths.max().item())
            real_state = torch.zeros(size[0], max_len, size[2], device=device)
            for b in range(size[0]):
                L = int(lengths[b].item())
                real_state[b, :L] = state[b, size[1] - L:size[1]]
            packed = nn.utils.rnn.pack_padded_sequence(
                real_state, lengths.cpu(),
                batch_first=True, enforce_sorted=False
            )
            h0 = torch.zeros(1, size[0], self.lstm_hidden_dim, device=device)
            c0 = torch.zeros(1, size[0], self.lstm_hidden_dim, device=device)
            _, (hn, cn) = self.lstm(packed, (h0, c0))

        hn = hn.squeeze(0)
        joint_state = torch.cat([self_state, hn], dim=1)
        value = self.mlp(joint_state)
        return value


class ValueNetwork2(nn.Module):
    def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp_dims, lstm_hidden_dim):
        super().__init__()
        self.self_state_dim = self_state_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.mlp1 = mlp(input_dim, mlp1_dims)
        self.mlp = mlp(self_state_dim + lstm_hidden_dim, mlp_dims)
        self.lstm = nn.LSTM(mlp1_dims[-1], lstm_hidden_dim, batch_first=True)

    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans, length of a joint state)
        :return:
        """
        # See ValueNetwork1.forward for the rationale on padding-aware
        # LSTM execution. Same logic applied here, except the LSTM input is
        # mlp1's output rather than the raw state.
        size = state.shape
        device = state.device

        real_mask = state.abs().sum(dim=2) > 0
        lengths = real_mask.sum(dim=1).clamp(min=1)

        last_idx = (size[1] - 1)
        self_state = state[:, last_idx, :self.self_state_dim]

        state_flat = torch.reshape(state, (-1, size[2]))
        mlp1_output = self.mlp1(state_flat)
        mlp1_output = torch.reshape(mlp1_output, (size[0], size[1], -1))

        if torch.all(lengths == size[1]):
            h0 = torch.zeros(1, size[0], self.lstm_hidden_dim, device=device)
            c0 = torch.zeros(1, size[0], self.lstm_hidden_dim, device=device)
            _, (hn, cn) = self.lstm(mlp1_output, (h0, c0))
        else:
            max_len = int(lengths.max().item())
            real_mlp1 = torch.zeros(size[0], max_len, mlp1_output.shape[-1], device=device)
            for b in range(size[0]):
                L = int(lengths[b].item())
                real_mlp1[b, :L] = mlp1_output[b, size[1] - L:size[1]]
            packed = nn.utils.rnn.pack_padded_sequence(
                real_mlp1, lengths.cpu(),
                batch_first=True, enforce_sorted=False
            )
            h0 = torch.zeros(1, size[0], self.lstm_hidden_dim, device=device)
            c0 = torch.zeros(1, size[0], self.lstm_hidden_dim, device=device)
            _, (hn, cn) = self.lstm(packed, (h0, c0))

        hn = hn.squeeze(0)
        joint_state = torch.cat([self_state, hn], dim=1)
        value = self.mlp(joint_state)
        return value


class LstmRL(MultiHumanRL):
    def __init__(self):
        super().__init__()
        self.name = 'LSTM-RL'
        self.with_interaction_module = None
        self.interaction_module_dims = None

    def configure(self, config):
        self.set_common_parameters(config)
        mlp_dims = [int(x) for x in config.get('lstm_rl', 'mlp2_dims').split(', ')]
        global_state_dim = config.getint('lstm_rl', 'global_state_dim')
        self.with_om = config.getboolean('lstm_rl', 'with_om')
        with_interaction_module = config.getboolean('lstm_rl', 'with_interaction_module')
        if with_interaction_module:
            mlp1_dims = [int(x) for x in config.get('lstm_rl', 'mlp1_dims').split(', ')]
            self.model = ValueNetwork2(self.input_dim(), self.self_state_dim, mlp1_dims, mlp_dims, global_state_dim)
        else:
            self.model = ValueNetwork1(self.input_dim(), self.self_state_dim, mlp_dims, global_state_dim)
        self.multiagent_training = config.getboolean('lstm_rl', 'multiagent_training')
        logging.info('Policy: {}LSTM-RL {} pairwise interaction module'.format(
            'OM-' if self.with_om else '', 'w/' if with_interaction_module else 'w/o'))

    def set_device(self, device):
        self.device = device
        self.model.to(device)

    def predict(self, state):
        """
        Input state is the joint state of robot concatenated with the observable state of other agents

        To predict the best action, agent samples actions and propagates one step to see how good the next state is
        thus the reward function is needed

        """

        def dist(human):
            # sort human order by decreasing distance to the robot
            return np.linalg.norm(np.array(human.position) - np.array(state.self_state.position))

        state.human_states = sorted(state.human_states, key=dist, reverse=True)
        return super().predict(state)

    def transform(self, state):
        """
        Enforce the closest-human-last ordering here too, not just in predict().

        During IL, the acting policy is ORCA (not LstmRL), so explorer.py calls
        target_policy.transform(state) directly on ORCA's raw, unsorted
        state.human_states — predict()'s sort above never runs. Without this
        override, IL demonstrations were fed to the LSTM in arbitrary
        environment order while RL experience (sorted via predict()) was fed
        in closest-last order, an IL/RL feature-distribution mismatch that
        degraded training. Sorting here makes transform() correct regardless
        of which policy produced the state.
        """
        def dist(human):
            return np.linalg.norm(np.array(human.position) - np.array(state.self_state.position))

        state.human_states = sorted(state.human_states, key=dist, reverse=True)
        return super().transform(state)

    def _order_next_human_states(self, next_self_state, next_human_states):
        """
        Sort candidate next-states by decreasing distance to the candidate's
        next self position, matching the closest-human-last convention
        predict() applies to state.human_states. Without this, next_human_states
        built from env.onestep_lookahead() (query_env=True) arrives in the
        environment's fixed order, which mismatches the ordering the LSTM was
        trained on and degrades action-value estimates used to pick the action.
        """
        def dist(human):
            return np.linalg.norm(np.array(human.position) - np.array(next_self_state.position))

        return sorted(next_human_states, key=dist, reverse=True)