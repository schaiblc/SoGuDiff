import torch
import torch.nn as nn
from torch.nn.functional import softmax
import logging
from crowd_nav.policy.cadrl import mlp
from crowd_nav.policy.multi_human_rl import MultiHumanRL


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims, attention_dims, with_global_state,
                 cell_size, cell_num):
        super().__init__()
        self.self_state_dim = self_state_dim
        self.global_state_dim = mlp1_dims[-1]
        self.mlp1 = mlp(input_dim, mlp1_dims, last_relu=True)
        self.mlp2 = mlp(mlp1_dims[-1], mlp2_dims)
        self.with_global_state = with_global_state
        if with_global_state:
            self.attention = mlp(mlp1_dims[-1] * 2, attention_dims)
        else:
            self.attention = mlp(mlp1_dims[-1], attention_dims)
        self.cell_size = cell_size
        self.cell_num = cell_num
        mlp3_input_dim = mlp2_dims[-1] + self.self_state_dim
        self.mlp3 = mlp(mlp3_input_dim, mlp3_dims)
        self.attention_weights = None

    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans, length of a rotated state)
        :return:
        """
        # ── Padding-aware forward ─────────────────────────────────────────
        # MultiHumanRL.transform() pads the humans dimension up to a fixed
        # max_humans by PREPENDING all-zero rows. Two corrections are needed
        # for SARL:
        #   1. self_state must be read from a REAL human row, not row 0
        #      (which is padding when num_humans < max_humans). Every row
        #      carries the same self_state in its first self_state_dim
        #      columns by construction in transform(), so we read from the
        #      LAST row (always a real human after prepended padding).
        #   2. The attention mask must actually mask padding rows. The
        #      original `(scores != 0)` heuristic does not — padding rows
        #      pass through mlp1+attention and produce non-zero scores
        #      (because of biases), so they get included in the softmax.
        #      We detect padding rows by all-zero input and mask their
        #      contribution by setting their score to -inf before softmax.
        size = state.shape

        # Per-sample mask of real (non-padding) rows.
        real_mask = state.abs().sum(dim=2) > 0          # (batch, num_humans)

        # self_state from the last row (always a real human after prepended
        # padding). Falls back gracefully when there is no padding.
        last_idx = size[1] - 1
        self_state = state[:, last_idx, :self.self_state_dim]

        mlp1_output = self.mlp1(state.view((-1, size[2])))
        mlp2_output = self.mlp2(mlp1_output)

        assert state.shape[-1] == self.self_state_dim + 7, \
            f"expected {self.self_state_dim + 7}, got {state.shape[-1]}"

        if self.with_global_state:
            # Global state should average only over REAL humans, not over
            # padding rows. Use the mask to compute a masked mean.
            mlp1_grid = mlp1_output.view(size[0], size[1], -1)
            mask_f = real_mask.unsqueeze(-1).float()
            real_counts = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
            global_state = (mlp1_grid * mask_f).sum(dim=1, keepdim=True) / real_counts
            global_state = global_state.expand((size[0], size[1], self.global_state_dim)).\
                contiguous().view(-1, self.global_state_dim)
            attention_input = torch.cat([mlp1_output, global_state], dim=1)
        else:
            attention_input = mlp1_output
        scores = self.attention(attention_input).view(size[0], size[1], 1).squeeze(dim=2)

        # Proper masked softmax: set padding-row scores to -inf so they
        # contribute exactly zero weight after softmax. The original
        # `(scores != 0)` heuristic did not actually mask padding (padding
        # produces non-zero scores after mlp1+attention biases).
        scores = scores.masked_fill(~real_mask, float('-inf'))
        weights = torch.softmax(scores, dim=1).unsqueeze(2)
        self.attention_weights = weights[0, :, 0].data.cpu().numpy()

        # output feature is a linear combination of input features
        features = mlp2_output.view(size[0], size[1], -1)
        weighted_feature = torch.sum(torch.mul(weights, features), dim=1)

        # concatenate agent's state with global weighted humans' state
        joint_state = torch.cat([self_state, weighted_feature], dim=1)
        value = self.mlp3(joint_state)
        return value


class SARL(MultiHumanRL):
    def __init__(self):
        super().__init__()
        self.name = 'SARL'

    def configure(self, config):
        self.set_common_parameters(config)
        mlp1_dims = [int(x) for x in config.get('sarl', 'mlp1_dims').split(', ')]
        mlp2_dims = [int(x) for x in config.get('sarl', 'mlp2_dims').split(', ')]
        mlp3_dims = [int(x) for x in config.get('sarl', 'mlp3_dims').split(', ')]
        attention_dims = [int(x) for x in config.get('sarl', 'attention_dims').split(', ')]
        self.with_om = config.getboolean('sarl', 'with_om')
        with_global_state = config.getboolean('sarl', 'with_global_state')
        self.model = ValueNetwork(self.input_dim(), self.self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims,
                                  attention_dims, with_global_state, self.cell_size, self.cell_num)
        self.multiagent_training = config.getboolean('sarl', 'multiagent_training')
        if self.with_om:
            self.name = 'OM-SARL'
        logging.info('Policy: {} {} global state'.format(self.name, 'w/' if with_global_state else 'w/o'))

    def get_attention_weights(self):
        return self.model.attention_weights