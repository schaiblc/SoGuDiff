import itertools
import logging
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch.nn import Parameter
from crowd_nav.policy.cadrl import mlp
from crowd_nav.policy.multi_human_rl import MultiHumanRL


class ValueNetwork(nn.Module):
    """
    GCN-based value network, ported from RelationalGraphLearning's
    crowd_nav/policy/gcn.py (the ICRA-benchmark "RGL" policy). Builds a small
    graph of [self-node, human-node x N], computes a learned similarity
    (attention) matrix between nodes, and does `num_layer` rounds of graph
    convolution before reading off the self-node's final feature as input to
    a value MLP.
    """

    def __init__(self, input_dim, self_state_dim, num_layer, X_dim, wr_dims, wh_dims, final_state_dim,
                 gcn2_w1_dim, planning_dims, similarity_function, layerwise_graph, skip_connection):
        super().__init__()
        self.similarity_function = similarity_function
        self.self_state_dim = self_state_dim
        self.human_state_dim = input_dim - self_state_dim
        self.num_layer = num_layer
        self.X_dim = X_dim
        self.layerwise_graph = layerwise_graph
        self.skip_connection = skip_connection

        self.w_r = mlp(self_state_dim, wr_dims, last_relu=True)
        self.w_h = mlp(self.human_state_dim, wh_dims, last_relu=True)

        if self.similarity_function == 'embedded_gaussian':
            self.w_a = Parameter(torch.randn(self.X_dim, self.X_dim))
        elif self.similarity_function == 'concatenation':
            self.w_a = mlp(2 * X_dim, [2 * X_dim, 1], last_relu=True)

        if num_layer == 1:
            self.w1 = Parameter(torch.randn(self.X_dim, final_state_dim))
        elif num_layer == 2:
            self.w1 = Parameter(torch.randn(self.X_dim, gcn2_w1_dim))
            self.w2 = Parameter(torch.randn(gcn2_w1_dim, final_state_dim))
        else:
            raise NotImplementedError

        self.value_net = mlp(final_state_dim, planning_dims)

        # for visualization
        self.A = None

    def compute_similarity_matrix(self, X, real_mask):
        """
        real_mask: (batch, num_nodes) bool — True for the self-node (always)
        and real (non-padding) human nodes, False for padding-human nodes.
        Padding nodes are masked out of every node's attention distribution
        (not just the self-node's) so a real human's intermediate embedding
        can't be polluted by attending to padding before it's aggregated
        into the self-node's final feature over 2 graph-conv layers.

        Only the softmax-normalized similarity functions (embedded_gaussian,
        gaussian, cosine_softmax) are masked here — those are the only ones
        that silently misbehave with padding (same failure mode SARL's
        original `(scores != 0)` heuristic had: padding rows produce
        nonzero post-bias scores that leak into the softmax). The
        non-softmax variants (cosine, concatenation, squared,
        equal_attention, diagonal) aren't used by the configured
        'embedded_gaussian' default and haven't been padding-audited — avoid
        selecting them with variable human counts without adding masking
        here first.
        """
        neighbor_mask = (~real_mask).unsqueeze(1)  # (batch, 1, num_nodes) — broadcast over query dim

        if self.similarity_function == 'embedded_gaussian':
            A = torch.matmul(torch.matmul(X, self.w_a), X.permute(0, 2, 1))
            A = A.masked_fill(neighbor_mask, float('-inf'))
            normalized_A = softmax(A, dim=2)
        elif self.similarity_function == 'gaussian':
            A = torch.matmul(X, X.permute(0, 2, 1))
            A = A.masked_fill(neighbor_mask, float('-inf'))
            normalized_A = softmax(A, dim=2)
        elif self.similarity_function == 'cosine_softmax':
            A = torch.matmul(X, X.permute(0, 2, 1))
            magnitudes = torch.norm(A, dim=2, keepdim=True)
            norm_matrix = torch.matmul(magnitudes, magnitudes.permute(0, 2, 1))
            A = torch.div(A, norm_matrix)
            A = A.masked_fill(neighbor_mask, float('-inf'))
            normalized_A = softmax(A, dim=2)
        elif self.similarity_function == 'cosine':
            A = torch.matmul(X, X.permute(0, 2, 1))
            magnitudes = torch.norm(A, dim=2, keepdim=True)
            norm_matrix = torch.matmul(magnitudes, magnitudes.permute(0, 2, 1))
            normalized_A = torch.div(A, norm_matrix)
        elif self.similarity_function == 'concatenation':
            indices = [pair for pair in itertools.product(list(range(X.size(1))), repeat=2)]
            selected_features = torch.index_select(X, dim=1, index=torch.LongTensor(indices).reshape(-1))
            pairwise_features = selected_features.reshape((-1, X.size(1) * X.size(1), X.size(2) * 2))
            A = self.w_a(pairwise_features).reshape(-1, X.size(1), X.size(1))
            normalized_A = A
        elif self.similarity_function == 'squared':
            A = torch.matmul(X, X.permute(0, 2, 1))
            squared_A = A * A
            normalized_A = squared_A / torch.sum(squared_A, dim=2, keepdim=True)
        elif self.similarity_function == 'equal_attention':
            normalized_A = (torch.ones(X.size(1), X.size(1)) / X.size(1)).expand(X.size(0), X.size(1), X.size(1))
        elif self.similarity_function == 'diagonal':
            normalized_A = (torch.eye(X.size(1), X.size(1))).expand(X.size(0), X.size(1), X.size(1))
        else:
            raise NotImplementedError

        return normalized_A

    def forward(self, state_input):
        if isinstance(state_input, tuple):
            state, lengths = state_input
        else:
            state = state_input

        size = state.shape

        # ── Padding-aware self_state + node mask ────────────────────────────
        # MultiHumanRL.transform() pads the humans dimension up to a fixed
        # max_humans by PREPENDING all-zero rows (see multi_human_rl.py).
        # self_state must therefore be read from a REAL human row, not row 0
        # (which is padding when num_humans < max_humans) — every row carries
        # the same self_state block by construction, so the LAST row (always
        # real after prepended padding) is used instead.
        real_mask = state.abs().sum(dim=2) > 0                    # (batch, num_humans)
        last_idx = size[1] - 1
        self_state = state[:, last_idx, :self.self_state_dim]
        human_states = state[:, :, self.self_state_dim:]

        # compute feature matrix X: [self-node, human-node x num_humans]
        self_state_embedings = self.w_r(self_state)
        human_state_embedings = self.w_h(human_states)
        X = torch.cat([self_state_embedings.unsqueeze(1), human_state_embedings], dim=1)

        # self-node is always real; prepend True to the human real_mask
        self_mask = torch.ones((size[0], 1), dtype=torch.bool, device=state.device)
        node_real_mask = torch.cat([self_mask, real_mask], dim=1)  # (batch, 1 + num_humans)

        # compute matrix A
        normalized_A = self.compute_similarity_matrix(X, node_real_mask)
        self.A = normalized_A[0, :, :].data.cpu().numpy()

        # graph convolution
        if self.num_layer == 0:
            feat = X[:, 0, :]
        elif self.num_layer == 1:
            h1 = torch.relu(torch.matmul(torch.matmul(normalized_A, X), self.w1))
            feat = h1[:, 0, :]
        else:
            if not self.skip_connection:
                h1 = torch.relu(torch.matmul(torch.matmul(normalized_A, X), self.w1))
            else:
                h1 = torch.relu(torch.matmul(torch.matmul(normalized_A, X), self.w1)) + X
            if self.layerwise_graph:
                normalized_A2 = self.compute_similarity_matrix(h1, node_real_mask)
            else:
                normalized_A2 = normalized_A
            if not self.skip_connection:
                h2 = torch.relu(torch.matmul(torch.matmul(normalized_A2, h1), self.w2))
            else:
                h2 = torch.relu(torch.matmul(torch.matmul(normalized_A2, h1), self.w2)) + h1
            feat = h2[:, 0, :]

        value = self.value_net(feat)
        return value


class RGL(MultiHumanRL):
    """
    GCN-based "RGL" baseline (Relational Graph Learning, Chen et al. ICRA
    2020), ported from RelationalGraphLearning/crowd_nav/policy/gcn.py.

    Inherits from *our* MultiHumanRL (crowd_nav.policy.multi_human_rl), not
    the upstream repo's — this is what gives RGL the same unicycle
    kinodynamic-limit handling (_feasible_action_space, _clamp_and_update,
    prev_v/prev_omega tracking, the appended (v, omega) state features, and
    the padding-aware max_humans handling) that CADRL/SARL/LSTM-RL already
    get "for free" from the shared base class, instead of reimplementing any
    of it. Only the value network architecture (GCN vs. SARL's attention vs.
    LSTM-RL's recurrence) differs between these three multi-human policies.
    """

    def __init__(self):
        super().__init__()
        self.name = 'RGL'

    def configure(self, config):
        self.set_common_parameters(config)
        # The GCN's own graph aggregation over human nodes is what the
        # upstream RGL architecture uses in place of an occupancy map — om
        # was never a tunable for this policy, so it's fixed off rather than
        # exposed as a config key like SARL/LSTM-RL's with_om.
        self.with_om = False
        self.multiagent_training = config.getboolean('rgl', 'multiagent_training')
        num_layer = config.getint('rgl', 'num_layer')
        X_dim = config.getint('rgl', 'X_dim')
        wr_dims = [int(x) for x in config.get('rgl', 'wr_dims').split(', ')]
        wh_dims = [int(x) for x in config.get('rgl', 'wh_dims').split(', ')]
        final_state_dim = config.getint('rgl', 'final_state_dim')
        gcn2_w1_dim = config.getint('rgl', 'gcn2_w1_dim')
        planning_dims = [int(x) for x in config.get('rgl', 'planning_dims').split(', ')]
        similarity_function = config.get('rgl', 'similarity_function')
        layerwise_graph = config.getboolean('rgl', 'layerwise_graph')
        skip_connection = config.getboolean('rgl', 'skip_connection')

        self.model = ValueNetwork(self.input_dim(), self.self_state_dim, num_layer, X_dim, wr_dims, wh_dims,
                                  final_state_dim, gcn2_w1_dim, planning_dims, similarity_function, layerwise_graph,
                                  skip_connection)
        logging.info('Policy: RGL (GCN layers: %d, similarity: %s)', num_layer, similarity_function)

    def get_matrix_A(self):
        return self.model.A
