import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorchBaselines.a2c_ppo_acktr.utils import AddBias, init

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

#
# Standardize distribution interfaces
#

# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entrop(self):
        return super.entropy().sum(-1)

    def mode(self):
        return self.mean


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())


class FixedTanhNormal:
    """tanh-squashed Normal: action = tanh(z) * scale, z ~ Normal(mean, std).

    Used only for the unicycle accel-rate-limited action heads (DSRNN's SRNN
    and HEIGHT's selfAttn_merge_SRNN, wired in by model.py/height_model.py)
    — not a general replacement for DiagGaussian, which both networks keep
    using everywhere else (e.g. the holonomic action space, unaffected by
    this). Unlike DiagGaussian + an external hard np.clip (what DSRNN's own
    unicycle scheme and this project's accel-rate-limited accumulator both
    used), the action PPO's log_prob is computed on is always the same one
    the environment sees — never an unclipped raw value that can drift
    arbitrarily far outside the clip boundary into a zero-gradient region.
    That drift is what caused training to collapse to a saturated,
    state-independent action under the accumulator scheme (see
    crowd_sim.py's calc_reward comment and config.py's entropy_coef comment
    for the full diagnosis) — hard-clipping a raw unbounded Gaussian output
    only breaks down once the accumulator makes "stuck at the clip boundary"
    a viable low-cost equilibrium, which neither DSRNN's original
    non-accumulating unicycle scheme nor its holonomic action space ever
    created.
    """

    def __init__(self, mean, std, scale):
        self.normal = torch.distributions.Normal(mean, std)
        self.scale = scale

    def sample(self):
        z = self.normal.rsample()
        return torch.tanh(z) * self.scale

    def mode(self):
        return torch.tanh(self.normal.mean) * self.scale

    def log_probs(self, action):
        a = torch.clamp(action / self.scale, -1 + 1e-3, 1 - 1e-3)
        z = 0.5 * (torch.log1p(a) - torch.log1p(-a))  # atanh
        log_prob_z = self.normal.log_prob(z)
        # Jacobian of a = tanh(z)*scale: da/dz = scale*(1 - tanh(z)^2).
        # log(1-tanh(z)^2) computed via the numerically stable identity
        # 2*(log(2) - |z| - softplus(-2|z|)) instead of log(1-tanh(z)^2+eps)
        # directly — the direct form loses all precision once tanh(z) is
        # within float32 epsilon of +-1 (an eps-floor doesn't prevent that,
        # it only prevents log(0); the *input* to log is still garbage from
        # catastrophic cancellation). This form never computes tanh(z)^2 at
        # all, so it stays accurate and finite for any z.
        log_sech2 = 2.0 * (torch.log(torch.tensor(2.0, device=z.device))
                            - z.abs() - torch.nn.functional.softplus(-2 * z.abs()))
        jacobian = torch.log(self.scale) + log_sech2
        return (log_prob_z - jacobian).sum(-1, keepdim=True)

    def entropy(self):
        # No closed form for a tanh-squashed Normal's entropy. Using the
        # underlying Normal's entropy as the exploration-bonus proxy (same
        # simplification common in SAC-style implementations) — it's only
        # ever used as ppo.entropy_coef's regularization term, not for
        # anything requiring an exact value.
        return self.normal.entropy().sum(-1)


class TanhGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, action_scale):
        super(TanhGaussian, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))
        self.register_buffer('action_scale', torch.as_tensor(action_scale, dtype=torch.float32))

    def forward(self, x):
        action_mean = self.fc_mean(x)
        # Bound the raw (pre-tanh) mean too, not just std. Nothing else
        # constrains it: the *executed* action is always safely squashed by
        # tanh regardless of how large the mean gets, so the environment
        # never signals anything is wrong. But log_probs() evaluates the
        # Normal at z=atanh(stored action) (always within a few units of 0,
        # since the atanh clamp bounds it), and if mean has drifted far from
        # that range, (z-mean)^2/var overflows float32 and the resulting
        # inf/nan gradient corrupts the whole network (observed in practice:
        # crashed two real 20M-step runs, both around update ~5500-6000).
        # tanh(5) is already ~0.9999 — clamping here to +-5 costs no real
        # expressiveness while making mean divergence structurally
        # impossible.
        action_mean = action_mean.clamp(min=-5.0, max=5.0)

        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        # Floor/ceiling std (unlike DSRNN's own DiagGaussian, whose logstd
        # is completely unconstrained) — this matters here specifically
        # because log_probs() evaluates the Normal at z=atanh(stored
        # action), a fixed point that can end up far from the *current*
        # mean as PPO takes multiple epochs/mini-batch steps per rollout.
        # If std collapses toward 0 in the interim, (z-mean)^2/std^2
        # explodes and the resulting Inf/NaN gradient corrupts the shared
        # base network on the next optimizer step (observed in practice: a
        # full HEIGHT run crashed this way after ~500 updates). DSRNN's own
        # scheme never evaluates log_prob at a transformed fixed point like
        # this, so it never faced this risk.
        action_logstd = action_logstd.clamp(min=-5.0, max=2.0)  # std in [~0.0067, ~7.39]
        scale = self.action_scale.to(x.device)
        return FixedTanhNormal(action_mean, action_logstd.exp(), scale)


class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Bernoulli, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)
