import torch.nn as nn


def get_vec_normalize(venv):
    # Imported here rather than at module level: training/networks/envs.py
    # does `from baselines import bench`, and OpenAI baselines cannot be
    # pip-installed (its sdist builds mujoco-py). Evaluation reaches this
    # module only for AddBias/init via distributions.py -> model.py, so a
    # module-level import made the HEIGHT policy unloadable for anyone
    # without a prebuilt baselines. Training, which needs baselines anyway,
    # is unaffected.
    from training.networks.envs import VecNormalize

    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module



