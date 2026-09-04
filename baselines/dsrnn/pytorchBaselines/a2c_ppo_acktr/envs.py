import os

import gym
import numpy as np
import torch
from gym.spaces.box import Box
from gym.spaces.dict import Dict

from baselines import bench
try:
    # Only used inside the `if is_atari:` branch below, which is
    # structurally unreachable for CrowdSimDict-v0 (is_atari requires
    # env.unwrapped to be a gym.envs.atari.atari_env.AtariEnv instance) —
    # so this doesn't need to succeed for the crowd-nav task this repo
    # actually runs. cv2 (opencv), which atari_wrappers imports, isn't
    # pip-installable on this cluster (provided as a system module instead,
    # see the error from `pip install opencv-python-headless`) and isn't
    # worth pulling in just to satisfy an otherwise-dead import.
    from baselines.common.atari_wrappers import make_atari, wrap_deepmind
except ImportError:
    make_atari = wrap_deepmind = None
from baselines.common.vec_env import VecEnv, VecEnvWrapper
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv as DummyVecEnv_
# pytorchBaselines' own ShmemVecEnv (shmem_vec_env.py) is written against a
# newer baselines API than what's installed on this cluster (missing
# baselines.common.vec_env.util and .vec_env.clear_mpi_env_vars in the
# available 0.1.5). SubprocVecEnv is bundled in the installed package and
# functionally equivalent — separate-process parallel env stepping, just
# without ShmemVecEnv's shared-memory transport optimization — so it's used
# here instead rather than backporting several more compatibility shims for
# a pure performance optimization.
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv as _SubprocVecEnv
from baselines.common.vec_env.vec_normalize import \
    VecNormalize as VecNormalize_


def _flatten_dict_obs(observation_space, obs_list):
    """Stack per-process dict observations into batched arrays, normalizing
    each component to the observation space's declared shape and dtype.

    The shape normalization is load-bearing for the upstream CrowdSimDict env:
    its generate_ob() returns robot_node as a flat 7-list and temporal_edges as
    a bare (2,) array — NOT the (1,7)/(1,2) the gym Dict space declares. The
    author's own ShmemVecEnv normalized this implicitly (workers np.copyto()
    raw values into shared buffers preallocated at the space's shape, then
    _decode_obses() reshaped on read); baselines' SubprocVecEnv/DummyVecEnv
    have no such step, so without it the batch comes out [nenv,7] instead of
    [nenv,1,7] and train.py's rollouts.obs.copy_() fails on a shape mismatch.
    """
    keys = obs_list[0].keys()
    out = {}
    for k in keys:
        space = observation_space.spaces[k]
        out[k] = np.stack(
            [np.asarray(o[k], dtype=space.dtype).reshape(space.shape) for o in obs_list])
    return out


class SubprocVecEnv(_SubprocVecEnv):
    """
    This installed baselines version's SubprocVecEnv predates gym.spaces.Dict
    observation support in vec envs — reset()/step_wait() do a blanket
    np.stack(obs) over the list of per-process observations, which produces
    a garbage dtype=object array when each observation is itself a dict
    (CrowdSimDict-v0's robot_node/temporal_edges/spatial_edges obs). Newer
    baselines versions handle this with a `_flatten_obs` helper that
    branches on the observation space type; only that piece is needed here
    (not the rest of the ShmemVecEnv/util.py machinery), so it's overridden
    directly rather than pulling in more compatibility shims.
    """

    def _flatten_obs(self, obs_list):
        if isinstance(self.observation_space, Dict):
            return _flatten_dict_obs(self.observation_space, obs_list)
        return np.stack(obs_list)

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return self._flatten_obs([remote.recv() for remote in self.remotes])

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return self._flatten_obs(obs), np.stack(rews), np.stack(dones), infos


class DummyVecEnv(DummyVecEnv_):
    """
    Same Dict-obs + shape-normalization fix as SubprocVecEnv above, for the
    num_processes == 1 path (used by test.py). The installed baselines 0.1.5
    DummyVecEnv predates dict observations entirely (it buffers Tuple obs only
    — with a Dict space its __init__ crashes), so __init__/step_wait/reset are
    overridden wholesale. Semantics follow the NEWER baselines DummyVecEnv the
    author's code was written against: no auto-reset on done (unlike the
    Shmem/Subproc workers, whose auto-reset upstream relied on for training).
    """

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool_)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None

    def _flatten_obs(self, obs_list):
        if isinstance(self.observation_space, Dict):
            return _flatten_dict_obs(self.observation_space, obs_list)
        return np.stack(obs_list)

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for env, a in zip(self.envs, self.actions)]
        obs, rews, dones, infos = zip(*results)
        # Auto-reset envs whose episode just ended — mirrors upstream
        # ShmemVecEnv's worker (`if done: obs = env.reset()`). REQUIRED by
        # pytorchBaselines/evaluation.py: its per-episode `while not done`
        # loops never call reset(), so the done-step itself must return the
        # NEXT episode's first observation (and installed baselines 0.1.5's
        # Monitor raises on step-after-done without it).
        obs = [env.reset() if d else o
               for env, o, d in zip(self.envs, obs, dones)]
        return self._flatten_obs(obs), np.stack(rews), np.stack(dones), list(infos)

    def reset(self):
        return self._flatten_obs([env.reset() for env in self.envs])

try:
    import dm_control2gym
except ImportError:
    pass

try:
    import roboschool
except ImportError:
    pass

try:
    import pybullet_envs
except ImportError:
    pass


def make_env(env_id, seed, rank, log_dir, allow_early_resets, config=None, envNum=1, ax=None, test_case=-1):
    def _thunk():
        if env_id.startswith("dm"):
            _, domain, task = env_id.split('.')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        else:
            env = gym.make(env_id)

        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)

        env.configure(config)

        # gym.make() returns a wrapped env (OrderEnforcing, etc. depending
        # on gym version) — gym.Wrapper.__setattr__ doesn't proxy plain
        # attribute assignment through to the underlying env (only method/
        # attribute *reads* fall back via __getattr__), so setting these
        # directly on `env` would silently set them on the wrapper instead
        # of the actual CrowdSimDict instance that reset()/step() read them
        # from — same gotcha crowdnav_env's evaluate.py works
        # around with env.unwrapped for the identical reason.
        envSeed = seed + rank if seed is not None else None
        # environment.render_axis = ax
        env.unwrapped.thisSeed = envSeed
        env.unwrapped.nenv = envNum
        if envNum > 1:
            env.unwrapped.phase = 'train'
        else:
            env.unwrapped.phase = 'test'

        if ax:
            env.unwrapped.render_axis = ax
            if test_case >= 0:
                env.unwrapped.test_case = test_case
        env.seed(seed + rank)

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)

        # if log_dir is not None:
        env = bench.Monitor(
            env,
            None,
            allow_early_resets=allow_early_resets)
        print(env)

        if isinstance(env.observation_space, Box):
            if is_atari:
                if len(env.observation_space.shape) == 3:
                    env = wrap_deepmind(env)
            elif len(env.observation_space.shape) == 3:
                raise NotImplementedError(
                    "CNN models work only for atari,\n"
                    "please use a custom wrapper for a custom pixel input env.\n"
                    "See wrap_deepmind for an example.")

            # If the input has shape (W,H,3), wrap for PyTorch convolutions

            obs_shape = env.observation_space.shape
            if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
                env = TransposeImage(env, op=[2, 0, 1])

        return env

    return _thunk


def make_vec_envs(env_name,
                  seed,
                  num_processes,
                  gamma,
                  log_dir,
                  device,
                  allow_early_resets,
                  num_frame_stack=None,
                  config=None,
                  ax=None, test_case=-1):
    envs = [
        make_env(env_name, seed, i, log_dir, allow_early_resets, config=config,
                 envNum=num_processes, ax=ax, test_case=test_case)
        for i in range(num_processes)
    ]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)  # patched subclass — see class docstring

    if isinstance(envs.observation_space, Box):
        if len(envs.observation_space.shape) == 1:
            if gamma is None:
                envs = VecNormalize(envs, ret=False, ob=False)
            else:
                envs = VecNormalize(envs, gamma=gamma, ob=False, ret=False)

    envs = VecPyTorch(envs, device)

    if num_frame_stack is not None:
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    elif isinstance(envs.observation_space, Box):
        if len(envs.observation_space.shape) == 3:
            envs = VecPyTorchFrameStack(envs, 4, device)

    return envs


# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# Can be used to test recurrent policies for Reacher-v2
class MaskGoal(gym.ObservationWrapper):
    def observation(self, observation):
        if self.env._elapsed_steps > 0:
            observation[-2:] = 0
        return observation


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        if isinstance(obs, dict):
            for key in obs:
                obs[key]=torch.from_numpy(obs[key]).to(self.device)
        else:
            obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        if isinstance(obs, dict):
            for key in obs:
                obs[key] = torch.from_numpy(obs[key]).to(self.device)
        else:
            obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info

    def render_traj(self, path, episode_num):
        if self.venv.num_envs == 1:
            return self.venv.envs[0].env.render_traj(path, episode_num)
        else:
            for i, curr_env in enumerate(self.venv.envs):
                curr_env.env.render_traj(path, str(episode_num) + '.' + str(i))


class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        if self.ob_rms:
            if self.training and update:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) /
                          np.sqrt(self.ob_rms.var + self.epsilon),
                          -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs, ) +
                                       low.shape).to(device)

        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:].clone()
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()
