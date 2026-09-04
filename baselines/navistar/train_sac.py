import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import time
import shutil
from offpolicy.video import VideoRecorder
from offpolicy.logger import Logger
from offpolicy.replay_buffer import ReplayBuffer
from offpolicy.agent.sac import SACAgent
import offpolicy.utils as utils
from crowd_nav.configs.config import Config
import crowd_sim
import os
from collections import deque

class Workspace(object):
    def __init__(self):

        self.config = Config()
        utils.set_seed_everywhere(self.config.env.seed)

        env_name = self.config.env.env_name
        task = self.config.env.task
        policy_name = self.config.robot.policy + '_sac'
        self.output_dir = os.path.join(self.config.training.output_dir, task, policy_name)
        # save policy to output_dir
        if os.path.exists(self.output_dir) and self.config.training.overwrite:  # if I want to overwrite the directory
            shutil.rmtree(self.output_dir)  # delete an entire directory tree

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # baselines/navistar: guard for chained resubmission — on a resume the
        # config snapshot from the original run is already there, and
        # copytree into an existing directory raises FileExistsError.
        configs_snapshot = os.path.join(self.output_dir, 'configs')
        if not os.path.exists(configs_snapshot):
            shutil.copytree('crowd_nav/configs', configs_snapshot)

        self.model_dir = os.path.join(self.output_dir, 'checkpoints')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.work_dir = self.output_dir
        print(f'workspace: {self.work_dir}')

        self.logger = Logger(self.work_dir,
                             save_tb=True,
                             log_frequency=10000,
                             agent='sac')


        self.device = torch.device("cuda" if self.config.training.cuda and torch.cuda.is_available() else "cpu")
        self.env = utils.make_env(self.config)
        self.eval_env = utils.make_eval_env(self.config)

        obs_shape = self.env.observation_space.spaces
        action_shape = self.env.action_space.shape
        self.agent = SACAgent(self.config, obs_shape, action_shape, self.device)

        self.replay_buffer = ReplayBuffer(obs_shape,
                                          action_shape,
                                          int(self.config.sac.num_train_steps),
                                          self.device)

        self.video_recorder = VideoRecorder(
            self.work_dir if self.config.sac.save_video else None)
        self.step = 0
        self.interaction = 0
        self.episode = 0

        self.max_success_rate = 0.

        # baselines/navistar: resume support for self-chaining sbatch resubmission
        # (train_baselines/navistar.slurm) — a 24h walltime can't fit
        # sac.num_train_steps=1e6 in one job, and this run has no way to pick
        # back up without it. Replay buffer content is NOT restored (SAC's
        # buffer is uncapped-by-config-size and checkpointing it would mean
        # writing/reading up to num_train_steps transitions every save,
        # dwarfing the actor/critic checkpoint cost); we resume the policy,
        # critics, and step counter and let the buffer refill from the
        # partially-trained policy, which is standard practice for this kind
        # of resume.
        self.training_state_path = os.path.join(self.model_dir, 'training_state.pt')
        if os.path.exists(self.training_state_path):
            state = torch.load(self.training_state_path, map_location=self.device)
            self.agent.actor.load_state_dict(state['actor'])
            self.agent.critic.load_state_dict(state['critic'])
            self.agent.critic_target.load_state_dict(state['critic_target'])
            self.agent.log_alpha.data.copy_(state['log_alpha'])
            self.agent.actor_optimizer.load_state_dict(state['actor_optimizer'])
            self.agent.critic_optimizer.load_state_dict(state['critic_optimizer'])
            self.agent.log_alpha_optimizer.load_state_dict(state['log_alpha_optimizer'])
            self.step = state['step']
            self.episode = state['episode']
            self.max_success_rate = state['max_success_rate']
            print(f'resumed from {self.training_state_path}: step={self.step} '
                  f'episode={self.episode} max_success_rate={self.max_success_rate}')

    def save_training_state(self):
        state = {
            'actor': self.agent.actor.state_dict(),
            'critic': self.agent.critic.state_dict(),
            'critic_target': self.agent.critic_target.state_dict(),
            'log_alpha': self.agent.log_alpha.data,
            'actor_optimizer': self.agent.actor_optimizer.state_dict(),
            'critic_optimizer': self.agent.critic_optimizer.state_dict(),
            'log_alpha_optimizer': self.agent.log_alpha_optimizer.state_dict(),
            'step': self.step,
            'episode': self.episode,
            'max_success_rate': self.max_success_rate,
        }
        # write to a tmp file first — a job killed by SLURM walltime mid-save
        # must never leave a corrupt training_state.pt for the next resubmit
        # to trip over.
        tmp_path = self.training_state_path + '.tmp'
        torch.save(state, tmp_path)
        os.replace(tmp_path, self.training_state_path)

    def evaluate(self):
        success = 0
        collision = 0
        timeout = 0
        average_episode_reward = 0
        # baselines/navistar: was `self.eval_env.case_counter['test'] = 0` INSIDE
        # the loop, forced before every episode. crowd_sim.py's reset()
        # reseeds np.random from case_counter[phase] and only advances
        # case_counter AFTER seeding (case_counter = (case_counter+1) %
        # case_size) — so resetting it to 0 every iteration undid that
        # advance before the next reset() ever saw it. All
        # num_eval_episodes "episodes" were therefore the exact same
        # deterministic scenario run num_eval_episodes times, not a sample
        # of num_eval_episodes different ones — the SR/CR/TR logged during
        # training was "does the policy solve this one fixed layout right
        # now", not a population statistic. Reset the counter once, before
        # the loop, so each episode gets the next scenario in sequence like
        # the actual 500-episode harness (evaluate.py) does.
        self.eval_env.case_counter['test'] = 0
        for episode in range(self.config.sac.num_eval_episodes):
            obs = self.eval_env.reset()
            self.agent.reset()
            done = False
            episode_reward = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, info = self.eval_env.step(action)
                done = done[0]
                episode_reward += reward[0]

            average_episode_reward += episode_reward
            status = str(info['info'])
            if status == 'Reaching goal':
                success += 1
            elif status == 'Collision':
                collision += 1
            elif status == 'Timeout':
                timeout += 1
        average_episode_reward /= self.config.sac.num_eval_episodes
        success_rate = success / self.config.sac.num_eval_episodes
        collision_rate = collision / self.config.sac.num_eval_episodes
        timeout_rate = timeout / self.config.sac.num_eval_episodes

        if success_rate > self.max_success_rate:
            self.max_success_rate = success_rate
            torch.save(self.agent.actor.state_dict(), os.path.join(self.model_dir, './best_sac_actor.pt'))

        print('eval', average_episode_reward)
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.log('eval/success_rate', success_rate,
                        self.step)
        self.logger.log('eval/collision_rate', collision_rate,
                        self.step)
        self.logger.log('eval/timeout_rate', timeout_rate,
                        self.step)
        self.logger.dump(self.step)

    def run(self):
        episode, episode_step, episode_reward, done = self.episode, 0, 0, True
        start_time = time.time()
        episode_rewards = deque(maxlen=100)
        reward_list = []
        # baselines/navistar: MetersGroup._dump_to_csv (offpolicy/logger.py:82-86)
        # locks its CSV fieldnames from the first row it's ever asked to
        # write (save=True) and crashes on any later call whose dict has
        # keys outside that locked set. `save=(self.step >
        # num_seed_steps)` was meant as a proxy for "update() has started
        # logging actor/critic loss keys, so the row is actually complete" —
        # true together on a fresh run, since both flip at the same
        # boundary. On a RESUMED run self.step restores past num_seed_steps
        # immediately, but update() is separately gated on the replay buffer
        # actually holding a batch (buffer isn't persisted across resumes —
        # see Workspace.__init__), so `save` went True several dump() calls
        # before any loss keys existed, locked the CSV to the narrower set,
        # and the first dump() after update() finally ran crashed with
        # "dict contains fields not in fieldnames". Simply moving the
        # premature dump() later only delays the crash; the fix is to track
        # whether update() has actually run rather than inferring it from
        # self.step.
        first_iteration = True
        agent_updated_once = False
        while self.step < self.config.sac.num_train_steps:
            if done:
                if not first_iteration:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(self.step, save=agent_updated_once)
                first_iteration = False


                if self.interaction > self.config.sac.save_interval:
                    self.interaction = 0
                    filename = 'sac_actor' + str(self.step) + '.pt'
                    torch.save(self.agent.actor.state_dict(), os.path.join(self.model_dir, filename))
                    np.save(os.path.join(self.output_dir, 'reward.npy'), reward_list)
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()
                    self.episode = episode
                    self.save_training_state()

                self.logger.log('train/episode_reward', episode_reward,
                                self.step)
                if self.step >= self.config.sac.num_seed_steps:
                    episode_rewards.append(episode_reward)
                    reward_list.append(np.mean(episode_rewards))
                    print('%d/%d, %d, %f' % (self.step, self.config.sac.num_train_steps, episode_step, np.mean(episode_rewards)))

                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.config.sac.num_seed_steps:
                action = self.env.action_space.sample()
                action = utils.clip_action(action, clip_norm=True, max_norm=self.config.robot.v_pref)
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)
                    action = utils.clip_action(action, clip_norm=True, max_norm=self.config.robot.v_pref)

            # run training update
            # baselines/navistar: gating on self.step alone crashes every resumed
            # job — self.step restores to wherever training left off (well
            # past num_seed_steps), but the replay buffer is NOT restored
            # (see Workspace.__init__), so it's empty right after resume and
            # ReplayBuffer.sample()'s np.random.randint(0, 0, ...) raises
            # ValueError: high <= 0. Every resumed segment of a chained run
            # therefore crashed on its first update() call, and the chain's
            # "no new checkpoint" guard then correctly stopped re-queuing --
            # so the failure surfaced as a silently stalled chain rather than
            # an error. Requiring the buffer to actually hold a batch's worth
            # of transitions fixes the resumed case; for a fresh run
            # len(replay_buffer) always exceeds
            # batch_size well before self.step reaches num_seed_steps, so
            # this changes nothing about the original from-scratch schedule.
            if (self.step >= self.config.sac.num_seed_steps
                    and len(self.replay_buffer) >= self.config.sac.batch_size):
                self.agent.update(self.replay_buffer, self.logger, self.step)
                agent_updated_once = True


            next_obs, reward, done, info = self.env.step(action)

            # allow infinite bootstrap
            done = float(done[0])
            # baselines/navistar: bootstrap the critic across time-limit
            # truncations instead of treating them as terminals — the intent
            # of the comment above, which the original `done_no_max = done`
            # defeated. At our eval-matched time_limit=25, truncations are
            # common early in training and terminal-izing them poisons Q.
            # ReachGoal/Collision are genuine terminals and keep not_done=0.
            done_no_max = 0.0 if type(info['info']).__name__ == 'Timeout' else done
            episode_reward += reward[0]

            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1
            self.interaction += 1

        # loop exited because self.step reached num_train_steps (not a
        # walltime kill, which never reaches this line) — persist final state
        # and drop a DONE marker so the self-chaining slurm script knows to
        # stop resubmitting.
        self.episode = episode
        self.save_training_state()
        open(os.path.join(self.model_dir, 'DONE'), 'w').close()


def main():
    workspace = Workspace()
    workspace.run()


if __name__ == '__main__':
    main()
