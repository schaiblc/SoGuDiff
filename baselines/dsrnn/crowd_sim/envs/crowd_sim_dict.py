import gym
import numpy as np
from numpy.linalg import norm
import copy
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs import CrowdSim

# The class for the simulation environment used for training a DSRNN policy

class CrowdSimDict(CrowdSim):
    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.
        """
        super().__init__()

        self.desiredVelocity=[0.0,0.0]


    # define the observation space and the action space
    def set_robot(self, robot):
        self.robot = robot

        # set observation space and action space
        # we set the max and min of action/observation space as inf
        # clip the action and observation as you need

        d={}
        # robot node: px, py, r, gx, gy, v_pref, theta
        d['robot_node'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,7,), dtype = np.float32)
        # only consider the robot temporal edge and spatial edges pointing from robot to each human
        d['temporal_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,2,), dtype=np.float32)
        # FIXED network input size = observed_human_num (NOT the variable actual
        # human_num); generate_ob feeds the closest observed_human_num + padding.
        d['spatial_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.observed_human_num, 2), dtype=np.float32)
        self.observation_space=gym.spaces.Dict(d)

        high = np.inf * np.ones([2, ])
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)


    # generate observation for each timestep
    # reset = True: reset calls this function; reset = False: step calls this function
    def generate_ob(self, reset):
        ob = {}

        # nodes
        visible_humans, num_visibles, human_visibility = self.get_num_human_in_fov()

        # set_robot() declares robot_node/temporal_edges with a leading
        # size-1 "node" dimension (shape (1,7)/(1,2)) — wrapped here to
        # actually match, rather than returning flat (7,)/(2,) arrays.
        # DummyVecEnv (single-process, what this repo's own test.py uses)
        # tolerates the mismatch silently; RolloutStorage's fixed-shape
        # training buffers do not.
        ob['robot_node'] = np.array([self.robot.get_full_state_list_noV()])

        self.update_last_human_states(human_visibility, reset=reset)

        # edges
        # temporal edge: robot's velocity
        ob['temporal_edges'] = np.array([[self.robot.vx, self.robot.vy]])
        # spatial edges: vector from robot to each human's position, for the
        # CLOSEST observed_human_num humans, with far-away padding for empty
        # slots — IDENTICAL to the eval adapter (crowd_nav/policy/dsrnn.py
        # predict()) so DSRNN sees the same input distribution at train and
        # eval. With full-circle FOV (humans.FOV=2pi) every human is visible, so
        # last_human_states holds the current true states.
        obs_n = self.observed_human_num
        ob['spatial_edges'] = np.full((obs_n, 2), self.pad_offset, dtype=np.float32)
        if self.human_num > 0:
            rel = self.last_human_states[:self.human_num, :2] - np.array([self.robot.px, self.robot.py])
            order = np.argsort(np.hypot(rel[:, 0], rel[:, 1]))
            for slot, i in enumerate(order[:obs_n]):
                ob['spatial_edges'][slot] = rel[i]

        return ob


    # reset function
    def reset(self, phase='train', test_case=None):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """

        if self.phase is not None:
            phase = self.phase
        if self.test_case is not None:
            test_case=self.test_case

        if self.robot is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']
        if test_case is not None:
            self.case_counter[phase] = test_case # test case is passed in to calculate specific seed to generate case
        self.global_time = 0

        self.desiredVelocity = [0.0, 0.0]
        self.humans = []
        # train, val, and test phase should start with different seed.
        # case capacity: the maximum number for train(max possible int -2000), val(1000), and test(1000)
        # val start from seed=0, test start from seed=case_capacity['val']=1000
        # train start from self.case_capacity['val'] + self.case_capacity['test']=2000
        counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                          'val': 0, 'test': self.case_capacity['val']}

        # here we use a counter to calculate seed. The seed=counter_offset + case_counter
        np.random.seed(counter_offset[phase] + self.case_counter[phase] + self.thisSeed)
        # Variable ACTUAL human count per episode for the evaluation scenario,
        # matching crowdnav_env's eval (randint(1,11) -> 1..10). Drawn
        # AFTER the seed so it stays deterministic. The network input size
        # (observed_human_num) is unchanged — generate_ob feeds the closest
        # observed_human_num humans + padding.
        if getattr(self.config.sim, 'scenario', 'circle_crossing') == 'evaluation':
            self.human_num = int(np.random.randint(self.min_human_num, self.max_human_num + 1))
        # last_human_states tracks the ACTUAL humans this episode; size to match.
        self.last_human_states = np.zeros((self.human_num, 5))
        self.generate_robot_humans(phase)


        # If configured to randomize human policies, do so
        if self.random_policy_changing:
            self.randomize_human_policies()

        # case size is used to make sure that the case_counter is always between 0 and case_size[phase]
        self.case_counter[phase] = (self.case_counter[phase] + int(1*self.nenv)) % self.case_size[phase]

        # get robot observation
        ob = self.generate_ob(reset=True)

        # initialize potential
        self.potential = -abs(np.linalg.norm(np.array([self.robot.px, self.robot.py]) - np.array([self.robot.gx, self.robot.gy])))


        return ob


    # step function
    def step(self, action, update=True):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
        """
        action = self.robot.policy.clip_action(action, self.robot.v_pref)

        if self.robot.kinematics == 'unicycle':
            # v: forward-only (matches the other unicycle baselines in
            # crowdnav_env — no reverse), accel-rate-limited via
            # clip_action's bound on the increment above.
            self.desiredVelocity[0] = np.clip(self.desiredVelocity[0]+action.v, 0.0, self.robot.v_pref)
            # omega: same persistent-accumulator pattern already used for v
            # above, extended to the turning rate (desiredVelocity[1] was
            # otherwise unused). action.r here is now a delta-omega (see
            # srnn.py's clip_action), so it accumulates into a rate-limited
            # persistent omega rather than being applied as a raw per-step
            # theta change like the original unconstrained version did.
            max_wrot = self.config.action_space.max_wrot
            self.desiredVelocity[1] = np.clip(self.desiredVelocity[1]+action.r, -max_wrot, max_wrot)
            action=ActionRot(self.desiredVelocity[0], self.desiredVelocity[1]*self.time_step)


        human_actions = self.get_human_actions()

        # compute reward and episode info
        reward, done, episode_info = self.calc_reward(action)


        # apply action and update all agents
        self.robot.step(action)
        for i, human_action in enumerate(human_actions):
            self.humans[i].step(human_action)
        self.global_time += self.time_step # max episode length=time_limit/time_step


        # compute the observation
        ob = self.generate_ob(reset=False)

        info={'info':episode_info}
        # Proper time-limit handling: a Timeout is a *truncation*, not a true
        # terminal — the robot is still mid-navigation with positive expected
        # future return. Flag it so train.py / train_height.py set bad_masks=0
        # and RolloutStorage.compute_returns() bootstraps V(s) for this step
        # instead of the (wrong) V=0 terminal target. Without this the critic
        # is trained toward 0 on exactly the states that dominate early
        # training under the accel-limited (double-integrator) dynamics — where
        # episodes time out far more often than under the original single-
        # integrator scheme — which poisons the value targets (observed:
        # value_loss stuck ~6-9, flat reward). ReachGoal/Collision are genuine
        # terminals and correctly keep bad_mask=1. Requires
        # training.use_proper_time_limits = True (config.py / height_config.py).
        if type(episode_info).__name__ == 'Timeout':
            info['bad_transition'] = True


        # Update all humans' goals randomly midway through episode
        if self.random_goal_changing:
            if self.global_time % 5 == 0:
                self.update_human_goals_randomly()
        
        # Update a specific human's goal once its reached its original goal
        if self.end_goal_changing:
            for human in self.humans:
                if norm((human.gx - human.px, human.gy - human.py)) < human.radius:
                    self.update_human_goal(human)


        return ob, reward, done, info

