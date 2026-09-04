import logging
import gym
import matplotlib.lines as mlines
import numpy as np
import rvo2
from matplotlib import patches
from numpy.linalg import norm
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.utils import point_to_segment_dist
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.state import ObservableState
import math
import os

import shutil
class CrowdSim(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.

        """
        self.time_limit = None
        self.time_step = None
        self.robot = None
        self.humans = None
        self.global_time = None
        self.human_times = None
        # reward function
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.discomfort_penalty_factor = None
        # simulation configuration
        self.config = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None
        self.train_val_sim = None
        self.test_sim = None
        self.square_width = None
        self.circle_radius = None
        self.human_num = None
        # for visualization
        self.states = None
        self.action_values = None
        self.attention_weights = None
        self.predicted_trajs = None
        self.testoffset = 0
        self.minobsdist = float('inf')
        self.pathlength = 0
        self.avgobsdist = []
        self.npz_num=0
        # Static map state — populated by load_npz_scenario / _draw_from_npz_pool
        self._npz_occ_world_xy  = None                              # (N,2) world-frame cell centers
        self._npz_has_map       = 0.0
        self._npz_map_extent    = 10.0
        self._npz_cell_res      = 0.2                               # m/cell
        # Human data (npz_hard path) — initialized here so the attribute always
        # exists; populated by load_npz_scenario / _draw_from_npz_pool.
        self._npz_humans_data   = np.zeros((0, 4), dtype=np.float32)
        self._npz_map_eval_type = 'none'

    def configure(self, config):
        self.config = config
        self.time_limit = config.getint('env', 'time_limit')
        self.time_step = config.getfloat('env', 'time_step')
        self.randomize_attributes = config.getboolean('env', 'randomize_attributes')
        self.success_reward = config.getfloat('reward', 'success_reward')
        self.collision_penalty = config.getfloat('reward', 'collision_penalty')
        self.discomfort_dist = config.getfloat('reward', 'discomfort_dist')
        self.discomfort_penalty_factor = config.getfloat('reward', 'discomfort_penalty_factor')
        if self.config.get('humans', 'policy') == 'orca':
            self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
            self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': config.getint('env', 'val_size'),
                              'test': config.getint('env', 'test_size')}
            self.train_val_sim = config.get('sim', 'train_val_sim')
            self.test_sim = config.get('sim', 'test_sim')
            self.square_width = config.getfloat('sim', 'square_width')
            self.circle_radius = config.getfloat('sim', 'circle_radius')
            self.human_num = config.getint('sim', 'human_num')
            self.testoffset = config.getint('sim', 'testoffset')
        else:
            raise NotImplementedError
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        logging.info('human number: {}'.format(self.human_num))
        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")
        logging.info('Training simulation: {}, test simulation: {}'.format(self.train_val_sim, self.test_sim))
        logging.info('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))

    def set_robot(self, robot):
        self.robot = robot

    def load_npz_scenario(self, npz_path):
        """
        Load a pre-generated hard scenario from an npz file.
        Populates self._npz_humans_data, self._npz_goal, self._npz_v0,
        self._npz_w0.
 
        Expected npz schema:
          start_state: [v0, w0]      (legacy: [v0] also accepted; w0 → 0)
          goal:        [gx, gy]
          obstacles:   [N, 4]        (x, y, vx, vy per row)
          threat_type: scalar str    (informational)
 
        Call this before reset() when using npz_hard evaluation mode.
        """
        data = np.load(npz_path, allow_pickle=True)
 
        obstacles = data['obstacles']  # [N, 4] or [0, 4]
        goal      = data['goal']       # [2]
 
        # start_state may be [v0] (old) or [v0, w0] (new). Default w0 to 0
        # for backward compatibility with scenarios generated before w0 was
        # included.
        if 'start_state' in data:
            ss = data['start_state']
            v0 = float(ss[0])
            w0 = float(ss[1]) if len(ss) > 1 else 0.0
        else:
            v0 = 1.0
            w0 = 0.0

        if len(obstacles) > 0:
            valid_mask = np.any(obstacles != 0, axis=1)
            obstacles  = obstacles[valid_mask]

        # NEW: pick up static map if present
        if 'occupancy_map' in data.files and 'has_map' in data.files:
            occ_map  = np.asarray(data['occupancy_map'], dtype=np.float32)
            has_map  = float(data['has_map'])

            # Convert occupied cells to WORLD coordinates ONCE, at scenario load.
            # The npz map is ego-centered at the robot's starting pose, with
            # +x = forward, +y = left, robot at center. Since the robot starts
            # at world (0, 0) with heading 0 in npz_hard mode (see reset()),
            # the ego frame at t=0 coincides with the world frame — so the
            # cell→world transform is just the cell→ego transform with the
            # robot at (0, 0, 0).
            H, W = occ_map.shape
            map_extent = 10   # must match training data generator
            res = map_extent / W
            occ_rows, occ_cols = np.where(occ_map > 0.5)
            # Convention: row indexes y, col indexes x
            x_world = (occ_cols - W / 2 + 0.5) * res
            y_world = (occ_rows - H / 2 + 0.5) * res
            occ_world_xy = np.stack([x_world, y_world], axis=1).astype(np.float32)
        else:
            occ_map      = None
            occ_world_xy = None
            has_map      = 0.0

        self._npz_humans_data = obstacles
        self._npz_goal        = goal
        self._npz_v0          = v0
        self._npz_w0          = w0
        self._npz_path        = npz_path
        self._npz_occ_map      = occ_map         # kept for visualization only
        self._npz_occ_world_xy = occ_world_xy    # used at runtime — world coords
        self._npz_has_map      = has_map
        self._npz_map_extent   = 10             # must match training
        if occ_map is not None:
            _, W = occ_map.shape
            self._npz_cell_res = 10.0 / W       # meters per cell
        else:
            self._npz_cell_res = 0.2            # default: 10m / 50 cells

    def set_npz_pool(self, npz_dir, hard_only=True):
        """
        Register a directory of npz scenarios as a pool. When the active
        sim rule for a phase is 'npz_hard' and a pool is registered, each
        reset() pulls the next scenario from the pool (cycling with a
        per-cycle shuffle) so all examples are seen during training.

        This enables npz_hard for *training* and *val*, not just *test*.
        The single-file load_npz_scenario() path still works — when called
        explicitly (as test.py does), reset() uses that pre-loaded scenario
        instead of pulling from the pool.

        Filters use the same hard-only criterion as test.py:
          difficulty == 'hard' OR
          threat_type in {frontal, side, rear, mixed, pincer, group}
        Pass hard_only=False to disable the filter.
        """
        if npz_dir is None or not os.path.isdir(npz_dir):
            self._npz_pool = []
            self._npz_pool_pos = 0
            self._npz_pool_dir = None
            if npz_dir is not None:
                logging.warning('npz_dir %r not found; pool empty', npz_dir)
            return

        all_files = sorted(
            os.path.join(npz_dir, f)
            for f in os.listdir(npz_dir)
            if f.endswith('.npz')
        )

        # Pre-parse each file once so per-episode reset is just a dict lookup.
        # Each entry is the same triple (humans_data, goal, v0) load_npz_scenario
        # would set on the env — so the existing reset() path can use it without
        # touching disk.
        pool = []
        for f in all_files:
            try:
                d = np.load(f, allow_pickle=True)
                if hard_only:
                    difficulty = str(d['difficulty']) if 'difficulty' in d else None
                    threat = str(d['threat_type']) if 'threat_type' in d else None
                    is_hard = (
                        difficulty == 'hard' or
                        threat in ['frontal', 'side', 'rear', 'mixed', 'pincer', 'group']
                    )
                    if not is_hard:
                        continue

                obstacles = d['obstacles']
                goal      = d['goal']
                if 'start_state' in d:
                    ss = d['start_state']
                    v0 = float(ss[0])
                    w0 = float(ss[1]) if len(ss) > 1 else 0.0
                else:
                    v0 = 1.0
                    w0 = 0.0
 
                if len(obstacles) > 0:
                    valid_mask = np.any(obstacles != 0, axis=1)
                    obstacles  = obstacles[valid_mask]
                
                if 'occupancy_map' in d.files and 'has_map' in d.files:
                    occ_map  = np.asarray(d['occupancy_map'], dtype=np.float32)
                    has_map  = float(d['has_map'])

                    # Convert occupied cells to WORLD coordinates ONCE, at scenario load.
                    # The npz map is ego-centered at the robot's starting pose, with
                    # +x = forward, +y = left, robot at center. Since the robot starts
                    # at world (0, 0) with heading 0 in npz_hard mode (see reset()),
                    # the ego frame at t=0 coincides with the world frame — so the
                    # cell→world transform is just the cell→ego transform with the
                    # robot at (0, 0, 0).
                    H, W = occ_map.shape
                    map_extent = 10   # must match training data generator
                    res = map_extent / W
                    occ_rows, occ_cols = np.where(occ_map > 0.5)
                    # Convention: row indexes y, col indexes x
                    x_world = (occ_cols - W / 2 + 0.5) * res
                    y_world = (occ_rows - H / 2 + 0.5) * res
                    occ_world_xy = np.stack([x_world, y_world], axis=1).astype(np.float32)
                    cell_res = res
                else:
                    occ_map      = None
                    occ_world_xy = None
                    has_map      = 0.0
                    cell_res     = 0.2   # default: 10m / 50 cells

                pool.append({
                    'path':             f,
                    'humans':           obstacles,
                    'goal':             np.asarray(goal),
                    'v0':               v0,
                    'w0':               w0,
                    'npz_occ_map':      occ_map,         # kept for visualization only
                    'npz_occ_world_xy': occ_world_xy,    # used at runtime — world coords
                    'npz_has_map':      has_map,
                    'npz_map_extent':   10,              # must match training
                    'npz_cell_res':     cell_res,
                })
            except Exception as e:
                logging.warning('Skipping %s (load failed): %s', f, e)

        self._npz_pool      = pool
        self._npz_pool_pos  = 0
        self._npz_pool_dir  = npz_dir
        logging.info('NPZ pool: %d scenarios from %s', len(pool), npz_dir)

    def _draw_from_npz_pool(self, rng):
        """
        Pull the next scenario from the pool. Cycles through the full list
        and reshuffles at each cycle boundary using the given seeded rng,
        so every scenario is seen each pass and the schedule is deterministic
        in the master seed.
        """
        if not getattr(self, '_npz_pool', None):
            raise ValueError(
                'npz_hard requested but no pool registered. '
                'Call env.set_npz_pool(dir) before training, or '
                'env.load_npz_scenario(path) before reset() for single-file use.'
            )
        if self._npz_pool_pos >= len(self._npz_pool):
            # End of cycle — reshuffle for the next pass
            order = rng.permutation(len(self._npz_pool))
            self._npz_pool = [self._npz_pool[i] for i in order]
            self._npz_pool_pos = 0

        entry = self._npz_pool[self._npz_pool_pos]
        self._npz_pool_pos += 1

        # Populate the same attributes load_npz_scenario() would, so the
        # existing reset() branch can use them without modification.
        self._npz_humans_data  = entry['humans']
        self._npz_goal         = entry['goal']
        self._npz_v0           = entry['v0']
        self._npz_w0           = entry['w0']
        self._npz_path         = entry['path']
        self._npz_occ_map      = entry.get('npz_occ_map')         # kept for visualization only
        self._npz_occ_world_xy = entry.get('npz_occ_world_xy')    # used at runtime — world coords
        self._npz_has_map      = entry.get('npz_has_map', 0.0)
        self._npz_map_extent   = entry.get('npz_map_extent', 10)
        self._npz_cell_res     = entry.get('npz_cell_res', 0.2)

    def generate_random_human_position(self, human_num, rule, rng):
        """
        Generate human position according to certain rule
        Rule square_crossing: generate start/goal position at two sides of y-axis
        Rule circle_crossing: generate start position on a circle, goal position is at the opposite side

        :param human_num:
        :param rule:
        :return:
        """
        # initial min separation distance to avoid danger penalty at beginning
        if rule == 'square_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_square_crossing_human())
        elif rule == 'circle_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_circle_crossing_human())
        elif rule == 'evaluation':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_evaluation_human(rng))
        elif rule == 'map_eval':
            self.humans = []
            for _ in range(human_num):
                h = self._generate_map_eval_human(rng)
                if h is not None:
                    self.humans.append(h)
            self.human_num = len(self.humans)
        elif rule == 'npz_hard':
            # humans already loaded and stored in self._npz_humans_data
            # by load_npz_scenario() — just instantiate them
            self.humans = []
            for obs_data in self._npz_humans_data:
                human = self.generate_npz_hard_human(obs_data, rng)  # pass rng
                self.humans.append(human)
            self.human_num = len(self.humans)
        elif rule == 'mixed':
            # mix different raining simulation with certain distribution
            static_human_num = {0: 0.05, 1: 0.2, 2: 0.2, 3: 0.3, 4: 0.1, 5: 0.15}
            dynamic_human_num = {1: 0.3, 2: 0.3, 3: 0.2, 4: 0.1, 5: 0.1}
            static = True if np.random.random() < 0.2 else False
            prob = np.random.random()
            for key, value in sorted(static_human_num.items() if static else dynamic_human_num.items()):
                if prob - value <= 0:
                    human_num = key
                    break
                else:
                    prob -= value
            self.human_num = human_num
            self.humans = []
            if static:
                # randomly initialize static objects in a square of (width, height)
                width = 4
                height = 8
                if human_num == 0:
                    human = Human(self.config, 'humans')
                    human.set(0, -10, 0, -10, 0, 0, 0)
                    self.humans.append(human)
                for i in range(human_num):
                    human = Human(self.config, 'humans')
                    if np.random.random() > 0.5:
                        sign = -1
                    else:
                        sign = 1
                    while True:
                        px = np.random.random() * width * 0.5 * sign
                        py = (np.random.random() - 0.5) * height
                        collide = False
                        for agent in [self.robot] + self.humans:
                            if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                                collide = True
                                break
                        if not collide:
                            break
                    human.set(px, py, px, py, 0, 0, 0)
                    self.humans.append(human)
            else:
                # the first 2 two humans will be in the circle crossing scenarios
                # the rest humans will have a random starting and end position
                for i in range(human_num):
                    if i < 2:
                        human = self.generate_circle_crossing_human()
                    else:
                        human = self.generate_square_crossing_human()
                    self.humans.append(human)
        else:
            raise ValueError("Rule doesn't exist")

    def generate_circle_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            px_noise = (np.random.random() - 0.5) * human.v_pref
            py_noise = (np.random.random() - 0.5) * human.v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False
            for agent in [self.robot] + self.humans:
                min_dist = human.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, -px, -py, 0, 0, 0)
        return human

    def generate_square_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        if np.random.random() > 0.5:
            sign = -1
        else:
            sign = 1
        while True:
            px = np.random.random() * self.square_width * 0.5 * sign
            py = (np.random.random() - 0.5) * self.square_width
            collide = False
            for agent in [self.robot] + self.humans:
                if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        while True:
            gx = np.random.random() * self.square_width * 0.5 * -sign
            gy = (np.random.random() - 0.5) * self.square_width
            collide = False
            for agent in [self.robot] + self.humans:
                if norm((gx - agent.gx, gy - agent.gy)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, gx, gy, 0, 0, 0)
        return human

    def generate_evaluation_human(self, rng):
        """
        Evaluation mode:
        - Random start inside square
        - Random goal inside square
        - No initial collisions
        - Deterministic if seed provided
        """

        human = Human(self.config, 'humans')

        if self.randomize_attributes:
            human.sample_random_attributes()

        half_w = self.square_width / 2

        # ---------
        # Sample Start
        # ---------
        while True:
            px = rng.uniform(-half_w, half_w)
            py = rng.uniform(-half_w, half_w)

            collide = False
            for agent in [self.robot] + self.humans:
                min_dist = 2*(human.radius + agent.radius) + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist:
                    collide = True
                    break

            if not collide:
                break

        # ---------
        # Sample Goal
        # ---------
        while True:
            gx = rng.uniform(-half_w, half_w)
            gy = rng.uniform(-half_w, half_w)

            # Avoid trivial goal (too close to start)
            # if norm((gx - px, gy - py)) < 1.0:
            #     continue

            collide = False
            for agent in [self.robot] + self.humans:
                min_dist = 2*(human.radius + agent.radius) + self.discomfort_dist
                if norm((gx - agent.gx, gy - agent.gy)) < min_dist:
                    collide = True
                    break

            if not collide:
                break

        human.set(px, py, gx, gy, 0, 0, 0)

        return human

    def generate_npz_hard_human(self, obs_data, rng):
        """
        Create a Human agent from a pre-generated npz obstacle entry.
        obs_data: [x, y, vx, vy] — obstacle state at t=0
        """
        human = Human(self.config, 'humans')
        
        if self.randomize_attributes:
            human.sample_random_attributes()
        
        px, py, vx, vy = float(obs_data[0]), float(obs_data[1]), \
                        float(obs_data[2]), float(obs_data[3])
        
        # Goal: propagate current velocity forward to estimate where this
        # human is heading. Use a fixed time horizon so the goal is meaningful.
        propagation_time = self.time_limit + rng.uniform(5.0, 10.0)
        speed = np.sqrt(vx**2 + vy**2)

        # Use the npz-recorded speed as this human's preferred speed instead of
        # the Human class default (or randomize_attributes' sampled value).
        # Floor it so a stationary/near-stationary human doesn't get v_pref≈0,
        # which ORCA/goal-seeking logic elsewhere assumes is nonzero.
        human.v_pref = max(speed, 0.2)
        
        if speed > 0.05:
            # Human is moving — goal is where they're heading
            gx = px + vx * propagation_time
            gy = py + vy * propagation_time
            # Clamp to arena bounds
            # gx = np.clip(gx, -self.square_width/2, self.square_width/2)
            # gy = np.clip(gy, -self.square_width/2, self.square_width/2)
        else:
            # Stationary human — goal is their current position (they stay put)
            gx, gy = px, py
        
        # Set human: px, py, gx, gy, vx, vy, theta
        theta = np.arctan2(vy, vx) if speed > 0.05 else 0.0
        human.set(px, py, gx, gy, vx, vy, theta)
        
        return human

    # =================================================================
    # map_eval helpers
    # =================================================================

    def _map_gen(self):
        """Lazily build the SceneMapGenerator using the robot's policy config."""
        if not hasattr(self, '_cached_map_gen') or self._cached_map_gen is None:
            from crowd_sim.envs.utils.map_scene_generator import SceneMapGenerator
            map_size = (getattr(self.robot.policy, 'map_size', 50)
                        if self.robot is not None else 50)
            self._cached_map_gen = SceneMapGenerator(
                map_size=map_size, map_extent=10.0)
        return self._cached_map_gen

    def _generate_map_eval_scene(self, rng):
        """
        Generate one map_eval scene: sample a layout type, place robot at
        origin, choose a free reachable goal. Returns a dict with all the
        map attributes needed by the reset block.
        """
        gen = self._map_gen()

        # Sample goal in free space with a challenging map layout.
        # Longer goal distances (3-5.5 m) give the map room to obstruct the path.
        goal = None
        occ = gen._empty()
        map_type = "open"
        occ_world_xy = gen.occ_to_world_xy(occ)

        for _ in range(20):
            dist  = rng.uniform(3.0, 5.5)
            angle = rng.uniform(-np.pi, np.pi)
            g_try = np.array([dist * np.cos(angle), dist * np.sin(angle)])
            g_try = np.clip(g_try, -gen.half + 0.5, gen.half - 0.5)
            occ_try, world_xy_try, type_try = gen.generate_scene(
                rng, goal=g_try, challenge=True, min_obstruction=0.20)
            robot_r = self.robot.radius if self.robot else 0.30
            if gen.is_free(occ_try, g_try, robot_r + 0.1):
                goal         = g_try
                occ          = occ_try
                occ_world_xy = world_xy_try
                map_type     = type_try
                break

        if goal is None:
            # Absolute fallback: open map, simple straight goal
            goal         = np.array([3.0, 0.0])
            occ          = gen._empty()
            occ_world_xy = gen.occ_to_world_xy(occ)
            map_type     = "open"

        v0 = float(rng.uniform(0.0, 1.0))

        return dict(
            goal         = goal,
            v0           = v0,
            occ_map      = occ,
            occ_world_xy = occ_world_xy,
            has_map      = 1.0,
            map_type     = map_type,
            map_extent   = 10.0,
            cell_res     = 10.0 / gen.map_size,
        )

    def _generate_map_eval_human(self, rng):
        """
        Place a single human in free map space with a reachable free-space goal.
        Returns a Human object or None if placement failed after many attempts.
        """
        gen     = self._map_gen()
        occ     = self._npz_occ_map
        human   = Human(self.config, 'humans')
        r_clear = human.radius + 0.15

        # Exclude positions already occupied (robot + existing humans)
        exclude = [np.array([self.robot.px, self.robot.py])]
        for h in self.humans:
            exclude.append(np.array([h.px, h.py]))

        start = gen.sample_free_position(
            occ, rng, clearance_m=r_clear,
            exclude_positions=exclude,
            exclude_radius=self.discomfort_dist + human.radius + 0.1)
        if start is None:
            return None

        # Goal: free space, not too close to start
        goal = gen.sample_free_position(
            occ, rng, clearance_m=r_clear,
            exclude_positions=[start], exclude_radius=0.8)
        if goal is None:
            goal = start  # stationary human

        d = np.linalg.norm(goal - start)
        if d > 0.1:
            speed = float(rng.uniform(0.3, human.v_pref))
            direction = (goal - start) / d
            vx, vy = direction * speed
        else:
            vx, vy = 0.0, 0.0

        theta = float(np.arctan2(vy, vx)) if (abs(vx) + abs(vy)) > 0.01 else 0.0
        human.set(float(start[0]), float(start[1]),
                  float(goal[0]),  float(goal[1]),
                  float(vx), float(vy), theta)
        return human

    def _get_wall_obs_for_human(self, human, max_cells=8, radius=3.0):
        """
        Return nearby wall cells as ObservableState objects for ORCA.
        The fake agents have zero velocity so ORCA treats them as static
        obstacles and steers around them naturally.
        """
        if (self._npz_occ_world_xy is None
                or self._npz_has_map < 0.5
                or self._npz_occ_world_xy.shape[0] == 0):
            return []
        pts  = self._npz_occ_world_xy
        dx   = pts[:, 0] - human.px
        dy   = pts[:, 1] - human.py
        d    = np.sqrt(dx ** 2 + dy ** 2)
        sel  = d < radius
        if not sel.any():
            return []
        d_sel   = d[sel]
        pts_sel = pts[sel]
        order   = np.argsort(d_sel)[:max_cells]
        # Effective radius: half-diagonal of a cell, so ORCA gives the cell
        # the right amount of "space"
        cell_r = getattr(self, '_npz_cell_res', 0.2) * 0.5 * (2 ** 0.5)
        return [ObservableState(float(pts_sel[i, 0]), float(pts_sel[i, 1]),
                                0.0, 0.0, cell_r)
                for i in order]

    # =================================================================

    def get_human_times(self):
        """
        Run the whole simulation to the end and compute the average time for human to reach goal.
        Once an agent reaches the goal, it stops moving and becomes an obstacle
        (doesn't need to take half responsibility to avoid collision).

        :return:
        """
        # centralized orca simulator for all humans
        if not self.robot.reached_destination():
            raise ValueError('Episode is not done yet')
        params = (10, 10, 5, 5)
        sim = rvo2.PyRVOSimulator(self.time_step, *params, 0.3, 1)
        sim.addAgent(self.robot.get_position(), *params, self.robot.radius, self.robot.v_pref,
                     self.robot.get_velocity())
        for human in self.humans:
            sim.addAgent(human.get_position(), *params, human.radius, human.v_pref, human.get_velocity())

        max_time = 1000
        while not all(self.human_times):
            for i, agent in enumerate([self.robot] + self.humans):
                vel_pref = np.array(agent.get_goal_position()) - np.array(agent.get_position())
                if norm(vel_pref) > 1:
                    vel_pref /= norm(vel_pref)
                sim.setAgentPrefVelocity(i, tuple(vel_pref))
            sim.doStep()
            self.global_time += self.time_step
            if self.global_time > max_time:
                logging.warning('Simulation cannot terminate!')
            for i, human in enumerate(self.humans):
                if self.human_times[i] == 0 and human.reached_destination():
                    self.human_times[i] = self.global_time

            # for visualization
            self.robot.set_position(sim.getAgentPosition(0))
            for i, human in enumerate(self.humans):
                human.set_position(sim.getAgentPosition(i + 1))
            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])

        del sim
        return self.human_times

    def reset(self, **kwargs):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """
        self.predicted_trajs = []
        self.minobsdist = float('inf')
        self.pathlength = 0
        self.avgobsdist = []

        # Determine the phase and the active simulation rule for this episode
        # *first*, since downstream blocks (unicycle reset, scenario selection)
        # depend on them. test uses test_sim; train and val use train_val_sim.
        phase = kwargs.get('phase', 'test')
        test_case = kwargs.get('test_case', None)
        if self.robot is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']
        if test_case is not None:
            self.case_counter[phase] = test_case

        active_sim = self.test_sim if phase == 'test' else self.train_val_sim

        # Build the rng now so the npz pool draw and downstream sampling
        # share a single deterministic source for this episode.
        counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                              'val': 0, 'test': self.case_capacity['val']}
        if phase == 'test' and self.test_sim == "evaluation":
            seed = counter_offset[phase] + self.testoffset
        else:
            seed = counter_offset[phase] + self.case_counter[phase] + self.testoffset
        rng = np.random.RandomState(seed)
        self.testoffset += 1

        # If this episode runs npz_hard and we're in train/val, pull the next
        # scenario from the pool. In phase='test' we defer to whatever the
        # caller pre-loaded via load_npz_scenario() — that's how test.py
        # already drives this env, and we don't want to silently override it.
        if active_sim == 'npz_hard' and phase != 'test':
            self._draw_from_npz_pool(rng)

        # Reset unicycle dynamics state.
        # Use _reset_unicycle_state() when available (CADRL / MultiHumanRL
        # subclasses) so the method is the single source of truth.
        # For npz_hard the robot may start with a non-zero velocity v0 AND
        # a non-zero angular velocity w0; pass both through so the first
        # feasible-action set and the first (v, omega) feature both match.
        _v0    = 0.0
        _omega0 = 0.0
        if active_sim == 'npz_hard':
            if hasattr(self, '_npz_v0'):
                _v0 = float(self._npz_v0)
            if hasattr(self, '_npz_w0'):
                _omega0 = float(self._npz_w0)

        if hasattr(self.robot.policy, '_reset_unicycle_state'):
            self.robot.policy._reset_unicycle_state(_v0, _omega0)
        else:
            # Fallback for policies that still use the old attribute names.
            if hasattr(self.robot.policy, 'prev_v'):
                self.robot.policy.prev_v = _v0
            if hasattr(self.robot.policy, 'prev_omega'):
                self.robot.policy.prev_omega = _omega0

        if hasattr(self.robot.policy, 'prev_theta'):
            del self.robot.policy.prev_theta
        if hasattr(self.robot.policy, '_v0_from_last_step'):
            self.robot.policy._v0_from_last_step = _v0
        if hasattr(self.robot.policy, '_w0_from_last_step'):
            self.robot.policy._w0_from_last_step = _omega0

        # Reset the robot's own omega attribute too — Agent.set() called below
        # resets px, py, gx, gy, vx, vy, theta, but NOT omega. Without this,
        # robot.omega carries over the LAST omega of the previous episode, and
        # the (v, omega) feature in transform() on the first observation of
        # each new episode is wrong (it reads state.self_state.omega which
        # comes from robot.omega).
        self.robot.omega = _omega0

        # active_sim was computed above before the npz pool draw and unicycle reset.

        if active_sim == "evaluation":
            # up to 10 humans
            human_num = rng.randint(1, 11)
            self.human_num = human_num

        self.global_time = 0
        if phase == 'test' or active_sim in ('evaluation', 'npz_hard', 'map_eval'):
            # These modes have variable human counts; size human_times to match.
            self.human_times = [0] * (self.human_num if active_sim != 'npz_hard'
                                      else len(self._npz_humans_data))
        else:
            self.human_times = [0] * (self.human_num if self.robot.policy.multiagent_training else 1)
        # The original fallback below force-overrides train_val_sim to
        # circle_crossing for single-agent-trained policies (e.g. CADRL).
        # That would silently undo an explicit evaluation/npz_hard opt-in,
        # so we skip the override when an explicit mode is selected.
        if (not self.robot.policy.multiagent_training
                and active_sim not in ('evaluation', 'npz_hard', 'map_eval')):
            self.train_val_sim = 'circle_crossing'

        # Clear static map so stale state never bleeds across episodes.
        # map_eval and npz_hard re-populate these below.
        if active_sim not in ('npz_hard', 'map_eval'):
            self._npz_occ_world_xy = None
            self._npz_has_map      = 0.0

        if self.config.get('humans', 'policy') == 'trajnet':
            raise NotImplementedError
        else:
            
            self.robot.set(0, -self.circle_radius, 0, self.circle_radius, 0, 0, np.pi / 2)

            if active_sim == 'npz_hard':
                # ── NPZ hard scenario mode ──────────────────────────────
                # Scenario was either pre-loaded by an external caller via
                # load_npz_scenario() (test.py path) or auto-drawn from the
                # pool earlier in this reset() (training/val path).
                # Robot starts at origin with heading toward goal.
                goal = self._npz_goal
                v0   = self._npz_v0
                self.npz_num += 1

                goal_angle = np.arctan2(goal[1], goal[0])
                self.robot.set(0, 0, float(goal[0]), float(goal[1]), v0, 0, 0)
                self.generate_random_human_position(
                    len(self._npz_humans_data), 'npz_hard', rng
                )
                # human_times needs to match actual human count
                self.human_times = [0] * len(self.humans)
                self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]

                # NEW: hand the static map to the policy for this episode
                if hasattr(self.robot.policy, 'set_static_map'):
                    occ_world_xy = getattr(self, '_npz_occ_world_xy', None)
                    has_map      = getattr(self, '_npz_has_map', 0.0)
                    map_extent   = getattr(self, '_npz_map_extent', 10)
                    self.robot.policy.set_static_map(occ_world_xy, has_map, map_extent)

            elif active_sim == 'map_eval':
                # ── Map-eval: generate a structured layout with occupancy map ──
                scene = self._generate_map_eval_scene(rng)
                goal  = scene['goal']
                v0    = scene['v0']
                self.robot.set(0, 0, float(goal[0]), float(goal[1]), v0, 0, 0)

                # Populate map attributes (same attributes as npz_hard)
                self._npz_occ_map       = scene['occ_map']
                self._npz_occ_world_xy  = scene['occ_world_xy']
                self._npz_has_map       = 1.0
                self._npz_map_extent    = scene['map_extent']
                self._npz_cell_res      = scene['cell_res']
                self._npz_map_eval_type = scene['map_type']  # for evaluate

                # Generate humans in free space
                human_num = rng.randint(1, 11)
                self.generate_random_human_position(human_num, 'map_eval', rng)
                self.human_times = [0] * len(self.humans)
                self.case_counter[phase] = (
                    self.case_counter[phase] + 1) % self.case_size[phase]

                if hasattr(self.robot.policy, 'set_static_map'):
                    self.robot.policy.set_static_map(
                        scene['occ_world_xy'], 1.0, scene['map_extent'])

            elif active_sim == 'evaluation':
                px_r=rng.uniform(-self.square_width/2, self.square_width/2)
                py_r=rng.uniform(-self.square_width/2, self.square_width/2)
                while True:
                    gx_r=rng.uniform(-self.square_width/2, self.square_width/2)
                    gy_r=rng.uniform(-self.square_width/2, self.square_width/2)
                    if math.sqrt((gx_r - px_r)**2 + (gy_r - py_r)**2) > 5.0:
                        break
                self.robot.set(
                    px_r,
                    py_r,
                    gx_r,
                    gy_r,
                    0, 0, rng.uniform(-np.pi, np.pi)
                )
                self.generate_random_human_position(self.human_num, 'evaluation', rng)
                self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]

                if hasattr(self.robot.policy, 'set_static_map'):
                    self.robot.policy.set_static_map(None, 0.0, 0.0)

            else:
                self.robot.set(0, -self.circle_radius, 0, self.circle_radius, 0, 0, np.pi / 2)
                if hasattr(self.robot.policy, 'set_static_map'):
                    self.robot.policy.set_static_map(None, 0.0, 0.0)

                if self.case_counter[phase] >= 0:
                    if phase in ['train', 'val']:
                        human_num = self.human_num if self.robot.policy.multiagent_training else 1
                        self.generate_random_human_position(human_num, self.train_val_sim, rng)
                    else:
                        self.generate_random_human_position(self.human_num, self.test_sim, rng)
                    self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
                else:
                    assert phase == 'test'
                    if self.case_counter[phase] == -1:
                        self.human_num = 3
                        self.humans = [Human(self.config, 'humans') for _ in range(self.human_num)]
                        self.humans[0].set(0, -6, 0, 5, 0, 0, np.pi / 2)
                        self.humans[1].set(-5, -5, -5, 5, 0, 0, np.pi / 2)
                        self.humans[2].set(5, -5, 5, 5, 0, 0, np.pi / 2)
                    else:
                        raise NotImplementedError

        for agent in [self.robot] + self.humans:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        self.states = list()
        if hasattr(self.robot.policy, 'action_values'):
            self.action_values = list()
        if hasattr(self.robot.policy, 'get_attention_weights'):
            self.attention_weights = list()

        # get current observation
        if self.robot.sensor == 'coordinates':
            ob = [human.get_observable_state() for human in self.humans]
        elif self.robot.sensor == 'RGB':
            raise NotImplementedError

        return ob

    def onestep_lookahead(self, action):
        return self.step(action, update=False)

    def step(self, action, update=True):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)

        """
        # Build per-human wall-cell observations (used by ORCA to steer around
        # walls).  Only populated when a static map is active — no-op otherwise.
        has_wall_map = (self._npz_occ_world_xy is not None
                        and self._npz_has_map > 0.5
                        and self._npz_occ_world_xy.shape[0] > 0)

        human_actions = []
        for human in self.humans:
            # observation for humans is always coordinates
            ob = [other_human.get_observable_state()
                  for other_human in self.humans if other_human != human]
            if self.robot.visible:
                ob += [self.robot.get_observable_state()]
            wall_obs = self._get_wall_obs_for_human(human) if has_wall_map else []
            human_actions.append(human.act(ob, static_obs=wall_obs))

        # Fallback: if ORCA still returns an action that would put a human
        # inside a wall cell, stop that human.  This catches edge cases where
        # the wall cells are too far for the ORCA look-ahead.
        if has_wall_map:
            cell_res = getattr(self, '_npz_cell_res', 0.2)
            coll_thr = (self.humans[0].radius if self.humans else 0.25) + cell_res * 0.5
            pts = self._npz_occ_world_xy
            for i, (human, ha) in enumerate(zip(self.humans, human_actions)):
                nx, ny = human.compute_position(ha, self.time_step)
                d_cells = np.sqrt((pts[:, 0] - nx) ** 2 + (pts[:, 1] - ny) ** 2)
                if d_cells.min() < coll_thr:
                    human_actions[i] = ActionXY(0.0, 0.0)

        # collision detection
        dmin = float('inf')
        collision = False
        for i, human in enumerate(self.humans):
            px = human.px - self.robot.px
            py = human.py - self.robot.py
            if self.robot.kinematics == 'holonomic':
                vx = human.vx - action.vx
                vy = human.vy - action.vy
            else:
                vx = human.vx - action.v * np.cos(action.r + self.robot.theta)
                vy = human.vy - action.v * np.sin(action.r + self.robot.theta)
            ex = px + vx * self.time_step
            ey = py + vy * self.time_step
            # closest distance between boundaries of two agents
            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - self.robot.radius
            if closest_dist < 0:
                collision = True
                self.minobsdist=0
                dmin=0
                # logging.debug("Collision: distance between robot and p{} is {:.2E}".format(i, closest_dist))
                break
            elif closest_dist < dmin:
                dmin = closest_dist
            if closest_dist < self.minobsdist:
                self.minobsdist=closest_dist

        # End position needed for both static collision check and reaching_goal below.
        end_position = np.array(self.robot.compute_position(action, self.time_step))
        if update:
            self.pathlength += math.sqrt(
                (end_position[0] - self.robot.px) ** 2 +
                (end_position[1] - self.robot.py) ** 2
            )

        # Static map collision detection — check robot end position against
        # occupied cells from the loaded npz occupancy map. Static cells
        # have zero velocity so we only need to check the end position.
        static_collision = False
        min_static_clearance = float('inf')
        if (not collision
                and self._npz_occ_world_xy is not None
                and self._npz_has_map > 0.5
                and self._npz_occ_world_xy.shape[0] > 0):
            cell_res  = getattr(self, '_npz_cell_res', 0.2)
            coll_thr  = self.robot.radius + cell_res * 0.5
            ex_r = float(end_position[0])
            ey_r = float(end_position[1])
            pts  = self._npz_occ_world_xy
            d_cells = np.sqrt((pts[:, 0] - ex_r) ** 2 + (pts[:, 1] - ey_r) ** 2)
            min_cell_dist       = float(np.min(d_cells))
            min_static_clearance = min_cell_dist - coll_thr
            if min_static_clearance < 0:
                static_collision = True
                self.minobsdist  = 0
            elif min_static_clearance < self.minobsdist:
                self.minobsdist  = min_static_clearance

        effective_dmin = min(dmin, min_static_clearance) if min_static_clearance < float('inf') else dmin
        if update:
            self.avgobsdist.append(effective_dmin)

        # collision detection between humans
        human_num = len(self.humans)
        for i in range(human_num):
            for j in range(i + 1, human_num):
                dx = self.humans[i].px - self.humans[j].px
                dy = self.humans[i].py - self.humans[j].py
                dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.humans[i].radius - self.humans[j].radius
                if dist < 0:
                    # detect collision but don't take humans' collision into account
                    logging.debug('Collision happens between humans in step()')

        # check if reaching the goal (end_position already computed above)
        reaching_goal = norm(end_position - np.array(self.robot.get_goal_position())) < self.robot.radius

        if self.global_time >= self.time_limit - 1:
            reward = 0
            done = True
            info = Timeout(0)
        elif collision:
            reward = self.collision_penalty
            done = True
            info = Collision(self.collision_penalty)
        elif static_collision:
            reward = self.collision_penalty
            done = True
            info = WallCollision(self.collision_penalty)
        elif reaching_goal:
            reward = self.success_reward
            done = True
            info = ReachGoal(self.success_reward)
        elif dmin < self.discomfort_dist:
            # only penalize agent for getting too close if it's visible
            # adjust the reward based on FPS
            reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            done = False
            info = Danger(dmin)
        else:
            reward = 0
            done = False
            info = Nothing()

        if update:
            # store state, action value and attention weights
            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])
            if hasattr(self.robot.policy, 'action_values'):
                self.action_values.append(self.robot.policy.action_values)
            if hasattr(self.robot.policy, 'get_attention_weights'):
                self.attention_weights.append(self.robot.policy.get_attention_weights())
            if hasattr(self.robot.policy, 'predicted_traj'):
                self.predicted_trajs.append(self.robot.policy.predicted_traj)
            else:
                self.predicted_trajs.append(None)


            # update all agents
            self.robot.step(action)
            for i, human_action in enumerate(human_actions):
                self.humans[i].step(human_action)
            self.global_time += self.time_step
            for i, human in enumerate(self.humans):
                # only record the first time the human reaches the goal
                if self.human_times[i] == 0 and human.reached_destination():
                    self.human_times[i] = self.global_time

            # compute the observation
            if self.robot.sensor == 'coordinates':
                ob = [human.get_observable_state() for human in self.humans]
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError
        else:
            if self.robot.sensor == 'coordinates':
                ob = [human.get_next_observable_state(action) for human, action in zip(self.humans, human_actions)]
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError

        return ob, reward, done, info

    def render(self, mode='human', **kwargs):
        output_file = kwargs.get('output_file', None)
        if output_file:
            base_policy, ext = os.path.splitext(output_file)  # ext includes the dot, e.g. '.mp4'
            base_policy = kwargs.get('basepolicy', None)
        else:
            base_policy = None
        phase = kwargs.get('phase', 'test')
        test_case = kwargs.get('test_case', None)
        if phase == 'test' and self.test_sim == "evaluation":
            testnum = kwargs.get('testnum', None)
            if isinstance(testnum, int) and testnum >= 0:
                testnum += 1
        else:
            # testnum kwarg (0-indexed loop counter) takes priority over test_case
            testnum = kwargs.get('testnum', test_case)
            if isinstance(testnum, int) and testnum >= 0:
                testnum += 1   # display as 1-indexed
        from matplotlib import animation
        import matplotlib.pyplot as plt
        # Video writing needs an ffmpeg binary. Prefer whatever is on PATH;
        # set SOGUDIFF_FFMPEG to point at a specific build (useful on clusters
        # where ffmpeg is provided by a module rather than installed system
        # wide). Falls back to matplotlib's own default if neither is found.
        _ffmpeg = os.environ.get('SOGUDIFF_FFMPEG') or shutil.which('ffmpeg')
        if _ffmpeg:
            plt.rcParams['animation.ffmpeg_path'] = _ffmpeg

        x_offset = 0.11
        y_offset = 0.11
        cmap = plt.cm.get_cmap('hsv', 10)
        robot_color = 'yellow'
        goal_color = 'red'
        arrow_color = 'red'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        if mode == 'human':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            for human in self.humans:
                human_circle = plt.Circle(human.get_position(), human.radius, fill=False, color='b')
                ax.add_artist(human_circle)
            ax.add_artist(plt.Circle(self.robot.get_position(), self.robot.radius, fill=True, color='r'))
            plt.show()
        elif mode == 'traj':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            robot_positions = [self.states[i][0].position for i in range(len(self.states))]
            human_positions = [[self.states[i][1][j].position for j in range(len(self.humans))]
                               for i in range(len(self.states))]
            for k in range(len(self.states)):
                if k % 4 == 0 or k == len(self.states) - 1:
                    robot = plt.Circle(robot_positions[k], self.robot.radius, fill=True, color=robot_color)
                    humans = [plt.Circle(human_positions[k][i], self.humans[i].radius, fill=False, color=cmap(i))
                              for i in range(len(self.humans))]
                    ax.add_artist(robot)
                    for human in humans:
                        ax.add_artist(human)
                # add time annotation
                global_time = k * self.time_step
                if global_time % 4 == 0 or k == len(self.states) - 1:
                    agents = humans + [robot]
                    times = [plt.text(agents[i].center[0] - x_offset, agents[i].center[1] - y_offset,
                                      '{:.1f}'.format(global_time),
                                      color='black', fontsize=14) for i in range(self.human_num + 1)]
                    for time in times:
                        ax.add_artist(time)
                if k != 0:
                    nav_direction = plt.Line2D((self.states[k - 1][0].px, self.states[k][0].px),
                                               (self.states[k - 1][0].py, self.states[k][0].py),
                                               color=robot_color, ls='solid')
                    human_directions = [plt.Line2D((self.states[k - 1][1][i].px, self.states[k][1][i].px),
                                                   (self.states[k - 1][1][i].py, self.states[k][1][i].py),
                                                   color=cmap(i), ls='solid')
                                        for i in range(self.human_num)]
                    ax.add_artist(nav_direction)
                    for human_direction in human_directions:
                        ax.add_artist(human_direction)
            if self.predicted_trajs and self.predicted_trajs[-1] is not None:
                traj = self.predicted_trajs[-1]
                # Handle both the new dict shape (projection-aware policy)
                # and the old list/array shape. Prefer projection >
                draw = None
                if isinstance(traj, dict):
                    draw = traj.get('projection')
                    if draw is None:
                        draw = traj.get('selected_sample')
                else:
                    try:
                        draw = traj[0]
                    except Exception:
                        draw = None
                if draw is not None:
                    import numpy as _np
                    draw = _np.asarray(draw)
                    ax.plot(draw[:, 0], draw[:, 1],
                            linestyle='--', linewidth=2, color='orange',
                            alpha=0.8, label='Predicted traj')

            # plt.legend([robot], ['Robot'], fontsize=16)
            plt.legend(fontsize=16)
            plt.show()
        elif mode == 'video':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-7.5, 7.5)
            ax.set_ylim(-7.5, 7.5)

            # NEW: show static map if present
            occ = getattr(self, '_npz_occ_map', None)
            has = getattr(self, '_npz_has_map', 0.0)
            if occ is not None and has > 0.5:
                half = getattr(self, '_npz_map_extent', 10) / 2
                # Map is centered at world origin since robot starts at (0,0) with theta=0
                ax.imshow(occ,
                        extent=[-half, half, -half, half],
                        origin='lower', cmap='gray_r', alpha=0.35, zorder=0,
                        interpolation='nearest')

            # ax.axis('off')
            # fig.subplots_adjust(left=0, right=0, top=0, bottom=0)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)
            if self.test_sim == 'npz_hard':
                ax.set_title(f'{base_policy} - Test {self.npz_num}', fontsize=18, fontweight='bold')
            else:
                ax.set_title(f'{base_policy} - Test {testnum}', fontsize=18, fontweight='bold')

            # add robot and its goal
            robot_positions = [state[0].position for state in self.states]
            goal_pos = self.robot.get_goal_position()
            goal = mlines.Line2D([goal_pos[0]], [goal_pos[1]], color=goal_color, marker='*', linestyle='None', markersize=15, label='Goal')
            robot = plt.Circle(robot_positions[0], self.robot.radius, fill=True, color=robot_color)
            ax.add_artist(robot)
            ax.add_artist(goal)

             # Robot trailing trajectory (dashed line, grows each frame)
            robot_trail, = ax.plot([], [], '--', color=arrow_color, linewidth=1.5, alpha=0.7, zorder=1)
            num_samples = getattr(self.robot.policy, 'num_samples', 0) or 0
 
            # Background: K-1 non-selected diffusion samples (orange).
            pred_lines_bg = [
                ax.plot([], [], '-', color='orange', linewidth=2, alpha=0.8, zorder=1)[0]
                for _ in range(max(num_samples-1, 0))
            ]
 
            # Mid: selected diffusion sample (purple) — what the scoring
            # function picked PRE-projection.
            pred_line_fg, = ax.plot(
                [], [], '-', color='purple', linewidth=2, alpha=0.8, zorder=2,
                label='Selected Sample'
            )
 
            # Top: feasibility-projected trajectory (blue) — what the
            # robot actually executes when projection succeeds. Drawn last
            # so it sits on top of the purple selected sample. On brake
            # fallback frames this line is cleared, visually flagging the
            # event.
            pred_line_proj, = ax.plot(
                [], [], '-', color='blue', linewidth=2.5, alpha=0.9, zorder=3,
                label='Feasible Projection'
            )
 
            legend_handles = [
                robot,
                goal,
                pred_line_fg,
                pred_line_proj,
            ]
            legend_labels = [
                'Robot',
                'Goal',
                'Selected Sample',
                'Feasible Projection',
            ]
            ax.legend(legend_handles, legend_labels, fontsize=16)

            # add humans and their numbers
            human_positions = [[state[1][j].position for j in range(len(self.humans))] for state in self.states]
            humans = [plt.Circle(human_positions[0][i], self.humans[i].radius, fill=False)
                      for i in range(len(self.humans))]
            human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] - y_offset, str(i),
                                      color='black', fontsize=12) for i in range(len(self.humans))]
            for i, human in enumerate(humans):
                ax.add_artist(human)
                ax.add_artist(human_numbers[i])

            # add time annotation
            time = plt.text(-1, 5, 'Time: {}'.format(0), fontsize=16)
            ax.add_artist(time)

            # compute attention scores
            if self.attention_weights is not None:
                attention_scores = [
                    plt.text(-5.5, 5 - 0.5 * i, 'Human {}: {:.2f}'.format(i + 1, self.attention_weights[0][i]),
                             fontsize=16) for i in range(len(self.humans))]

            # compute orientation in each step and use arrow to show the direction
            radius = self.robot.radius
            if self.robot.kinematics == 'unicycle':
                orientation = [((state[0].px, state[0].py), (state[0].px + radius * np.cos(state[0].theta),
                                                             state[0].py + radius * np.sin(state[0].theta))) for state
                               in self.states]
                orientations = [orientation]
            else:
                orientations = []
                for i in range(self.human_num + 1):
                    orientation = []
                    for state in self.states:
                        if i == 0:
                            agent_state = state[0]
                        else:
                            agent_state = state[1][i - 1]
                        theta = np.arctan2(agent_state.vy, agent_state.vx)
                        orientation.append(((agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(theta),
                                             agent_state.py + radius * np.sin(theta))))
                    orientations.append(orientation)
            arrows = [patches.FancyArrowPatch(*orientation[0], color=arrow_color, arrowstyle=arrow_style)
                      for orientation in orientations]
            for arrow in arrows:
                ax.add_artist(arrow)
            global_step = 0

            def update(frame_num):
                nonlocal global_step
                nonlocal arrows
                global_step = frame_num
                robot.center = robot_positions[frame_num]
                # Grow robot trail
                trail_x = [robot_positions[k][0] for k in range(frame_num + 1)]
                trail_y = [robot_positions[k][1] for k in range(frame_num + 1)]
                robot_trail.set_data(trail_x, trail_y)

                for i, human in enumerate(humans):
                    human.center = human_positions[frame_num][i]
                    human_numbers[i].set_position((human.center[0] - x_offset, human.center[1] - y_offset))
                    for arrow in arrows:
                        arrow.remove()
                    arrows = [patches.FancyArrowPatch(*orientation[frame_num], color=arrow_color,
                                                      arrowstyle=arrow_style) for orientation in orientations]
                    for arrow in arrows:
                        ax.add_artist(arrow)
                    if self.attention_weights is not None:
                        human.set_color(str(self.attention_weights[frame_num][i]))
                        attention_scores[i].set_text('human {}: {:.2f}'.format(i, self.attention_weights[frame_num][i]))
                
                trajs = self.predicted_trajs[frame_num]
                
                # Default: clear everything. Re-populate based on what
                # `trajs` actually contains for this frame.
                for line in pred_lines_bg:
                    line.set_data([], [])
                pred_line_fg.set_data([], [])
                pred_line_proj.set_data([], [])
 
                if trajs is None:
                    pass  # nothing to draw this frame
 
                elif isinstance(trajs, dict):
                    # New dict shape (projection-aware policy):
                    #   'selected_sample': (T, 2) world-frame  — purple
                    #   'projection'     : (T, 2) world-frame  — blue
                    #                      (None on brake fallback)
                    #   'all_samples'    : list of K (T, 2) arrays — orange
                    #                      (index 0 = selected sample)
                    #   'used_projection': bool
                    selected    = trajs.get('selected_sample')
                    projection  = trajs.get('projection')
                    all_samples = trajs.get('all_samples') or []
 
                    # Background: K-1 non-selected samples (skip index 0,
                    # which IS the selected sample — already in purple).
                    num_bg = min(len(pred_lines_bg), max(len(all_samples) - 1, 0))
                    for i in range(num_bg):
                        s = all_samples[i + 1]
                        pred_lines_bg[i].set_data(s[:, 0], s[:, 1])
 
                    # Selected sample (purple) — shown ALWAYS when present,
                    # including on brake/fallback frames, so we can see
                    # what the policy proposed but couldn't execute.
                    if selected is not None:
                        pred_line_fg.set_data(selected[:, 0], selected[:, 1])
 
                    # Feasible projection (blue) — only on success frames.
                    # On brake fallback the blue line stays cleared,
                    # signaling the event visually.
                    if projection is not None:
                        pred_line_proj.set_data(projection[:, 0], projection[:, 1])
 
                else:
                    # Backward-compat: old list/array shape. Index 0 is
                    # the executed path (purple); 1.. are background.
                    try:
                        num_samples_local = getattr(
                            self.robot.policy, 'num_samples', 0) or 0
                        for i in range(max(num_samples_local - 1, 0)):
                            if i + 1 < len(trajs):
                                pred_lines_bg[i].set_data(
                                    trajs[i + 1][:, 0], trajs[i + 1][:, 1])
                        if len(trajs) > 0:
                            pred_line_fg.set_data(trajs[0][:, 0], trajs[0][:, 1])
                    except Exception:
                        pass  # malformed predicted_traj — skip frame


                time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))

            def plot_value_heatmap():
                assert self.robot.kinematics == 'holonomic'
                for agent in [self.states[global_step][0]] + self.states[global_step][1]:
                    print(('{:.4f}, ' * 6 + '{:.4f}').format(agent.px, agent.py, agent.gx, agent.gy,
                                                             agent.vx, agent.vy, agent.theta))
                # when any key is pressed draw the action value plot
                fig, axis = plt.subplots()
                speeds = [0] + self.robot.policy.speeds
                rotations = self.robot.policy.rotations + [np.pi * 2]
                r, th = np.meshgrid(speeds, rotations)
                z = np.array(self.action_values[global_step % len(self.states)][1:])
                z = (z - np.min(z)) / (np.max(z) - np.min(z))
                z = np.reshape(z, (16, 5))
                polar = plt.subplot(projection="polar")
                polar.tick_params(labelsize=16)
                mesh = plt.pcolormesh(th, r, z, vmin=0, vmax=1)
                plt.plot(rotations, r, color='k', ls='none')
                plt.grid()
                cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8])
                cbar = plt.colorbar(mesh, cax=cbaxes)
                cbar.ax.tick_params(labelsize=16)
                plt.show()

            def on_click(event):
                anim.running ^= True
                if anim.running:
                    anim.event_source.stop()
                    if hasattr(self.robot.policy, 'action_values'):
                        plot_value_heatmap()
                else:
                    anim.event_source.start()

            fig.canvas.mpl_connect('key_press_event', on_click)
            anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 1000)
            anim.running = True

            if output_file is not None:
                ffmpeg_writer = animation.writers['ffmpeg']
                writer = ffmpeg_writer(fps=8, metadata=dict(artist='Me'), bitrate=1800)
                anim.save(output_file, writer=writer)
            else:
                plt.show()
        else:
            raise NotImplementedError