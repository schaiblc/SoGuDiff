
class BaseConfig(object):
    def __init__(self):
        pass


class Config(object):
    # environment settings
    env = BaseConfig()
    env.env_name = 'CrowdSimDict-v0'  # name of the environment
    env.time_limit = 25 # 25 to MATCH crowdnav_env's eval time_limit.
                        # Was 50: the robot learned slow/indirect paths that
                        # reach goals within 50s but time out under the eval's
                        # 25s cap (observed: ~75% timeout / ~3% success at eval
                        # despite a healthy +38 training reward). The other
                        # baselines (CADRL/SARL/RGL) are trained+tested at 25s.
    env.time_step = 0.25 # length of each timestep/control frequency (second)
    env.val_size = 100
    env.test_size = 500 # number of episodes for test.py
    # False so humans.radius/v_pref below actually hold at the common
    # 0.25/1 values (sample_random_attributes() would otherwise overwrite
    # them with radius~U(0.3,0.5), v_pref~U(0.5,1.5) at every spawn,
    # silently breaking the common-radius requirement despite the fixed
    # config values below).
    env.randomize_attributes = False
    env.seed = 0  # random seed for environment

    # reward function
    reward = BaseConfig()
    reward.success_reward = 10
    reward.collision_penalty = -20
    # discomfort distance for the front half of the robot
    reward.discomfort_dist_front = 0.25
    # discomfort distance for the back half of the robot
    reward.discomfort_dist_back = 0.25
    reward.discomfort_penalty_factor = 10
    reward.gamma = 0.99  # discount factor for rewards

    # environment settings
    sim = BaseConfig()
    sim.render = False # show GUI for visualization
    sim.circle_radius = 6 # UNUSED by the 'evaluation' scenario (robot start/goal
                          # are random within the square box, not on a circle);
                          # kept only for the inactive circle_crossing path.
    sim.human_num = 5 # network input size (observed_human_num): the SRNN's
                      # spatial-edge graph is built for exactly this many slots.
    # Variable ACTUAL human count per episode for the 'evaluation' scenario,
    # matching crowdnav_env's eval (randint(1,11) -> 1..10). The env
    # feeds the network the closest sim.human_num of these + padding. Set both
    # equal to sim.human_num to recover the old fixed-count behaviour.
    sim.min_human_num = 1
    sim.max_human_num = 10
    # Group environment: set to true; FoV environment: false
    sim.group_human = False
    # "circle_crossing" (DSRNN's own default scenario, unchanged) or
    # "evaluation" (matches crowdnav_env's env.config evaluation
    # scenario: robot start/goal random inside a square, goal >5m from
    # start, humans randomly placed in the same square) — for training this
    # baseline on the same scenario distribution as the other baselines.
    # human_num stays fixed (see sim.human_num above) rather than the other
    # baselines' per-episode 1-10 draw: DSRNN's SRNN network has a
    # fixed-size observation/action graph (spatial_edges is (human_num, 2),
    # built once at network-construction time), unlike the padding-aware
    # MultiHumanRL family, so it can't take a different human count per
    # episode without a separate architecture change.
    sim.scenario = "evaluation"
    sim.square_width = 15  # matches crowdnav_env's env.config sim.square_width

    # human settings
    humans = BaseConfig()
    humans.visible = True # a human is visible to other humans and the robot
    # policy to control the humans: orca or social_force
    humans.policy = "orca"
    humans.radius = 0.25 # radius of each human — held common, see robot.radius above
    humans.v_pref = 1 # max velocity of each human
    # FOV = this values * PI
    humans.FOV = 2.

    # a human may change its goal before it reaches its old goal
    # This fork: False (author default True/0.25). crowdnav_env's eval
    # never re-draws a human goal mid-episode, so training against re-randomized
    # goals taught anticipatory behavior the eval never exercises. Aligning the
    # TASK is part of "train on our evaluation setup"; this is not an
    # algorithmic change.
    humans.random_goal_changing = False
    humans.goal_change_chance = 0.25

    # a human may change its goal after it reaches its old goal
    # This fork: False for the same reason (eval humans keep one fixed goal).
    humans.end_goal_changing = False
    humans.end_goal_change_chance = 1.0

    # a human may change its radius and/or v_pref after it reaches its current goal
    humans.random_radii = False
    humans.random_v_pref = False

    # one human may have a random chance to be blind to other agents at every time step
    humans.random_unobservability = False
    humans.unobservable_chance = 0.3

    humans.random_policy_changing = False

    # robot settings
    robot = BaseConfig()
    robot.visible = False  # the robot is visible to humans
    # robot policy: srnn for now
    robot.policy = 'srnn'
    # Held common across all baselines being compared (CADRL/SARL/LSTM-RL/RGL
    # in crowdnav_env) rather than DSRNN's own default of 0.3 — this
    # is a physical robot/human size, not part of the DSRNN algorithm itself.
    robot.radius = 0.25  # radius of the robot
    robot.v_pref = 1  # max velocity of the robot
    # robot FOV = this values * PI
    robot.FOV = 2.

    # add noise to observation or not
    noise = BaseConfig()
    noise.add_noise = False
    # uniform, gaussian
    noise.type = "uniform"
    noise.magnitude = 0.1

    # robot action type
    action_space = BaseConfig()
    # holonomic or unicycle
    # This fork — MINIMAL PORT: the author's ORIGINAL dynamics.
    # Every prior unicycle conversion of this baseline failed at eval
    # (5-44% success across 4 attempts) and each failure traced to a change
    # made to DSRNN's own action semantics (omega accumulator, clip-range
    # curriculum, post-hoc clamps). So this attempt keeps the author's
    # holonomic ActionXY policy + their clip-to-v_pref exactly as published,
    # and changes ONLY the task around it: our 'evaluation' scenario, our
    # radii/v_pref/time_limit, no mid-episode goal changing. The kinematic
    # difference vs the other baselines is documented as an eval caveat
    # rather than "fixed" by re-engineering the policy's dynamics.
    # (The limit fields below stay only because crowd_sim_dict/srnn.py read
    # them on the unicycle code path; they are inert when kinematics is
    # holonomic.)
    action_space.kinematics = "holonomic"
    # Kinodynamic limits — held common with crowdnav_env's
    # policy.config [action_space] section (enforce_lims=true, max_vel=1,
    # max_wrot=3.14, max_accel=1.5, max_w_accel=3.14). Wired into
    # srnn.py's clip_action() and crowd_sim_dict.py's step().
    # INERT for the holonomic attempt-B run.
    action_space.max_vel = 1.0
    action_space.max_wrot = 3.14
    action_space.max_accel = 1.5
    action_space.max_w_accel = 3.14

    # config for ORCA
    orca = BaseConfig()
    orca.neighbor_dist = 10
    orca.safety_space = 0.15
    orca.time_horizon = 5
    orca.time_horizon_obst = 5

    # config for social force
    sf = BaseConfig()
    sf.A = 2.
    sf.B = 1
    sf.KI = 1

    # cofig for RL ppo
    ppo = BaseConfig()
    ppo.num_mini_batch = 2  # number of batches for ppo
    ppo.num_steps = 30  # number of forward steps
    ppo.recurrent_policy = True  # use a recurrent policy
    ppo.epoch = 5  # number of ppo epochs
    ppo.clip_param = 0.2  # ppo clip parameter
    ppo.value_loss_coef = 0.5  # value loss coefficient
    # This fork: expression self-resolves to the author's published holonomic
    # value of 0.0 now that kinematics = "holonomic". The history below is from
    # the abandoned unicycle conversions and no longer applies to this run:
    # those raised entropy to 0.01/0.001 only as band-aids for instability that
    # came from re-engineering the action dynamics (omega accumulator + clip
    # curriculum), which this attempt removes entirely.
    ppo.entropy_coef = 0.01 if action_space.kinematics == 'unicycle' else 0.0  # entropy term coefficient
    ppo.use_gae = True  # use generalized advantage estimation
    ppo.gae_lambda = 0.95  # gae lambda parameter

    # SRNN config
    SRNN = BaseConfig()
    # RNN size
    SRNN.human_node_rnn_size = 128  # Size of Human Node RNN hidden state
    SRNN.human_human_edge_rnn_size = 256  # Size of Human Human Edge RNN hidden state

    # Input and output size
    SRNN.human_node_input_size = 3  # Dimension of the node features
    SRNN.human_human_edge_input_size = 2  # Dimension of the edge features
    SRNN.human_node_output_size = 256  # Dimension of the node output

    # Embedding size
    SRNN.human_node_embedding_size = 64  # Embedding size of node features
    SRNN.human_human_edge_embedding_size = 64  # Embedding size of edge features

    # Attention vector dimension
    SRNN.attention_size = 64  # Attention size

    # training config
    training = BaseConfig()
    training.lr = 4e-5  # learning rate (default: 7e-4)
    training.eps = 1e-5  # RMSprop optimizer epsilon
    training.alpha = 0.99  # RMSprop optimizer alpha
    training.max_grad_norm = 0.5  # max norm of gradients
    # This fork: author's published holonomic training budget (upstream config:
    # "10e6 for holonomic, 20e6 for unicycle"). The 30e6 below was the
    # unicycle-conversion budget; with the author's own dynamics restored, their
    # own budget applies. best.pt (train.py) captures the reward peak either way.
    training.num_env_steps = 10e6
    training.use_linear_lr_decay = False  # author's published holonomic setting (True was a unicycle-conversion choice)
    training.save_interval = 200  # save interval, one save per n updates
    training.log_interval = 20  # log interval, one log per n updates
    # This fork keeps True — the ONE deliberate deviation from upstream PPO
    # settings, and it's forced by a task change, not an algorithm tweak:
    # upstream's False was tuned at their time_limit=50 where truncations are
    # rare; at our eval-matched time_limit=25 truncated episodes are common
    # early in training, and treating them as V=0 terminals corrupts value
    # targets (v2 evidence: stuck value_loss / flat reward until fixed).
    # See crowd_sim_dict.py step()'s bad_transition flag + storage.compute_returns().
    training.use_proper_time_limits = True  # compute returns taking into account time limits
    training.cuda_deterministic = False  # sets flags for determinism when using CUDA (potentially slow!)
    training.cuda = True  # use CUDA for training
    training.num_processes = 12 # how many training CPU processes to use
    training.output_dir = 'data/trained'  # the saving directory for train.py
    # Warm-start the SRNN encoder + critic from the original author's working
    # unicycle checkpoint (it learned to eprewmean ~+23 under the *easier*
    # single-integrator heading), then let PPO adapt to our accel-limited
    # (double-integrator) dynamics. From scratch the double-integrator has a
    # very weak early learning signal — random angular-accel commands make the
    # heading random-walk, so there is little net goal progress to reward — and
    # training stalls near zero; starting from a competent navigator skips that
    # dead zone. The action head is reinitialized (reinit_action_head) because
    # its output semantics changed (raw delta-theta -> pre-tanh delta-omega),
    # so those weights are not transferable — see train.py's warm-start block.
    # resume=False: FROM SCRATCH on the fair task. The warm-start source
    # (example_model_unicycle) was trained on the OLD fixed-goal task (always
    # (0,-4)->(0,+4), i.e. "drive up") and is a HARMFUL init for random goals —
    # it produced a 12.8%-success / 60.8%-timeout model that could not navigate
    # to arbitrary goals. From scratch, the network learns general goal-reaching
    # from the start. load_path/reinit_action_head are ignored when resume=False.
    training.resume = False  # resume training from an existing checkpoint or not
    training.load_path = 'data/example_model_unicycle/checkpoints/55554.pt'  # (unused when resume=False)
    training.reinit_action_head = True  # (unused when resume=False)
    training.overwrite = True  # whether to overwrite the output directory in training
    training.num_threads = 1  # number of threads used for intraop parallelism on CPU