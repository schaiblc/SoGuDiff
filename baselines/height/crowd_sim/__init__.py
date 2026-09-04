from gym.envs.registration import register

register(
    id='CrowdSim-v0',
    entry_point='crowd_sim.envs:CrowdSim',
)

register(
    id='CrowdSimVarNum-v0',
    entry_point='crowd_sim.envs:CrowdSimVarNum',
)

# Cluster compatibility: the TurtleBot envs import pybullet at
# module level, and pybullet is not always installable on a cluster;
# crowd_sim/envs/__init__.py
# guards their imports, so these registrations are guarded to match. They are
# unused — this repo's training/eval only instantiates CrowdSimVarNum-v0.
try:
    from crowd_sim.envs import CrowdSim3DTB  # noqa: F401
    register(
        id='CrowdSim3DTB-v0',
        entry_point='crowd_sim.envs:CrowdSim3DTB',
    )

    register(
        id='CrowdSim3DTbObs-v0',
        entry_point='crowd_sim.envs.crowd_sim_tb2_obs:CrowdSim3DTbObs',
    )

    register(
        id='CrowdSim3DTbObsHie-v0',
        entry_point='crowd_sim.envs:CrowdSim3DTbObsHie',
    )

    register(
        id='rosTurtlebot2iEnv-v0',
        entry_point='crowd_sim.envs.ros_turtlebot2i_env:rosTurtlebot2iEnv',
    )
except ImportError:  # pragma: no cover - pybullet unavailable on cluster
    pass
