from .crowd_sim import CrowdSim
from .crowd_sim_var_human import CrowdSimVarNum
# Cluster compatibility: the four TurtleBot envs below import
# pybullet at module level, which is not always installable on a cluster. They're
# unused here — training/eval only ever instantiate CrowdSimVarNum-v0 /
# CrowdSim-v0 — so their imports are guarded and the gym registrations in
# crowd_sim/__init__.py are guarded to match.
try:
    from .crowd_sim_tb2 import CrowdSim3DTB
    from .crowd_sim_tb2_obs import CrowdSim3DTbObs
    from .crowd_sim_tb2_sim2real import CrowdSim3DTB_Sim2real
    from .crowd_sim_tb2_obs_hierarchy import CrowdSim3DTbObsHie
except ImportError:  # pragma: no cover - pybullet unavailable on cluster
    CrowdSim3DTB = CrowdSim3DTbObs = CrowdSim3DTB_Sim2real = None
