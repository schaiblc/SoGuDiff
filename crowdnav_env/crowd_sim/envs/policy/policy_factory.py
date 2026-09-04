from crowd_sim.envs.policy.linear import Linear
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.policy.orca_unicycle import ORCAUnicycle
from crowd_sim.envs.policy.SB3_policy import SB3
from crowd_sim.envs.policy.social_force import SFM
from crowd_sim.envs.policy.sfm_unicycle import SFMUnicycle

def none_policy():
    return None

policy_factory = dict()
policy_factory['none'] = none_policy
policy_factory['linear'] = Linear
policy_factory['orca'] = ORCA
policy_factory['orca_unicycle'] = ORCAUnicycle
policy_factory['SB3'] = SB3
policy_factory['sfm'] = SFM
policy_factory['sfm_unicycle'] = SFMUnicycle