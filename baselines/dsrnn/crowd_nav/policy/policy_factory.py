policy_factory = dict()
def none_policy():
    return None

from crowd_nav.policy.orca import ORCA
from crowd_nav.policy.social_force import SOCIAL_FORCE
from crowd_nav.policy.srnn import SRNN

policy_factory['orca'] = ORCA
policy_factory['none'] = none_policy
policy_factory['social_force'] = SOCIAL_FORCE
policy_factory['srnn'] = SRNN
# Robot.policy is just a thin per-agent wrapper (clip_action + kinematics
# bookkeeping) — it's independent of which network train_height.py actually
# uses (HeightPolicy / selfAttn_merge_SRNN, wired up separately). SRNN's
# clip_action() already does exactly the rate-limiting HEIGHT's baseline
# needs here, so it's reused rather than duplicated.
policy_factory['selfAttn_merge_srnn'] = SRNN

