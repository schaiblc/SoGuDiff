class FullState(object):
    def __init__(self, px, py, vx, vy, radius, gx, gy, v_pref, theta, omega=None):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.gx = gx
        self.gy = gy
        self.v_pref = v_pref
        self.theta = theta
        self.omega = omega

        self.position = (self.px, self.py)
        self.goal_position = (self.gx, self.gy)
        self.velocity = (self.vx, self.vy)

    def __add__(self, other):
        # NOTE: omega is intentionally NOT included in the addition tuple.
        # rotate() in cadrl.py assumes a fixed 9-field robot block followed by
        # the 5-field human block (so px1=index 9, py1=10, ..., r1=13). If we
        # inserted omega at index 9, every human-state index would shift by one
        # the moment Agent.step() runs (which is the moment self.omega becomes
        # non-None). That created two bugs at once: (a) the human-state features
        # the value network sees were garbage from step 2 onward, and (b) the
        # tensor length flipped between 14, 15, 16 depending on whether omega
        # had been set, causing inconsistent layouts between IL and RL and
        # between predict-lookahead (FullState built by propagate(), omega=None)
        # and transform (FullState from get_full_state(), omega set).
        # Code paths that need (v, omega) features should append them
        # explicitly after the join — see cadrl.transform / multi_human_rl.transform.
        return other + (self.px, self.py, self.vx, self.vy, self.radius,
                        self.gx, self.gy, self.v_pref, self.theta)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy,
                                          self.v_pref, self.theta, self.omega]])

# Overwriting to account for overwritten FullState
class JointState(object):
    def __init__(self, self_state, human_states, static_obs=[]):
        assert isinstance(self_state, FullState)
        for human_state in human_states:
            assert isinstance(human_state, ObservableState)

        self.self_state = self_state
        self.human_states = human_states
        self.static_obs = static_obs


class FullyObservableJointState(object):
    def __init__(self, self_state, human_states, static_obs=[]):
        assert isinstance(self_state, FullState)
        for human_state in human_states:
            assert isinstance(human_state, FullState)

        self.self_state = self_state
        self.human_states = human_states
        self.static_obs = static_obs


class ObservableState(object):
    def __init__(self, px, py, vx, vy, radius):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius

        self.position = (self.px, self.py)
        self.velocity = (self.vx, self.vy)

    def __add__(self, other):
        return other + (self.px, self.py, self.vx, self.vy, self.radius)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius]])