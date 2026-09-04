import numpy as np
import rvo2
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY


class ORCA(Policy):
    def __init__(self):
        """
        timeStep        The time step of the simulation.
                        Must be positive.
        neighborDist    The default maximum distance (center point
                        to center point) to other agents a new agent
                        takes into account in the navigation. The
                        larger this number, the longer the running
                        time of the simulation. If the number is too
                        low, the simulation will not be safe. Must be
                        non-negative.
        maxNeighbors    The default maximum number of other agents a
                        new agent takes into account in the
                        navigation. The larger this number, the
                        longer the running time of the simulation.
                        If the number is too low, the simulation
                        will not be safe.
        timeHorizon     The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        other agents. The larger this number, the
                        sooner an agent will respond to the presence
                        of other agents, but the less freedom the
                        agent has in choosing its velocities.
                        Must be positive.
        timeHorizonObst The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        obstacles. The larger this number, the
                        sooner an agent will respond to the presence
                        of obstacles, but the less freedom the agent
                        has in choosing its velocities.
                        Must be positive.
        radius          The default radius of a new agent.
                        Must be non-negative.
        maxSpeed        The default maximum speed of a new agent.
                        Must be non-negative.
        velocity        The default initial two-dimensional linear
                        velocity of a new agent (optional).

        ORCA first uses neighborDist and maxNeighbors to find neighbors that need to be taken into account.
        Here set them to be large enough so that all agents will be considered as neighbors.
        Time_horizon should be set that at least it's safe for one time step

        In this work, obstacles are not considered. So the value of time_horizon_obst doesn't matter.

        """
        super().__init__()
        self.name = 'ORCA'
        self.trainable = False
        self.multiagent_training = None
        self.kinematics = 'holonomic'
        self.safety_space = 0
        self.neighbor_dist = 10
        self.max_neighbors = 10
        self.time_horizon = 2.0
        self.time_horizon_obst = 0.50
        self.radius = 0.3
        self.max_speed = 1
        self.sim = None

    def configure(self, config):
        # self.time_step = config.getfloat('orca', 'time_step')
        # self.neighbor_dist = config.getfloat('orca', 'neighbor_dist')
        # self.max_neighbors = config.getint('orca', 'max_neighbors')
        # self.time_horizon = config.getfloat('orca', 'time_horizon')
        # self.time_horizon_obst = config.getfloat('orca', 'time_horizon_obst')
        # self.radius = config.getfloat('orca', 'radius')
        # self.max_speed = config.getfloat('orca', 'max_speed')
        return

    def set_phase(self, phase):
        return

    def predict(self, state):
        """
        Create a rvo2 simulation at each time step and run one step.

        static_obs support
        ------------------
        state.static_obs (list of ObservableState, default []) may contain
        wall-cell fake agents passed by the env.  Each is added to the RVO2
        sim as a zero-velocity, zero-preferred-velocity agent so ORCA treats
        them as permanent static obstacles and steers around them naturally.
        """
        self_state  = state.self_state
        static_obs  = getattr(state, 'static_obs', [])
        params      = (self.neighbor_dist, self.max_neighbors,
                       self.time_horizon, self.time_horizon_obst)

        n_expected = 1 + len(state.human_states) + len(static_obs)
        if self.sim is not None and self.sim.getNumAgents() != n_expected:
            del self.sim
            self.sim = None

        if self.sim is None:
            self.sim = rvo2.PyRVOSimulator(self.time_step, *params,
                                            self.radius, self.max_speed)
            # Index 0 — controlled agent (robot or this human)
            self.sim.addAgent(self_state.position, *params,
                              self_state.radius + 0.01 + self.safety_space,
                              self_state.v_pref, self_state.velocity)
            # Indices 1 .. n_humans — dynamic neighbors
            for hs in state.human_states:
                self.sim.addAgent(hs.position, *params,
                                  hs.radius + 0.01 + self.safety_space,
                                  self.max_speed, hs.velocity)
            # Indices n_humans+1 .. — static wall cells (zero max speed)
            for s in static_obs:
                self.sim.addAgent((s.px, s.py), *params,
                                  s.radius + 0.01, 0.0, (0.0, 0.0))
        else:
            self.sim.setAgentPosition(0, self_state.position)
            self.sim.setAgentVelocity(0, self_state.velocity)
            for i, hs in enumerate(state.human_states):
                self.sim.setAgentPosition(i + 1, hs.position)
                self.sim.setAgentVelocity(i + 1, hs.velocity)
            # Static obs don't move — no position/velocity update needed

        # Preferred velocity: unit speed toward goal
        velocity = np.array([self_state.gx - self_state.px,
                              self_state.gy - self_state.py])
        speed    = np.linalg.norm(velocity)
        pref_vel = velocity / speed if speed > 1 else velocity

        self.sim.setAgentPrefVelocity(0, tuple(pref_vel))
        for i in range(len(state.human_states)):
            self.sim.setAgentPrefVelocity(i + 1, (0, 0))
        # Static wall agents always have pref_vel (0,0) — already set by addAgent

        self.sim.doStep()
        action          = ActionXY(*self.sim.getAgentVelocity(0))
        self.last_state = state
        self.sim        = None   # reset for SB3 compatibility
        return action