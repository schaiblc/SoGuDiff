
from crowd_nav.policy.policy import Policy
import numpy as np
from crowd_sim.envs.utils.action import ActionRot, ActionXY


class SRNN(Policy):
	def __init__(self, config):
		super().__init__(config)
		self.time_step = self.config.env.time_step # Todo: is this needed?
		self.name = 'srnn'
		self.trainable = True
		self.multiagent_training = True


	# clip the self.raw_action and return the clipped action
	def clip_action(self, raw_action, v_pref):
		"""
        Input state is the joint state of robot concatenated by the observable state of other agents

        To predict the best action, agent samples actions and propagates one step to see how good the next state is
        thus the reward function is needed

        """
		# quantize the action
		holonomic = True if self.config.action_space.kinematics == 'holonomic' else False
		# clip the action
		if holonomic:
			act_norm = np.linalg.norm(raw_action)
			if act_norm > v_pref:
				raw_action[0] = raw_action[0] / act_norm * v_pref
				raw_action[1] = raw_action[1] / act_norm * v_pref
			return ActionXY(raw_action[0], raw_action[1])
		else:
			# Kinodynamic rate limits (max_accel, max_w_accel), matching the
			# unicycle-limited baselines in crowdnav_env. Replaces
			# the original hardcoded sim2real bounds (+-0.1) with bounds
			# derived from the configured physical limits: action[0] is the
			# per-step change of v, bounded by max_accel*dt; action[1] is the
			# per-step change of omega (not raw theta — see
			# crowd_sim_dict.py's step(), which accumulates this into a
			# persistent desired omega exactly like it already does for v),
			# bounded by max_w_accel*dt.
			max_dv = self.config.action_space.max_accel * self.time_step
			max_domega = self.config.action_space.max_w_accel * self.time_step
			raw_action[0] = np.clip(raw_action[0], -max_dv, max_dv)          # action[0] is change of v
			raw_action[1] = np.clip(raw_action[1], -max_domega, max_domega)  # action[1] is change of omega

			return ActionRot(raw_action[0], raw_action[1])


