import numpy as np

class MineRLEnvWrapper():
	def __init__(self, env, g):
		self.env = env
		# using g we can create action representations for pretraining
		self.g = g

def split_action(action):
	"""
	input action is an action from MineRL environment
	"""
	camera = np.zeros(2)
	for act in ['camera', 'forward', 'jump', 'left', 'back']:
		tmp = ""
		if act == 'camera':
			camera[0] = action[act][0]
			camera[1] = action[act][1]
		else:
			tmp += str(action[act])
	discrete = int(tmp, 2)
	return camera, discrete

def convert_action(action):
	"""
	input action is an action from MineRL environment
	"""
	new = np.zeros(18)
	for act in ['camera', 'forward', 'jump', 'left', 'back']:
		tmp = ""
		if act == 'camera':
			new[0] = action[act][0]
			new[1] = action[act][1]
		else:
			tmp += str(action[act])
	new[int(tmp, 2)] = 1
	return new

