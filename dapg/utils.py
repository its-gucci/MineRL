import numpy as np

class MineRLEnvWrapper():
	def __init__(self, env, g, f):
		self.env = env
		# using g we can get action representations if necessary from just states
		self.g = g
		# using f we can get actions that we can perform in the MineRL environment
		self.f = f

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

def get_minerl_action(action_rep, f, env, eps=0.05):
	act = f(action_rep).numpy()
	action = env.action_space.noop()

	# attack is always one
	action['attack'] = 1

	# update camera action
	action['camera'] = act[:2]

	# epsilon-greedy action update
	if np.random() > eps:
		i = np.argmax(act[2:])
	else:
		i = np.random.randint(0, 16)
	action_string = '{0:04b}'.format(i)
	action['forward'] = int(action_string[0])
	action['jump'] = int(action_string[1])
	action['left'] = int(action_string[2])
	action['right'] = int(action_string[3])
	return action, i