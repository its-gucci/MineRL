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

import numpy as np

#######################################################################
# Code borrowed from 
# https://github.com/aravindr93/mjrl/blob/master/mjrl/utils/cg_solve.py
#######################################################################
def cg_solve(f_Ax, b, x_0=None, cg_iters=10, residual_tol=1e-10):
	"""
	conjugate gradient solver
	"""
    x = np.zeros_like(b) #if x_0 is None else x_0
    r = b.copy() #if x_0 is None else b-f_Ax(x_0)
    p = r.copy()
    rdotr = r.dot(r)

    for i in range(cg_iters):
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p

        rdotr = newrdotr
        if rdotr < residual_tol:
            break

    return x
#######################################################################
# End
#######################################################################