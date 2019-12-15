import numpy as np

def split_action(action):
    """
    input action is an action from MineRL environment
    """
    camera = action['camera']
    discrete = []
    for i in range(len(camera)):
      s = str(action['forward'][i]) + str(action['jump'][i]) + str(action['left'][i]) + str(action['right'][i])
      onehot = np.zeros(16)
      onehot[int(s, 2)] = 1
      discrete.append(onehot)
    return camera, np.array(discrete)

def convert_action(action):
    """
    input action is an action from MineRL environment
    """
    new = np.zeros(18)
    for act in ['camera', 'forward', 'jump', 'left', 'right']:
        tmp = ""
        if act == 'camera':
            new[0] = action[act][0]
            new[1] = action[act][1]
        else:
            tmp += str(action[act])
    new[int(tmp, 2)] = 1
    return new

def get_minerl_action(action_rep, f, env, eps=0.05):
    act = f(action_rep).detach().numpy()[0]
    action = env.action_space.noop()

    # attack is always one
    action['attack'] = 1

    # update camera action
    action['camera'] = act[:2]

    # epsilon-greedy action update
    if np.random.rand() > eps:
        i = np.argmax(act[2:])
    else:
        i = np.random.randint(0, 16)
    action_string = '{0:04b}'.format(i)
    action['forward'] = int(action_string[0])
    action['jump'] = int(action_string[1])
    action['left'] = int(action_string[2])
    action['right'] = int(action_string[3])
    return action, i