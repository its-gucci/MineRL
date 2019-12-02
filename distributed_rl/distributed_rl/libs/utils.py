import sys
import random
from collections import namedtuple
import numpy as np
from PIL import Image
if sys.version_info.major == 3:
    import _pickle as cPickle
else:
    import cPickle
import torch
import lz4.frame
_USE_COMPRESS = True

class Transition(namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))):
    __slots__ = ()
    def __new__(cls, state, action, reward, next_state=None, done=None):
        return super(Transition, cls).__new__(cls, state, action, reward, next_state, done)

Sequence = namedtuple('Sequence', ('transitions', 'recurrent_state'))

_outsize = lambda x, f, p, s: int(x - f + 2 * p) / s + 1

def outsize(x, f, p=0, s=1):
    return (_outsize(x[0], f, p, s), _outsize(x[1], f, p, s))

def preprocess(img, shape=None, gray=False):
    pil_img = Image.fromarray(img)
    if not shape is None:
        pil_img = pil_img.resize(shape)
    if gray:
        img_ary = np.asarray(pil_img.convert("L"))
    else:
        img_ary = np.asarray(pil_img).transpose((2, 0, 1))
    return np.ascontiguousarray(img_ary, dtype=np.float32) / 255

def epsilon_greedy(state, policy_net, eps=0.1):
    if random.random() > eps:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1).cpu()
    else:
        return torch.tensor([random.randrange(policy_net.n_action)], dtype=torch.long)

################################################################################
# Handling states, action spaces for MineRL
#
# Here, we need to some special action-space mapping to both reduce the size of
# the action space and handle the continuous actions (i.e., camera movement). 
#
################################################################################
def minerl_action_to_enc(action):
    """
    Takes an action output by the MineRL Treechop-v0 environment and outputs an
    encoding of the action. 
    """
    enc = [action["attack"]]
    if action["camera"][0] != 0:
        enc.append(0.1)
    else:
        enc.append(0)
    if action["camera"][1] != 0:
        enc.append(0.1)
    else:
        enc.append(0)
    enc = enc + [action["forward"], action["jump"], action["left"], action["right"]] 
    return enc

def minerl_demo_action_to_enc(action):
    """
    Takes an action output by the MineRL Treechop-v0 environment and outputs an
    encoding of the action. 
    """
    action_map = {
        0: "attack",
        1: "camera",
        2: "camera",
        3: "forward",
        4: "jump",
        5: "left",
        6: "right"
    }
    nonzero_actions = []
    for i in action_map:
        if i not in [1, 2]:
            if action[action_map[i]] != 0:
                nonzero_actions.append(i)
        else:
            if action["camera"][0] != 0:
                nonzero_actions.append(1)
            if action["camera"][1] != 0:
                nonzero_actions.append(2)
    if nonzero_actions == []:
        return [random.randint(0, 6)]
    act = random.choice(nonzero_actions)
    return [act]

def minerl_enc_to_action(i, env):
    """
    Takes an encoding of an action (say, produced by our policy net) and turns
    that into an action that can be fed back in to the MineRL environment.

    Right now the action space mapping is probably too simple:

    We will have run, attack always on. Only one action will be taken per step. 
    Camera will be discretized to 10 degrees. The back, right, sneak, and sprint
    actions will all be disabled.

    Actions will be in the following order:
    [attack, camera1, camera2, forward, jump, left, right]

    Reverse mapping of minerl_action_to_enc
    """
    action = env.action_space.noop()
    # disable these "always on" actions so we match the demonstrations
    # action["attack"] = 1
    # action["forward"] = 1
    action_map = {
        0: "attack",
        1: "camera",
        2: "camera",
        3: "forward",
        4: "jump",
        5: "left",
        6: "right"
    }
    int_i = int(i.item())
    if action_map[int_i] != "camera":
        action[action_map[int_i]] = 1
    else:
        if int_i == 1:
            action["camera"] = [0.1, 0]
        elif int_i == 2:
            action["camera"] = [0, 0.1]
    return action

################################################################################
# Above is the old hand-crafted action mapping 
# Below are utils for action representations 
# reference: https://arxiv.org/pdf/1902.00183.pdf
################################################################################

def minerl_embed_to_action(f_output, env):
    """
    Takes the output of our action mapping f:E -> A and outputs an action that
    can be carried out in the MineRL environment
    """
    action = env.action_space.noop()
    action_map = {
        0: "attack",
        1: "camera",
        2: "camera",
        3: "forward",
        4: "jump",
        5: "left",
        6: "right"
    }
    for i in range(7):
        if i == 1:
            action["camera"] = [f_output[0][1].item(), f_output[0][2].item()]
        elif i == 2:
            continue
        elif f_output[0][i].item() > 0:
            action[action_map[i]] = 1
    return action

################################################################################
# End of code modification
################################################################################

def dumps(data):
    if _USE_COMPRESS:
        return lz4.frame.compress(cPickle.dumps(data))
    else:
        return cPickle.dumps(data)

def loads(packed):
    if _USE_COMPRESS:
        return cPickle.loads(lz4.frame.decompress(packed))
    else:
        return cPickle.loads(packed)

def rescale(x, eps=1.0e-3):
    return x.sign() * ((x.abs() + 1.0).sqrt() - 1.0) + eps * x

def inv_rescale(x, eps=1.0e-3):
    if eps == 0:
        return x.sign() * (x * x + 2.0 * x.abs())
    else:
        return x.sign() * ((((1.0 + 4.0 * eps * (x.abs() + 1.0 + eps)).sqrt() - 1.0) / (2.0 * eps)).pow(2) - 1.0)
