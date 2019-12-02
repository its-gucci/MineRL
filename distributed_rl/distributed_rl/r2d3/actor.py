# -*- coding: utf-8 -*-
import numpy as np
from itertools import count
from collections import deque
import redis
import torch
from ..ape_x import actor
from ..libs import replay_memory, utils

class Actor(actor.Actor):
    EPS_BASE = 0.4
    EPS_ALPHA = 7.0
    def __init__(self, name, env, policy_net, target_net, g, f, vis, hostname='localhost',
                 batch_size=20, nstep_return=5, gamma=0.997,
                 clip=lambda x: x,
                 target_update=400, eps_decay=10000000,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(Actor, self).__init__(name, env, policy_net, vis, hostname,
                                    batch_size, nstep_return, gamma, clip,
                                    target_update, eps_decay, device)
        self._target_net = target_net
        self._target_net.load_state_dict(self._policy_net.state_dict())
        self._target_net.eval()
        self._n_burn_in = self._policy_net.n_burn_in

        # need to add action representation networks g, f
        self._g = g
        self._f = f

    def _pull_gparams(self):
        params = self._connect.get('gparams')
        if not params is None:
            print("[%s] Sync gparams." % self._name)
            self._f.load_state_dict(utils.loads(params))

    def _pull_fparams(self):
        params = self._connect.get('fparams')
        if not params is None:
            print("[%s] Sync fparams." % self._name)
            self._f.load_state_dict(utils.loads(params))

    def run(self, n_overlap=40, n_sequence=80):
        assert n_sequence > 1, "n_sequence must be more than 1."
        assert n_overlap < n_sequence, "n_overlap must be less than n_sequence."
        state = self._env.reset()
        step_buffer = deque(maxlen=self._nstep_return)
        sequence_buffer = []
        recurrent_state_buffer = []
        n_total_sequence = self._n_burn_in + n_sequence
        n_total_overlap = self._n_burn_in + n_overlap
        gamma_nsteps = [self._gamma ** i for i in range(self._nstep_return + 1)]
        sum_rwd = 0
        n_episode = 0
        for t in count():
            recurrent_state_buffer.append(self._policy_net.get_state())
            # Select and perform an action
            eps = self.EPS_BASE ** (1.0 + t / (self._eps_decay - 1.0) * self.EPS_ALPHA)
            action = utils.epsilon_greedy(torch.from_numpy(np.reshape(state['pov'].copy(), (3, 64, 64))).type(torch.FloatTensor).unsqueeze(0).to(self._device),
                                          self._policy_net, eps)
            ##########################################################
            # Need to remap actions that come our of our policy net
            ##########################################################
            # new action representation approach
            # TODO: fix hard-coded embed_dim
            mrl_action = torch.zeros(1, 10)
            mrl_action[0, action] = 1
            mrl_action = self._f(mrl_action.float().to(self._device))

            # old action mapping code
            mrl_action = utils.minerl_embed_to_action(mrl_action, self._env)

            # perform action and get next state
            next_state, reward, done, _ = self._env.step(mrl_action)
            ##########################################################
            # End of change
            ##########################################################
            sum_rwd += reward
            reward = torch.tensor([self._clip(reward)])
            done = torch.tensor([float(done)])
            step_buffer.append(utils.Transition(torch.from_numpy(np.reshape(state['pov'].copy(), (3, 64, 64))).type(torch.FloatTensor), 
                action, reward, done=done))
            if len(step_buffer) == step_buffer.maxlen:
                r_nstep = sum([gamma_nsteps[-(i + 2)] * step_buffer[i].reward for i in range(step_buffer.maxlen)])
                sequence_buffer.append(utils.Transition(step_buffer[0].state,
                                                        step_buffer[0].action, r_nstep, done=done))
            if len(sequence_buffer) == n_total_sequence:
                self._local_memory.push(utils.Sequence(sequence_buffer,
                                                       recurrent_state_buffer[0]))
                sequence_buffer = sequence_buffer[-n_total_overlap:] if n_total_overlap > 0 else []
                recurrent_state_buffer = recurrent_state_buffer[-(n_total_overlap + step_buffer.maxlen - 1):] if n_total_overlap + step_buffer.maxlen - 1 > 0 else []
            elif done and len(sequence_buffer) > n_total_overlap:
                self._local_memory.push(utils.Sequence(sequence_buffer,
                                                       recurrent_state_buffer[0]))
            # disable visdom images for now
            # self._vis.image(utils.preprocess(self._env.env._get_image()), win=self._win1)
            state = next_state.copy()
            if done:
                self._vis.line(X=np.array([n_episode]), Y=np.array([sum_rwd]),
                               win=self._win2, update='append')
                state = self._env.reset()
                sum_rwd = 0
                step_buffer.clear()
                sequence_buffer = []
                recurrent_state_buffer = []
                self._policy_net.reset(done)
                n_episode += 1
            if len(self._local_memory) >= self._batch_size:
                samples = self._local_memory.sample(self._batch_size)
                recurrent_state = self._policy_net.get_state()
                _, prio = self._policy_net.calc_priorities(self._target_net, samples,
                                                           gamma=gamma_nsteps[-1],
                                                           require_grad=False,
                                                           device=self._device)
                self._policy_net.set_state(recurrent_state, self._device)
                print("[%s] Publish experience." % self._name)
                self._connect.rpush('experience',
                                    utils.dumps((samples, prio.squeeze(1).cpu().numpy().tolist())))
                self._local_memory.clear()

            if t > 0 and t % self._target_update == 0:
                self._pull_params()
                ################################################################################
                # Need to pull g, f params so our action representations are synced
                ################################################################################ 
                self._pull_gparams()
                self._pull_fparams()
                ################################################################################
                # End of code modification
                ################################################################################

                self._target_net.load_state_dict(self._policy_net.state_dict())
