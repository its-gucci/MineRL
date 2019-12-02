# -*- coding: utf-8 -*-
import os
import time
import numpy as np
from itertools import count
import redis
import torch
from ..libs import utils, replay_memory
from . import replay
import minerl
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Learner(object):
    """Learner of Ape-X

    Args:
        policy_net (torch.nn.Module): Q-function network
        target_net (torch.nn.Module): target network
        optimizer (torch.optim.Optimizer): optimizer
        vis (visdom.Visdom): visdom object
        replay_size (int, optional): size of replay memory
        hostname (str, optional): host name of redis server
        beta_decay (int, optional): Decay of annealing bias
        use_memory_compress (bool, optional): use the compressed replay memory for saved memory
    """
    def __init__(self, policy_net, target_net, g, f, optimizer, goptimizer, foptimizer,
                 vis, replay_size=30000, hostname='localhost',
                 beta_decay=1000000,
                 use_memory_compress=False):
        self._vis = vis
        self._policy_net = policy_net
        self._target_net = target_net
        self._target_net.load_state_dict(self._policy_net.state_dict())
        self._target_net.eval()
        self._beta_decay = beta_decay
        self._connect = redis.StrictRedis(host=hostname)
        self._connect.delete('params')
        self._optimizer = optimizer
        self._win = self._vis.line(X=np.array([0]), Y=np.array([0]),
                                   opts=dict(title='Memory size'))
        self._win2 = self._vis.line(X=np.array([0]), Y=np.array([0]),
                                    opts=dict(title='Q loss'))
        self._memory = replay.Replay(replay_size, self._connect,
                                     use_compress=use_memory_compress)
        self._memory.start()

        # add g, f networks for action representations
        self._g = g
        self._f = f

        # action network optimizers
        self._foptimizer = foptimizer
        self._goptimizer = goptimizer

    def _sleep(self):
        mlen = self._connect.llen('experience')
        time.sleep(0.01 * mlen)

    def _wait_memory(self, memory_size):
        while True:
            if len(self._memory) > memory_size:
                break
            time.sleep(0.1)

    def optimize_loop(self, batch_size=512, gamma=0.999**3, rho=1/256.,
                      beta0=0.4, max_grad_norm=40, time_skip=1,
                      start_memory_size=10000,
                      fit_timing=100, target_update=1000, actor_device=device,
                      save_timing=10000, save_model_dir='./models',
                      env='MineRLTreechop-v0', minerl_data_path='../../minerl_data/dataset/'):
        self._wait_memory(max(batch_size, start_memory_size))

        ################################################################################
        # Modification of original R2D2 code to produce R2D3
        #
        # The main change from R2D2 to R2D3 is the addition of a demonstration replay
        # buffer that is sampled from with probability rho, the "demo-ratio". Here we
        # create the demonstration replay buffer.
        #
        ################################################################################
        self._demo_memory = replay_memory.PrioritizedMemory(100000)
        data = minerl.data.make(env, data_dir=minerl_data_path)
        trajectories = data.get_trajectory_names()
        for trajectory in trajectories:
            current_sequence = []
            priority = 0
            for state, action, reward, next_state, done in load_data(trajectory):
                # "action" is the action in the real action space, not the action representation space!
                # we will fix this later, during sampling?
                current_sequence.append(utils.Transition(torch.from_numpy(np.reshape(state['pov'].copy(), (3, 64, 64))).type(torch.FloatTensor), 
                    torch.LongTensor(action), reward, done=done))
                if len(current_sequence) == n_sequence:
                    delta, prio = self._policy_net.calc_priorities(self._target_net,
                                                           current_sequence, gamma=gamma,
                                                           device=device)
                    self._demo_memory.push(current_sequence, prio)
                    current_sequence = current_sequence[n_sequence - n_overlap:]
        ################################################################################
        # End of code modification
        ################################################################################
        for t in count():
            ################################################################################
            # Modification of original R2D2 code to produce R2D3
            #
            # Here we sample from the demo replay buffer with probability rho, and from the
            # agent replay buffer with probability 1 - rho. If rho=0, we recover R2D2.
            #
            # Because of our use of action representations, need to "fix" the 
            # expert transitions using the latest value of the g network. 
            #
            ################################################################################
            expert_transitions, expert_prios, expert_indices = self._demo_memory.sample(batch_size)
            # fix expert transitions
            for i in range(len(expert_transitions)):
                state, action, reward, next_state, done = expert_transitions[i]
                action = self._g(torch.cat([state.reshape(-1), next_state.reshape(-1)], 0))
                expert_transitions[i] = utils.Transition(state, action, reward, next_state, done)
            agent_transitions, agent_prios, agent_indices = self._memory.sample(batch_size)
            if np.random.uniform() < rho:
                transitions, prios, indices = expert_transitions, expert_prios, expert_indices
            else:
                transitions, prios, indices = agent_transitions, agent_prios, agent_indices
            ################################################################################
            # End of code modification
            ################################################################################
            total = len(self._memory)
            beta = min(1.0, beta0 + (1.0 - beta0) / self._beta_decay * t)
            weights = (total * np.array(prios) / self._memory.total_prios) ** (-beta)
            weights /= weights.max()
            delta, prio = self._policy_net.calc_priorities(self._target_net,
                                                           transitions, gamma=gamma,
                                                           device=device)
            loss = (delta * torch.from_numpy(np.expand_dims(weights, 1).astype(np.float32)).to(device)).mean()

            # Optimize the model
            self._optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._policy_net.parameters(), max_grad_norm)
            self._optimizer.step()

            ################################################################################
            # Learn action representation via backprop on g, f
            #
            ################################################################################
            # calculate action representation loss 
            g_input = []
            action_mask = []
            for i in range(time_skip, len(transitions)):
                s_tk, a_tk, r_tk, _, d_tk = transitions[i]
                s_t, a_t, r_t, _, d_t = transitions[i - time_skip]
                g_input.append(torch.cat(s_t.reshape(-1), s_tk.reshape(-1), 0))
                action = [0 for i in range(len(7))]
                action[a_t] = 1
                action_mask.append()
            g_input = torch.from_numpy(g_input)
            action_mask = torch.from_numpy(action_mask).byte()
            # need to select the entry corresponding to a_t
            action_loss = torch.nn.functional.softmax(self._f(self._g(g_input)))
            action_loss = torch.masked_select(action_loss, action_mask).mean()

            # backpropagate with the action representation optimizer
            self._goptimizer.zero_grad()
            action_loss.backward()
            self._goptimizer.step()

            self._foptimizer.zero_grad()
            action_loss.backward()
            self._foptimizer.step()
            ################################################################################
            # End of code modification
            ################################################################################

            ################################################################################
            # Modification of original R2D2 code to produce R2D3
            #
            # We need to update the priorities for both memory buffers
            #
            ################################################################################
            _, agent_prio = self._policy_net.calc_priorities(self._target_net,
                                                           agent_transitions, gamma=gamma,
                                                           device=device)

            _, expert_prio = self._policy_net.calc_priorities(self._target_net,
                                                           expert_transitions, gamma=gamma,
                                                           device=device)

            self._memory.update_priorities(agent_indices,
                                           agent_prio.squeeze(1).cpu().numpy().tolist())

            self._demo_memory.update_priorities(expert_indices,
                                                expert_prio.squeeze(1).cpu().numpy().tolist())
            ################################################################################
            # End of code modification
            ################################################################################

            self._connect.set('params', utils.dumps(self._policy_net.to(actor_device).state_dict()))
            ################################################################################
            # Need to save g, f params so we can sync across all actors
            ################################################################################
            self._connect.set('gparams', utils.dumps(self._g.to(actor_device).state_dict()))
            self._connect.set('fparams', utils.dumps(self._f.to(actor_device).state_dict()))
            ################################################################################
            # End of code modification
            ################################################################################
            self._policy_net.to(device)

            self._vis.line(X=np.array([t]), Y=np.array([loss.detach().cpu().numpy()]),
                           win=self._win2, update='append')
            if t % fit_timing == 0:
                print('[Learner] Remove to fit.')
                self._memory.remove_to_fit()
                self._vis.line(X=np.array([t]), Y=np.array([len(self._memory)]),
                               win=self._win, update='append')
            if t % target_update == 0:
                print('[Learner] Update target.')
                self._target_net.load_state_dict(self._policy_net.state_dict())
            if t % save_timing == 0:
                print('[Learner] Save model.')
                torch.save(self._policy_net.state_dict(), os.path.join(save_model_dir, 'model_%d.pth' % t))
                ################################################################################
                # Need to save g, f models as well
                ################################################################################
                torch.save(self._g.state_dict(), os.path.join(save_model_dir), 'g_%d.pth' % t)
                torch.save(self._g.state_dict(), os.path.join(save_model_dir), 'f_%d.pth' % t)
                ################################################################################
                # End of code modification
                ################################################################################
            self._sleep()
