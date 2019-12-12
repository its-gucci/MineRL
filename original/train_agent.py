import argparse

import minerl
import gym
import networks
import utils

import torch
import torch.nn as nn
import torch.nn.functional as F

def main():
	# add flags
	parser = argparse.ArgumentParser(description='Agent training for MineRL.')
	parser.add_argument('-e', '--env', type=str, default='MineRLTreechop-v0', help='Environment name.')
	parser.add_argument('-d', '--dim', type=int, default=10, help="Embedding dimension for action representations.")
	parser.add_argument('-k', '--timeskip', type=int, default=1, help="k parameter for action representations.")
	parser.add_argument('--pretrain', type=int, default=1, help="Number of pretraining epochs.")
	parser.add_arugment('--bc', type=int, default=10000, help="Number of behavioral cloning steps.")
	parser.add_argument('--iters', type=int, default=100000, help="Number of training iterations.")
	parser.add_argument('--datadir', type=str, default='../distributed_rl/minerl_data/dataset/', help='Path to demonstration data.')
	parser.add_argument('--outdir', type=str, default='/models/', help='Path to model storage folder.')
	args = parser.parse_args()

	# initialize stuff
	g = networks.GNetwork(args.dim, k=args.timeskip)
	f = networks.FNetwork(n_action, args.dim, k=args.timeskip)
	pi = networks.PolicyNetwork(args.dim)

	g_opt = torch.optim.Adam(g.parameters(), lr=0.0001, eps=1.0e-3)
	f_opt = torch.optim.Adam(f.parameters(), lr=0.0001, eps=1.0e-3)
	pi_opt = torch.optim.Adam(pi.parameters(), lr=0.0001, eps=1.0e-3)

	# prepare environment
	core_env = gym.make(args.env)
	env = utils.wrap_env(env)

	# pre-train action representations on demonstration data
	data = minerl.data.make(args.datadir, args.env)
	for i in range(args.pretrain):
		for current_state, action, reward, next_state, done in data.sarsd_iter(num_epochs=1):
			# convert states to torch tensors
			current_state = torch.from_numpy(np.reshape(current_state['pov'].copy(), (3, 64, 64))).type(torch.FloatTensor)
			next_state = torch.from_numpy(np.reshape(next_state['pov'].copy(), (3, 64, 64))).type(torch.FloatTensor)
			camera, discrete = utils.split_action(action)
			
			# feed through network
			net_output = f(g(current_state, next_state))

			# split output into continuous, discrete
			continuous = net_output[:, :2]
			discrete = net_output[:, 2:]

			# calculate loss, which is a mixed log_prob + MSE loss
			loss = -F.log_softmax(discrete, dim=-1)[:, discrete].mean() + F.mse_loss(continuous, camera).mean()
		    
		    # train f, g networks
		    g_opt.zero_grad()
		    f_opt.zero_grad()
		    loss.backward()
		    g_opt.step()
		    f_opt.step()


	# pre-train policy network on demonstration data + action reps with behavioral cloning
	for i in range(args.bc):
		for current_state, action, reward, next_state, done in data.sarsd_iter(num_epochs=1):
			# convert states, actions to torch tensors
			current_state = torch.from_numpy(np.reshape(current_state['pov'].copy(), (3, 64, 64))).type(torch.FloatTensor)
			next_state = torch.from_numpy(np.reshape(next_state['pov'].copy(), (3, 64, 64))).type(torch.FloatTensor)
			
			# get action from our g network, since we want to pre-train in action representation space
			action_rep = g(current_state, next_state)

			# get output of policy net
			policy_act = pi(current_state)

			# compute loss, which is just the negative log likelihodd
			loss = 

			# update policy network params
			pi_opt.zero_grad()
			loss.backward()
			pi_opt.step()

	# train policy network with action representation updates + DAPG
	
	return

if __name__ == '__main__':
    main()