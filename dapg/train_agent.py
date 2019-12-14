import argparse
import os
import numpy as np
from scipy.sparse.linalg import cg

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
  parser.add_argument('-d', '--dim', type=int, default=10, help='Embedding dimension for action representations.')
  parser.add_argument('-k', '--timeskip', type=int, default=1, help='k parameter for action rep resentations.')
  parser.add_argument('-l', '--load', action='store_true', help='Set true to load models from modeldir.')
  parser.add_argument('--train', action='store_false', help='Set to true to train model.')
  parser.add_argument('--pretrain', type=int, default=10000, help='Number of pretraining epochs.')
  parser.add_arugment('--bc', type=int, default=10000, help='Number of behavioral cloning steps.')
  parser.add_argument('--episodes', type=int, default=10000, help='Number of training episodes.')
  parser.add_argument('--datadir', type=str, default='../distributed_rl/minerl_data/dataset/', help='Path to demonstration data.')
  parser.add_argument('--modeldir', type=str, default='/models', help='Path to model storage folder.')
  parser.add_argument('--outdir', type=str, default='/output', help='Path to store output data.')
  parser.add_argument('--eval', action='store_true', help='Evaluate a trained model.')
  parser.add_argument('--eval-runs', type=int, default=100, help='Number of evaluation runs.')
  args = parser.parse_args()

  # initialize stuff
  delta = 0.05
  gamma = 0.995
  lam_0 = 1e-2
  lam_1 = 0.95

  g = networks.GNetwork(args.dim, k=args.timeskip)
  f = networks.FNetwork(n_action, args.dim, k=args.timeskip)
  pi = networks.PolicyNetwork(args.dim)

  g_opt = torch.optim.Adam(g.parameters(), lr=0.0001, eps=1.0e-3)
  f_opt = torch.optim.Adam(f.parameters(), lr=0.0001, eps=1.0e-3)
  pi_opt = torch.optim.Adam(pi.parameters(), lr=0.0001, eps=1.0e-3)

  if args.load:
    checkpoint = torch.load(os.path.join(args.modeldir, 'saved_model.pth'))
    g.load_state_dict(checkpoint['g_state_dict'])
    f.load_state_dict(checkpoint['f_state_dict'])
    pi.load_state_dict(checkpoint['pi_state_dcit'])

  # prepare environment
  env = gym.make(args.env)

  # pre-train action representations on demonstration data
  if args.train:
    data = minerl.data.make(args.datadir, args.env)
    for i in range(args.pretrain):
      for current_state, action, reward, next_state, done in data.sarsd_iter(num_epochs=1):
        # convert states to torch tensors
        current_state = torch.from_numpy(np.reshape(current_state['pov'].copy(), (3, 64, 64))).type(torch.FloatTensor)
        next_state = torch.from_numpy(np.reshape(next_state['pov'].copy(), (3, 64, 64))).type(torch.FloatTensor)
        camera, dis = utils.split_action(action)
        
        # feed through network
        net_output = f(g(current_state, next_state))

        # split output into continuous, discrete
        continuous = net_output[:, :2]
        discrete = net_output[:, 2:]

        # calculate loss, which is a mixed log_prob + MSE loss
        loss = -F.log_softmax(discrete, dim=-1)[:, dis].mean() + F.mse_loss(continuous, camera).mean()
          
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

        # compute loss, which is just the negative log likelihood
        nll = -policy_act.log_prob(action_rep)

        # update policy network params
        pi_opt.zero_grad()
        nll.backward()
        pi_opt.step()

    # train policy network with action representation updates + DAPG
    for i in range(args.episodes):
      # sample a single trajectory using the policy network
      obs = env.reset()
      state = torch.from_numpy(np.reshape(obs['pov'].copy(), (3, 64, 64))).type(torch.FloatTensor)
      done = False
      observations = []
      actions = []
      rewards = []
      while not done:
        # sample action embedding from policy, then convert to MineRL action
        action = pi(state)
        minerl_action, act = utils.get_minerl_action(action, f, args.env)
        obs, reward, done, _ = env.step(minerl_action)
        state = torch.from_numpy(np.reshape(obs['pov'].copy(), (3, 64, 64))).type(torch.FloatTensor)
        observations.append(state)
        actions.append(action)
        rewards.append(reward)

        # train f, g further
        net_output = f(g(observations[-2], observations[-1]))

        # split output into continuous, discrete
        discrete = net_output[:, 2:]

        # calculate softmax loss (TODO: add continuous action loss)
        action_rep_loss = -F.log_softmax(discrete)[:, act].mean()

        # update f, g params
          g_opt.zero_grad()
          f_opt.zero_grad()
          action_rep_loss.backward()
          g_opt.step()
          f_opt.step()

      # convert to np arrays
      observations = np.array(observations)
      actions = np.array(actions)
      rewards = np.array(rewards)

      # need to whiten advantages? this is how it's done in original DAPG code
      # also not using baseline at current time -> advantages = rewards
      rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-6)

      # need a demonstration trajectory as well
      demo_obs = []
      demo_actions = []
      for current_state, action, reward, next_state, done in data.sarsd_iter(num_epochs=1):
        # convert states, actions to torch tensors
        current_state = torch.from_numpy(np.reshape(current_state['pov'].copy(), (3, 64, 64))).type(torch.FloatTensor)
        next_state = torch.from_numpy(np.reshape(current_state['pov'].copy(), (3, 64, 64))).type(torch.FloatTensor)
        action_rep = g(current_state, next_state)
        demo_obs.append(current_state)
        demo_actions.append(action_rep)

      # convert to np array
      demo_obs = np.array(demo_obs)
      demo_actions = np.array(demo_actions)

      # create all observations, all actions
      all_obs = np.concatenate([observations, demo_obs])
      all_actions = np.concatenate([actions, demo_actions])

      # unsure about whether this is correct, but this is how it's done in the 
      # original DAPG code
      demo_adv = lam_0 * (self.lam_1 ** i) * np.ones(demo_obs.shape[0])
      all_adv = 1e-2*np.concatenate([advantages/(np.std(advantages) + 1e-8), demo_adv])

      # calculate g, the REINFORCE gradient
      likelihood = pi(all_obs).log_prob(all_actions)
      likelihood_grad = torch.autograd.grad(likelihood, pi.trainable_params).numpy()
      reinforce_grad = torch.mean(likelihood_grad * all_adv, dim=0).numpy()

      # create Fischer Information matrix
      FIM = np.matmul(likelihood_grad, likelihood_grad.T)/all_adv.shape[0]
      Finv_g = cg(FIM, reinforce_grad, x0=reinforce_grad, maxiter=10)
      alpha = np.sqrt(np.abs(delta/(np.dot(likelihood_grad.T, Finv_g) + 1e-20)))

      # update params
      curr_params = pi.get_params()
      new_params = curr_params + alpha * Finv_g
      pi.set_params(new_params)

    # save model at the end of training
    torch.save({
      'f_state_dict': f.state_dict(),
      'g_state_dict': g.state_dict(),
      'pi_state_dcit': pi.state_dict(),
      }, os.path.join(args.modeldir, 'saved_model.pth')) 

  if args.eval:
    all_rewards = []
    for i in range(args.eval_runs):
      rewards = []
      total_reward = 0
      obs = env.reset()
      state = torch.from_numpy(np.reshape(obs['pov'].copy(), (3, 64, 64))).type(torch.FloatTensor)
      done = False
      while not done:
        action = pi(state)
        minerl_action, act = utils.get_minerl_action(action, f, args.env)
        obs, reward, done, _ = env.step(minerl_action)
        total_reward += reward 
        rewards.append(reward)
      print("Total Reward for Episode {}: {}".format(i ,total_reward))
      all_rewards.append(rewards)
    all_rewards = np.array(all_rewards)
    avg_rewards = np.mean(all_rewards, axis=-1) 
    print("Final score: {} +- {}".format(np.mean(avg_reward), np.std(avg_reward)))
    data = {
      'all_rewards': all_rewards,
      'avg_rewards': avg_rewards,
      'avg_reward': np.mean(avg_reward),
      'std_reward': np.std(avg_reward)
    }
    np.save(os.path.join(args.outdir, 'results.npy'), data)
  return

if __name__ == '__main__':
    main()