# -*- coding: utf-8 -*-
import argparse
import gym
import torch
import visdom
from distributed_rl.libs import wrapped_env, models

def main():
    parser = argparse.ArgumentParser(description='Actor process for distributed reinforcement.')
    parser.add_argument('-n', '--name', type=str, default='actor1', help='Actor name.')
    parser.add_argument('-e', '--env', type=str, default='MultiFrameBreakout-v0', help='Environment name.')
    parser.add_argument('-a', '--algorithm', type=str, default='ape_x', choices=['ape_x', 'r2d2'], help='Select an algorithm.')
    parser.add_argument('-d', '--eps_decay', type=int, default=10000000, help='Decay of random action rate in e-greedy.')
    parser.add_argument('-r', '--redisserver', type=str, default='localhost', help="Redis's server name.")
    parser.add_argument('-v', '--visdomserver', type=str, default='localhost', help="Visdom's server name.")
    args = parser.parse_args()
    vis = visdom.Visdom(server='http://' + args.visdomserver)
    env = gym.make(args.env)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.algorithm == 'ape_x':
        from distributed_rl.ape_x.actor import Actor
        actor = Actor(args.name, env, models.DuelingDQN(env.action_space.n).to(device),
                      vis, hostname=args.redisserver, eps_decay=args.eps_decay,
                      device=device)
    elif args.algorithm == 'r2d2':
        from distributed_rl.r2d2.actor import Actor
        nstep_return = 5
        actor = Actor(args.name, env,
                      models.DuelingLSTMDQN(env.action_space.n, 1,
                                            nstep_return=nstep_return).to(device),
                      models.DuelingLSTMDQN(env.action_space.n, 1,
                                            nstep_return=nstep_return).to(device),
                      vis, hostname=args.redisserver, eps_decay=args.eps_decay,
                      device=device)
    else:
        raise ValueError('Unknown the algorithm: %s.' % args.algorithm)
    actor.run()
   
if __name__ == '__main__':
    main()
