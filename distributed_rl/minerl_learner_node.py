# -*- coding: utf-8 -*-
import argparse
import minerl
import gym
import torch
import torch.optim as optim
import visdom
from distributed_rl.libs import models, wrapped_env
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser(description='Learner process for distributed reinforcement.')
    parser.add_argument('-e', '--env', type=str, default='MineRLTreechop-v0', help='Environment name.')
    parser.add_argument('-a', '--algorithm', type=str, default='r2d3', choices=['ape_x', 'r2d2', 'r2d3'], help='Select an algorithm.')
    parser.add_argument('-r', '--redisserver', type=str, default='localhost', help="Redis's server name.")
    parser.add_argument('-v', '--visdomserver', type=str, default='localhost', help="Visdom's server name.")
    parser.add_argument('-d', '--actordevice', type=str, default='', help="Actor's device.")
    parser.add_argument('-s', '--replaysize', type=int, default=100000, help="Replay memory size.")
    args = parser.parse_args()
    env = gym.make(args.env)
    vis = visdom.Visdom(server='http://' + args.visdomserver)
    actordevice = ("cuda" if torch.cuda.is_available() else "cpu") if args.actordevice == '' else args.actordevice
    if args.algorithm == 'ape_x':
        from distributed_rl.ape_x.learner import Learner
        nstep_return = 3
        model = models.DuelingDQN(env.action_space.n).to(device)
        learner = Learner(model,
                          models.DuelingDQN(env.action_space.n).to(device),
                          optim.RMSprop(model.parameters(), lr=0.00025 / 4, alpha=0.95, eps=1.5e-7),
                          vis, replay_size=args.replaysize, hostname=args.redisserver,
                          use_memory_compress=True)
        learner.optimize_loop(gamma=0.999**nstep_return, actor_device=torch.device(actordevice))
    elif args.algorithm == 'r2d2':
        from distributed_rl.ape_x.learner import Learner
        batch_size = 64
        nstep_return = 5
        model = models.DuelingLSTMDQN(env.action_space.n, batch_size,
                                      nstep_return=nstep_return).to(device)
        learner = Learner(model,
                          models.DuelingLSTMDQN(env.action_space.n, batch_size,
                                                nstep_return=nstep_return).to(device),
                          optim.Adam(model.parameters(), lr=0.00048, eps=1.0e-3),
                          vis, replay_size=args.replaysize, hostname=args.redisserver,
                          use_memory_compress=True)
        learner.optimize_loop(batch_size=batch_size, gamma=0.997**nstep_return,
                              beta0=0.6, target_update=2500,
                              actor_device=torch.device(actordevice))
    elif args.algorithm == 'r2d3':
        from distributed_rl.ape_x.r2d3_learner import Learner
        batch_size = 64
        nstep_return = 5
        model = models.DuelingLSTMDQN2(7, batch_size,
                                      nstep_return=nstep_return).to(device)
        g = models.GNetwork(10).to(device)
        f = models.FNetwork(7, 10).to(device)
        learner = Learner(model,
                          models.DuelingLSTMDQN2(7, batch_size,
                                                nstep_return=nstep_return).to(device),
                          g,
                          f,
                          optim.Adam(model.parameters(), lr=0.00048, eps=1.0e-3),
                          optim.Adam(g.parameters(), lr=0.0001, eps=1.0e-3),
                          optim.Adam(f.parameters(), lr=0.0001, eps=1.0e-3),
                          vis, replay_size=args.replaysize, hostname=args.redisserver,
                          use_memory_compress=True)
        learner.optimize_loop(batch_size=batch_size, gamma=0.997**nstep_return,
                              beta0=0.6, target_update=2500,
                              actor_device=torch.device(actordevice))
    else:
        raise ValueError('Unknown the algorithm: %s.' % args.algorithm)

if __name__ == '__main__':
    main()
