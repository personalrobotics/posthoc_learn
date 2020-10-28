#!/usr/bin/env python

"""
Initiates the services `getAction` and `publishLoss`.

TODO:
  * Run this script as a node in a launch script
"""

import argparse
from posthoc_learn.algoserver import create_server, N_FEATURES
from posthoc_learn.banalg import HardConstraint, Greedy, EpsilonGreedy, LinUCB
from posthoc_learn.config import posthoc_config as config
from posthoc_learn.conban_dataset import ConBanDataset
import os
import rospy

if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument('-a', '--algo', default="greedy",
            type=str, help='algorithm to use: greedy, epsilon, linUCB, random')
    ap.add_argument('-alp', '--alpha', default=0.01,
                    type=float, help='alpha for LinUCB')
    ap.add_argument('-e', '--epsilon', default=0.1,
                    type=float, help='epsilon for epsilon greedy')
    ap.add_argument('-g', '--gpu', default='0', type=str, help='GPU ID')
    ap.add_argument('-p', '--posthoc', action='store_true', help='Whether to use Post Hoc Context')
    ap.add_argument('-d', '--dataset', default=config.data_name, type=str, help='Dataset name')
    ap.add_argument('-t', '--trialName', default='test', type=str, help='Name of trial (if exists, will use previous data)')
    ap.add_argument('-x', '--exclude', nargs='?', const='', default=None, type=str, help='Name of item to exclude from pretrain, use empty to pretrain with everything')

    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.posthoc:
        print("Using Post Hoc Data")
        lambG = config.posthoc_lambda
    else:
        lambG = 0.0

    # Initialize algorithm
    if  args.algo == 'greedy':
        print('Greedy exploration')
        algo = Greedy(config.n_actions, N_FEATURES, config.n_haptic_features, config.context_lambda, lambG)

    elif args.algo == 'epsilon':
        print('Epsilon-greedy with epsilon {}'.format(args.epsilon))
        algo = EpsilonGreedy(config.n_actions, N_FEATURES, config.n_haptic_features, config.context_lambda, lambG, args.epsilon)

    elif args.algo == 'linUCB':
        print('LinUCB with alpha {}'.format(args.alpha))
        algo = LinUCB(config.n_actions, N_FEATURES, config.n_haptic_features, config.context_lambda, lambG, args.alpha)

    elif args.algo == 'random':
        print('Random exploration')
        algo = HardConstraint(config.n_actions, N_FEATURES, config.n_haptic_features, config.context_lambda, lambG)
    else:
        exit('"{}" is not a valid algorithm type'.format(args.algo))

    dataset = ConBanDataset(args.dataset, config.visual_model, config.haptic_model)

    create_server(algo, dataset, args.trialName, args.exclude)

    try:
        print('Running server for trial {}'.format(args.trialName))
        rospy.spin()
    except KeyboardInterrupt:
        pass
    print('\nShutting down trial {}...'.format(args.trialName))
