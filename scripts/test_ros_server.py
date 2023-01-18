#!/usr/bin/python3
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import posthoc_learn.banalg as banalg
from posthoc_learn.config import posthoc_config as config
from posthoc_learn.conban_dataset import ConBanDataset
from posthoc_learn.srv import GetAction, PublishLoss, GetActionResponse, PublishLossResponse

import rospy

def main(name, nTrials):

    # Load Dataset
    print("Loading Dataset...")
    dataset = ConBanDataset(name)
    env, _ = dataset.train_test_split(1.0)
    contexts = env[0]
    posthocs = env[1]
    losses = env[2]

    # Constants
    K = losses.shape[1]
    dF = contexts.shape[1]
    dG = posthocs.shape[1]
    fLambda = config.context_lambda
    gLambda = config.posthoc_lambda

    # Define bandits

    # Vanilla LinUCB
    bandit = banalg.LinUCB(K, dF, dG, fLambda, 0, 0.01)


    print("Run: ./run_ros_server.py -a linUCB -t test")

    input("Press Enter to continue...")

    # Start Service Clients
    rospy.wait_for_service('GetAction')
    rospy.wait_for_service('PublishLoss')

    print("Services Started!")

    get_action = rospy.ServiceProxy('GetAction', GetAction)
    publish_loss = rospy.ServiceProxy('PublishLoss', PublishLoss)

    # Run experiment
    print("Running Test...")
    for i in range(nTrials):
        print("Trial: " + str(i))
        ## TODO: This will fail (expecting image), update to accept features directly too
        resp = get_action(contexts[i].tolist())
        assert resp.p_t[resp.a_t] == 1, "Incorrect max probability"
        assert np.all(np.array(resp.p_t) >= 0.0) and np.all(np.array(resp.p_t) <= 1.0), "Malformed probability bounds"
        assert np.isclose(np.sum(np.array(resp.p_t)), 1.0), "Malformed probability sum"

        local, _ = bandit.choice(i, contexts[i])
        assert resp.a_t == local, "Bandit choice {0} does not match local choice {1}".format(resp.a_t, local)
        loss = np.random.randint(2)

        bandit.update(contexts[i], resp.a_t, loss, np.zeros(dG))
        resp = publish_loss(contexts[i].tolist(), np.zeros(7 * 64).tolist(), resp.a_t, loss, resp.p_t)

        assert resp, "Publish loss failed"

    print("Success! Check algoserver_results/test for {0} npz files".format(nTrials))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: main.py <dataset_name> <num_trials>")
        sys.exit(-1)
    main(str(sys.argv[1]), int(sys.argv[2]))
