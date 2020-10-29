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

#=================
# Plot Cumulative Regret
#=================

def plot(title, regret, labels):
    """
    @param title: graph title
    @param regret: T+1 x len(bandits) cumulative regret
    @param labels: label[i] for bandits[i]

    Plots regret curve.
    """
    plt.title(title)
    t = np.arange(regret.shape[0])
    for i, l in enumerate(labels):
        plt.plot(t, regret[:, i], label=l)

    T = (regret.shape[0] - 1)
    plt.xlim(0, T)
    plt.xlabel("Attempts")
    plt.ylabel("Cumulative Regret")
    plt.legend()
    plt.show()

#=================
# General Bandit Alg
#=================

def run(bandits, contexts, posthocs, loss, noise=0):
    """
    @param bandits: list of initialized bandit algorithms
    @param contexts: T x dF
    @param posthocs: T x dG
    @param loss: len(K)
    """

    # Define constants
    T = contexts.shape[0]
    regrets = np.zeros((T + 1, len(bandits)))

    print("Running for {0} rounds!".format(T))
    for t in range(T):
        print("Round: %d" % t)

        # Choose arm for each bandit
        I_t = []
        for bandit in bandits:
            ret, _ = bandit.choice(t, contexts[t, :])
            I_t.append(ret)

        # Update bandits
        for i, bandit in enumerate(bandits):
            regrets[t+1, i] = loss[t, I_t[i]] - np.amin(loss[t, :])
            bandit.update(contexts[t, :], I_t[i], loss[t, I_t[i]] + np.random.normal(0, noise), posthocs[t, :])

    # Finished, return error array
    print("Finished T=%d rounds!" % T)
    cum_regrets = np.cumsum(regrets, axis=0)
    return cum_regrets


def gen_posthoc(dG, rewards):
    # Generate a random invertible matrix reward[i, :] = A * posthocs[i, :]
    # (n x T) = (n x dg) * (dg x T)
    T, n = rewards.shape

    # Generate random matrix
    A = np.random.rand(n, dG)

    # Make sure it inverts
    Ainv = np.linalg.inv(A.T @ A) @ A.T
    assert Ainv.shape == (dG, n)

    posthocs = (Ainv @ rewards.T).T
    assert posthocs.shape == (T, dG)
    return posthocs, Ainv

def main(dF, dG):

    cacheFile = "cache_"+str(dF)+".npz"

    if path.exists(cacheFile):
        print("Loading from Cache...")
        with np.load(cacheFile) as data:
            contexts = data["contexts"]
            rewards = data["rewards"]
            contexts_test = data["contexts_test"]
            rewards_test = data["rewards_test"]
    else:
        # Import MNIST
        print("Loading MNIST...")
        mndata = MNIST('./mnist')
        mndata.gz = True

        # Load Contexts / Rewards
        contexts, rewards, contexts_test, rewards_test = gen_context(mndata, dF)
        print("Saving Cache...")
        np.savez_compressed(cacheFile, contexts=contexts, rewards=rewards, contexts_test=contexts_test, rewards_test=rewards_test)

    T, K = rewards.shape

    # Constants
    fLambda = 1E-7
    gLambda = 1E-7

    # Define bandits
    bandits = []

    # Vanilla Greedy
    bandits.append(banalg.EpsilonGreedy(K, dF, dG, fLambda, 0, 0.1))

    # Vanilla TS
    #bandits.append(banalg.Thompson(K, dF, dG, fLambda, 0, var))
    # Post Hoc Greedy
    bandits.append(banalg.EpsilonGreedy(K, dF, dG, fLambda, gLambda, 0.1))

    # Post Hoc Only
    #bandits.append(banalg.Thompson(K, dF, dG, 0, gLambda, 0.01))

    # Run experiment
    print("Running Experiment...")
    regrets = run(bandits, contexts, posthocs, losses)

    # Plot Cumulative Regret
    labels = []
    for bandit in bandits:
        labels.append(bandit.label)

    title = "Augmented Contextual Bandit Regret"
    plot(title, regrets, labels)

    # Save Regret Data
    np.savez("regrets_{0}_{1}_{2}_{3}.npz".format(T, K, dF, dG),
        regrets = regrets)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: main.py <dF> <dG>")
        sys.exit(-1)
    main(int(sys.argv[1]), int(sys.argv[2]))
