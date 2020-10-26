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
            I_t.append(bandit.choice(t, contexts[t, :]))

        # Update bandits
        for i, bandit in enumerate(bandits):
            regrets[t+1, i] = loss[t, I_t[i]] - np.amin(loss[t, :])
            bandit.update(contexts[t, :], I_t[i], loss[t, I_t[i]] + np.random.normal(0, noise), posthocs[t, :])

    # Finished, return error array
    print("Finished T=%d rounds!" % T)
    cum_regrets = np.cumsum(regrets, axis=0)
    return cum_regrets


def gen_data(T, K, dF, dG):
    # Generate a random invertible matrix (dG x K)

    # Generate random matrix
    A = np.random.rand(K, dG)

    # Make sure it inverts
    Ainv = np.linalg.inv(A.T @ A) @ A.T
    assert Ainv.shape == (dG, K)

    contexts = np.random.rand(T, dF)
    assert contexts.shape == (T, dF)

    posthocs = contexts[:, :dG]
    assert posthocs.shape == (T, dG)

    losses = posthocs @ Ainv
    assert losses.shape == (T, K)

    return contexts, posthocs, losses

def main(T, K, dF, dG):

    # Load Dataset
    print("Loading Dataset...")
    contexts, posthocs, losses = gen_data(T, K, dF, dG)

    # Constants
    fLambda = 1E-7
    gLambda = 1E-7

    # Define bandits
    bandits = []

    # Vanilla Greedy
    bandits.append(banalg.Greedy(T, K, dF, dG, fLambda, 0))

    # Post Hoc Greedy
    bandits.append(banalg.Greedy(T, K, dF, dG, fLambda, gLambda))

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
    if len(sys.argv) != 5:
        print("Usage: main.py <T> <K> <dF> <dG>")
        sys.exit(-1)
    main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
