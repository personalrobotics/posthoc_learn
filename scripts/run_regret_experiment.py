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

def plot_ci(title, means, lower, upper):
    """
    @param title: graph title
    @param regret: T+1 x len(bandits) cumulative regret
    @param labels: label[i] for bandits[i]

    Plots regret curve.
    """
    plt.title(title)
    t = np.arange(means.shape[0])
    plt.plot(t, means, label="Mean Difference (Vanilla - Post Hoc)")
    plt.fill_between(t, lower, upper, alpha=0.2, label="{0}% CI".format(int(100 * config.ci)))
    plt.plot(t, np.zeros(t.shape), color='black', linewidth=4)

    T = (means.shape[0] - 1)
    plt.xlim(0, T)
    plt.xlabel("Attempts")
    plt.ylabel("Cumulative Regret Diff")
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

def main(name, nRuns):

    # Load Dataset
    print("Loading Dataset...")
    dataset = ConBanDataset(name)
    env, _ = dataset.train_test_split(1.0)
    contexts = env[0]
    posthocs = env[1]
    losses = env[2]

    # Constants
    T = contexts.shape[0]
    K = losses.shape[1]
    dF = contexts.shape[1]
    dG = posthocs.shape[1]
    fLambda = 1000.0
    gLambda = 100.0

    # Define bandits
    bandits = []

    # Vanilla Greedy
    bandits.append(banalg.LinUCB(T, K, dF, dG, fLambda, 0))

    # Post Hoc Greedy
    bandits.append(banalg.LinUCB(T, K, dF, dG, fLambda, gLambda))

    # Run experiment
    print("Running Experiment...")
    if nRuns == 1:
        regrets = run(bandits, contexts, posthocs, losses)

        # Plot Cumulative Regret
        labels = []
        for bandit in bandits:
            labels.append(bandit.label)

        title = "Augmented Contextual Bandit Regret"
        plot(title, regrets, labels)

        # Save Regret Data
        np.savez("regrets_{0}_{1}.npz".format(name, nRuns),
            regrets = regrets)
    else:
         # Do Bootstrap
        error_diff = np.zeros((nRuns, T+1))
        regrets_save = np.zeros((nRuns, T+1, len(bandits)))
        for i in range(nRuns):
            for bandit in bandits:
                bandit.reinit()
            print("Running Experiment {0}:".format(i))
            env = dataset.sample_with_replacement()
            regrets = run(bandits, env[0], env[1], env[2])
            diff = regrets[:, 0] - regrets[:, 1]
            error_diff[i, :] = diff
            regrets_save[i, :, :] = regrets[:, :]

        means = np.mean(error_diff, axis=0)
        sorted_idx = np.argsort(error_diff, axis=0)
        sorted_ed = np.take_along_axis(error_diff, sorted_idx, axis=0)
        print("Confidence Interval: " + str(config.ci))
        lower_idx = int(nRuns * (1.0-config.ci) / 2.0)
        upper_idx = int(nRuns - 1.0 - lower_idx)
        print("Lower Index: " + str(lower_idx))
        print("Upper Index: " + str(upper_idx))
        lower = sorted_ed[lower_idx, :]
        upper = sorted_ed[upper_idx, :]

        title = "Post Hoc Bandit Difference"
        plot_ci(title, means, lower, upper)
        print("Saving errors_diff.npz:")
        np.savez_compressed("errors_diff.npz", 
            error_diff = error_diff,
            regrets = regrets_save)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: main.py <dataset_name> <num_runs>")
        sys.exit(-1)
    main(str(sys.argv[1]), int(sys.argv[2]))
