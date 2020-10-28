#!/usr/bin/python3
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import time

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
    plt.ylabel("MSE")
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
    plt.ylabel("MSE Diff")
    plt.legend()
    plt.show()

#=================
# General Bandit Alg
#=================

def run(bandits, training, test):
    """
    @param bandits: list of initialized bandit algorithms
    """

    # Define constants
    T = training[0].shape[0]
    errors = np.zeros((T + 1, len(bandits)))

    print("Running for {0} rounds!".format(T))
    start = time.time()
    for t in range(T):
        print("Round: %d" % t)
        for i, bandit in enumerate(bandits):
            rate = bandit.report_mse(test[0], test[2])
            print("Bandit %d MSE %f" % (i, rate))
            errors[t, i] = rate

        # Choose arm for each bandit
        I_t = []
        for bandit in bandits:
            ret, _ = bandit.choice(t, training[0][t, :])
            I_t.append(ret)

        # Update bandits
        for i, bandit in enumerate(bandits):
            bandit.update(training[0][t, :], I_t[i], training[2][t, I_t[i]], training[1][t, :])

    # Final Error Rate
    print("Final Error Rate Calculation")
    for i, bandit in enumerate(bandits):
        rate = bandit.report_mse(test[0], test[2])
        print("Bandit %d MSE %f" % (i, rate))
        errors[T, i] = rate

    # Finished, return error array
    print("Finished T=%d rounds!" % T)
    print("Time Taken (s): " + str(time.time() - start))
    return errors


def main(name, nRuns):

    # Load Dataset
    print("Loading Dataset...")
    dataset = ConBanDataset(name)
    training, test = dataset.train_test_split(config.train_split)

    # Dimensions
    T, n = training[2].shape
    T_test = test[0].shape[0]
    dF = training[0].shape[1]
    dG = training[1].shape[1]
    fLambda = 1000.0
    gLambda = 100.0

    print("Training Rounds: " + str(T))
    print("Testing Rounds: " + str(T_test))

    # Best Linear Fit Possible
    print("Best Context Linear Fit...")
    A = fLambda * np.eye(dF)
    b = np.zeros((dF, n))
    for t in range(T):
        A += np.outer(training[0][t, :], training[0][t, :])
        b += np.outer(training[0][t, :], training[2][t, :])
    assert A.shape == (dF, dF)
    assert b.shape == (dF, n)
    print("Doing Fit")
    phi = np.linalg.solve(A, b)
    assert phi.shape == (dF, n)

    print("Testing...")
    mse = 0.0
    for i in range(T_test):
        ans = test[0][i, :] @ phi
        assert ans.size == n
        mse += np.linalg.norm(ans - test[2][i, :])

    print("Context MSE: " + str(mse / float(T_test)))
    print()

    # Best PostHoc Fit Possible
    print("Best Posthoc Linear Fit...")
    A = gLambda * np.eye(dG)
    b = np.zeros((dG, n))
    for t in range(T):
        A += np.outer(training[1][t, :], training[1][t, :])
        b += np.outer(training[1][t, :], training[2][t, :])
    assert A.shape == (dG, dG)
    assert b.shape == (dG, n)
    print("Doing Fit")
    phi = np.linalg.solve(A, b)
    assert phi.shape == (dG, n)

    print("Testing...")
    mse = 0.0
    for i in range(T_test):
        ans = test[1][i, :] @ phi
        assert ans.size == n
        mse += np.linalg.norm(ans - test[2][i, :])

    print("Posthoc MSE: " + str(mse / float(T_test)))
    print()

    # Define bandits
    bandits = []

    # Test 1: Vanilla vs Hard Constraint
    bandits.append(banalg.HardConstraint(n, dF, dG, fLambda, 0))
    bandits.append(banalg.HardConstraint(n, dF, dG, fLambda, gLambda))

    # Run experiment
    if nRuns == 1:
        print("Running Experiment...")
        errors = run(bandits, training, test)

        # Plot Cumulative Regret
        labels = []
        for bandit in bandits:
            labels.append(bandit.label)

        title = "Augmented Contextual Bandit Learning"
        plot(title, errors, labels)
    else:
        # Do Bootstrap
        error_diff = np.zeros((nRuns, T+1))
        for i in range(nRuns):
            for bandit in bandits:
                bandit.reinit()
            print("Running Experiment {0}:".format(i))
            training, test = dataset.train_test_split(config.train_split)
            errors = run(bandits, training, test)
            diff = errors[:, 0] - errors[:, 1]
            error_diff[i, :] = diff

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
            error_diff = error_diff)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: main.py <dataset_name> <num_runs>")
        sys.exit(-1)
    main(str(sys.argv[1]), int(sys.argv[2]))
