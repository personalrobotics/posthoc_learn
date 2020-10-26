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
    # plt.title(title)
    fig = plt.figure()
    st = fig.suptitle(title, fontsize="x-large")
    ax1 = fig.add_subplot(311)

    t = np.arange(regret.shape[0])
    for i in range(0,2):
        ax1.plot(t, regret[:, i], label=labels[i])
    ax1.set_title("Greedy")
    ax1.set_xlabel("Attempts")
    # ax1.set_ylabel("Cumulative Regret")
    ax2 = fig.add_subplot(312)
    for i in range(2,4):
        ax2.plot(t, regret[:, i], label=labels[i])
    ax2.set_title("Epsilon-Greedy")
    ax2.set_xlabel("Attempts")
    ax2.set_ylabel("Cumulative Regret")
    ax3 = fig.add_subplot(313)
    for i in range(4,6):
        ax3.plot(t, regret[:, i], label=labels[i])
    ax3.set_title("LinUCB")
    ax3.set_xlabel("Attempts")
    # ax3.set_ylabel("Cumulative Regret")

    T = (regret.shape[0] - 1)
    ax1.set_xlim(0, T)
    ax2.set_xlim(0, T)
    ax3.set_xlim(0, T)

    ax1.legend()
    # ax2.legend()
    # ax3.legend()
    fig.savefig('regret.pdf')
    # plt.show()

def plot_ci(title, means, lower, upper):
    """
    @param title: graph title
    @param regret: T+1 x len(bandits) cumulative regret
    @param labels: label[i] for bandits[i]

    Plots regret curve.
    """
    plt.title(title)
    t = np.arange(means.shape[0])
    plt.plot(t, means, label="Cumulative Regret")
    plt.fill_between(t, lower, upper, alpha=0.2, label="{0}% CI".format(int(100 * config.ci)))
    plt.plot(t, np.zeros(t.shape), color='black', linewidth=4)

    T = (means.shape[0] - 1)
    plt.xlim(0, T)
    plt.xlabel("Attempts")
    plt.ylabel("Cumulative Regret")
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
    # print(training[0].shape)
    # print(training[1].shape)
    # print(training[2].shape)
    # print(test[0].shape)
    # print(test[1].shape)
    # print(test[2].shape)
    errors = np.zeros((T + 1, len(bandits)))

    for t in range(T):
        print("Round: %d" % t)

        # Choose arm for each bandit
        I_t = []
        # I_test = np.zeros((test[2].shape[0], len(bandits)), dtype=int)
        # for tst in range(test[2].shape[0]):
        #     for x, bandit in enumerate(bandits):
        #         I_test[tst][x] = bandit.choice(tst, test[0][tst, :])

        # print(I_test[6])
    
        for bandit in bandits:
            I_t.append(bandit.choice(t, training[0][t, :]))
        # print(I_t)
        
        for i, bandit in enumerate(bandits):
            # rate = bandit.report_mse(test[0], test[2])
            rate = training[2][t, I_t[i]] - np.min(training[2][t])
            print("Bandit %d Regret %f" % (i, rate))
            errors[t, i] = errors[max(0,t-1), i] + rate

        # Update bandits
        for i, bandit in enumerate(bandits):
            bandit.update(training[0][t, :], I_t[i], training[2][t, I_t[i]], training[1][t, :])

    # Final Error Rate
    print("Final Error Rate Calculation")
    for i, bandit in enumerate(bandits):
        # rate = bandit.report_mse(test[0], test[2])
        rate = training[2][t, I_t[i]] - np.min(training[2][t])
        print("Bandit %d MSE %f" % (i, rate))
        errors[T, i] = errors[max(0,T-1), i] + rate

    # Finished, return error array
    print("Finished T=%d rounds!" % T)
    # np.savez_compressed('errors_NB.npz', errors)
    return errors


def main(name, nRuns):

    # Load Dataset
    print("Loading Dataset...")
    dataset = ConBanDataset(name)
    training, test = dataset.train_test_split(config.train_split)

    test = training
    # Dimensions
    T, n = training[2].shape
    T_test = test[0].shape[0]
    dF = training[0].shape[1]
    dG = training[1].shape[1]
    fLambda = 1000.0
    gLambda = 100.0

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
    # bandits.append(banalg.HardConstraint(T, n, dF, dG, fLambda, 0))
    # bandits.append(banalg.HardConstraint(T, n, dF, dG, fLambda, gLambda))
    bandits.append(banalg.Greedy(T, n, dF, dG, fLambda, gLambda))
    bandits.append(banalg.Greedy(T, n, dF, dG, fLambda, 0))
    bandits.append(banalg.EpsilonGreedy(T, n, dF, dG, fLambda, gLambda, 0.1))
    bandits.append(banalg.EpsilonGreedy(T, n, dF, dG, fLambda, 0, 0.1))
    bandits.append(banalg.LinUCB(T, n, dF, dG, fLambda, gLambda, 0.01))
    bandits.append(banalg.LinUCB(T, n, dF, dG, fLambda, 0, 0.01))

    labels = []
    for bandit in bandits:
        labels.append(bandit.label)

    # Run experiment
    if nRuns == 1:
        print("Running Experiment...")
        errors = run(bandits, training, test)

        # Plot Cumulative Regret
        

        title = "Augmented Contextual Bandit Learning"
        plot(title, errors, labels)
    else:
        # Do Bootstrap
        error_boot = np.zeros((nRuns, T+1, len(bandits)))
        for i in range(nRuns):
            #
            for bandit in bandits:
                bandit.reinit()
            print("Running Experiment {0}:".format(i))
            training = dataset.sample_with_replacement()
            test = training
            errors = run(bandits, training, test)
            # diff = errors[:, 0] - errors[:, 1]
            error_boot[i] = errors
        print("Saving error_boot.npz:")
        np.savez_compressed("error_boot.npz", error_boot=error_boot)
        
        # error_boot = np.load("error_boot.npz", allow_pickle=True)
        # error_boot = error_boot['error_boot']
        title = "Post Hoc Bandit Difference"
        plt.title(title)
        for i in range(0,2):                                            # GREEDY
            means = np.mean(error_boot[:,:,i], axis=0)
            sorted_idx = np.argsort(error_boot[:,:,i], axis=0)
            sorted_ed = np.take_along_axis(error_boot[:,:,i], sorted_idx, axis=0)
            print("Confidence Interval: " + str(config.ci))
            lower_idx = int(nRuns * (1.0-config.ci) / 2.0)
            upper_idx = int(nRuns - 1.0 - lower_idx)
            print("Lower Index: " + str(lower_idx))
            print("Upper Index: " + str(upper_idx))
            lower = sorted_ed[lower_idx, :]
            upper = sorted_ed[upper_idx, :]

            
            # # plot_ci(title, means, lower, upper)
            
            t = np.arange(means.shape[0])
            plt.plot(t, means, label=labels[i])
            plt.fill_between(t, lower, upper, alpha=0.2, label="{0}% CI".format(int(100 * config.ci)))
            plt.plot(t, np.zeros(t.shape), color='black', linewidth=4)
        
        T = (means.shape[0] - 1)
        plt.xlim(0, T)
        plt.xlabel("Attempts")
        plt.ylabel("Cumulative Regret")
        plt.legend()
        plt.show()

        # UNCOMMENT DURING PLOTTING
        # for i in range(2,4):                                         # EPSILON - GREEDY
        #     means = np.mean(error_boot[:,:,i], axis=0)
        #     sorted_idx = np.argsort(error_boot[:,:,i], axis=0)
        #     sorted_ed = np.take_along_axis(error_boot[:,:,i], sorted_idx, axis=0)
        #     print("Confidence Interval: " + str(config.ci))
        #     lower_idx = int(nRuns * (1.0-config.ci) / 2.0)
        #     upper_idx = int(nRuns - 1.0 - lower_idx)
        #     print("Lower Index: " + str(lower_idx))
        #     print("Upper Index: " + str(upper_idx))
        #     lower = sorted_ed[lower_idx, :]
        #     upper = sorted_ed[upper_idx, :]

            
        #     # # plot_ci(title, means, lower, upper)
            
        #     t = np.arange(means.shape[0])
        #     plt.plot(t, means, label=labels[i])
        #     plt.fill_between(t, lower, upper, alpha=0.2, label="{0}% CI".format(int(100 * config.ci)))
        #     plt.plot(t, np.zeros(t.shape), color='black', linewidth=4)
        
        # T = (means.shape[0] - 1)
        # plt.xlim(0, T)
        # plt.xlabel("Attempts")
        # plt.ylabel("Cumulative Regret")
        # plt.legend()
        # plt.show()

        # for i in range(4,6):                                                     # LIN-UCB
        #     means = np.mean(error_boot[:,:,i], axis=0)
        #     sorted_idx = np.argsort(error_boot[:,:,i], axis=0)
        #     sorted_ed = np.take_along_axis(error_boot[:,:,i], sorted_idx, axis=0)
        #     print("Confidence Interval: " + str(config.ci))
        #     lower_idx = int(nRuns * (1.0-config.ci) / 2.0)
        #     upper_idx = int(nRuns - 1.0 - lower_idx)
        #     print("Lower Index: " + str(lower_idx))
        #     print("Upper Index: " + str(upper_idx))
        #     lower = sorted_ed[lower_idx, :]
        #     upper = sorted_ed[upper_idx, :]

            
        #     # # plot_ci(title, means, lower, upper)
            
        #     t = np.arange(means.shape[0])
        #     plt.plot(t, means, label=labels[i])
        #     plt.fill_between(t, lower, upper, alpha=0.2, label="{0}% CI".format(int(100 * config.ci)))
        #     plt.plot(t, np.zeros(t.shape), color='black', linewidth=4)
        
        # T = (means.shape[0] - 1)
        # plt.xlim(0, T)
        # plt.xlabel("Attempts")
        # plt.ylabel("Cumulative Regret")
        # plt.legend()
        # plt.show()
        


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: main.py <dataset_name> <num_runs>")
        sys.exit(-1)
    main(str(sys.argv[1]), int(sys.argv[2]))
