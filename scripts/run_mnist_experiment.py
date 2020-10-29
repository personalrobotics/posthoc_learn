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
        if t % 100 == 0:
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

VAL_SIZE = 10000
def gen_context(mndata, dF, trueTest=False):
    print("Loading Training Set...")
    images, labels = mndata.load_training()

    if trueTest:
        print("Loading Test Set...")
        images_test, labels_test = mndata.load_testing()
    else:
        print("Loading Validation Set...")
        images_test = images[len(images) - VAL_SIZE:len(images)]
        images = images[0:len(images) - VAL_SIZE]
        labels_test = labels[len(labels) - VAL_SIZE:len(labels)]
        labels = labels[0:len(labels) - VAL_SIZE]

    # Format labels
    labels = np.array(labels)
    labels_test = np.array(labels_test)
    Ttrain = len(labels)
    Ttest = len(labels_test)
    print("T_train=%d" % Ttrain)
    print("T_val=%d" % Ttest)
    n = labels.max() + 1

    # Create 1-hot rewards
    rewards = np.zeros((Ttrain, n))
    rewards[np.arange(labels.size),labels] = 1
    rewards_test = np.zeros((Ttest, n))
    rewards_test[np.arange(labels_test.size),labels_test] = 1

    # PCA Contexts
    images = np.array(images)
    images_test = np.array(images_test)

    print("Performing PCA...")
    pca = PCA(n_components=dF)
    contexts = pca.fit_transform(images)
    contexts_test = pca.transform(images_test)
    assert contexts.shape == (Ttrain, dF)
    assert contexts_test.shape == (Ttest, dF)

    return contexts, rewards, contexts_test, rewards_test

def main(dF, dG):

    cacheFile = "cache_"+str(dF)+".npz"

    if Path(cacheFile).exists():
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

    # Convert rewards to losses
    losses = 1 - rewards
    losses_test = 1 - rewards_test

    T, K = losses.shape

    # Best Linear Fit Possible
    """
    print("Best Linear Fit...")
    print("Generating Matrices")
    A = np.eye(dF)
    b = np.zeros((dF, K))
    for t in range(contexts.shape[0]):
        A += np.outer(contexts[t, :], contexts[t, :])
        b += np.outer(contexts[t, :], losses[t, :])
    assert A.shape == (dF, dF)
    assert b.shape == (dF, K)
    print("Doing Fit")
    phi = np.linalg.solve(A, b)
    assert phi.shape == (dF, K)

    print("Testing...")
    err_count = 1.0
    for i in range(contexts_test.shape[0]):
        ans = contexts_test[i, :] @ phi
        assert ans.size == K
        if np.argmin(ans) != np.argmin(losses_test[i, :]):
            err_count += 1.0

    print("Error Rate: " + str(err_count / float(contexts_test.shape[0])))
    """

    # Generate Post-Hoc Contexts
    posthocs, _ = gen_posthoc(dG, losses_test)

    # Constants
    fLambda = 1E7
    gLambda = 1E-7

    # Define bandits
    bandits = []

    # Vanilla Greedy
    bandits.append(banalg.LinUCB(K, dF, dG, fLambda, 0, 0.01))

    bandits.append(banalg.LinUCB(K, dF, dG, fLambda, gLambda, 0.01))

    # Vanilla TS
    #bandits.append(banalg.Thompson(K, dF, dG, fLambda, 0, var))
    # Post Hoc Greedy
    #bandits.append(banalg.EpsilonGreedy(K, dF, dG, fLambda, gLambda, 0.1))

    # Post Hoc Only
    #bandits.append(banalg.Thompson(K, dF, dG, 0, gLambda, 0.01))

    # Run experiment
    print("Running Experiment...")

    # Sample with replacement for bootstrapping
    indices = np.random.randint(contexts_test.shape[0], size=contexts_test.shape[0])

    regrets = run(bandits, contexts_test[indices, :], posthocs[indices, :], losses_test[indices, :])

    # Plot Cumulative Regret
    labels = []
    for bandit in bandits:
        labels.append(bandit.label)

    title = "Augmented Contextual Bandit Regret"
    #plot(title, regrets, labels)

    # Save Regret Data
    np.savez("regrets_{0}_{1}_{2}_{3}.npz".format(T, K, dF, dG),
        regrets = regrets)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: main.py <dF> <dG>")
        sys.exit(-1)
    main(int(sys.argv[1]), int(sys.argv[2]))
