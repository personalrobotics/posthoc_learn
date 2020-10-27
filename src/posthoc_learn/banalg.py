#!/usr/bin/python3
from __future__ import print_function

import numpy as np

class HardConstraint(object):
    """
    Bandit with Hard Constraint
    Optimizing (f - l)^2 + (g - l)^2 s.t. f=g
    """
    def __init__(self, T, n, dF, dG, lambF=1E-2, lambG=0):
        self.lambF = lambF # F L2 Regularization
        self.lambG = lambG # G L2 Regularization
        self.n = n
        self.T = T
        self.dF = dF
        self.dG = dG

        assert (lambF != 0) or (lambG != 0)

        if lambF == 0:
            self.label = "Post Hoc Only"
            self.usePosthoc = 2
        elif lambG != 0:
            self.label = "Post Hoc"
            self.usePosthoc = 1
        else:
            self.label = "Normal"
            self.usePosthoc = 0

        self.reinit()

    def reinit(self):
        # Final Linear Model
        self.phiF = np.zeros((self.n, self.dF))

        # F Linear Regression Params
        self.AF = np.array([np.eye(self.dF) for i in range(self.n)]) * self.lambF
        self.AFfull = np.eye(self.dF)
        self.bF = np.zeros((self.n, self.dF))

        # G Linear Regression Params
        self.phiG = np.zeros((self.n, self.dG))
        self.AG = np.array([np.eye(self.dG) for i in range(self.n)]) * self.lambG
        self.AGfull = np.eye(self.dG)
        self.bG = np.zeros((self.n, self.dG))
        self.AFG = np.zeros((self.dF, self.dG))

    # Default to Random Choice
    def choice(self, t, context):
        return (t % self.n)

    def update(self, context, arm, loss, posthoc):
        # Record Context
        self.AF[arm, :] += np.outer(context, context)
        self.AFfull += np.outer(context, context)
        self.bF[arm, :] += loss * context

        # Record Posthoc
        self.AG[arm, :] += np.outer(posthoc, posthoc)
        self.AGfull += np.outer(posthoc, posthoc)
        self.bG[arm, :] += loss * posthoc

        # Cross Parameters
        XTP = np.outer(posthoc, context)
        assert XTP.T.shape == (self.dF, self.dG)
        self.AFG += XTP.T

        # Linear Regression
        if self.usePosthoc == 2:
            # Post Hoc Only
            self.AFinv = np.linalg.inv(self.AFfull)
            self.phiG[arm, :] = np.linalg.solve(self.AG[arm, :], self.bG[arm, :])
            for i in range(self.n):
                self.phiF[i, :] = self.AFinv @ self.AFG @ self.phiG[i, :]
            
        elif self.usePosthoc == 1:
            # Post Hoc Regression
            self.AGinv = np.linalg.inv(self.AGfull)
            for i in range(self.n):
                A = self.AF[i, :] + self.AFG @ self.AGinv @ self.AG[i, :] @ self.AGinv @ self.AFG.T
                B = self.bF[i, :] + self.AFG @ self.AGinv @ self.bG[i, :]
                self.phiF[i, :] = np.linalg.solve(A, B)
        else:
            
            # Normal Ridge Regression
            self.phiF[arm, :] = np.linalg.solve(self.AF[arm, :], self.bF[arm, :])
            

    def report_mse(self, test_contexts, test_losses):
        mse = 0.0
        for i in range(test_contexts.shape[0]):
            ans = self.phiF @ test_contexts[i, :].reshape((self.dF, 1))
            assert ans.size == self.n
            mse += np.linalg.norm(ans - test_losses[i, :])
        return mse / float(test_contexts.shape[0])

class Greedy(HardConstraint):
    def __init__(self, T, n, dF, dG, lambF=1E-2, lambG=0):
        super(Greedy, self).__init__(T, n, dF, dG, lambF, lambG)
        self.label = self.label = ", Greedy"

    def choice(self, t, context):
        return np.argmin(np.dot(context, self.phiF.T))

class EpsilonGreedy(HardConstraint):
    def __init__(self, T, n, dF, dG, lambF=1E-2, lambG=0, epsilon=0.1):
        super(EpsilonGreedy, self).__init__(T, n, dF, dG, lambF, lambG)
        self.epsilon = epsilon
        self.label = self.label + ", {0}-Greedy".format(epsilon)

    def choice(self, t, context):
        p_dist = np.dot(context, self.phiF.T)

        argmin_k = np.argmin(p_dist)
        prob_vec = np.zeros(self.n) + (self.epsilon) / (self.n)
        prob_vec[argmin_k] += (1.0 - self.epsilon)

        assert np.isclose(np.sum(prob_vec), 1.0)

        return np.random.choice(np.arange(self.n), None, p=prob_vec)

class LinUCB(HardConstraint):
    def __init__(self, T, n, dF, dG, lambF=1E-2, lambG=0, alpha=0.01):
        super(LinUCB, self).__init__(T, n, dF, dG, lambF, lambG)
        self.alpha = alpha
        self.label = self.label + ", LinUCB"

        self.Ainv = np.array([np.eye(self.dF) for i in range(self.n)]) * self.lambF

    def choice(self, t, context):
        con_shape = context.shape[0]
        lcb = np.dot(self.phiF, context) - (self.alpha) * np.sqrt(np.dot(np.dot(context.T, self.Ainv), context))

        return np.argmin(lcb)

    def update(self, context, arm, loss, posthoc):
        super(LinUCB, self).update(context, arm, loss, posthoc)

        if self.usePosthoc == 2:
            # TODO, figure this out, low priority
            raise NotImplementedError
        elif self.usePosthoc == 1:
            # Post hoc
            for i in range(self.n):
                A = self.AF[i, :] + self.AFG @ self.AGinv @ self.AG[i, :] @ self.AGinv @ self.AFG.T
                self.Ainv[i] = 2.0 * np.linalg.inv(A)
        else:
            # Normal Linear Regression
            self.Ainv[arm] = np.linalg.inv(self.AF[arm])
