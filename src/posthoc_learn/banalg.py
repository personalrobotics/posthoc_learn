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
        self.usePosthoc = (lambG != 0)

        if self.usePosthoc:
            self.label = "Post Hoc Hard Constraint"
        else:
            self.label = "Normal"

        # F Linear Regression Params
        self.phiF = np.zeros((n, dF))
        self.AF = np.array([np.eye(dF) for i in range(n)]) * lambF
        self.AFfull = np.eye(dF)
        self.bF = np.zeros((n, dF))

        # G Linear Regression Params
        self.phiG = np.zeros((n, dG))
        self.AG = np.array([np.eye(dG) for i in range(n)]) * lambG
        self.AGfull = np.eye(dG)
        self.bG = np.zeros((n, dG))
        self.AFG = np.zeros((dF, dG))

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
        if not self.usePosthoc:
            # Normal Ridge Regression
            self.phiF[arm, :] = np.linalg.solve(self.AF[arm, :], self.bF[arm, :])
        else:
            # Post Hoc Regression
            AGinv = np.linalg.inv(self.AGfull)
            A = self.AF[arm, :] + self.AFG @ AGinv @ self.AG[arm, :] @ AGinv @ self.AFG.T
            B = self.bF[arm, :] + self.AFG @ AGinv @ self.bG[arm, :]
            self.phiF[arm, :] = np.linalg.solve(A, B)
            

    def report_mse(self, test_contexts, test_losses):
        mse = 0.0
        for i in range(test_contexts.shape[0]):
            ans = self.phiF @ test_contexts[i, :].reshape((self.dF, 1))
            assert ans.size == self.n
            mse += np.linalg.norm(ans - test_losses[i, :])
        return mse / float(test_contexts.shape[0])


