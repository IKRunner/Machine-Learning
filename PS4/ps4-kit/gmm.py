import numpy as np
import scipy.linalg
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


class GaussianMixtureModel():
    def __init__(self, K, init_mean, covariances, mixing_coeff=None):
        """ Gaussian Mixture Model

        NOTE: You should NOT modify this constructor or the signature of any class functions

        Args:
            k (int): Number of mixture components

            init_mean (numpy.array):
                The initial mean parameter for the mixture components.
                It should have shape (k, d) where d is the dimension of the
                data

            covariances (numpy.array):
                The initial covariance parameter of the mixture components
                It should have shape (k, d, d)

            mixing_coef (numpy.array):
                The initial mixing coefficient. Default is None
                If provided, it should have shape (k,)

        """
        # Some housekeeping things to make sure the dimensions agree
        (num_k, d) = init_mean.shape
        assert (num_k == K)
        assert (init_mean.shape == (K, d))
        assert (covariances.shape == (K, d, d))

        self.K = K
        # If mixing coefficient is not specified, initialize to uniform distribution
        if mixing_coeff is None:
            mixing_coeff = np.ones((K,)) * 1. / K
        assert (mixing_coeff.shape == (K,))

        self.mixing_coeff = np.copy(mixing_coeff)
        self.initmean = np.copy(init_mean)
        self.mus = np.copy(init_mean)
        self.covariances = np.copy(covariances)
        self.d = d

    def fit(self, X, num_iters=1000, eps=1e-6):
        """ Learn the GMM via Expectation-Maximization

        NOTE: you should NOT modify this function

        Args:
            X (numpy.array): Input data matrix

            num_iters (int): Number of EM iterations

        Returns:
            None

        """
        (N, d) = X.shape
        assert (d == self.d)

        self.llh = []

        for it in range(num_iters):
            p = self.E_step(X)
            self.M_step(X, p)

            log_ll = self.compute_llh(X)
            # Early stopping
            self.llh.append(log_ll)
            if it > 0 and np.abs(log_ll - self.llh[-2]) < eps:
                self.it = it
                break

    def E_step(self, X):
        """ Do the estimation step of the EM algorithm

        Arg:
            X (numpy.array):
                The input feature matrix of shape (N, d)

        Returns
            p (numpy.array):
                The "membership" matrix of shape (N, k)
                Where p[i, j] is the "membership proportion" of instance
                `i` in mixture component `j`
                with `0 <= i <= N-1`
                     `0 <= j <= k-1`

        """
        N = X.shape[0]

        # Posterior distribution container
        p = np.zeros((N, self.K))

        # Perform E-step
        for i in range(N):
            for k in range(self.K):
                p[i, k] = self.mixing_coeff[k] * multivariate_normal.pdf(X[i], self.mus[k], self.covariances[k],
                                                                         allow_singular=False)
            p[i] = p[i] / np.sum(p[i])

        # Adjust ill-conditioned membership matrix
        return p + 0.000000001

    def M_step(self, X, p):
        """  Do the maximization step of the EM algorithm
        Update the mixing coefficients
        Update the mean and covariance matrix of each component

        Arg:
            X (numpy.array):
                The input feature matrix of shape (N, d)

            p (numpy.array):
                The "membership" matrix of shape (N, k)
                Where p[i, j] is the "membership proportion" of instance
                `i` in mixture component `j`
                with `0 <= i <= N-1`
                     `0 <= j <= k-1`

        Returns:
            None

        """
        N = X.shape[0]

        # Update parameters
        for k in range(self.K):
            m_k = np.sum(p[:, k])
            self.mixing_coeff[k] = m_k / N
            self.mus[k] = (p[:, k][None] @ X) / m_k
            self.covariances[k, :, :] = (p[:, k] * (X - self.mus[k]).T @ (X - self.mus[k])) / m_k
        return None

    def compute_llh(self, X):
        """ Compute the log likelihood under the GMM
        Arg:
            X (numpy.array): Input feature matrix of shape (N, d)

        Returns:
            llh (float): Log likelihood of the given data
                under the learned GMM

        """
        N = X.shape[0]

        # Compute log-likelihood
        outer = 0
        for i in range(N):
            inner = 0
            for k in range(self.K):
                inner += self.mixing_coeff[k] * multivariate_normal.pdf(X[i], self.mus[k], self.covariances[k],
                                                                        allow_singular=False)
            outer += np.log(inner)

        # Normalize
        llh = outer / N
        return llh
