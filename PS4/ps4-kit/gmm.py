import numpy as np
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
        # Posterior distribution container
        p = np.zeros((X.shape[0], self.K))

        # Perform E-step
        for k in range(self.K):
            # Generate posterior distribution
            p[:, k] = self.mixing_coeff[k] * multivariate_normal.pdf(X, self.mus[k, :], self.covariances[k])

        # Normalize posterior distribution
        gamma_sum = np.sum(p, axis=1)[:, None]
        p /= gamma_sum

        return p

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
        # fig, ax3 = plt.subplots()
        # ax3.scatter(self.initmean[:, 0], self.initmean[:, 1], c='b')
        # ax3.scatter(self.mus[:, 0], self.mus[:, 1], c="r")
        # plt.title('test')
        # ax3.set_xlabel('$\mu_1$')
        # ax3.set_ylabel('$\mu_2$')
        # plt.show()

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
        k_terms = np.zeros((self.K, ))
        m_terms = np.zeros((N, ))

        # Compute log-likelihood
        for i in range(N):
            for k in range(self.K):
                exp_term = np.exp(-0.5 * (X[i] - self.mus[k]).T @
                                  np.linalg.inv(self.covariances[k]) @ (X[i] - self.mus[k]))
                num = self.mixing_coeff[k] * exp_term
                den = ((2 * np.pi) ** (self.d / 2)) * np.sqrt(np.linalg.det(self.covariances[k]))
                k_terms[k] = num / den
            m_terms[i] = np.log(np.sum(k_terms))

        # Normalize
        llh = np.sum(m_terms) / N
        return llh
