import numpy as np
import numpy.random as rd
import scipy.stats as st


class GMM:
    def __init__(self, n_components=1, max_iter=100):
        self.n_components = n_components
        self.max_iter = max_iter
        self.mu = None
        self.pi = None
        self.sigma = None

    def fit(self, X):
        # initialize pi
        self.pi = np.zeros(self.n_components)
        for k in range(self.n_components):
            if k == self.n_components - 1:
                self.pi[k] = 1 - np.sum(self.pi)
            else:
                self.pi[k] = 1 / self.n_components
        print('init pi:', self.pi)

        # initialize mu
        max_x, min_x = np.max(X[:, 0]), np.min(X[:, 0])
        max_y, min_y = np.max(X[:, 1]), np.min(X[:, 1])
        self.mu = np.c_[rd.uniform(low=min_x, high=max_x, size=self.n_components), rd.uniform(low=min_y, high=max_y,
                                                                                              size=self.n_components)]
        print('init mu:\n', self.mu)

        # initialize sigma
        self.sigma = np.asanyarray(
            [[[0.1, 0], [0, 0.1]],
             [[0.1, 0], [0, 0.1]],
             [[0.1, 0], [0, 0.1]]])

        likelihood = self._calc_likelihood(X)

        for step in range(self.max_iter):
            # E step
            # 負担率(gamma)の計算
            likelihood = self._calc_likelihood(X)
            gamma = (likelihood.T / np.sum(likelihood, axis=1)).T
            N_k = [np.sum(gamma[:, k]) for k in range(self.n_components)]
            N = np.sum(X.shape[0])  # TODO
            D = X.shape[1]

            # M step
            # piの計算
            self.pi = N_k / N

            # muの計算
            tmp_mu = np.zeros((self.n_components, D))
            for k in range(self.n_components):
                for i in range(len(X)):
                    tmp_mu[k] += gamma[i, k] * X[i]
                tmp_mu[k] = tmp_mu[k] / N_k[k]

            mu_prev = self.mu.copy()
            self.mu = tmp_mu

            # sigmaの計算
            tmp_sigma = np.zeros((self.n_components, D, D))
            for k in range(self.n_components):
                tmp_sigma[k] = np.zeros((D, D))
                for i in range(N):
                    tmp = np.asanyarray(X[i] - self.mu[k])[:, np.newaxis]
                    tmp_sigma[k] += gamma[i, k] * np.dot(tmp, tmp.T)
                tmp_sigma[k] = tmp_sigma[k] / N_k[k]

            sigma = tmp_sigma.copy()
            # calculate likelihood
            prev_likelihood = likelihood
            likelihood = self._calc_likelihood(X)
            prev_sum_log_likelihood = np.sum(np.log(prev_likelihood))
            sum_log_likelihood = np.sum(np.log(likelihood))
            diff = prev_sum_log_likelihood - sum_log_likelihood

            print('diff:', diff)

            if np.abs(diff) < 0.0001:
                print('likelihood is converged.')
                return gamma

    def _calc_likelihood(self, data):
        likelihood = np.zeros((data.shape[0], 3))
        for k in range(self.n_components):
            # 多次元正規分布
            likelihood[:, k] = [self.pi[k] * st.multivariate_normal.pdf(d, self.mu[k], self.sigma[k]) for d in data]
        return likelihood
