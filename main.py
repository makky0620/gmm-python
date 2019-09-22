import numpy as np
import numpy.random as rd
import scipy.stats as st
import matplotlib.pyplot as plt

from model import GMM

if __name__ == '__main__':
    seed = 77
    n_dim = 2
    n = [200, 150, 150]  # 各データのやつ
    mu_true = np.asanyarray(
        [[0.2, 0.5],
         [1.2, 0.5],
         [2.0, 0.5]])

    n_dim = mu_true.shape[1]

    sigma_true = np.asanyarray(
        [[[0.1, 0.085], [0.085, 0.1]],
         [[0.1, -0.085], [-0.085, 0.1]],
         [[0.1, 0.085], [0.085, 0.1]]
         ])
    c = ['r', 'g', 'b']

    rd.seed(seed)
    org_data = None  # np.empty((np.sum(n), 3))
    for i in range(3):
        if org_data is None:
            org_data = np.c_[
                st.multivariate_normal.rvs(mean=mu_true[i], cov=sigma_true[i], size=n[i]), np.ones(n[i]) * i]
        else:
            org_data = np.r_[org_data, np.c_[st.multivariate_normal.rvs(mean=mu_true[i],
                                                                        cov=sigma_true[i],
                                                                        size=n[i]), np.ones(n[i]) * i]]
    # drop true cluster label
    data = org_data[:, 0:2].copy()

    model = GMM(3)
    gamma = model.fit(data)

    for i in range(len(data)):
        plt.scatter(data[i, 0], data[i, 1], s=30, c=gamma[i], alpha=0.5, marker="+")

    plt.show()
