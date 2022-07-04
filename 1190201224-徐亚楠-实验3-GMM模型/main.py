from copy import deepcopy

import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import DataMaker
from sklearn import mixture
if __name__ == '__main__':
    k = 3
    n = 200
    dim = 2
    epsilon=1e-6
    # mu_list = np.array([ [1,2], [1,3], [3,2],[3,4] ])
    # sigma_list = np.array([ [[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]],[[1,0],[0,1]] ])
    # mu_list = np.array([[2, 6], [8, 10], [8, 2]])
    # sigma_list = np.array([[[1, 0], [0, 3]], [[3, 0], [0, 2]], [[3, 0], [0, 3]]])
    # X = DataMaker.generate_data(k, n, dim, mu_list, sigma_list)
    # real_lable = deepcopy(X[:, -1])
    # # print(real_lable)
    # DataMaker.show(X, mu_list, title='The Real Distribution')
    # X, center, times=DataMaker.k_means(X,k,dim,epsilon)
    # DataMaker.show(X, center, title='The K-means result')
    # print('k-means迭代次数：', times)
    # accuracy =DataMaker.accuracy(real_lable, X[:, -1], k)
    # print(accuracy)
    # # # print(X[:, -1])
    # X, center,iterations=DataMaker.GMM(X,k,dim,epsilon)
    # DataMaker.show(X, center, title='The GMM result')
    # accuracy = DataMaker.accuracy(real_lable, X[:, -1], k)
    # print('GMM迭代次数：',iterations)
    # print(accuracy)
    # gmm = GaussianMixture(n_components = 4, max_iter = 200, covariance_type='diag', n_init = 500)
    #
    # tain=gmm.fit(X[:,:-1],y=None)
    # plt.style.use('Solarize_Light2')
    # plt.scatter(X[:, 0], X[:, 1], c=tain, marker='.', s=25, cmap="Dark2_r")
    # plt.show()





    X = DataMaker.uci_data('./iris.csv')
    real_lable = deepcopy(X[:, -1])

    X, center, times = DataMaker.k_means(X, k, dim, epsilon)

    print('k-means迭代次数：', times)
    accuracy = DataMaker.accuracy(real_lable, X[:, -1], k)
    print(accuracy)
    X, center, iterations = DataMaker.GMM(X, k, dim, epsilon)

    accuracy = DataMaker.accuracy(real_lable, X[:, -1], k)
    print('GMM迭代次数：', iterations)
    print(accuracy)





