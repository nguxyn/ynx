from copy import deepcopy

import numpy as np

import DataMaker

if __name__ == '__main__':
    k = 3
    n = 200
    dim = 2
    epsilon=1e-5
    mu_list = np.array([[2, 6], [8, 10], [8, 2]])
    sigma_list = np.array([[[1, 0], [0, 3]], [[3, 0], [0, 2]], [[3, 0], [0, 3]]])
    X = DataMaker.generate_data(k, n, dim, mu_list, sigma_list)
    real_lable = deepcopy(X[:, -1])
    # print(real_lable)


    DataMaker.show(X, mu_list, title='The Real Distribution')
    X, center, times=DataMaker.k_means(X,k,dim,epsilon)
    DataMaker.show(X, center, title='The K-means result')
    accuracy =DataMaker.accuracy(real_lable, X[:, -1], k)
    print(accuracy)
    # print(X[:, -1])



