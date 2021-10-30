import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal
from itertools import permutations

'''
使用高斯分布产生一组数据，共k个高斯分布，每个分布产生n个数据
@:param mu_list 均值列表
@:param sigma_list 方差列表
@:param dimension 属性个数
'''
def generate_data(k, n, dim, mu_list, sigma_list):

 X = np.zeros((n * k, dim + 1))
 for i in range(k):
  X[i * n: (i + 1) * n, :dim] = np.random.multivariate_normal(mu_list[i], sigma_list[i], size=n)
  X[i * n: (i + 1) * n, dim: dim + 1] = i
 #print(X)
 return X
#
def k_means(X, k, dim,epsilon):
 center = np.zeros((k, dim)) #聚类中心矩阵 k*属性个数
 times=0
 size=X.shape[0]

 #随机选定k个中心点
 for i in range(k):
  center[i, :] = X[np.random.randint(0, high=X.shape[0]), :-1]
 while True:
   times+=1
   dis = np.zeros(k)
   for i in range(size):
#根据中心重新给每个点贴分类标签
    for j in range(k):
     dis[j] = np.linalg.norm(X[i, :-1] - center[j, :])
    X[i, -1] = np.argmin(dis)#返回指定数组的下标 重新给数据打标签
# 根据每个点新的标签计算它的中心
   new_center = np.zeros((k, X.shape[1]-1))
   count = np.zeros(k)
   for i in range(X.shape[0]):
    new_center[int(X[i, -1]), :] += X[i, :-1]  # 对每个类的所有点坐标求和
    count[int(X[i, -1])] += 1
   for i in range(k):
    new_center[i, :] = new_center[i, :] / count[i]  # 对每个类的所有点坐标求平均值
   if np.linalg.norm(new_center - center) < epsilon:  # 用差值的二范数表示精度
    break
   else:
    center = new_center
   return X, center, times
def GMM(X, k, dim,epsilon):
    train_x= X[:, :-1]
    pi_list = np.ones(k) * (1.0 / k) #随机初始化参数pi pi的和为1
    sigma_list = np.array([0.1 * np.eye(train_x.shape[1])] * k)
    # 随机选第1个初始点，依次选择与当前mu中样本点距离最大的点作为初始簇中心点
    mu_list = [train_x[np.random.randint(0, k) + 1]]
    for times in range(k - 1):
        temp=[]
        for i in range(train_x.shape[0]):
            temp.append(np.sum([np.linalg.norm(train_x[i] - mu_list[j]) for j in range(len(mu_list))]))
        mu_list.append(train_x[np.argmax(temp)])
    mu_list = np.array(mu_list)
    old_log = likelihood(train_x, mu_list, sigma_list, pi_list)
    while True:
        gamma_z = E_step(train_x, mu_list, sigma_list, pi_list)
        mu_list, sigma_list, pi_list = M_step(train_x, mu_list, gamma_z)
        new_log=likelihood(train_x, mu_list, sigma_list, pi_list)
        if old_log < new_log & (new_log - old_log) < epsilon:
            break
        old_log=new_log
        # 计算标签
    for i in range(X.shape[0]):
        X[i, -1] = np.argmax(gamma_z[i, :])
    return X


def E_step(x, mu_list, sigma_list, pi_list):
    """
        e步，求每个样本由各个混合高斯成分生成的后验概率
    """
    k = mu_list.shape[0]
    gamma_z=np.zeros((x.shape[0], k))
    for i in range(x.shape[0]):
        sum=0
        pi= np.zeros(k)

        for j in range(k):
            pi[j]=pi_list[j] * multivariate_normal.pdf(x[i], mean=mu_list[j], cov=sigma_list[j])
            sum+=pi[j]

        for j in range(k):
            gamma_z[i, j] = pi[j] / sum

    return gamma_z

def M_step(x, mu_list, gamma_z):
    """
    m步，根据公式更新参数
    """
    k = mu_list.shape[0]
    dim = x.shape[1]
    n = x.shape[0]
    mu_list_new = np.zeros(mu_list.shape)
    sigma_list_new = np.zeros((k, dim, dim))
    pi_new = np.zeros(k)
    for j in range(k):
      n_j = np.sum(gamma_z[:, j])
      pi_new[j] = n_j / n  # 计算新的pi
      gamma = gamma_z[:, j]
      gamma = gamma.reshape(n, 1)
      mu_list_new[j, :] = (gamma.T @ x) / n_j  # 计算新的mu
      sigma_list_new[j] = ((x - mu_list[j]).T @ np.multiply((x - mu_list[j]), gamma)) / n_j  # 计算新的sigma
    return mu_list_new, sigma_list_new, pi_new
# 计算极大似然
def likelihood(x, mu_list, sigma_list, pi_list):
    log_sum= 0
    k=mu_list.shape[0]
    for i in range(x.shape[0]):
        sum=0
        for j in range(k):
            sum+= pi_list[j] * multivariate_normal.pdf(x[j], mean=mu_list[j], cov=sigma_list[j])
    return log_sum

#计算准确率
def accuracy(real_lable, train_lable, k):
    real_list=real_lable.tolist()
    train_list=train_lable.tolist()
    counts=0
    size=len(real_list)
    for i in range(size):
        if real_list[i]==train_list[i]:
            counts+=1

    accuracy=counts*1.0/size
    return accuracy

def show(X, center, title):
 plt.style.use('Solarize_Light2')
 plt.scatter(X[:, 0], X[:, 1], c=X[:, 2], marker='.', s=25, cmap="Dark2_r")
 if not center is None:
  plt.scatter(center[:, 0], center[:, 1], c='r', marker='x', s=250)
 if not title is None:
  plt.title(title)
 plt.show()



   
