import math

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from PIL import Image


def pca(x,k):
    n = x.shape[0]
    mu = np.sum(x, axis=0) / n
    x_center = x - mu
    cov = (x_center.T @ x_center) / n

    values, vectors = np.linalg.eig(cov)
    index = np.argsort(values)  # 从小到大排序后的下标序列
    vectors = vectors[:, index[:-(k + 1):-1]].T  # 把序列逆向排列然后取前k个，转为行向量
    vectors=np.real(vectors)#为了防止当维度过高时产生复数
    return x_center, mu, vectors


#生成高斯分布数据

def generate_data(mu,sigma,size):
    """
        生成高斯分布数据
        @:param mu
        @:param sigma
    """
    x = np.random.multivariate_normal(mu, sigma, size)
    return x
#二维降到一维
def show_2D(x):
    x_center, mu, vectors = pca(x, 1)

    x_pca = x_center @ vectors.T @ vectors + mu
    # show
    plt.style.use('seaborn')
    plt.scatter(x[:, 0], x[:, 1], c="b", label="Origin Data")
    plt.scatter(x_pca[:, 0], x_pca[:, 1], c='r', label='PCA Data')
    plt.plot(x_pca[:, 0], x_pca[:, 1], c='k', label='vector', alpha=0.5)
    plt.legend(loc='best')
    plt.ylim(np.min(x[:, 1]) - 1, np.max(x[:, 1]) + 1)
    plt.show()
    plt.style.use('default')
#将三维降到二维
def show_3D(x):
    x_center, mu, vectors = pca(x, 2)
    x_pca = x_center @ vectors.T @ vectors + mu
    # show
    plt.style.use('seaborn')
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], c="b", label='Origin Data')
    ax.scatter(x_pca[:, 0], x_pca[:, 1], x_pca[:, 2], c='r', label='PCA Data')
    ax.plot_trisurf(x_pca[:, 0], x_pca[:, 1], x_pca[:, 2], color='k', alpha=0.3)
    ax.legend(loc='best')
    plt.show()
    plt.style.use('default')
#读取指定目录下的照片
def face_process(path,k_list):
    file_list = os.listdir(path)
    x_list = []
    for file in file_list:
        file_path = os.path.join(path, file)
        pic = Image.open(file_path).convert('L')  # 读入图片，并将三通道转换为灰度图
        x_list.append(np.asarray(pic))
    for x in x_list:
        x_pca_list = []
        x_psnr_list = []
        for k in k_list:
            x_centerlized, mu, vectors = pca(x, k) # PCA降维'
            #print(x_centerlized)
            #print(vectors)
            x_pca = x_centerlized @ vectors.T @ vectors + mu # 重建数据
            # x_pca = np.real(x_pca)
            #print(x_pca)
            x_pca_list.append(x_pca)
            x_psnr_list.append(psnr(x, x_pca))
        show_pic(x, x_pca_list, k_list, x_psnr_list)
#计算信噪比
def psnr(source, target):
    """
    计算峰值信噪比
    """
    rmse = np.sqrt(np.mean((source - target) ** 2))
    return 20 * np.log10(255.0 / rmse)
def show_pic(x, x_pca_list, k_list, x_psnr_list):
    #plt.figure(figsize=(8,5), frameon=False)
    size = math.ceil((len(k_list) + 1) / 2)
    print(size)
    plt.subplot(2, size, 1)
    plt.title('Real Image')
    #print(x)
    plt.imshow(x)
    plt.axis('off')  # 去掉坐标轴
    #print(x_pca_list[1])

    for i in range(len(k_list)):
        plt.subplot(2, size, i+2)

        plt.title('k = ' + str(k_list[i]) + ', PSNR = ' + '{:.2f}'.format(x_psnr_list[i]))
        plt.imshow(x_pca_list[i])

        plt.axis('off')
    plt.show()

