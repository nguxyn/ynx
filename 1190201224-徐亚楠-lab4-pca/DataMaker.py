import math

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from PIL import Image
# size = (50, 50) # 由于较大的数据在求解特征值和特征向量时很慢，故统一压缩图像为size大小
import cv2

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
    plt.scatter(x_pca[:, 0].tolist(), x_pca[:, 1].tolist(), c=x_pca[:, 0].tolist(), cmap=plt.cm.gnuplot)

    plt.style.use('seaborn')
    plt.scatter(x[:, 0], x[:, 1], c="b", label="Origin Data")
    plt.scatter(x_pca[:, 0], x_pca[:, 1], c='r', label='PCA Data')
    plt.plot(x_pca[:, 0], x_pca[:, 1], c='k', label='vector', alpha=0.5)
    plt.legend(loc='best')
    plt.ylim(np.min(x[:, 1]) - 1, np.max(x[:, 1]) + 1)
    plt.show()
    # plt.style.use('default')
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
        img = plt.imread(file_path)
        plt.imshow(img)
        plt.axis('off')
        # plt.show()
        pic = Image.open(file_path).convert('L')  # 读入图片，并将三通道转换为灰度图

        x_list.append(np.asarray(pic))

    # n_samples, n_features = x_list.shape
    # print(data)
    for k in k_list:

        x_pca_list = []
        x_psnr_list = []
        for x in x_list:
            x_centerlized, mu, vectors = pca(x, k)  # PCA降维
            x_pca = x_centerlized @ vectors.T @ vectors + mu  # 重建数据
            x_pca_list.append(x_pca)
            x_psnr_list.append(psnr(x, x_pca))
        print(len(x_pca_list))
        show_pic(k, x_pca_list, x_list, x_psnr_list)
#计算信噪比
def psnr(source, target):
    """
    计算峰值信噪比
    """
    rmse = np.sqrt(np.mean((source - target) ** 2))
    return 20 * np.log10(255.0 / rmse)
def show_pic(k, x_pca_list, x_list, x_psnr_list):
    #plt.figure(figsize=(8,5), frameon=False)
    size = math.ceil((len(x_list) + 1) / 2)
    # print(size)
    plt.subplot(2, size, 1)
    plt.title('Real Image')
    #print(x)
    # plt.imshow(x)
    # plt.axis('off')  # 去掉坐标轴
    #print(x_pca_list[1])
    # print(len(x_list))
    # print(len(x_pca_list))
    for i in range(len(x_list)):
        plt.subplot(3,4,i+1)
        # print(x_pca_list[i])
        # plt.title('k = ' + str(x_pca_list[i]) + ', PSNR = ' + '{:.2f}'.format(x_psnr_list[i]))
        # print(i)
        plt.imshow(x_pca_list[i])
        print('图', i+1, '的信噪比: ', x_psnr_list[i])
        plt.axis('off')
    plt.show()

