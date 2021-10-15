#根据所需数据集维数多项式阶数添加高斯噪声生成相应数据集
import numpy as np
from matplotlib import pyplot as plt
import random
# exponent:多项式最高次幂 size:数据集包含数据个数
def generate_data(exponent, size,begin, end):
    X = np.linspace(begin, end, size)  #
    Y = np.sin(2*X*np.pi)
    # 对输入数据加入gauss噪声
    # 定义gauss噪声的均值和方差
    mu = 0
    sigma = 0.12
    for i in range(X.size):
        X[i] += random.gauss(mu, sigma)
        Y[i] += random.gauss(mu, sigma)
    train_x = np.zeros((exponent + 1, size))  # 创建（n+1)*m矩阵 X
    train_y = Y.reshape(size, 1)  # 标签矩阵 m*1  Y
    train_x = np.vander(X, exponent + 1, increasing=True)# 此函数生成的是转置的范德蒙德行列式
    #print(train_x)
    return train_x, train_y, X, Y
#计算w
def cal_w(train_x, train_y, lamda,exponent):
    return np.linalg.inv(np.dot(train_x.T,train_x)+lamda*np.identity(exponent+1, dtype=float)).dot(train_x.T).dot(train_y)
 # 最小二乘法求解析解
def lsm(train_x, train_y, lamda,exponent):
     #lamda=0时不带正则项，lamda!=0时带正则项
    w=cal_w(train_x, train_y, lamda,exponent)
    poly_fitting = np.poly1d(w[::-1].reshape(train_x.shape[1])) #多项式函数
    return poly_fitting
#损失函数
def loss(train_x, train_y, w, lamda):
    loss = train_x.dot(w) - train_y
    loss = 1 / 2 * np.dot(loss.T, loss) + lamda / 2 * np.dot(w.T, w)
    return loss
#梯度下降
#alpha:步长 epsilon:精度 times:最高迭代次数
def gradient_descent(train_x, train_y,lamda,alpha,epsilon,exponent,times):
    #w = np.zeros((train_x.shape[1], 1))
    w = cal_w(train_x, train_y, lamda, exponent)
    new_loss = abs(loss(train_x, train_y, w, lamda))
    for i in range(times):
        old_loss = new_loss
        gradient_w = np.dot(train_x.T, train_x).dot(w) - np.dot(train_x.T, train_y) + lamda * w  # 损失函数对w求导即梯度
        old_w = w
        w -= gradient_w * alpha
        new_loss = abs(loss(train_x, train_y, w, lamda))
        gradient_w = np.dot(train_x.T, train_x).dot(w) - np.dot(train_x.T, train_y) + lamda * w
        if old_loss < new_loss:  # 不下降了，说明步长过大
            w = old_w
            alpha /= 2
        if (old_loss - new_loss < epsilon)&(np.linalg.norm(gradient_w)<=epsilon) :
            break
    poly_fitting = np.poly1d(w[::-1].reshape(train_x.shape[1]))  # 多项式函数

    return poly_fitting, i
#共轭梯度下降
def conjugate_descent(train_x, train_y,lamda,epsilon,exponent):
    # 把loss 记为Aw=b的形式，其中A = X^T * X + lamda，b = X^T * Y
    A=np.dot(train_x.T, train_x)+lamda*np.identity(exponent+1, dtype=float)
    b=np.dot(train_x.T,train_y)
    w = np.zeros((train_x.shape[1], 1))  # 初始化w为 n+1 * 1 的零阵
    r = b
    p = b
    i = 0
    while True:
        i = i + 1
        norm_2 = np.dot(r.T, r)
        a = norm_2 / np.dot(p.T, A).dot(p)
        w = w + a * p
        r = r - (a * A).dot(p)
        if r.T.dot(r) < epsilon:
            break
        b = np.dot(r.T, r) / norm_2
        p = r + b * p
    print(i)
    poly_fitting = np.poly1d(w[::-1].reshape(train_x.shape[1]))
    return  poly_fitting
def plt_show(x, y, poly_fit):

    plot1 = plt.plot(x, y, 'co', label='training_data')
    real_x = np.linspace(0,1)
    real_y = np.sin(real_x * 2 * np.pi)
    fit_y = poly_fit(real_x)
    plot2 = plt.plot(real_x, fit_y, 'b', label='fit_poly')
    plot3 = plt.plot(real_x, real_y, 'r', label='real_ploy')
    #选择最佳位置写图像标注
    plt.legend(loc=0)
    plt.show()
    print(poly_fit)
#利用均方根判断lamda合适取值 解析法
# test_x, test_y 验证集合 size是训练集大小  test_size是测试集大小
def Rmse(train_x, train_y,train_size,test_x,test_y,exponent,test_size):

    ln_lamda = np.linspace(-10,0,50)

    rms_train = np.zeros(50)
    rms_test = np.zeros(50)
    for i in range(0, 50):
        lamda = np.exp(ln_lamda[i])
        w = cal_w(train_x, train_y, lamda,exponent)
        Ew_train = loss(train_x, train_y,w,lamda)
        rms_train[i] = np.sqrt(2 * Ew_train / train_size)
        #w_test=cal_w(test_x, test_y,lamda,exponent)
        Ew_test=loss(test_x, test_y, w,lamda)
        rms_test[i] = np.sqrt(2 * Ew_test / test_size)
    train_plot = plt.plot(ln_lamda, rms_train, 'b', label='train')
    test_plot = plt.plot(ln_lamda, rms_test, 'r', label='test')
    # 横坐标是lamda
    plt.xlabel('lamda')
    plt.ylabel('Rms')
    plt.legend(loc=0)
    plt.show()
#有正则项与无正则项图像对比
def comparsion_show(x, y, poly_fit,punish_poly):

    plot1 = plt.plot(x, y, 'co', label='training_data')
    real_x = np.linspace(0,1)
    real_y = np.sin(real_x * 2 * np.pi)
    fit_y = poly_fit(real_x)
    plot2 = plt.plot(real_x, fit_y, 'b', label='fit_poly')
    plot3 = plt.plot(real_x, real_y, 'r', label='real_ploy')
    polt4=plt.plot(real_x,punish_poly(real_x),'k',label='punish_poly')
    #选择最佳位置写图像标注
    plt.legend(loc=0)
    plt.show()
    print(poly_fit)


