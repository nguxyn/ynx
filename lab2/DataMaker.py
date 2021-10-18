import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
'''
生成数据
@:param mu_neg 反例均值 
@:param mu_pos 正例均值

@:param cov_xy 
@:param var_neg 反例的方差
@:param 
'''
def generate_data(mu_neg, var_neg, size_neg, mu_pos, var_pos, size_pos, cov_xy=0.0):
    train_x = np.zeros((size_pos + size_neg, 2))
    train_y = np.zeros(size_pos + size_neg)
    train_x[:size_neg, :] = np.random.multivariate_normal(mu_neg, [[var_neg, cov_xy], [cov_xy, var_neg]], size=size_neg)
    train_x[size_neg:, :] = np.random.multivariate_normal(mu_pos, [[var_pos, cov_xy], [cov_xy, var_pos]], size=size_pos)
    train_y[size_neg:] = 1 #正例的标签
    return train_x, train_y

def sigmoid(x):
    if x.all()>0:
        return 1 / (1 + np.exp(-x))
    else:
        return (np.exp(-x))/(1 + np.exp(x))
'''
利用极大条件似然得到loss
在计算e的指数幂会溢出 所以一定要注意归一化处理

'''
def loss(train_x, train_y, w, lamda):
    size = train_x.shape[0]
    loss_sum=0
    W=np.zeros((size, 1))
    for i in range(size):
        W[i]=w @ train_x[i].T

    mean=np.mean(W)
    var=np.std(W)
    for i in range(size):
        W[i]=(W[i]-mean)/var

    for i in range(size):
        loss_sum += np.log(1 + np.e ** W[i])
    loss_mcle = train_y @ W - loss_sum
    return -loss_mcle / size

'''
 梯度下降
    
    @:papam alpha: 步长
    @:param epsilon: 精度，算法终止距离
    @:param times: 最大迭代次数
    @:param lamda:正则项
'''
def gradient_descent(train_x, train_y, lamda, alpha, epsilon, times):
    size = train_x.shape[0]
    dimension = train_x.shape[1] #train_x 的列数
    X = np.ones((size, dimension + 1))
    for i in range(dimension):
       X[:, i + 1] = train_x[:, i]
    w = np.ones((1, X.shape[1]))
    new_loss = loss(X, train_y, w, lamda)
    for i in range(times):
        old_loss = new_loss
        t = np.zeros((size, 1))
        for j in range(size):
            t[j] = w @ X[j].T
        gradient_w = - (train_y - sigmoid(t.T)) @ X / size
        old_w = w
        w = w - alpha * lamda * w - alpha * gradient_w
        new_loss = loss(X, train_y, w, lamda)
        if old_loss < new_loss:  # 不下降了，说明步长过大
            w = old_w
            alpha /= 2
            continue
        #np.linalg.norm(gradient_w)<=epsilon
        if abs(new_loss - old_loss)<epsilon:
            break
    print(i)
    w = w.reshape(dimension + 1)  # 得到的w
    coefficient = -(w / w[dimension])[0:dimension]  # 对w做归一化得到方程系数
    return coefficient, w
"""
    计算准确率
    @:param x: 测试数据
    @:param y: 测试数据标签
    @:param w: w
    """
def accuracy(x, y, w):
    size = x.shape[0]
    dimension = x.shape[1]
    correct_count = 0
    X = np.ones((size, dimension + 1))  # 构造X矩阵，第一维都设置成1，方便与w相乘
    X[:, 1:dimension + 1] = x
    for i in range(size):
        label = 0
        if w @ X[i].T >= 0:
            label = 1
        if label == y[i]:
            correct_count += 1
    accuracy = correct_count / size
    return accuracy

def show_poly(x, y, poly):

    plt.scatter(x[:, 0], x[:, 1], c=y, s=30, marker='o', cmap=plt.cm.Spectral)
    real_x = min(x[:, 0]) + (max(x[:, 0]) - min(x[:, 0])) * np.random.random(50)
    real_y = poly(real_x)
    plt.plot(real_x, real_y, 'r', label='classify_poly')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc=0)
    plt.show()

def uci_data(path):

    data_uci=np.loadtxt(path, dtype=np.float32)
    np.random.shuffle(data_uci) # 打乱原数据，以便分成训练集和测试集
    data_size=np.size(data_uci, axis=0) # 一共多少行
    train_data = data_uci[:int(0.4* data_size), :]
    test_data = data_uci[int(0.4 * data_size):, :] #训练集测试集 4:6
    dim = np.size(data_uci, axis=1) - 1  # 训练集样本维度 分离最后一列
    train_x = train_data[:, 0:dim]
    train_y = train_data[:, dim:dim + 1]
    train_size = np.size(train_x, axis=0)
    train_y = train_y.reshape(train_size)  # 矩阵转化为行向量
    test_x = test_data[:, 0:dim]
    test_y = test_data[:, dim:dim + 1]
    test_size = np.size(test_x, axis=0)
    test_y = test_y.reshape(test_size)  # 矩阵转化为行向量
    return train_x,train_y,test_x,test_y


# 三维的
def uci_show(train_x, train_y, poly):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(train_x[:, 0], train_x[:, 1], train_x[:, 2], c=train_y, cmap=plt.cm.Spectral)
    real_x = np.linspace(np.min(train_x[:,0])-20, np.max(train_x[:,0])+20, 255)
    real_y = np.linspace(np.min(train_x[:,0])-20, np.max(train_x[:,0])+20, 255)
    real_X, real_Y = np.meshgrid(real_x, real_y)
    real_z = poly[0] + poly[1] * real_X + poly[2] * real_Y
    ax.plot_surface(real_x, real_y, real_z, rstride=1, cstride=1)
    ax.legend(loc='best')
    plt.show()


