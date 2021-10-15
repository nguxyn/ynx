
import DataMaker
import numpy as np
epsilon = 1e-9
import matplotlib.pyplot as plt

if __name__=='__main__':
    '''# 最小二乘法求解析解(无正则项) lamda=0
    begin, end, exponent,lamda,size = 0, 1, 9, np.exp(-10),20
    train_x, train_y, x,y=DataMaker.generate_data(exponent,size,begin,end)
    poly = DataMaker.lsm(train_x, train_y,lamda,exponent)
    DataMaker.plt_show(x,y,poly)'''
     # 最小二乘法求解析解(有正则项)
    #梯度下降法求优化解
    begin, end, exponent, lamda, size = 0, 1, 9, 0, 10
    alpha=0.01
    train_x, train_y, x, y = DataMaker.generate_data(exponent, size, begin, end)
    poly,i=DataMaker.gradient_descent(train_x, train_y,lamda,alpha,epsilon,exponent,times=10000000)
    print(i)
    DataMaker.plt_show(x, y, poly)
    # 共轭梯度法求优化解
    '''begin, end, exponent, lamda, size = 0, 1, 50, np.exp(-10),50
    train_x, train_y, x, y = DataMaker.generate_data(exponent, size, begin, end)
    poly=DataMaker.conjugate_descent(train_x, train_y,lamda,epsilon,exponent)
    DataMaker.plt_show(x, y, poly)
    begin, end, exponent, lamda, size = 0, 1, 9, 0, 10
    train_x, train_y, x, y = DataMaker.generate_data(exponent, size, begin, end)
    test_x, test_y,X,Y=DataMaker.generate_data(exponent,20,0, 1)
    DataMaker.Rmse(train_x,train_y,10,test_x,test_y,exponent,20)
    begin, end, exponent, lamda, size = 0, 1, 9, np.exp(-10), 10
    train_x, train_y, x, y = DataMaker.generate_data(exponent, size, begin, end)
    poly = DataMaker.lsm(train_x, train_y, 0, exponent)
    punish_poly = DataMaker.lsm(train_x, train_y, lamda, exponent)
    DataMaker.comparsion_show(x,y,poly,punish_poly)'''

















