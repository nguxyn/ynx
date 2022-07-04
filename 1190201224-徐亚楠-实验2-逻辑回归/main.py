import numpy as np
import DataMaker

if __name__ == '__main__':
    mean_neg,mean_pos = [-0.7, -0.3],[0.3, 0.5]
    var = 0.2
    cov_xy = 0.2
    size_pos,size_neg, lamda,epsilon,alpha= 50,50,np.exp(-8),1e-6,0.01
    # train_x,train_y=DataMaker.generate_data(mean_neg, var, size_neg, mean_pos, var, size_pos,0.0)
    # coefficient, w = DataMaker.gradient_descent(train_x, train_y, lamda, alpha, epsilon,100000)
    # poly = np.poly1d(coefficient[::-1])
    # DataMaker.show_poly(train_x,train_y,poly)
    # test_x, test_y=DataMaker.generate_data(mean_neg, var, 200, mean_pos, var, 200,0.0)
    # accuracy = DataMaker.accuracy(test_x, test_y, w)
    # print(poly)
    # print('accuracy=',accuracy)
    # DataMaker.show_poly(test_x, test_y, poly)


    train_x,train_y,test_x,test_y=DataMaker.uci_data('haberman.data')
    coefficient, w = DataMaker.gradient_descent(train_x, train_y, lamda, alpha, epsilon, 100000)
    print(coefficient)
    accuracy=DataMaker.accuracy(test_x,test_y,w)
    print('accuracy=',accuracy)
    DataMaker.uci_show(train_x,train_y,coefficient)





