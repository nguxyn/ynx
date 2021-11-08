import DataMaker

if __name__ == '__main__':
    size = 100
    # mu = [1, 2, 3]
    # sigma = [[0.1, 0, 0], [0, 10, 0], [0, 0, 10]]
    #
    # train_x=DataMaker.generate_data(mu,sigma,size)
    # DataMaker.show_3D(train_x)

    mu = [2, 2]
    sigma = [[10, 0], [0, 0.1]]
    train_x = DataMaker.generate_data(mu, sigma, size)
    DataMaker.show_2D(train_x)
    # , 10, 5, 3, 1
    # k_list = [50]
    # DataMaker.face_process('photo', k_list)






