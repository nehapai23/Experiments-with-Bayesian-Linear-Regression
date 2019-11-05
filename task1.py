import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt     

def calculate_w(l, phi, t, N_rows, M_cols):
    LI = np.eye(M_cols) * l
    innerPrdt = (LI + np.matmul(np.transpose(phi),phi))
    w = np.matmul(inv(innerPrdt),np.matmul(np.transpose(phi),t))
    return w

def regularization(train, trainR, test, testR, dataset_name):
    #train = [[1,0,5],[2,1,6],[3,4,0]]
    #trainR = [[1],[2],[3]]
    N_rows_train = np.shape(train)[0]
    M_features_train = np.shape(train)[1]
    N_rows_test = np.shape(test)[0]
    mse_train = []
    mse_test = []
    lambda_values = [x for x in range(0,151)]
    for l in lambda_values:
        w = calculate_w(l, train, trainR, N_rows_train, M_features_train)
        mse_train.append(sum((np.matmul(train,w) - trainR) ** 2) / N_rows_train)
        mse_test.append(sum((np.matmul(test,w) - testR) ** 2) / N_rows_test)
    
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

    plt.rc('font', **font) 

    plt.plot(lambda_values,mse_train, label = "MSE(Training set)", linewidth=2, markersize=12)
    plt.plot(lambda_values,mse_test, label = "MSE(Test set)", linewidth=2, markersize=12)
    plt.xlabel('Lambda',fontsize=30) 
    plt.ylabel('MSE',fontsize=30) 
    plt.title('Regularization, Dataset: '+dataset_name,fontsize=40) 
    plt.legend(fontsize=30) 
    plt.show() 

    print(min(mse_test), mse_test.index(min(mse_test)))
