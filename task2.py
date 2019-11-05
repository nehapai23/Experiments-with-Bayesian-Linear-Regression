import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt   
import random  

def calculate_w(l, phi, t, N_rows, M_cols):
    LI = np.eye(M_cols) * l
    innerPrdt = (LI + np.matmul(np.transpose(phi),phi))
    w = np.matmul(inv(innerPrdt),np.matmul(np.transpose(phi),t))
    return w

def avg(lst):
    return sum(lst)/len(lst)

def learning_curves(train, trainR, test, testR, dataset_name):
    l1 = 5
    l2 = 27
    l3 = 145
    mse_lambda1 = []
    mse_lambda2 = []
    mse_lambda3 = []
    N_rows_test = np.shape(test)[0]
    M_features_train = np.shape(train)[1]
    #training_set_sizes = [10,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000]
    training_set_sizes = [x for x in range(10,1001,15)]
    for t in training_set_sizes:
        N_rows_train = t
        w1,w2,w3 = [],[],[]
        for trial in range(0,10):
            train_sub, trainR_sub = zip(*random.sample(list(zip(train, trainR)), t))
            w1.append(calculate_w(l1, train_sub, trainR_sub, N_rows_train, M_features_train))
            w2.append(calculate_w(l2, train_sub, trainR_sub, N_rows_train, M_features_train))
            w3.append(calculate_w(l3, train_sub, trainR_sub, N_rows_train, M_features_train))
        mse_lambda1.append(sum((np.matmul(test,avg(w1)) - testR) ** 2) / N_rows_test)
        mse_lambda2.append(sum((np.matmul(test,avg(w2)) - testR) ** 2) / N_rows_test)
        mse_lambda3.append(sum((np.matmul(test,avg(w3)) - testR) ** 2) / N_rows_test)
    
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

    plt.rc('font', **font)
    #plt.subplot(3, 1, 1)
    plt.plot(training_set_sizes,mse_lambda1, label = "MSE(Lambda = "+str(l1)+")")
    plt.xlabel('Training Set Sizes',fontsize=30) 
    plt.ylabel('MSE',fontsize=30) 
    plt.title('Learning Curves, Dataset: '+dataset_name,fontsize=40) 
    plt.legend(fontsize=30) 
    plt.show()
    
    #plt.subplot(3, 1, 2)
    plt.plot(training_set_sizes,mse_lambda2, label = "MSE(Lambda = "+str(l2)+")")
    plt.xlabel('Training Set Sizes',fontsize=30) 
    plt.ylabel('MSE',fontsize=30) 
    plt.legend(fontsize=30) 
    plt.show() 
    
    #plt.subplot(3, 1, 3)
    plt.plot(training_set_sizes,mse_lambda3, label = "MSE(Lambda = "+str(l3)+")")
    plt.xlabel('Training Set Sizes',fontsize=30) 
    plt.ylabel('MSE',fontsize=30) 
    plt.legend(fontsize=30) 
    plt.show() 


