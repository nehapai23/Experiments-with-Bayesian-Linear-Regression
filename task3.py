import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA
import time

def calculate_w(l, phi, t, N_rows, M_cols):
    LI = np.eye(M_cols) * l
    innerPrdt = (LI + np.matmul(np.transpose(phi),phi))
    w = np.matmul(inv(innerPrdt),np.matmul(np.transpose(phi),t))
    return w

def avg(lst):
    return sum(lst)/len(lst)

def model_selection_using_cross_validation(train, trainR, test, testR, dataset_name):
    start_time = time.time()
    len_train = np.shape(train)[0]
    step = len_train//10 
    mse_for_lambdas = []    
    for l in range(0,151):
        mse_folds = []
        for i in range(0,len_train, step):
            current_train = np.delete(train,slice(i,i+step),0)
            current_trainR = np.delete(trainR,slice(i,i+step),0)
            current_N_rows = np.shape(current_train)[0]
            current_M_features_train = np.shape(current_train)[1]
            current_test_rows = step
            w = calculate_w(l,current_train,current_trainR,current_N_rows,current_M_features_train)
            mse_folds.append(sum((np.matmul(train[i:i+step],w) - trainR[i:i+step]) ** 2) / current_test_rows)
        mse_for_lambdas.append(avg(mse_folds))
    
    print("Dataset: ", dataset_name)
    print("\n--MODEL SELECTION USING CROSS VALIDATION--")
    print("Lambda: ", str(mse_for_lambdas.index(min(mse_for_lambdas))))

    M_features_train = np.shape(train)[1]
    N_rows_test = np.shape(test)[0]
    
    w = calculate_w(mse_for_lambdas.index(min(mse_for_lambdas)),train,trainR,len_train,M_features_train)
    mse = (sum((np.matmul(test,w) - testR) ** 2) / N_rows_test)
    print("MSE : ",mse)
    print("--- %s seconds ---" % (time.time() - start_time))


def bayesian_model_selection(train, trainR, test, testR, dataset_name):
    start_time = time.time()
    alpha = 2.34
    beta = 3.22
    prev_alpha = 0
    prev_beta = 0
    N_rows_train = np.shape(train)[0]
    M_features_train = np.shape(train)[1]
    N_rows_test = np.shape(test)[0]
    i = 0
    while abs(prev_alpha-alpha) > 0.0001 and abs(prev_beta-beta) > 0.0001:
        prev_alpha = alpha
        prev_beta = beta
        eigen_phiT_phi = LA.eigvals(np.matmul(np.transpose(train),train))
        Sn_inv = alpha * np.eye(M_features_train) + beta * np.matmul(np.transpose(train),train)
        Mn = beta * np.matmul(inv(Sn_inv),np.matmul(np.transpose(train),trainR))
        lamda = beta * eigen_phiT_phi
        alpha_lamda = LA.eigvals(Sn_inv)
        gamma = sum(lamda/alpha_lamda)
        alpha = gamma / np.matmul(np.transpose(Mn),Mn)
        beta = 1/(sum((trainR - np.matmul(train,Mn)) ** 2) / (N_rows_train-gamma))
        i+=1

    w = calculate_w(alpha/beta,train,trainR,N_rows_train,M_features_train)
    mse = (sum((np.matmul(test,w) - testR) ** 2) / N_rows_test)
    print("\n--BAYESIAN MODEL SELECTION--")
    print("Alpha: ",alpha)
    print("Beta:", beta)
    print("Lambda: ",alpha/beta)
    print("No. of iterations: ",i)
    print("Test Set MSE: ",mse)
    print("--- %s seconds ---" % (time.time() - start_time))
