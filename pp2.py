#!/usr/local/bin/python3
import numpy as np 
import sys

def read_csv(dataset_name):
    train = np.loadtxt("pp2data/train-"+dataset_name+".csv", delimiter=",") 
    trainR = np.loadtxt("pp2data/trainR-"+dataset_name+".csv", delimiter=",") 
    test = np.loadtxt("pp2data/test-"+dataset_name+".csv", delimiter=",") 
    testR = np.loadtxt("pp2data/testR-"+dataset_name+".csv", delimiter=",") 
    return train, trainR, test, testR


if __name__ == "__main__":
    #filenames = ["wine", "crime", "1000-100", "100-100", "100-10"]
    #tasks = ["task1","task2","task3"]
    if(len(sys.argv) != 3):
        raise Exception('Error: expected 2 command line arguments!')

    dataset_name = sys.argv[1]
    task = sys.argv[2]
    train, trainR, test, testR = read_csv(dataset_name)
    if task == "task1":
        from task1 import regularization
        regularization(train, trainR, test, testR, dataset_name)
    if task == "task2":
        from task2 import learning_curves
        learning_curves(train, trainR, test, testR, dataset_name) 
    if task == "task3":
        from task3 import model_selection_using_cross_validation, bayesian_model_selection
        model_selection_using_cross_validation(train, trainR, test, testR, dataset_name)
        bayesian_model_selection(train, trainR, test, testR, dataset_name)
    print("\n\n..Done!")
