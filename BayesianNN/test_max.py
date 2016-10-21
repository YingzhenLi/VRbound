
import math

import time

import numpy as np

import sys, pickle

from black_box_alphamax import fit_q

def get_test_error(X, y, i, dataset, K):
    # We fix the random seed

    np.random.seed(1)
    
    # We load the indexes of the training and test sets
    path = 'data/' + dataset + '/'
    index_train = np.loadtxt(path + "index_train_{}.txt".format(i))
    index_test = np.loadtxt(path + "index_test_{}.txt".format(i))
    # load training and test data
    X_train = X[ index_train.tolist(), ]
    y_train = y[ index_train.tolist() ]
    X_test = X[ index_test.tolist(), ]
    y_test = y[ index_test.tolist() ]

    # We normalize the features
    std_X_train = np.std(X_train, 0)
    std_X_train[ std_X_train == 0 ] = 1
    mean_X_train = np.mean(X_train, 0)
    X_train = (X_train - mean_X_train) / std_X_train
    X_test = (X_test - mean_X_train) / std_X_train
    mean_y_train = np.mean(y_train)
    std_y_train = np.std(y_train)
    y_train = (y_train - mean_y_train) / std_y_train

    y_train = np.array(y_train, ndmin = 2).reshape((-1, 1))
    y_test = np.array(y_test, ndmin = 2).reshape((-1, 1))

    learning_rate = 0.001; v_prior = 1.0
    print 'learning rate', learning_rate, 'v_prior', v_prior

    # We iterate the method 

    batch_size = 32
    epochs = 500
    alpha = 0.0#1 - float(batch_size) / float(X_train.shape[0])
    hidden_layer_size = 100

    start_time = time.time()
    w, v_prior, get_error_and_ll = fit_q(X_train, y_train, hidden_layer_size, 
        batch_size, epochs, K, alpha, learning_rate, v_prior)
    running_time = time.time() - start_time

    # We obtain the test RMSE and the test ll

    error, ll = get_error_and_ll(w, v_prior, X_test, y_test, K, mean_y_train, std_y_train)

    return -ll, error, running_time

# Write a function like this called 'main'
def main(dataset):

    # We load the data
    datapath = 'data/' + dataset + '/'
    data = np.loadtxt(datapath + 'data.txt')
    index_features = np.loadtxt(datapath + 'index_features.txt')
    index_target = np.loadtxt(datapath + 'index_target.txt')

    X = data[ : , index_features.tolist() ]
    y = data[ : , index_target.tolist() ]

    n_splits = 50 
    K = 100
    savepath = 'results/'
    for i in range(n_splits):
        print 'split', i+1
        neg_test_ll, test_error, running_time = get_test_error(X, y, i+1, dataset, K)
        with open(savepath + dataset + "_test_ll_max_k{}.txt".format(K), 'a') as f:
            f.write(repr(neg_test_ll) + '\n')
        with open(savepath + dataset + "_test_error_max_k{}.txt".format(K), 'a') as f:
            f.write(repr(test_error) + '\n')
        with open(savepath + dataset + "_test_time_max_k{}.txt".format(K), 'a') as f:
            f.write(repr(running_time) + '\n')

if __name__ == '__main__':
    dataset = str(sys.argv[1])
    main(dataset)
    
