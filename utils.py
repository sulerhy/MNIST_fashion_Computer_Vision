"""
Copyright (C) Hoang Pham Duc - All Rights Reserved
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
Written by Hoang Pham Duc <phamduchoangeee@gmail.com>, May 2020
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats.stats as stats
from sklearn.metrics import mean_squared_error
from math import sqrt
import h5py


def get_labels_number():
    train_data = np.loadtxt("dataset/fashion-mnist_train.csv", delimiter=",", skiprows=1)
    Y_train = train_data[:, 0].astype(np.int32)
    labels = len(set(Y_train))
    return labels


def load_dataset():
    train_data = np.loadtxt("dataset/fashion-mnist_train.csv", delimiter=",", skiprows=1)
    np.random.shuffle(train_data)
    X_train = train_data[:, 1:].reshape([-1, 28, 28, 1])
    Y_train = train_data[:, 0].astype(np.int32)
    labels = len(set(Y_train))

    test_data = np.loadtxt("dataset/fashion-mnist_test.csv", delimiter=",", skiprows=1)
    np.random.shuffle(test_data)
    X_test = test_data[:, 1:].reshape([-1, 28, 28, 1])
    Y_test = test_data[:, 0].astype(np.int32)
    return labels, X_train, Y_train, X_test, Y_test


# save data as txt file
def savetxt(filename, data):
    np.savetxt(filename, data, delimiter=',', fmt='%3.5f')


def plot_compare(label, true_ground, predict, P_corr, RMSE):
    plt.scatter(true_ground, predict)
    plt.plot(true_ground, true_ground, 'r-')
    plt.title(label + '\n' + 'Pearson Correlation: ' + str(P_corr) + "| RMSE: " + str(RMSE))
    plt.xlabel('Ground Truth')
    plt.ylabel('Predict')
    plt.show()


def evaluate(ground_truth, predict):
    # print Pearson product-moment correlation coefficients
    R, _ = stats.pearsonr(ground_truth.flatten(), predict.flatten())
    R = round(R, 4)
    print('Pearson correlation coefficients: ' + str(R))
    # print ROOT MEAN SQUARED ERROR
    RMSE = sqrt(mean_squared_error(ground_truth.flatten(), predict.flatten()))
    RMSE = round(RMSE, 4)
    print("Root Mean Squared Error: " + str(RMSE))
    plot_compare('Proposed model', ground_truth, predict, R, RMSE)


def load_all_data():
    with h5py.File('dataset/X_train.h5', 'r') as hf:
        X_train = hf['X_train'][:]
    with h5py.File('dataset/y_train.h5', 'r') as hf:
        y_train = hf['y_train'][:]
    with h5py.File('dataset/X_test.h5', 'r') as hf:
        X_test = hf['X_test'][:]
    with h5py.File('dataset/y_test.h5', 'r') as hf:
        y_test = hf['y_test'][:]
    return X_train, y_train, X_test, y_test
