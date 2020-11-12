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
from keras import utils as np_utils


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

    # to categorical
    Y_train = np_utils.to_categorical(Y_train, num_classes=labels)
    Y_test = np_utils.to_categorical(Y_test, num_classes=labels)

    with h5py.File('dataset/X_train.h5', 'w') as hf:
        hf.create_dataset("X_train", data=X_train)
    with h5py.File('dataset/Y_train.h5', 'w') as hf:
        hf.create_dataset("Y_train", data=Y_train)
    with h5py.File('dataset/X_test.h5', 'w') as hf:
        hf.create_dataset("X_test", data=X_test)
    with h5py.File('dataset/Y_test.h5', 'w') as hf:
        hf.create_dataset("Y_test", data=Y_test)
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


def load_all_data():
    with h5py.File('dataset/X_train.h5', 'r') as hf:
        X_train = hf['X_train'][:]
    with h5py.File('dataset/Y_train.h5', 'r') as hf:
        y_train = hf['Y_train'][:]
    with h5py.File('dataset/X_test.h5', 'r') as hf:
        X_test = hf['X_test'][:]
    with h5py.File('dataset/Y_test.h5', 'r') as hf:
        y_test = hf['Y_test'][:]
    return 10, X_train, y_train, X_test, y_test


def get_label_name(key):
    dictionary = {
        0: "T - shirt / top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot"
    }
    return dictionary[key]


def starting_text():
    return """             __ _____  __  _____     ___                            _                    _     _             
  /\/\    /\ \ \\_   \/ _\/__   \   / __\___  _ __ ___  _ __  _   _| |_ ___ _ __  /\   /(_)___(_) ___  _ __  
 /    \  /  \/ / / /\/\ \   / /\/  / /  / _ \| '_ ` _ \| '_ \| | | | __/ _ \ '__| \ \ / / / __| |/ _ \| '_ \ 
/ /\/\ \/ /\  /\/ /_  _\ \ / /    / /__| (_) | | | | | | |_) | |_| | ||  __/ |     \ V /| \__ \ | (_) | | | |
\/    \/\_\ \/\____/  \__/ \/     \____/\___/|_| |_| |_| .__/ \__,_|\__\___|_|      \_/ |_|___/_|\___/|_| |_|
                                                       |_|                                                   """
