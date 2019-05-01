import json
import numpy as np
import os
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def load_data(dataset, fraction=1.0):
    """
    Loads a dataset and performs a random stratified split into training and
    test partitions.

    Arguments:
        dataset - (string) The name of the dataset to load. One of the
            following:
              'blobs': A linearly separable binary classification problem.
              'mnist-binary': A subset of the MNIST dataset containing only
                  0s and 1s.
              'mnist-multiclass': A subset of the MNIST dataset containing the
                  numbers 0 through (and including) 4.
              'synthetic': A small custom dataset for exploring properties of
                  gradient descent algorithms.
    Returns:
        train_features - (np.array) An Nxd array of features, where N is the
            number of examples and d is the number of features.
        test_features - (np.array) An Nxd array of features, where M is the
            number of examples and d is the number of features.
        train_targets - (np.array) A 1D array of targets of size N.
        test_targets - (np.array) A 1D array of targets of size M.
    """
    if dataset == 'blobs':
        path = os.path.join('data', 'blobs')
        features, targets = load_json_data(path)
    elif dataset == 'mnist-binary':
        features, targets = load_mnist(2)
    elif dataset == 'mnist-multiclass':
        features, targets = load_mnist(5)
    elif dataset == 'synthetic':
        path = os.path.join('data', 'synthetic')
        features, targets = load_json_data(path)
    else:
        raise ValueError('Dataset {} not found!'.format(dataset))

    # Split the data into training and testing sets
    np.random.seed(0)
    train_features, test_features, train_targets, test_targets = \
        train_test_split(
            features, targets, test_size=1.0 - fraction, stratify=targets)

    # Normalize the data using feature-independent whitening. Note that the
    # statistics are computed with respect to the training set and applied to
    # both the training and testing sets.
    if dataset != 'synthetic':
        mean = train_features.mean(axis=0, keepdims=True)
        std = train_features.std(axis=0, keepdims=True)
        train_features = (train_features - mean) / std
        test_features = (test_features - mean) / std

    return train_features, test_features, train_targets, test_targets


def load_json_data(path):
    """
    Loads data from JSON files.

    Args:
        path - (string) Path to json file containing the data
        normalize - (bool) Whether to whiten the features.
    Returns:
        features - (np.array) An Nxd array of features, where N is the
            number of examples and d is the number of features.
        targets - (np.array) A 1D array of targets of size N.
    """
    with open(path, 'rb') as file:
        data = json.load(file)
    features = np.array(data[0]).astype(float)
    targets = np.array(data[1]).astype(int)

    return features, targets


def load_mnist(threshold, examples_per_class=500):
    """
    Loads a subset of the MNIST dataset

    Arguments:
        threshold - (int) One greater than the maximum digit in the selected
            subset
        normalize - (bool) Whether to whiten the features.
        examples_per_class - (int) Number of examples to retrieve in each
            class.
    Returns:
        features - (np.array) An Nxd array of features, where N is the
            number of examples and d is the number of features.
        targets - (np.array) A 1D array of targets of size N.
    """
    mnist = fetch_openml('mnist_784')
    features = mnist['data']
    targets = mnist['target']

    idxs = np.array([False] * len(features))
    for c in range(threshold):
        idxs[np.where(targets == c)[:examples_per_class]] = True

    return features, targets[idxs]
