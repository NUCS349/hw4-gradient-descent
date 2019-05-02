import json
import numpy as np
import os
from code import load_mnist
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
        fraction - (float) Value between 0.0 and 1.0 representing the fraction
            of data to include in the training set. The remaining data is
            included in the test set. Unused if dataset == 'synthetic'.
    Returns:
        train_features - (np.array) An Nxd array of features, where N is the
            number of examples and d is the number of features.
        test_features - (np.array) An Nxd array of features, where M is the
            number of examples and d is the number of features.
        train_targets - (np.array) A 1D array of targets of size N.
        test_targets - (np.array) A 1D array of targets of size M.
    """
    if dataset == 'blobs':
        path = os.path.join('data', 'blobs.json')
        train_features, test_features, train_targets, test_targets = \
            load_json_data(path, fraction=fraction)
    elif dataset == 'mnist-binary':
        train_features, test_features, train_targets, test_targets = \
            load_mnist_data(2, fraction=fraction)
    elif dataset == 'mnist-multiclass':
        train_features, test_features, train_targets, test_targets = \
            load_mnist_data(5, fraction=fraction)
    elif dataset == 'synthetic':
        path = os.path.join('data', 'synthetic.json')
        train_features, test_features, train_targets, test_targets = \
            load_json_data(path)
    else:
        raise ValueError('Dataset {} not found!'.format(dataset))

    # Normalize the data using feature-independent whitening. Note that the
    # statistics are computed with respect to the training set and applied to
    # both the training and testing sets.
    if dataset != 'synthetic':
        mean = train_features.mean(axis=0, keepdims=True)
        std = train_features.std(axis=0, keepdims=True)
        train_features = (train_features - mean) / std
        test_features = (test_features - mean) / std

    return train_features, test_features, train_targets, test_targets


def load_json_data(path, fraction=1.0):
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

    # Split the data into training and testing sets
    np.random.seed(0)
    return train_test_split(
        features, targets, test_size=1.0 - fraction, stratify=targets)


def load_mnist_data(threshold, fraction=1.0, examples_per_class=500):
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

    train_examples = int(examples_per_class * fraction)
    train_features, train_targets = load_mnist(
        dataset='training', digits=range(threshold), path='data')
    train_features, train_targets = stratified_subset(
        train_features, train_targets, train_examples)

    test_examples = examples_per_class - train_examples
    test_features, test_targets = load_mnist(
        dataset='testing', digits=range(threshold), path='data')
    test_features, test_targets = stratified_subset(
        test_features, test_targets, test_examples)

    return train_features, test_features, train_targets, test_targets


def stratified_subset(features, targets, examples_per_class):
    """
    Evenly sample the dataset across unique classes. Requires each unique class
    to have at least examples_per_class examples.

    Arguments:
        features - (np.array) An Nxd array of features, where N is the
            number of examples and d is the number of features.
        targets - (np.array) A 1D array of targets of size N.
        examples_per_class - (int) The number of examples to take in each
            unique class.
    Returns:
        features - (np.array) A subset of the input features of size
            examples_per_class.
        tarets (np.array) A subset of the input targets of size
            examples_per_class.
    """
    idxs = np.array([False] * len(features))
    for target in np.unique(targets):
        idxs[np.where(targets == target)[:examples_per_class]] = True
    return features[idxs], targets[idxs]
