import json
import numpy as np
import os
from sklearn.datasets import fetch_openml


def load_data(dataset):
    """
    Loads a dataset.

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
        features - (np.array) An Nxd array of features, where N is the
            number of examples and d is the number of features.
        targets - (np.array) A 2D array of targets of size Nxc. Note that c = 1
            for all datasets except mnist-multiclass.
    """
    if dataset == 'blobs':
        return load_json_data(os.path.join('data', 'blobs'), normalize=True)
    elif dataset == 'mnist-binary':
        return load_mnist(2, normalize=True)
    elif dataset == 'mnist-multiclass':
        return load_mnist(5, normalize=True)
    elif dataset == 'synthetic':
        return load_json_data(os.path.join('data', 'synthetic'))
    else:
        raise ValueError('Dataset {} not found!'.format(dataset))


def load_json_data(path, normalize=False):
    """
    Loads data from JSON files.

    Args:
        path - (string) Path to json file containing the data
        normalize - (bool) Whether to whiten the features.
    Returns:
        features - (np.array) An Nxd array of features, where N is the
            number of examples and d is the number of features.
        targets - (np.array) A 2D array of targets of size Nxc.
    """
    with open(path, 'rb') as file:
        data = json.load(file)
    features = np.array(data[0]).astype(float)
    targets = np.array(data[1]).astype(float)

    return features, np.expand_dims(targets, axis=1)


def load_mnist(threshold, normalize=False):
    """
    Loads a subset of the MNIST dataset

    Arguments:
        threshold - (int) One greater than the maximum digit in the selected
            subset
        normalize - (bool) Whether to whiten the features.
    Returns:
        features - (np.array) An Nxd array of features, where N is the
            number of examples and d is the number of features.
        targets - (np.array) A 2D array of targets of size Nxc.
    """
    mnist = fetch_openml('mnist_784')
    features = mnist['data']
    targets = mnist['target']
    return features[targets < threshold], targets[targets < threshold]


def normalize(features):
    """
    Performs whitening independently on each feature. This allows our
    hyperparameters (e.g., learning_rate) to have a consistent magnitude of
    effect across datasets.

    Note: here we are normalizing using statistics computed over the entire
    dataset. When we are using multiple data partitions (e.g., train,
    validation, test), statistics are computed ONLY on the training set. Those
    statistics are then stored and applied to the normalization of the
    valieation and test sets.

    Arguments:
        features - (np.array) A Nxd array of features, where N is the
            number of examples and d is the number of features.
    Returns:
        normalized - (np.array) The whitened features. The dimensionality is
            preserved with respect to the input features.
    """
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return (features - mean) / std
