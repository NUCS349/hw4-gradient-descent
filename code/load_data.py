def load_data(dataset):
    """
    Loads a dataset.

    Arguments:
        dataset - (string) The name of the dataset to load. One of the
            following:
              'blobs': A linearly separable binary classification problem.
              'mnist-binary': A subset of the MNIST dataset containing only
                  0s and 1s.
              'mnist-multiclass': A subset of the MNISt dataset containing the
                  numbers 0 through (and including) 4.
              'synthetic': A small custom dataset for exploring properties of
                  gradient descent algorithms.
    Returns:
        features - (np.array) An Nxd array of features, where N is the
            number of examples and d is the number of features.
        targets - (np.array) A 2D array of targets of size Nxc. Note that c = 1
            for all datasets except mnist-multi.
    """
    raise NotImplementedError()
