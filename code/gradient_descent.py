import numpy as np
from code import HingeLoss, SquaredLoss
from matplotlib import pyplot as plt


class GradientDescent:
    """
    A linear gradient descent classifier with offset.

    TODO: Pseudocode

    Arguments:
        learning_rate - (float) The size of each gradient descent update step.
    """
    def __init__(self, loss, learning_rate=0.01, momentum=False):
        self.learning_rate = learning_rate
        self.momentum = momentum

        if loss == 'hinge':
            self.loss = HingeLoss()
        elif loss == 'squared':
            self.loss = SquaredLoss()
        else:
            raise ValueError('Loss function {} is not defined'.format(loss))

        self.model = None

    def fit(self, features, targets):
        """
        Fits a gradient descent learner to the features and targets.

        Arguments:
            features - (np.array) An Nxd array of features, where N is the
                number of examples and d is the number of features.
            targets - (np.array) A 1D array of targets of length N.
        Modifies:
            self.model - (np.array) A 1D array of model parameters of length d.
        """
        raise NotImplementedError()

    def predict(self, features):
        """
        Predicts the class labels of each example in features.

        Arguments:
            features - (np.array) A Nxd array of features, where N is the
                number of examples and d is the number of features.
        Returns:
            predictions - (np.array) A 1D array of predictions of length N,
                where index d corresponds to the prediction of row N of
                features.
        """
        raise NotImplementedError()
