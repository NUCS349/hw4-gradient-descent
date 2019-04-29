import numpy as np


class Loss:
    """
    An abstract base class for a loss function that computes both the original
    loss function (the forward pass) as well as its gradient (the backward
    pass).

    Arguments:
        regularization - (string or None) The type of regularization to
            perform. Should take on one of the following values:
              - 'l1': Performs l1 regularization
              - 'l2': Performs l2 regularization
              - None: Does not perform regularization
    """

    def __init__(self, regularization=None):
        self.regularization = regularization

    def forward(self, X, w, y):
        """
        Computes the forward pass through the loss function. If
        self.regularization is not None, also adds the forward pass of the
        regularization term to the loss.

        Arguments:
            X - (np.array) An Nx(d+1) array of features, where N is the number
                of examples and d is the number of features. The +1 refers to
                the bias term.
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
            y - (np.array) A 1D array of targets of length N.
        Returns:
            loss - (float) The average loss.
        """
        pass

    def backward(self, X, w, y):
        """
        Computes the gradient of the loss function with respect to the model
        parameters. If self.regularization is not None, also adds the backward
        pass of the regularization term to the loss.

        Arguments:
            X - (np.array) An Nx(d+1) array of features, where N is the number
                of examples and d is the number of features. The +1 refers to
                the bias term.
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
            y - (np.array) A 1D array of targets of length N.
        Returns:
            gradient - (np.array) The (d+1)-dimensional gradient of the loss
                function with respect to the model parameters. The +1 refers to
                the bias term.
        """
        pass


class SquaredLoss(Loss):
    """
    The squared loss function.
    """

    def forward(self, X, w, y):
        """
        Computes the forward pass through the loss function. If
        self.regularization is not None, also adds the forward pass of the
        regularization term to the loss.

        For squared

        Arguments:
            X - (np.array) An Nx(d+1) array of features, where N is the number
                of examples and d is the number of features. The +1 refers to
                the bias term.
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
            y - (np.array) A 1D array of targets of length N.
        Returns:
            loss - (float) The average loss.
        """
        raise NotImplementedError()

    def backward(self, X, w, y):
        """
        Computes the gradient of the loss function with respect to the model
        parameters. If self.regularization is not None, also adds the backward
        pass of the regularization term to the loss.

        Arguments:
            X - (np.array) An Nx(d+1) array of features, where N is the number
                of examples and d is the number of features. The +1 refers to
                the bias term.
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
            y - (np.array) A 1D array of targets of length N.
        Returns:
            gradient - (np.array) The (d+1)-dimensional gradient of the loss
                function with respect to the model parameters. The +1 refers to
                the bias term.
        """
        raise NotImplementedError()


class HingeLoss(Loss):
    """
    The hinge loss function.
    """

    def forward(self, X, w, y):
        """
        Computes the forward pass through the loss function. If
        self.regularization is not None, also adds the forward pass of the
        regularization term to the loss.

        Arguments:
            X - (np.array) An Nx(d+1) array of features, where N is the number
                of examples and d is the number of features. The +1 refers to
                the bias term.
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
            y - (np.array) A 1D array of targets of length N.
        Returns:
            loss - (float) The average loss.
        """
        raise NotImplementedError()

    def backward(self, X, w, y):
        """
        Computes the gradient of the loss function with respect to the model
        parameters. If self.regularization is not None, also adds the backward
        pass of the regularization term to the loss.

        Arguments:
            X - (np.array) An Nx(d+1) array of features, where N is the number
                of examples and d is the number of features. The +1 refers to
                the bias term.
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
            y - (np.array) A 1D array of targets of length N.
        Returns:
            gradient - (np.array) The (d+1)-dimensional gradient of the loss
                function with respect to the model parameters. The +1 refers to
                the bias term.
        """
        raise NotImplementedError()
