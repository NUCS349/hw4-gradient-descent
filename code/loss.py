import numpy as np


class Loss:
    """
    An abstract base class for a loss function that computes both the original
    loss function (the forward pass) as well as its gradient (the backward
    pass).
    """

    def __init__(self):
        self.losses = None
        self.X = None
        self.y = None

    def forward(self, X, y):
        """
        Computes the forward pass through the loss function.

        Arguments:
            X - (np.array) An Nxd array of features, where N is the number of
                examples and d is the number of features in each example.
            y - (np.array) A 1D array of targets of length N.
        Modifies:
            self.losses - (np.array) A 1D array of per-example losses of length
                N.
            self.X - (np.array) The stored features X passed in to the
                function. Used in self.backward().
            self.y - (np.array) The stored targets y passed in to the function.
                Used in self.backward().
        Returns:
            loss - (float) The average loss.
        """
        pass

    def backward(self):
        """
        Computes the gradient of the loss function with respect to the model
        parameters.

        Returns:
            gradient - (np.array) The d-dimensional gradient of the loss
                function with respect to the model parameters.
        """
        pass


class SquaredLoss(Loss):
    """
    The squared loss function.

    TODO: Pseudocode
    """

    def forward(self, X, y):
        """
        Computes the forward pass through the loss function.

        Arguments:
            X - (np.array) An Nxd array of features, where N is the number of
                examples and d is the number of features in each example.
            y - (np.array) A 1D array of targets of length N.
        Modifies:
            self.losses - (np.array) A 1D array of per-example losses of length
                N.
            self.X - (np.array) The stored features X passed in to the
                function. Used in self.backward().
            self.y - (np.array) The stored targets y passed in to the function.
                Used in self.backward().
        Returns:
            loss - (float) The average loss.
        """
        raise NotImplementedError()

    def backward(self):
        """
        Computes the gradient of the loss function with respect to the model
        parameters.

        Returns:
            gradient - (np.array) The d-dimensional gradient of the loss
                function with respect to the model parameters.
        """
        raise NotImplementedError()


class HingeLoss(Loss):
    """
    The hinge loss function.

    TODO: Pseudocode
    """

    def forward(self, X, y):
        """
        Computes the forward pass through the loss function.

        Arguments:
            X - (np.array) An Nxd array of features, where N is the number of
                examples and d is the number of features.
            y - (np.array) A 1D array of targets of length N.
        Modifies:
            self.losses - (np.array) A 1D array of per-example losses of length
                N.
            self.X - (np.array) The stored features X passed in to the
                function. Used in self.backward().
            self.y - (np.array) The stored targets y passed in to the function.
                Used in self.backward().
        Returns:
            loss - (float) The average loss.
        """
        raise NotImplementedError()

    def backward(self):
        """
        Computes the gradient of the loss function with respect to the model
        parameters.

        Returns:
            gradient - (np.array) The d-dimensional gradient of the loss
                function with respect to the model parameters.
        """
        raise NotImplementedError()
