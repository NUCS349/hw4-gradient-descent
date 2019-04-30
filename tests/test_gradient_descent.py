import numpy as np
from code import load_data


def test_gradient_descent_blobs():
    """
    Tests the ability of the gradient descent algorithm to classify a linearly
    separable dataset.
    """
    from code import GradientDescent

    def make_predictions(loss, regularization):
        np.random.seed(0)
        learner = GradientDescent(loss=loss, regularization=regularization,
                                  learning_rate=0.1, reg_param=0.05)
        learner.fit(features, targets)
        return learner.predict(features)

    features, targets = load_data('blobs')

    hinge = make_predictions('hinge', None)
    assert np.all(hinge == targets)

    l1_hinge = make_predictions('hinge', 'l1')
    assert np.all(l1_hinge == targets)

    l2_hinge = make_predictions('hinge', 'l2')
    assert np.all(l2_hinge == targets)

    squared = make_predictions('squared', None)
    assert np.all(squared == targets)

    l1_squared = make_predictions('squared', 'l1')
    assert np.all(l1_squared == targets)

    l2_squared = make_predictions('squared', 'l2')
    assert np.all(l2_squared == targets)


def test_gradient_descent_mnist_binary():
    from code import GradientDescent

    features, targets = load_data('mnist')


