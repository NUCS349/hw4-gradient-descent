import numpy as np
from code import load_data


def make_predictions(features, targets, loss, regularization):
    np.random.seed(0)
    learner = GradientDescent(loss=loss, regularization=regularization,
                              learning_rate=0.01, reg_param=0.05)
    learner.fit(features, targets, batch_size=None, max_iter=1000)
    return learner.predict(features)


def test_gradient_descent_blobs():
    """
    Tests the ability of the gradient descent algorithm to classify a linearly
    separable dataset.
    """
    from code import GradientDescent

    features, targets = load_data('blobs')

    hinge = make_predictions(features, targets, 'hinge', None)
    assert np.all(hinge == targets)

    l1_hinge = make_predictions(features, targets, 'hinge', 'l1')
    assert np.all(l1_hinge == targets)

    l2_hinge = make_predictions(features, targets, 'hinge', 'l2')
    assert np.all(l2_hinge == targets)

    squared = make_predictions(features, targets, 'squared', None)
    assert np.all(squared == targets)

    l1_squared = make_predictions(features, targets, 'squared', 'l1')
    assert np.all(l1_squared == targets)

    l2_squared = make_predictions(features, targets, 'squared', 'l2')
    assert np.all(l2_squared == targets)


def test_gradient_descent_mnist_binary():
    """
    Tests the ability of the gradient descent classifier to classify a
    non-trivial problem with a reasonable accuracy.
    """
    from code import GradientDescent, accuracy

    features, targets = load_data('mnist-binary')

    predictions = make_predictions(features, targets, 'squared', None)
    assert accuracy(targets, predictions) > 0.9

