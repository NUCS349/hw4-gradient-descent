import numpy as np
from your_code import load_data


def test_multiclass_gradient_descent_separable():
    """
    Tests the ability of the OVA multiclass gradient descent classifier to
    classify a simple, separable dataset.
    """
    from your_code import MultiClassGradientDescent

    np.random.seed(0)

    features = np.identity(4)
    targets = np.array(range(4))

    learner = MultiClassGradientDescent(loss='squared', regularization=None,
                                        learning_rate=0.01, reg_param=0.05)
    learner.fit(features, targets, batch_size=None, max_iter=1000)
    predictions = learner.predict(features)

    assert np.all(predictions == targets)


def test_multiclass_gradient_descent_blobs():
    """
    Tests that the multiclass classifier also works on binary tasks
    """
    from your_code import MultiClassGradientDescent

    np.random.seed(0)

    features, _, targets, _ = load_data('blobs')

    learner = MultiClassGradientDescent(loss='squared', regularization=None,
                                        learning_rate=0.01, reg_param=0.05)
    learner.fit(features, targets, batch_size=None, max_iter=1000)
    predictions = learner.predict(features)

    assert np.all(predictions == targets)


def test_multiclass_gradient_descent_mnist():
    """
    Tests the ability of the multiclass gradient descent classifier to classify
    a non-trivial problem with a reasonable accuracy.
    """
    from your_code import MultiClassGradientDescent, accuracy

    np.random.seed(0)

    train_features, test_features, train_targets, test_targets = \
        load_data('mnist-multiclass', fraction=0.8)

    learner = MultiClassGradientDescent(loss='squared', regularization=None,
                                        learning_rate=0.01, reg_param=0.05)
    learner.fit(train_features, train_targets, batch_size=None, max_iter=1000)
    predictions = learner.predict(test_features)

    assert accuracy(test_targets, predictions) >= 0.85
