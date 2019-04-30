# Coding (5 points)

Your task is to implement the gradient descent algorithm as well a multiclass classification wrapper for One-vs-All (OVA) classification. You will write the following code:

 - Two types of lp regularization and their gradients (in `code/regularization.py`)
 - Two loss functions and their gradients (in `code/loss.py`)
 - The gradient descent algorithm (in `code/gradient_descent.py`)
 - The OVA multiclass wrapper (in `code/multiclass_gradient_descent.py`)

Note that this is also the order in which we recommend you implement the code for this homework.

Your goal is to pass the test suite (contained in `tests/`). Once the tests are passed, you may move on to the next part - reporting your results.

Your grade for this section is defined by the autograder. If it says you got an 80/100, you get 4 points here.

To answer some of the free-response questions, you will have to write extra code (that is not covered by the test cases). You may include your experiments in new files in the `experiments` directory. See `experiments/example.py` for an example. You can run any experiments you create within this directory with `python -m experiments.<experiment_name>`. For example, `python -m experiments.example` runs the example experiment.

# Free-response questions (5 points)

## 1. () Visualizing Gradient Descent

## 2. () Loss Landscapes and the Effects of Batching
    - a. () Here we will setup an experiment that will allow us to visualize the loss landscape of the `synthetic` dataset. In your experiment, first load the `synthetic` dataset. Using only the bias term (i.e., set all other parameter values to 0), determine the squared loss over the entire dataset for bias values in [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]. Construct a plot of the loss landscape by plotting the bias on the X axis and the loss on the Y axis. Include your (labeled) plot and describe the minima of the loss landscape.
    - b. () By analyzing the `synthetic` dataset and the loss landscape plot you have generated, adversarially select 4 points from the dataset that cause the global minimum of the loss landscape to shift. Repeat your experiment from part (a) and demonstrate this shift.
    - c. () Based on your answers from part (a) and (b), explain what effect batching can have on the loss landscape and the convergence of gradient descent.

## 3. () Multiclass Classification with Gradient Descent
    - a. () Confusion matrix
    - b. () OVO vs OVA complexity

## 4. () Regularization and Feature Selection
    - a. () Here we will explore the use of L1 regularization as a means of feature selection. Setup an experiment using the `mnist-binary` dataset. Run gradient descent on the `mnist-binary` dataset using squared loss, using both 'l1' and 'l2' regularization. For each regularizer, run the algorithm 4 times for values of reg_param in [1e-3, 1e-2, 1e-1, 1]. Plot the number of non-zero model parameters of the learned model for each value of reg_param. Plot the trend line for both regularizers on one plot. Include your (labeled) plot and describe the trend in non-zero parameters for each regularizer.
    - b. () Compared to the l2 regularizer, what property of the l1 regularizer allows it to promote sparsity in the model parameters? Describe a situation in which this sparsity is useful.
    - c. () Which pixels (features) within the `mnist-binary` dataset were ignored by the l1 regularizer? Do not list feature indices. Instead, describe the properties these features have in common throughout the dataset. Why does it make sense for these features to be ignored?
