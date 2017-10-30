import numpy as np

def compute_loss_mse(y, tx, w):
    """Calculate the MSE loss.

    Keyword arguments:
    y -- the labels
    tx -- the features
    w -- the weights
    """
    N = y.shape[0]
    D = w.shape[0]
    y = y.reshape((N,1))
    w = w.reshape((D,1))

    e = y - np.dot(tx, w)
    loss =  np.dot(e.T, e) / (2 * N)
    return loss[0][0]


def compute_loss_mae(y, tx, w):
    """Calculate the MAE loss.

    Keyword arguments:
    y -- the labels
    tx -- the features
    w -- the weights
    """
    N = y.shape[0]
    D = w.shape[0]
    y = y.reshape((N,1))
    w = w.reshape((D,1))

    e = y - np.dot(tx, w)
    return np.sum(np.absolute(e)) / N


def compute_rmse(y, tx, w):
    """Computes the Root Mean Square Error.

    Keyword arguments:
    y -- the labels
    tx -- the features
    w -- the weights
    """
    mse = compute_loss_mse(y, tx, w)
    return np.sqrt(2 * mse)


def compute_gradient_mse(y, tx, w):
    """Compute the MSE gradient.

    Keyword arguments:
    y -- the labels
    tx -- the features
    w -- the weights
    """
    N = y.shape[0]
    D = w.shape[0]
    y = y.reshape((N,1))
    w = w.reshape((D,1))

    e = y - np.dot(tx, w)
    return (-1 / N) * np.dot(tx.T, e)


def compute_stochastic_subgradient_mae(y, tx, w):
    """Compute a stochastic subgradient from just few examples n and their corresponding y_n labels.

    Keyword arguments:
    y -- the labels
    tx -- the features
    w -- the weights
    """
    N = y.shape[0]
    D = w.shape[0]
    y = y.reshape((N,1))
    w = w.reshape((D,1))

    e = y - np.dot(tx, w)
    abs_e_subgrad = [np.sign(en) for en in e]  # Sign chosen for subgradient of absolute value function
    return (-1 / N) * np.dot(np.transpose(tx), abs_e_subgrad)


def batch_iterator(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>

    Keyword arguments:
    y -- the labels
    tx -- the features
    batch_size -- size of the mini-batches
    num_batches -- how many batches to return (default: True)
    shuffle -- to shuffle data (default: True)
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def split_data(x, y, ratio, seed=1):
    """
    Split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing.

    Keyword arguments:
    y -- the labels
    x -- the features
    ratio -- ratio of training to testing data
    seed -- seed for random number generator (default: 1)
    """
    np.random.seed(seed)  # set seed
    permuted_idxs = np.random.permutation(x.shape[0])
    train_size = int(ratio * x.shape[0])
    train_idxs, test_idxs = permuted_idxs[:train_size], permuted_idxs[train_size:]

    return x[train_idxs], x[test_idxs], y[train_idxs], y[test_idxs]


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred


def predict_labels_cutoff(weights, data, cutoff):
    """Generates class predictions given weights, a cutoff value,
    and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= cutoff)] = -1
    y_pred[np.where(y_pred > cutoff)] = 1

    return y_pred


def sigmoid(t):
    """Applies sigmoid function on array t."""
    # Handle overflow of exp() function
    t[t > 700] = 700
    t[t < -700] = -700
    return 1 / (1 + np.exp(-t))


def logistic_regression_loss(y, tx, w):
    """Computes the cost by negative log likelihood.

    Keyword arguments:
    y -- the labels
    tx -- the features
    w -- the weights
    """
    N = y.shape[0]
    D = w.shape[0]
    y = y.reshape((N,))
    w = w.reshape((D,))

    loss = 0
    for n in range(N):
        xnw = np.dot(tx[n], w)
        # To not run into overflow in exp()
        if (xnw > 700):
            loss += xnw - y[n]*xnw
        else:
            loss += np.log(1 + np.exp(xnw)) - y[n]*xnw

    return -loss


def logistic_regression_gradient(y, tx, w):
    """Compute the gradient of the negative log likelihood loss.

    Keyword arguments:
    y -- the labels
    tx -- the features
    w -- the weights
    """
    N = y.shape[0]
    D = w.shape[0]
    y = y.reshape((N,1))
    w = w.reshape((D,1))
    return np.dot(tx.T, sigmoid(np.dot(tx, w)) - y)


def sigmoid_diff(x):
    """The first derrivative of the sigmoid function."""
    return sigmoid(x) * (1 - sigmoid(x))


def logistic_regression_hessian(y, tx, w):
    """Returns the hessian of the loss function.

    Keyword arguments:
    y -- the labels
    tx -- the features
    w -- the weights
    """
    N = y.shape[0]
    D = w.shape[0]
    y = y.reshape((N,1))
    w = w.reshape((D,1))
    S = sigmoid_diff(np.dot(tx, w))
    return np.dot(tx.T, S * tx)


def newton_step(y, tx, w, gamma):
    """
    Does one step on Newton's method.
    Returns the loss and updated w.

    Keyword arguments:
    y -- the labels
    tx -- the features
    w -- the weights
    gamma -- learning rate
    """
    N = y.shape[0]
    D = w.shape[0]
    y = y.reshape((N,1))
    w = w.reshape((D,1))

    loss = logistic_regression_loss(y, tx, w)
    grad = logistic_regression_gradient(y, tx, w)
    hess = logistic_regression_hessian(y, tx, w)

    hess_inv = np.linalg.solve(hess, np.identity(D))
    w = w - gamma * np.dot(hess_inv, grad)

    return loss, w


def penalized_logistic_regression_step(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Returns the loss and updated w.

    Keyword arguments:
    y -- the labels
    tx -- the features
    w -- the weights
    gamma -- learning rate
    lambda_ -- shrinkage parameter
    """
    N = y.shape[0]
    D = w.shape[0]
    y = y.reshape((N,1))
    w = w.reshape((D,1))

    loss = logistic_regression_loss(y, tx, w) + lambda_ * np.linalg.norm(w)
    grad = logistic_regression_gradient(y, tx, w) + 2 * lambda_ * w
    hess = logistic_regression_hessian(y, tx, w) + 2 *lambda_ * np.identity(D)

    hess_inv = np.linalg.solve(hess, np.identity(D))
    w = w - gamma * np.dot(hess_inv, grad)

    return loss, w
