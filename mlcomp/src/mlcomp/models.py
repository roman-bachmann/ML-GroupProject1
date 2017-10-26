from mlcomp.helpers import compute_rmse, compute_gradient_mse, compute_loss_mae, compute_loss_mse, \
    compute_stochastic_subgradient_mae, newton_step
import numpy as np


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm using MSE."""
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient_mse(y, tx, w)
        loss = compute_loss_mse(y, tx, w)
        w = w - gamma * grad

    rmse = compute_rmse(y, tx, w)

    return w, rmse


def stochastic_subgradient_descent_mae(y, tx, initial_w, max_iters, gamma):
    """Stochastic subgradient descent algorithm using MAE."""
    batch_size = 1
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            g = compute_stochastic_subgradient_mae(minibatch_y, minibatch_tx, w)
            w = w - gamma * g
        loss = compute_loss_mae(y, tx, w)
        print("Stochastic Subgradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return w, loss


def least_squares(y, tx):
    """Calculates the explicit least squares solution.
    Returns rmse, optimal weights"""
    N = tx.shape[0]
    D = tx.shape[1] if tx.shape[1:] else 1
    rank_tx = np.linalg.matrix_rank(tx)

    # Check if tx is invertible. If so, find explicit solution
    # using real inverses.
    # If not, find explicit solution using pseudoinverses.
    if (rank_tx == max(tx.shape[0], tx.shape[1])):
        gramian_inv = np.linalg.solve(np.dot(tx.T, tx), np.identity(D))
        w = np.dot(gramian_inv, np.dot(tx.T, y))
    else:
        U, s, V_T = np.linalg.svd(tx)
        S_inv_T = np.zeros((D, N))
        S_inv_T[:len(s), :len(s)] = np.diag(1/s)
        w = np.dot(V_T.T, np.dot(S_inv_T, np.dot(U.T, y)))

    rmse = compute_rmse(y, tx, w)

    return w, rmse


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations"""

    if (lambda_ == 0):
        return least_squares(y, tx)

    N = tx.shape[0]
    D = tx.shape[1] if tx.shape[1:] else 1

    inv_inner = np.dot(tx.T, tx) + 2 * N * lambda_ * np.identity(D)
    inv = np.linalg.solve(inv_inner, np.identity(D))
    w = np.dot(inv, np.dot(tx.T, y)).reshape((D, 1))

    rmse = compute_rmse(y, tx, w)

    return w, rmse


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Calculates the best weights for a logistic regression using the Newton method.
    Returns the weights and the loss (negative log likelihood)"""
    # init parameters
    threshold = 1e-8
    losses = []
    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = newton_step(y, tx, w, gamma)
        # log info
        print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Calculates the best weights for a penalized logistic regression using the Newton method.
    Returns the weights and the loss (negative log likelihood)"""
    # init parameters
    threshold = 1e-8
    losses = []
    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = penalized_logistic_regression_step(y, tx, w, gamma, lambda_)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w, loss
