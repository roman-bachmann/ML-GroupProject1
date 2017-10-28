from mlcomp.helpers import compute_rmse, compute_gradient_mse, compute_loss_mae, compute_loss_mse, \
    compute_stochastic_subgradient_mae, newton_step, batch_iterator
import numpy as np


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm using MSE.
    Returns the optimal weights and MSE."""
    N = y.shape[0]
    D = initial_w.shape[0]
    y = y.reshape((N,1))
    w = initial_w.reshape((D,1))

    for n_iter in range(max_iters):
        grad = compute_gradient_mse(y, tx, w)
        w = w - gamma * grad

    loss = compute_loss_mse(y, tx, w)

    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic subgradient descent algorithm using MAE.
    Returns the optimal weights and MAE."""
    N = y.shape[0]
    D = initial_w.shape[0]
    y = y.reshape((N,1))
    w = initial_w.reshape((D,1))
    batch_size = 1

    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iterator(y, tx, batch_size):
            g = compute_stochastic_subgradient_mae(minibatch_y, minibatch_tx, w)
            w = w - gamma * g
        loss = compute_loss_mae(y, tx, w)

    return w, loss


def least_squares(y, tx):
    """Calculates the explicit least squares solution.
    Returns the optimal weights and RMSE."""
    N = tx.shape[0]
    D = tx.shape[1] if tx.shape[1:] else 1
    rank_tx = np.linalg.matrix_rank(tx)

    # Check if tx is invertible. If so, find explicit solution
    # using real inverses, if not, use pseudoinverses.
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
    """Ridge regression using normal equations.
    Returns the optimal weights and RMSE."""

    if (lambda_ == 0):
        return least_squares(y, tx)

    N = tx.shape[0]
    D = tx.shape[1] if tx.shape[1:] else 1

    inv_inner = np.dot(tx.T, tx) + 2 * N * lambda_ * np.identity(D)
    inv = np.linalg.solve(inv_inner, np.identity(D))
    w = np.dot(inv, np.dot(tx.T, y))

    rmse = compute_rmse(y, tx, w)

    return w, rmse


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Calculates the best weights for a logistic regression using the Newton method.
    Returns the weights and the loss (negative log likelihood)"""
    N = y.shape[0]
    D = initial_w.shape[0]
    y = y.reshape((N,1))
    w = initial_w.reshape((D,1))

    threshold = 1e-8
    losses = []

    for iter in range(max_iters):
        loss, w = newton_step(y, tx, w, gamma)
        print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        # Convergence criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Calculates the best weights for a penalized logistic regression using the Newton method.
    Returns the weights and the loss (negative log likelihood)"""
    N = y.shape[0]
    D = initial_w.shape[0]
    y = y.reshape((N,1))
    w = initial_w.reshape((D,1))

    threshold = 1e-8
    losses = []

    # start the logistic regression
    for iter in range(max_iters):
        loss, w = penalized_logistic_regression_step(y, tx, w, gamma, lambda_)
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # Convergence criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w, loss
