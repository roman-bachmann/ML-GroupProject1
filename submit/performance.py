import numpy as np
import matplotlib.pyplot as plt
from mlcomp.models import ridge_regression
from mlcomp.helpers import compute_rmse

def eval_correctness(yb, y_pred, verbose=False):
    """Takes inputs known y and predicted y and prints the ratio of correct predictions vs incorrect ones.

    Keyword arguments:
    yb -- true labels
    y_pred -- predicted labels
    verbose -- to print out correctness (default: False)
    """
    corrects = (y_pred == yb).sum()
    perc = corrects / len(y_pred) * 100
    if verbose:
        incorrect = len(y_pred) - corrects
        print("Total correct:", corrects, "\nTotal incorrect:", incorrect, "\nCorrect percentage:", perc, "%")

    return perc

def cross_validation_visualization(lambds, mse_tr, mse_te):
    """Helper function for visualization of the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")

def build_k_indices(y, k_fold, seed):
    """Build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def get_train_indices(k_indices, k):
    """Returns the train indices of a k-fold split."""
    train_indices = np.array([])
    for i in range(k_indices.shape[0]):
        if (i != k-1):
            train_indices = np.hstack((train_indices, k_indices[i]))
    return train_indices.astype(int)

def cross_validation_step(yb, tx, k_indices, k, lambda_):
    """Helper function that calculates one step of the cross validation."""
    train_indices = get_train_indices(k_indices, k)
    x_train = tx[train_indices]
    y_train = yb[train_indices]
    x_test = tx[k_indices[k-1]]
    y_test = yb[k_indices[k-1]]

    weights, rmse = ridge_regression(y_train, x_train, lambda_)

    loss_tr = compute_rmse(y_train, x_train, weights)
    loss_te = compute_rmse(y_test, x_test, weights)

    return loss_tr, loss_te

def cross_validation(yb, tx, k_fold, seed=1):
    """Returns the best lambda using k-fold cross-validation for ridge regression.

    Keyword arguments:
    yb -- the training labels
    tx -- the input data
    k_fold -- number of folds for cross-validation
    seed -- seed for randomizer (default: 1)
    """
    lambdas = np.logspace(-4, 0, 15)
    # split data in k fold
    k_indices = build_k_indices(yb, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    best_rmse_te = float("inf")
    best_ind = 0

    for ind, lambda_ in enumerate(lambdas):
        avg_loss_tr = 0
        avg_loss_te = 0
        for k in range(1, k_fold+1):
            print("k=", k)
            loss_tr, loss_te = cross_validation_step(yb, tx, k_indices, k, lambda_)
            avg_loss_tr += loss_tr
            avg_loss_te += loss_te
        rmse_tr.append(avg_loss_tr/k_fold)
        rmse_te.append(avg_loss_te/k_fold)
        if (rmse_te[ind] < best_rmse_te):
            best_rmse_te = rmse_te[ind]
            best_ind = ind
        print("lambda={l:.9f}, Training RMSE={tr:.3f}, Testing RMSE={te:.3f}".format(
               l=lambda_, tr=rmse_tr[ind], te=rmse_te[ind]))

    cross_validation_visualization(lambdas, rmse_tr, rmse_te)
    return lambdas[best_ind]
