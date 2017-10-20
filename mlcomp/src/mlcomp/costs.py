import numpy as np

from mlcomp.performance import predict


def compute_loss_mse(y, tx, w):
    """Calculate the MSE loss."""
    N = len(y)
    e = y - np.dot(tx, w)
    return np.dot(e, e) / (2 * N)


def compute_loss_mae(y, tx, w):
    """Calculate the MAE loss."""
    N = len(y)
    e = y - np.dot(tx, w)
    return np.sum(np.absolute(e)) / N


def compute_loss_rmse(y, tx, w):
    """Computes the Root Mean Square Error"""
    mse = compute_loss_mse(y, tx, w)
    return np.sqrt(2 * mse)


def compute_gradient_mse(y, tx, w):
    """Compute the MSE gradient."""
    N = len(y)
    e = y - np.dot(tx, w)
    return (-1 / N) * np.dot(np.transpose(tx), e)


def compute_mse(y_true, y_pred):
    N = len(y_true)
    e = y_true - y_pred
    return np.dot(e, e) / (2 * N)


def compuse_rse(y_true, y_pred):
    return np.sqrt(2 * compute_mse(y_true, y_pred))


def compute_mae(y_true, y_pred):
    N = len(y_true)
    e = y_true - y_pred
    return np.sum(np.absolute(e)) / N


def compute_correctness(y_true, y_pred, cutoff=0, verbose=False):
    """Takes inputs known y and predicted y and prints the ratio of correct predictions vs incorrect ones."""
    y_pred_labels = predict(y_pred, cutoff)
    corrects = (y_pred_labels == y_true).sum()
    perc = corrects / len(y_pred_labels) * 100
    if verbose:
        incorrect = len(y_pred_labels) - corrects
        print("Total correct:", corrects, "\nTotal incorrect:", incorrect, "\nCorrect percentage:", perc, "%")

    return perc
