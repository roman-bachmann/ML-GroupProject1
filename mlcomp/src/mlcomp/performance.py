import numpy as np


def correctness(yb, y_pred):
    """Takes inputs known y and predicted y and prints the ratio of correct predictions vs incorrect ones."""
    correct = 0
    for i in range(len(y_pred)):
        if (y_pred[i] == yb[i]):
            correct += 1

    incorrect = len(y_pred) - correct
    perc = correct / len(y_pred) * 100
    print("Total correct:", correct, "\nTotal incorrect:", incorrect, "\nCorrect percentage:", perc, "%")


def predict_values(weights, X):
    return np.dot(X, weights)


def predict(y_values, cutoff):
    labels = np.empty(len(y_values))
    labels[y_values <= cutoff] = -1
    labels[y_values > cutoff] = 1

    return labels


def correctness_perc(yb, y_pred, verbose=False):
    """Takes inputs known y and predicted y and prints the ratio of correct predictions vs incorrect ones."""
    corrects = (y_pred == yb).sum()
    perc = corrects / len(y_pred) * 100
    if verbose:
        incorrect = len(y_pred) - corrects
        print("Total correct:", corrects, "\nTotal incorrect:", incorrect, "\nCorrect percentage:", perc, "%")

    return perc


def correct_by_cutoff(y_true, y_pred, loss_fn, search_space):
    labels = list(map(lambda c: predict(y_pred, c), search_space))
    corrects = list(map(lambda l: loss_fn(y_true, l), labels))
    return corrects
