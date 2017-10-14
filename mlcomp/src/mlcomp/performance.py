import numpy as np


def predict_values(weights, X):
    return np.dot(X, weights)


def predict(y_values, cutoff):
    labels = np.empty(len(y_values))
    labels[y_values <= cutoff] = -1
    labels[y_values > cutoff] = 1

    return labels


def correctness(yb, y_pred, verbose=False):
    """Takes inputs known y and predicted y and prints the ratio of correct predictions vs incorrect ones."""
    corrects = (y_pred == yb).sum()
    perc = corrects / len(y_pred) * 100
    if verbose:
        incorrect = len(y_pred) - corrects
        print("Total correct:", corrects, "\nTotal incorrect:", incorrect, "\nCorrect percentage:", perc, "%")

    return perc


def correct_by_cutoff(y_true, y_pred, search_space):
    labels = list(map(lambda c: predict(y_pred, c), search_space))
    return list(map(lambda l: correctness(y_true, l), labels))


def best_cutoff(y_true, y_pred, search_space):
    all_correctness = correct_by_cutoff(y_true, y_pred, search_space)
    return search_space[all_correctness.index(max(all_correctness))]
