import numpy as np

def eval_correctness(yb, y_pred, verbose=False):
    """Takes inputs known y and predicted y and prints the ratio of correct predictions vs incorrect ones."""
    corrects = (y_pred == yb).sum()
    perc = corrects / len(y_pred) * 100
    if verbose:
        incorrect = len(y_pred) - corrects
        print("Total correct:", corrects, "\nTotal incorrect:", incorrect, "\nCorrect percentage:", perc, "%")

    return perc
