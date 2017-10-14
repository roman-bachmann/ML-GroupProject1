def correctness(yb, y_pred):
    """Takes inputs known y and predicted y and prints the ratio of correct predictions vs incorrect ones."""
    corrects = (y_pred == yb).sum()

    incorrect = len(y_pred) - corrects
    perc = corrects / len(y_pred) * 100
    print("Total correct:", corrects, "\nTotal incorrect:", incorrect, "\nCorrect percentage:", perc, "%")

    return perc