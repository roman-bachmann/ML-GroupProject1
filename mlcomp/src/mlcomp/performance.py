def correctness(yb, y_pred):
    """Takes inputs known y and predicted y and prints the ratio of correct predictions vs incorrect ones."""
    correct = 0
    for i in range(len(y_pred)):
        if (y_pred[i] == yb[i]):
            correct += 1

    incorrect = len(y_pred) - correct
    perc = correct / len(y_pred) * 100
    print("Total correct:", correct, "\nTotal incorrect:", incorrect, "\nCorrect percentage:", perc, "%")