# Setup and imports
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import PolynomialFeatures

from mlcomp import *
from mlcomp.config import DATA_PATH
from mlcomp.feature_eng import build_advanced_poly, build_simple_poly, build_mult_comb
from mlcomp.helpers import split_data, compute_rmse, predict_labels
from mlcomp.performance import correctness
from mlcomp.data import load_csv_data, create_csv_submission

def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations"""
    N = tx.shape[0]
    D = tx.shape[1]

    inv_inner = np.dot(tx.T, tx) + 2 * N * lambda_ * np.identity(D)
    inv = np.linalg.solve(inv_inner, np.identity(D))
    w = np.dot(inv, np.dot(tx.T, y)).reshape((D, 1))

    rmse = compute_rmse(y, tx, w)

    return w, rmse

def replace_nan_by_median(tx, nan_value):
    """Replaces values with a specified nan_value by the column median."""
    tx[tx == nan_value] = np.nan
    col_median = np.nanmedian(tx, axis=0)
    return np.where(np.isnan(tx), col_median, tx)


def calc_correctness(yb, y_pred):
    """Takes inputs known y and predicted y and prints the ratio of correct predictions vs incorrect ones."""
    corrects = (y_pred == yb).sum()
    perc = corrects / len(y_pred) * 100
    return perc

if __name__ == '__main__':
    TRAIN_PATH = os.path.join(DATA_PATH, 'train.csv')
    yb, input_data, ids = load_csv_data(TRAIN_PATH, sub_sample=False)
    yb = yb.reshape((yb.shape[0], 1))

    print("Data loaded. Begin data transform.")

    tx = replace_nan_by_median(input_data, -999)
    # important_cols = [0, 2, 7, 1, 11, 13, 5, 9, 19] # 82.592 %
    #important_cols = [0, 2, 7, 1, 11, 13, 5, 9, 19, 10] # 82.7284 %
    important_cols = [0, 2, 7, 1, 11, 13, 5, 9, 19, 10, 4] # 82.9896 %
    degree = 5
    poly = PolynomialFeatures(degree)
    x_poly = poly.fit_transform(tx[:, important_cols])

    print("Data transformed. Begin training.")

    weights, rsme = ridge_regression(yb, x_poly, 0.00001)

    print("Training done.")

    # Predict labels with found weights and print some useful information about quality of fit
    y_pred = predict_labels(weights, x_poly)
    correctness = calc_correctness(yb, y_pred)
    print("Correctness:", correctness)
    print("-----------------")
    print("RMSE:", rsme)

    print("Loading testing data")

    TEST_PATH = os.path.join(DATA_PATH, 'test.csv')
    yb_test, input_data_test, ids_test = load_csv_data(TEST_PATH, sub_sample=False)

    print("Loaded testing data. Transforming data...")

    input_data_test = replace_nan_by_median(input_data_test, -999)
    submit_poly = PolynomialFeatures(degree)
    x_submit_poly = submit_poly.fit_transform(input_data_test[:, important_cols])

    print("Data transformed. Predicting labels.")

    y_test_pred = predict_labels(weights, x_submit_poly)

    print("Labels predicted. Saving CSV.")

    # Save predictions of test data in csv file, ready for the upload on kaggle
    create_csv_submission(ids_test, y_test_pred, "test_output.csv")

    print("CSV saved!")
