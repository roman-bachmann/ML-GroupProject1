# Setup and imports
import numpy as np
import os

from config import DATA_PATH
from implementations import ridge_regression
from feature_eng import build_advanced_poly, replace_nan_by_median
from helpers import compute_rmse, predict_labels
from performance import eval_correctness
from data import load_csv_data, create_csv_submission



# Loading input data
print("Loading training data...")
TRAIN_PATH = os.path.join(DATA_PATH, 'train.csv')
yb, input_data, ids = load_csv_data(TRAIN_PATH, sub_sample=False)
yb = yb.reshape((yb.shape[0], 1))


# Feature Engineering
print("Data loaded. Begining feature engineering...")
tx = replace_nan_by_median(input_data, -999)
important_cols = [0, 2, 7, 1, 11, 13, 5, 9, 19, 10, 4]
degree = 5
x_poly = build_advanced_poly(tx, degree, important_cols)


# Training optimal weights
print("Data transformed. Begining training...")
lambda_ = 0.00001
weights, rmse = ridge_regression(yb, x_poly, lambda_)
print("Training done!\n")


# Predict training labels with found weights and print some useful information about quality of fit
y_pred = predict_labels(weights, x_poly)
eval_correctness(yb, y_pred, verbose=True)
print("-----------------")
print("Training RMSE:", rmse)


# Loading test data
print("\nLoading testing data...")
TEST_PATH = os.path.join(DATA_PATH, 'test.csv')
yb_test, input_data_test, ids_test = load_csv_data(TEST_PATH, sub_sample=False)


# Feature Engineering
print("Data loaded. Applying feature engineering to test data...")
input_data_test = replace_nan_by_median(input_data_test, -999)
x_submit_poly = build_advanced_poly(input_data_test, degree, important_cols)


# Predicting labels
print("Data transformed. Predicting labels...")
y_test_pred = predict_labels(weights, x_submit_poly)


# Save predictions of test data in csv file, ready for the upload on kaggle
print("Labels predicted. Saving CSV...")
create_csv_submission(ids_test, y_test_pred, "submission.csv")
print("Test data labels saved as submission.csv!")
