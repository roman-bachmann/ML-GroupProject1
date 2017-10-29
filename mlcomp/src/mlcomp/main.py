# Setup and imports
import numpy as np
import os
import gc
import timeit

from mlcomp.config import DATA_PATH
from mlcomp.models import ridge_regression, logistic_regression
from mlcomp.feature_eng import build_advanced_poly, replace_nan_by_median, standardize, replace_nan_by_mean, build_simple_poly
from mlcomp.helpers import split_data, compute_rmse, predict_labels
from mlcomp.performance import eval_correctness, cross_validation
from mlcomp.data import load_csv_data, create_csv_submission

if __name__ == '__main__':
    start = timeit.default_timer()
    TRAIN_PATH = os.path.join(DATA_PATH, 'train.csv')
    yb, input_data, ids = load_csv_data(TRAIN_PATH, sub_sample=False)
    yb = yb.reshape((yb.shape[0], 1))

    print("Data loaded. Begin data transform.")

    tx = replace_nan_by_median(input_data, -999)
    #important_cols = [0, 2, 7, 1, 11, 13, 5, 9, 19] # 82.592 %
    #important_cols = [0, 2, 7, 1, 11, 13, 5, 9, 19, 10] # 82.7284 %
    #important_cols = [0, 2, 7, 1, 11, 13, 5, 9, 19, 10, 4] # 82.9896 %
    #important_cols = [0, 2, 7, 1, 11, 13, 5, 9, 19, 10, 4, 3] # 83.2412 % not worth it
    #degree = 5
    #x_poly = build_advanced_poly(tx, degree, important_cols)

    print("Data transformed. Begining training...")

    #best_lambda = cross_validation(yb, x_poly, k_fold=4, seed=1)

    #print("Cross validation done. Best lambda:", best_lambda)
    initial_w = np.zeros((tx.shape[1], 1))
    weights, loss = logistic_regression(yb, tx, initial_w, max_iters = 100, gamma = 0.01)
    #weights, rsme = ridge_regression(yb, x_poly, 0.0001)
    #np.savetxt("weights.csv", weights, delimiter=",")

    print("Training done.")

    # Predict labels with found weights and print some useful information about quality of fit
    y_pred = predict_labels(weights, tx)
    eval_correctness(yb, y_pred, verbose=True)
    print("-----------------")
    print("RMSE:", rsme)

    gc.collect()

    print("Loading testing data")

    TEST_PATH = os.path.join(DATA_PATH, 'test.csv')
    yb_test, input_data_test, ids_test = load_csv_data(TEST_PATH, sub_sample=False)

    print("Loaded testing data. Transforming data...")

    input_data_test = replace_nan_by_median(input_data_test, -999)
    #x_submit_poly = build_advanced_poly(input_data_test, degree, important_cols)

    print("Data transformed. Predicting labels.")

    y_test_pred = predict_labels(weights, input_data_test)

    print("Labels predicted. Saving CSV.")

    # Save predictions of test data in csv file, ready for the upload on kaggle
    create_csv_submission(ids_test, y_test_pred, "test_output.csv")
    gc.collect()
    print("CSV saved!")
    stop = timeit.default_timer()

    print(stop-start)
