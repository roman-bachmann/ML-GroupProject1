import os
import numpy as np

from time import time

from mlcomp.config import DATA_PATH
from mlcomp.data import load_csv_data
from mlcomp.helpers import split_data
from mlcomp.feature_eng import apply_feature_eng
from mlcomp.costs import compute_correctness, compute_mae, compute_mse
from mlcomp.cross_validation import ridge_lambda_cv, get_best_parameter
from mlcomp.models import simple_ridge_regression
from mlcomp.performance import predict_values, predict, correct_by_cutoff

RATIO_SPLIT = 0.3
SEED_SPLIT = 872
LAMBDA_SPACE = np.logspace(-5, 0, 30)
TRAIN_PATH = os.path.join(DATA_PATH, 'train.csv')
LOSS_FN = compute_correctness
LOSS_GREATER_IS_BETTER = True
CUTOFF_SPACE = np.linspace(-4, 4, retstep=0.01)[0]


if __name__ == '__main__':
    print('Hello! This is the ml project \n')

    print('-- Reading Data -- \n')
    y, X, ids = load_csv_data(TRAIN_PATH)

    print('-- Spliting train {p_train} and test {p_test} --\n'.format(p_train=1-RATIO_SPLIT, p_test=RATIO_SPLIT))
    X_train, X_test, y_train, y_test = split_data(X, y, RATIO_SPLIT, seed=SEED_SPLIT)
    col_names = list(np.genfromtxt(TRAIN_PATH, delimiter=",", dtype=None, max_rows=1))
    col_names = list(map(lambda x: x.decode("utf-8"), col_names))[2:]


    print('-- Applying Feature Eng --')
    s = time()
    X_feat = apply_feature_eng(X)
    X_feat_train = apply_feature_eng(X_train)
    X_feat_test = apply_feature_eng(X_test)
    print('Train shape: {shape} \nTrain split shape: {tr_split}\nTest split shape: {te_split}'
          .format(shape=X_feat.shape, tr_split=X_feat_train.shape, te_split=X_feat_test.shape))
    print('Took {sec}s to do feature eng\n'.format(sec=round((time()-s), 1)))


    print('-- Ridge Regression --')
    loss_test = ridge_lambda_cv(y_train, X_feat_train, compute_correctness, LAMBDA_SPACE)[1]
    best_lambda = get_best_parameter(loss_test, LAMBDA_SPACE, LOSS_GREATER_IS_BETTER)

    best_w_train = simple_ridge_regression(y_train, X_feat_train, best_lambda)

    y_test_values = predict_values(best_w_train, X_feat_test)
    correctness_by_cutoff = correct_by_cutoff(y_test, y_test_values, LOSS_FN, CUTOFF_SPACE)
    best_cutoff = CUTOFF_SPACE[correctness_by_cutoff.index(max(correctness_by_cutoff))]
    y_hat_test = predict(y_test_values, best_cutoff)
    performance_test = compute_correctness(y_test, y_hat_test)

    best_w = simple_ridge_regression(y, X, best_lambda)

    print('Best lambda: {lambda_}\nBest Cutoff: {cutoff}\nPerformance in test set: {pf_test}\n'
      .format(lambda_=round(best_lambda, 3),
              cutoff=round(best_cutoff, 3),
              pf_test=round(performance_test, 3)))








