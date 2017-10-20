import numpy as np

from mlcomp.models import simple_ridge_regression
from mlcomp.performance import predict_values


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def ridge_lambda_cv(y, x,
                    loss_fn,
                    lambdas,
                    extra_param_loss={},
                    folds=4,
                    split_seed=1):
    k_indices = build_k_indices(y, folds, split_seed)

    losses_train = []
    losses_test = []
    for lambda_ in lambdas:
        loss_train = []
        loss_test = []
        for fold, _ in enumerate(k_indices):
            idx_all = [item for sublist in k_indices for item in sublist]
            idx_test = k_indices[fold]
            idx_train = list(set(idx_all) - set(idx_test))

            y_test, x_test, y_train, x_train = y[idx_test], x[idx_test], y[idx_train], x[idx_train]
            ws = simple_ridge_regression(y_train, x_train, lambda_)

            y_hat_test = predict_values(ws, x_test)
            y_hat_train = predict_values(ws, x_train)

            loss_train.append(loss_fn(y_train, y_hat_train, **extra_param_loss))
            loss_test.append(loss_fn(y_test, y_hat_test, **extra_param_loss))

        losses_train.append(loss_train)
        losses_test.append(loss_test)

    losses_train = list(map(lambda x: np.mean(x), losses_train))
    losses_test = list(map(lambda x: np.mean(x), losses_train))

    return losses_train, losses_test


def get_best_parameter(losses_test_cv, search_space, loss_greater_is_better=False):
    if loss_greater_is_better:
        best_index = losses_test_cv.index(max(losses_test_cv))
    else:
        best_index = losses_test_cv.index(min(losses_test_cv))

    return search_space[best_index]

