import copy
import numpy as np
import itertools


# TODO: Optimize. Very slow!
def build_mult_comb(tx, deg, cols=[]):
    """
    Returns all multiplicative combinations of the specified columns for degree deg.
    For len(col) = D', there are (D' choose deg) combinations of columns that get
    returned as a matrix.
    If cols is not given, it returns the combinations of all columns of tx.
    """
    N = tx.shape[0]
    if (cols == []):
        comb_iter = itertools.combinations_with_replacement(range(tx.shape[1]), deg)
    else:
        comb_iter = itertools.combinations_with_replacement(cols, deg)
    mult = []
    for comb in comb_iter:
        mult_col = np.ones(N)
        for idx in comb:
            tx_col = tx[:, idx]
            mult_col = np.multiply(mult_col, tx_col)
        mult.append(mult_col.tolist())
    return np.array(mult).T


def build_advanced_poly(tx, degree, cols=[]):
    """
    Builds full polynomial basis function for input data matrix tx, for j=0 up to j=degree,
    where the result will be a matrix of form:
    [1, tx, comb_mult(tx, 2), ..., comb_mult(tx, j)]
    comb_mult(tx, 2) denotes all multiplicative combinations of the selected columns of tx.
    If cols is not given, it returns the combinations of all columns of tx.
    """
    poly = np.ones((tx.shape[0], 1))

    for j in range(1, degree + 1):
        mult = build_mult_comb(tx, j, cols)
        poly = np.column_stack((poly, mult))

    return poly


def build_simple_poly(tx, degree):
    """
    Builds simple polynomial basis function for input data matrix tx, for j=0 up to j=degree,
    where the result will be a matrix of form [1, tx, tx^2, ..., tx^j].
    tx^j denotes that for each x_i,k in tx, the result will be (x_i,k)^j
    """
    poly = np.ones((tx.shape[0], 1))

    for j in range(1, degree + 1):
        poly = np.column_stack((poly, np.power(tx, j)))

    return poly


def replace_nan_by_median(tx, nan_value):
    """Replaces values with a specified nan_value by the column median."""
    new_tx = copy.deepcopy(tx)
    new_tx[new_tx == nan_value] = np.nan
    col_median = np.nanmedian(new_tx, axis=0)
    return np.where(np.isnan(new_tx), col_median, new_tx)


def new_transformations(X):
    N = len(X)

    # abs
    cols_to_abs = [14,  # PRI_tau_eta
                   17,  # PRI_lep_eta
                   24,  # PRI_jet_leading_eta
                   27,  # PRI_jet_subleading_eta
                   ]
    abs_cols = []
    for c in cols_to_abs:
        abs_cols.append((abs(X[:, c])).reshape((N, 1)))

    # abs of difference
    cols_to_abs_diff = [(14, 17), (14, 24), (14, 27),
                        (17, 24), (17, 27),
                        (24, 27)]
    abs_diff_cols = []
    for c1, c2 in cols_to_abs_diff:
        abs_diff_cols.append((abs(X[:, c1] - X[:, c2])).reshape(N, 1))

    # interactions
    cols_to_interaction = [(14, 17), (14, 24), (14, 27),
                           (17, 24), (17, 27),
                           (24, 27)]
    interaction_cols = []
    for c1, c2 in cols_to_interaction:
        interaction_cols.append((X[:, c1] * X[:, c2]).reshape(N, 1))

    deltaphi_tau_lep = np.where(X[:, 15] - X[:, 18] < np.pi, X[:, 15] - X[:, 18],
                                2 * np.pi - X[:, 15] - X[:, 18]).reshape(N, 1)
    deltaphi_tau_jet1 = np.where(X[:, 15] - X[:, 25] < np.pi, X[:, 15] - X[:, 25],
                                 2 * np.pi - X[:, 15] - X[:, 25]).reshape(N, 1)
    deltaphi_tau_jet2 = np.where(X[:, 15] - X[:, 28] < np.pi, X[:, 15] - X[:, 28],
                                 2 * np.pi - X[:, 15] - X[:, 28]).reshape(N, 1)
    deltaphi_lep_jet1 = np.where(X[:, 18] - X[:, 25] < np.pi, X[:, 18] - X[:, 25],
                                 2 * np.pi - X[:, 18] - X[:, 25]).reshape(N, 1)
    deltaphi_lep_jet2 = np.where(X[:, 18] - X[:, 28] < np.pi, X[:, 18] - X[:, 28],
                                 2 * np.pi - X[:, 18] - X[:, 28]).reshape(N, 1)
    deltaphi_jet_jet = np.where(X[:, 25] - X[:, 28] < np.pi, X[:, 25] - X[:, 28],
                                2 * np.pi - X[:, 25] - X[:, 28]).reshape(N, 1)

    distance_cols = []
    distance_cols.append(np.sqrt(np.square(abs(X[:, 14] - X[:, 17]).reshape(N, 1)) + np.square(deltaphi_tau_lep)))
    distance_cols.append(np.sqrt(np.square(abs(X[:, 14] - X[:, 24]).reshape(N, 1)) + np.square(deltaphi_tau_jet1)))
    distance_cols.append(np.sqrt(np.square(abs(X[:, 14] - X[:, 27]).reshape(N, 1)) + np.square(deltaphi_tau_jet2)))
    distance_cols.append(np.sqrt(np.square(abs(X[:, 17] - X[:, 24]).reshape(N, 1)) + np.square(deltaphi_lep_jet1)))
    distance_cols.append(np.sqrt(np.square(abs(X[:, 17] - X[:, 27]).reshape(N, 1)) + np.square(deltaphi_lep_jet2)))
    distance_cols.append(np.sqrt(np.square(abs(X[:, 24] - X[:, 27]).reshape(N, 1)) + np.square(deltaphi_jet_jet)))
    d = (X[:, 15] - X[:, 18]).reshape(N, 1)
    d = 1.0 - 2.0 * ((d > np.pi) | ((d < 0) & (d > -np.pi)))
    a = np.sin(X[:, 20] - X[:, 18]).reshape(N, 1)
    b = np.sin(X[:, 15] - X[:, 20]).reshape(N, 1)
    distance_cols.append(d * (a + b) / np.sqrt(np.square(a) + np.square(b)).reshape(N, 1))
    #     list_transf.append(np.exp(-4.0*np.square(X[:, 17]-(X[:, 24]+X[:, 27])/2).reshape(N,1)/np.square(X[:, 24]-X[:, 27]).reshape(N,1)))
    #     list_transf.append(np.exp(-4.0*np.square(X[:, 24]-(X[:, 24]+X[:, 27])/2).reshape(N,1)/np.square(X[:, 24]-X[:, 27]).reshape(N,1)))

    cols_to_metric = [(19, 20, 13, 15), (19, 20, 16, 18), (19, 20, 23, 25), (19, 20, 26, 28),
                      (13, 15, 16, 18), (13, 15, 16, 18), (13, 15, 23, 25), (13, 15, 26, 28),
                      (16, 18, 23, 25), (16, 18, 26, 28), (23, 25, 26, 28)]
    metric_cols = []
    for c1, c2, c3, c4 in cols_to_metric:
        m = (np.square(X[:, c1] * np.cos(X[:, c2]) + X[:, c3] * np.cos(X[:, c4]))
             + np.square(X[:, c1] * np.sin(X[:, c2]) + X[:, c3] * np.sin(X[:, c4]))).reshape(N, 1)
        metric_cols.append(m)

    metric_to_mass = [(19, 13, 0), (19, 16, 1), (19, 23, 2), (19, 26, 3),
                      (13, 16, 4), (13, 23, 5), (13, 26, 6),
                      (16, 23, 7), (16, 26, 8),
                      (23, 26, 9)]
    metric2_cols = []
    for c1, c2, c3 in metric_to_mass:
        m = np.sqrt(abs(np.square((X[:, 19] + X[:, 13]).reshape(N, 1)) - metric_cols[c3]))
        metric2_cols.append(m)

    cols_to_p2 = [(4, 13, 14, 16, 17), (5, 13, 14, 23, 24), (6, 13, 14, 26, 27),
                  (7, 16, 17, 23, 24), (8, 16, 17, 26, 27),
                  (9, 23, 24, 26, 27)]
    p2_cols = []
    for c1, c2, c3, c4, c5 in cols_to_p2:
        m = (metric_cols[c1] + (
            np.square(X[:, c2] * np.square(X[:, c3])).reshape(N, 1) + (X[:, c4] * np.sinh(X[:, c5])).reshape(N,
                                                                                                             1))).reshape(
            N,
            1)
        p2_cols.append(m)

    cosh_to_cols = [(13, 14), (16, 17), (23, 24), (26, 27)]
    cosh_cols = []
    for c1, c2 in cosh_to_cols:
        m = (X[:, c1] * np.cosh(X[:, c2])).reshape(N, 1)
        cosh_cols.append(m)

    cols_to_mass = [(0, 1, 0), (0, 2, 1), (0, 3, 2), (1, 2, 3), (1, 3, 4), (2, 3, 5)]
    mass_cols = []
    for c1, c2, c3 in cols_to_mass:
        m = np.sqrt((np.square(abs((cosh_cols[c1] + cosh_cols[c2]).reshape(N, 1)) - p2_cols[c3]))).reshape(N, 1)
        mass_cols.append(m)

    s_px = (X[:, 19] * np.cos(X[:, 20]) + X[:, 13] * np.cos(X[:, 15]) + X[:, 16] * np.cos(X[:, 18])).reshape(N, 1)
    s_py = (X[:, 19] * np.sin(X[:, 20]) + X[:, 13] * np.sin(X[:, 15]) + X[:, 16] * np.sin(X[:, 18])).reshape(N, 1)
    distance_cols.append(np.sqrt(np.square(s_px) + np.square(s_py)).reshape(N, 1))

    s_px_2 = s_px + (X[:, 23] * np.cos(X[:, 25])).reshape(N, 1)
    s_px_2[np.isnan(s_px_2)] = 0
    s_py_2 = s_py + (X[:, 23] * np.sin(X[:, 25])).reshape(N, 1)
    s_py_2[np.isnan(s_py_2)] = 0
    distance_cols.append(np.sqrt(np.square(s_px_2) + np.square(s_py_2)).reshape(N, 1))

    s_px_3 = s_px_2 + (X[:, 26] * np.cos(X[:, 28])).reshape(N, 1)
    s_px_3[np.isnan(s_px_3)] = 0
    s_py_3 = s_py_2 + (X[:, 26] * np.sin(X[:, 28])).reshape(N, 1)
    s_py_3[np.isnan(s_py_3)] = 0
    distance_cols.append(np.sqrt(np.square(s_px_3) + np.square(s_py_3)).reshape(N, 1))

    sums_cols = []
    sums_cols.append((X[:, 19] + X[:, 13] + X[:, 16]).reshape(N, 1))
    sums_cols.append((sums_cols[0] + X[:, 23].reshape(N, 1)).reshape(N, 1))
    #     sums_cols.append((sums_cols[1] + X[:, 26].reshape(N,1)).reshape(N,1))
    #     sums_cols.append((sums_cols[1] + X[:, 29].reshape(N,1)).reshape(N,1))
    sums_cols.append((X[:, 13] + X[:, 16] + X[:, 29]).reshape(N, 1))

    ratio_cols = []
    ratio_cols.append((X[:, 16] / X[:, 13]).reshape(N, 1))

    valid_transformations = [abs_cols,
                             abs_diff_cols,
                             interaction_cols,
                             distance_cols,
                             metric_cols,
                             metric2_cols,
                             p2_cols,
                             mass_cols,
                             sums_cols,
                             ratio_cols
                             #                              cosh_cols
                             ]
    transformations = [item for sublist in valid_transformations for item in sublist]

    array_transf = np.hstack(transformations)

    return np.concatenate((X, array_transf), axis=1)


def dummyrize(X, col_index, allowed_values):
    X_plus = copy.deepcopy(X)
    col_to_dummy = X[:, col_index - 1]
    dummies = np.empty((len(X_plus), len(allowed_values) - 1))

    for i, v in enumerate(allowed_values[0:-1]):
        dummies[:, i] = (col_to_dummy == v) * 1

    return np.delete(np.concatenate((X_plus, dummies), axis=1), [col_index - 2], axis=1)


def apply_feature_eng(X):
    X_plus = copy.deepcopy(X)

    X_plus = replace_nan_by_median(X_plus, -999)

    X_plus = new_transformations(X_plus)

    X_plus = (X_plus - X_plus.mean(axis=0)) / X_plus.std(axis=0)

    X_plus = build_simple_poly(X_plus, 3)

    # dummy_cols = {22: [1, 2, 3, 4]}
    # for c in dummy_cols:
    #     X_plus = dummyrize(X_plus, c, dummy_cols[c]) # decrese performance


    return X_plus
