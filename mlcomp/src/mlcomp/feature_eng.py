import itertools
import numpy as np
import copy


def replace_nan_by_median(tx, nan_value):
    """Replaces values with a specified nan_value by the column median."""
    new_tx = copy.deepcopy(tx)
    new_tx[new_tx == nan_value] = np.nan
    col_median = np.nanmedian(new_tx, axis=0)
    return np.where(np.isnan(new_tx), col_median, new_tx)


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
