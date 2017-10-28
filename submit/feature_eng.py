from itertools import chain, combinations_with_replacement
import numpy as np

def build_advanced_poly(tx, degree, cols=[]):
    """
    Builds full polynomial basis function for input data matrix tx, for j=0 up to j=degree,
    where the result will be a matrix of form:
    [1, tx, comb_mult(tx, 2), ..., comb_mult(tx, j)]
    comb_mult(tx, 2) denotes all multiplicative combinations of the selected columns of tx.
    If cols is not given, it returns the combinations of all columns of tx.
    Example: Given input tx=[a,b], the output will be [1,a,b,a*a,a*b,b*b]

    Keyword arguments:
    tx -- input data
    degree -- maximal degree for polynomial expansion
    cols -- Columns to consider for polynomial expansion (default: [])
    """
    if cols:
        tx = tx[:, cols]
    N = tx.shape[0]
    D = tx.shape[1] if tx.shape[1:] else 1

    comb = combinations_with_replacement
    combinations = chain.from_iterable(comb(range(D), i) for i in range(0, degree + 1))
    dimension_count_iter = chain.from_iterable(comb(range(D), i) for i in range(0, degree + 1))
    x_poly_D = sum(1 for _ in dimension_count_iter)

    x_poly = np.empty((N, x_poly_D), dtype=tx.dtype)
    for i, c in enumerate(combinations):
        x_poly[:, i] = tx[:, c].prod(1)

    return x_poly


def build_simple_poly(tx, degree):
    """
    Builds simple polynomial basis function for input data matrix tx, for j=0 up to j=degree,
    where the result will be a matrix of form [1, tx, tx^2, ..., tx^j].
    tx^j denotes that for each x_i,k in tx, the result will be (x_i,k)^j

    Keyword arguments:
    tx -- input data
    degree -- maximal degree for polynomial expansion
    """
    poly = np.ones((tx.shape[0], 1))

    for j in range(1, degree + 1):
        poly = np.column_stack((poly, np.power(tx, j)))

    return poly


def replace_nan_by_mean(tx, nan_value):
    """Replaces values with a specified nan_value by the column mean."""
    tx[tx == nan_value] = np.nan
    col_mean = np.nanmean(tx, axis=0)
    return np.where(np.isnan(tx), col_mean, tx)


def replace_nan_by_median(tx, nan_value):
    """Replaces values with a specified nan_value by the column median."""
    tx[tx == nan_value] = np.nan
    col_median = np.nanmedian(tx, axis=0)
    return np.where(np.isnan(tx), col_median, tx)


def drop_nan_rows(tx, nan_value):
    """Drop all rows that contain a nan equaling the specified nan_value."""
    tx[tx == nan_value] = np.nan
    return tx[~np.isnan(tx).any(axis=1)]


def standardize(tx, mean=None, std=None):
    """Standardizes the matrix and returns mean and standard deviation.
    All 0's in the std have been replaced by 1's to facilitate division by the std."""
    if mean is None:
        mean = np.mean(tx, axis=0)
    if std is None:
        std = np.std(tx, axis=0)
    std[std == 0] = 1
    tx = tx - mean
    tx = tx / std
    return tx, mean, std


def de_standardize(x, mean, std):
    """Reverse the procedure of standardization."""
    x = x * std
    x = x + mean
    return x
