def compute_loss_mse(y, tx, w):
    """Calculate the MSE loss."""
    N = len(y)
    e = y - np.dot(tx, w)
    return np.dot(e, e) / (2 * N)


def compute_loss_mae(y, tx, w):
    """Calculate the MAE loss."""
    N = len(y)
    e = y - np.dot(tx, w)
    return np.sum(np.absolute(e)) / N


def compute_rmse(y, tx, w):
    """Computes the Root Mean Square Error"""
    mse = compute_loss_mse(y, tx, w)
    return np.sqrt(2 * mse)


def compute_gradient_mse(y, tx, w):
    """Compute the MSE gradient."""
    N = len(y)
    e = y - np.dot(tx, w)
    return (-1 / N) * np.dot(np.transpose(tx), e)


def compute_stochastic_subgradient_mae(y, tx, w):
    """Compute a stochastic subgradient from just few examples n and their corresponding y_n labels."""
    N = len(y)
    e = y - np.dot(tx, w)
    abs_e_subgrad = [np.sign(en) for en in e]  # Sign chosen for subgradient of absolute value function
    return (-1 / N) * np.dot(np.transpose(tx), abs_e_subgrad)


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing
    """
    np.random.seed(seed)  # set seed
    permuted_idxs = np.random.permutation(x.shape[0])
    train_size = int(ratio * x.shape[0])
    train_idxs, test_idxs = permuted_idxs[:train_size], permuted_idxs[train_size:]

    return x[train_idxs], x[test_idxs], y[train_idxs], y[test_idxs]


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred
