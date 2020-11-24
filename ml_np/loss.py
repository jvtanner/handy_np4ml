import numpy as np


def L1(yhat, y):
    """
    L1 loss function: L1(yhat, y) = sum(|y(i) - yhat(i)|)

    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)

    Returns:
    loss -- the value of the L1 loss function
    """

    ### START CODE HERE ### (≈ 1 line of code)
    loss = np.sum(abs(y - yhat))
    ### END CODE HERE ###

    return loss


def L2(yhat, y):
    """
    L2 loss function: L2(yhat, y) = sum(y(i) - yhat(i))^2

    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)

    Returns:
    loss -- the value of the L2 loss function defined above
    """

    ### START CODE HERE ### (≈ 1 line of code)
    loss = np.dot(y - yhat, y - yhat)
    ### END CODE HERE ###

    return loss

