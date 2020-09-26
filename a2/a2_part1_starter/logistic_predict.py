"""
 Princeton University, COS 429, Fall 2020
"""


def logistic_predict(X, params):
    """Given a logistic model and some new data, predicts classification

    Args:
        X: datapoints (one per row, should include a column of ones
                       if the model is to have a constant)
        params: vector of parameters 

    Returns:
        z: predicted labels (0/1)
    """
    # We could evaluate the linear part, apply the nonlinearity, and
    # compare against a theshold of 0.5.  But that's equivalent to just
    # comparing the linear part, before the nonlinearity, against 0.
    return X @ params >= 0
