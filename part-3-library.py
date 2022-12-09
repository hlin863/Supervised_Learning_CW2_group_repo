#######################################################################################
### Library to implement the algorithms: perceptron, winnow, least-squares and 1-NN ###
#######################################################################################

import numpy as np

def winnow_algorithm(X, y):

    """
    
    Provides an implementation for the winnow algorithm.

    @param X: The input data.
    @param y: The labels.
    
    """

    # intialise a list to store the weights.

    weights = []

    w = 1 # initialize the weight vector to 1.

    weights.append(w)

    for t in range(len(X)): # iterate over the data from t = 1 to m.

        # receive patterns x = {0, 1}
        x = X[t]

        # predict y_pred
        y_pred = np.sign(np.dot(w, x))

        y_target = y[t]

        if y_pred != y_target: # if the prediction is wrong, update the weights.

            w = w * 2 ** (y_target * x)

        weights.append(w)

    return weights