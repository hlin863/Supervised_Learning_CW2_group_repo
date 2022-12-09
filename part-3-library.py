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

def perceptron_algorithm(X, y):

    """
    
    Provides an implementation for the perceptron algorithm.

    @param X: The input data.
    @param y: The labels.
    
    """

    weights = [] # intialise a list to store the weights.

    M = [] # initialise a list to store the margin.

    w = np.zeros(len(X[0])) # initialize the weight vector to 0.

    weights.append(w) # append the initial weight vector.

    M.append(0) # append the initial margin.

    for t in range(len(X)): # iterate over the data from t = 1 to m.

        # receive patterns x = {0, 1}
        x = X[t]

        # predict y_pred
        y_pred = np.sign(np.dot(w, x))

        y_target = y[t]

        # implement a mistake function to calculate the number of mistakes in the perceptron algorithm to update the weight. 

        if y_pred != y_target: # if the prediction is wrong, update the weights.

            w = w + y_target * x

        weights.append(w)

    return weights


def least_squares_algorithm(y_pred, y):

    """

    Provides an implementation for the least squares algorithm.

    @param y_pred: The predicted labels.

    @param y: The labels.

    """

    # find the difference between y_pred and y squared.
    difference_squared = (y_pred - y) ** 2

    # find the minimal difference between y_pred and y squared.
    min_difference_squared = np.min(difference_squared)

    return min_difference_squared
    

def mistake(y_pred, y_target):

    """
    
    Provides an implementation for the mistake function.

    @param y_pred: The predicted labels.
    @param y_target: The target labels.
    
    """

    pass