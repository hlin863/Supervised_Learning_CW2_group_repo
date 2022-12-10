#######################################################################################
### Library to implement the algorithms: perceptron, winnow, least-squares and 1-NN ###
#######################################################################################

import numpy as np # import the numpy library to perform mathematical operations.

import matplotlib.pyplot as plt # import the matplotlib library to plot the data.

def winnow_algorithm(X_train, X_test, y_train):

    """
    
    Provides an implementation for the winnow algorithm.

    @param X_train: The training dataset.
    @param X_test: The testing dataset.
    @param y: The labels.
    
    """

    # intialise a list to store the weights.

    weights = np.ones((X_train.shape[1])) # initialize the weight vector to 1.

    n = X_train.shape[1] # find the number of dimensions.

    # Training loop
    for i in range(X_train.shape[0]): # iterate over the training data.

        x_i, y_i = X_train[i], y_train[i] # receive patterns x = {0, 1} and y = {0, 1}.

        y_i_hat = 0 if weights @ x_i < n else 1 # predict the label.

        if (y_i_hat != y_i): # if the prediction is wrong, update the weights.
            weights *= np.float_power(2, ((y_i - y_i_hat) * x_i)) # update the weights.


    return np.where(X_test @ weights < n, 0, 1) # return the predicted labels.

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

classifiers = ["perceptron", "winnow", "least-squares", "1-NN"] # create a list to store the classifiers.


def plot_complexity(classifier_function, dimensions, samples_per_dimension):

    """
    
    Provides an implementation for plotting the complexity of the algorithms.

    @param classifier_function: The classifier function.

    @param dimensions: The dimensions.

    @param samples_per_dimension: The samples per dimension.
    
    """

    plt.plot(dimensions, samples_per_dimension, label=classifier_function.__name__) # plot the complexity of the algorithms.

    plt.xlabel("Dimensions") # set the x-axis label.

    plt.ylabel("Samples per dimension") # set the y-axis label.

    plt.savefig("complexity.png") # save the plot as a png file.

def sample_data(m, n):

    """
    
    Implement a function to sample data.

    @param m: The number of samples.

    @param n: The number of dimensions.
    
    """

    X = np.random.choice([-1, 1], size = (m, n)) # sample the data.

    y = np.random.choice([-1, 1], size = (m, 1)) # sample the labels.

    return X, y # return the data and labels.

def train_test_split(data):

    """
    
    Implement a function to split the data into training and testing data.

    @param data: The data.
    
    @return: The training and testing data.

    """

    return data[:int(len(data) * 0.8)], data[int(len(data) * 0.8):] # return the training and testing data.

for n in range(100):

    for m in range(40):

        X, y = sample_data(m, n) # sample the data.

        X_train, X_test = train_test_split(X) # split the data into training and testing data.

        y_train, y_test = train_test_split(y) # split the labels into training and testing labels.

        # y_pred = winnow_algorithm(X_train, X_test, y_train) # predict the labels using the winnow algorithm.

        y_pred = winnow_algorithm(X_train, X_test, y_train) # predict the labels using the winnow algorithm.

        n_errors = np.sum(y_pred != y_test) # find the number of errors.

        print("Number of errors: ", n_errors) # print the number of errors.

print("SUCCESS!")