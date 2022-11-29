# TODO - design a class for the multiclass perceptron

import numpy as np

def polynomial_kernel(x, y, p):
    return (np.dot(x, y.T)) ** p

class MultiClassPerceptron:

    def __init__(self, d):

        """
        
        @param: d represents the number of classes for each run
        
        """

        self.number_of_classes = d # initialises the number of classes

        self.X_train = None # initialises a trainingset for the data.

        self.y_train = None # initialises a trainingset for the labels.
    
    def train(self, X_train, y_train):

        """
        
        @param: X_train is the training data.
        @param: y_train is the training labels.

        """

        w = 0 # initialise the weight vector as 0.

        self.X_train = X_train # set the training data to the training data.
    
        self.y_train = y_train # set the training labels to the training labels.

    def polynomial_fitting(self, degree):

        """
        
        @param: degree is the degree of the polynomial to be fitted.

        """
        self.kernel = polynomial_kernel(self.X_train, self.y_train, degree) # set the kernel to the polynomial kernel.