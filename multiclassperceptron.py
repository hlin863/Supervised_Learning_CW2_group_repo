# TODO - design a class for the multiclass perceptron

import numpy as np # import the numpy library
from scipy.spatial.distance import cdist

def polynomial_kernel(x, y, p):

    """
    
    This function computes the polynomial kernel of degree p between two vectors x and y.
    @param x: a vector
    @param y: a vector
    @return: the polynomial kernel of degree p between x and y
    
    """

    x = np.array(x)
    y = np.array(y)
    return (np.dot(x, y.T)) ** p


def gaussian_kernel(x, y, sigma):

    """
    
    This function computes the gaussian kernel with standard deviation sigma between two vectors x and y.
    @param x: a vector
    @param y: a vector
    @return: the gaussian kernel with standard deviation sigma between x and y
    
    """

    x = np.array(x)
    y = np.array(y)
    
    # formula e^-sigma * ||x - y||^2
    return np.exp(-sigma * cdist(x, y) ** 2)

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

        self.X_train = X_train # set the training data to the training data.
    
        self.y_train = y_train # set the training labels to the training labels.

        n_X_train = len(X_train) # get the length of the training data.

        converged = False # initialise the converged variable as false.

        self.alpha = np.zeros((self.number_of_classes, n_X_train)) # initialise the alpha vector as 0.

        gram_matrix = self.kernel(X_train, X_train) # get the gram matrix.

        index = 0 # index tracks the number of epochs

        running_mistakes = 0 # initialise the number of mistakes as 0.

        train_scores = [] # initialise the training scores as an empty list.

        while not converged and index <= 50: # while the model has not converged and the number of epochs is less than 50.

            running_mistakes = 0 # initialise the number of mistakes as 0.

            # iterate through the training data
            for i in range(n_X_train):

                x = X_train[i] # get the ith training data.

                y = y_train[i] # get the ith training label.

                # print(gram_matrix[i])

                confidences = np.dot(self.alpha, gram_matrix[i]) # get the confidences.

                y_pred = np.argmax(confidences) # get the predicted label with the largest confidence score.

                # print("y_pred: ", y_pred)

                if y_pred != y: # if the predicted label is not the same as the actual label.

                    running_mistakes += 1 # increment the number of mistakes.

                    self.alpha[int(y), i] += 1 # increment the alpha value for the actual label.

                    self.alpha[y_pred, i] -= 1 # decrement the alpha value for the predicted label.
            
            train_score = (n_X_train - running_mistakes) / n_X_train # get the training score.

            train_scores.append(train_score) # append the training score to the training scores list.

            if index >= 10:

                if (np.mean(train_scores[-5:]) - np.mean(train_scores[-10:-5])) < 0.01:

                    converged = True

            index += 1 # increment the index.

            return train_scores
    
    def test(self, X_test, y_test):

        """
        
        @param: X_test is the testing data.
        @param: y_test is the testing labels.
        """

        n_X_test = len(X_test) # initialise the size of the testing data.

        gram_matrix = self.kernel(self.X_train, X_test) # get the gram matrix.

        running_mistakes = 0 # initialise the number of mistakes as 0.

        confusion_matrix = np.zeros((self.number_of_classes, self.number_of_classes)) # initialise the confusion matrix as 0.

        misclassifications = np.zeros((n_X_test)) # initialise the misclassifications as 0.

        # iterate through the testing data
        for i in range(n_X_test):
            
            y = y_test[i] # get the ith testing label.

            confidences = np.dot(self.alpha, gram_matrix[:, i]) # get the confidences.

            y_pred = np.argmax(confidences) # get the predicted label with the largest confidence score.

            if y_pred != y: # if the predicted label is not the same as the actual label.

                running_mistakes += 1 # increment the number of mistakes.

                confusion_matrix[int(y), y_pred] += 1 # increment the confusion matrix.

                misclassifications[i] += 1 # increment the misclassifications.

        test_score = (n_X_test - running_mistakes) / n_X_test # get the test score.

        return test_score, confusion_matrix, misclassifications # return the test score, confusion matrix and misclassifications.

    def polynomial_fitting(self, degree):

        """
        
        @param: degree is the degree of the polynomial to be fitted.
        """
        self.kernel = lambda a, b: polynomial_kernel(a, b, degree) # set the kernel to the polynomial kernel.
    
    def gaussian_fitting(self, sigma):

        """
        
        @param: sigma is the standard deviation of the gaussian to be fitted.
        """
        
        self.kernel = lambda a, b: gaussian_kernel(a, b, sigma) # set the kernel to the gaussian kernel.