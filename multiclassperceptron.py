# TODO - design a class for the multiclass perceptron

import numpy as np

def polynomial_kernel(x, y, p):
    x = np.array(x)
    y = np.array(y)
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

                confidences = np.dot(self.alpha, gram_matrix[i]) # get the confidences.

                y_pred = np.argmax(confidences) # get the predicted label with the largest confidence score.

                # print("y_pred: ", y_pred)

                if y_pred != y: # if the predicted label is not the same as the actual label.

                    self.alpha[int(y), i] += 1 # increment the alpha value for the actual label.

                    self.alpha[y_pred, i] -= 1 # decrement the alpha value for the predicted label.
            
            train_score = (n_X_train - running_mistakes) / n_X_train # get the training score.

            train_scores.append(train_score) # append the training score to the training scores list.

            if index >= 10:

                if (np.mean(train_scores[-5:]) - np.mean(train_scores[-10:-5])) < 0.01:

                    converged = True

            index += 1 # increment the index.

            return train_scores

    def polynomial_fitting(self, degree):

        """
        
        @param: degree is the degree of the polynomial to be fitted.

        """
        self.kernel = lambda a, b: polynomial_kernel(a, b, degree) # set the kernel to the polynomial kernel.