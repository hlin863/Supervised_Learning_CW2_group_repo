# TODO - design a class for the multiclass perceptron

class MultiClassPerceptron:

    def __init__(self, d):

        """
        
        @param: d represents the number of classes for each run
        
        """

        self.number_of_classes = d # initialises the number of classes

        self.X_train = None # initialises a trainingset for the data.
    
    def train(self, X_train, y_train):

        """
        
        @param: X_train is the training data.
        @param: y_train is the training labels.

        """

        w = 0 # initialise the weight vector as 0.

        self.X_train = X_train # set the training data to the training data.
    
    def polynomial_fitting(self, degree):

        """
        
        @param: degree is the degree of the polynomial to be fitted.

        """

        # TODO - implement the polynomial fitting
        pass