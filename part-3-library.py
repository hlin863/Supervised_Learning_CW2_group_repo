#######################################################################################
### Library to implement the algorithms: perceptron, winnow, least-squares and 1-NN ###
#######################################################################################

import numpy as np # import the numpy library to perform mathematical operations.

import matplotlib.pyplot as plt # import the matplotlib library to plot the data.

from sklearn.neighbors import KDTree

class PredictorBase():
    def __init__(self, name="") -> None:
        self.__model_name = name
        pass
    
    def fit(self, X, y):
        pass

    def pred(self, X):
        pass

    def get_name(self):
        return self.__model_name

class Winnow(PredictorBase):
    def __init__(self) -> None:
        super().__init__("Winnow")
    
    def fit(self, X, y):
        weights = np.ones((X.shape[1])) # initialize the weight vector to 1.

        self.__n = X.shape[1] # find the number of dimensions.

        X_ = (X == 1).astype(int)
        y_ = (y == 1).astype(int)

        # Training loop
        for i in range(X.shape[0]): # iterate over the training data.

            x_i, y_i = X_[i], y_[i] # receive patterns x = {0, 1} and y = {0, 1}.

            y_i_hat = -1 if weights @ x_i < n else 1 # predict the label.

            if (y_i_hat != y_i): # if the prediction is wrong, update the weights.
                weights *= np.float_power(2, ((y_i - y_i_hat) * x_i)) # update the weights.

        self.__w = weights

    def pred(self, X):
        X_ = (X == 1).astype(int)
        return np.where((X_ @ self.__w) < self.__n, -1, 1) # return the predicted labels.

class Perceptron(PredictorBase):
    def __init__(self) -> None:
        super().__init__("Perceptron")
    
    def fit(self, X, y):
        w = np.zeros(len(X[0])) # initialize the weight vector to 0.

        for t in range(len(X)): # iterate over the data from t = 1 to m.

            # receive patterns x = {0, 1}
            x = X[t]

            # predict y_pred
            y_pred = np.sign(np.dot(w, x))

            y_target = y[t]

            # implement a mistake function to calculate the number of mistakes in the perceptron algorithm to update the weight. 

            if y_pred != y_target: # if the prediction is wrong, update the weights.

                w = w + y_target * x

        self.__w = w

    def pred(self, X):
        return np.where(X @ self.__w <= 0, -1, 1)

class LeastSquare(PredictorBase):
    def __init__(self) -> None:
        super().__init__("Least Square")

    def fit(self, X, y):
        self.__w = np.linalg.pinv(X.T @ X) @ X.T @ y

    def pred(self, X):
        y_ = X @ self.__w
        return np.where(y_ <= 0, -1, 1)

class OneNearestNeighbor(PredictorBase):
    def __init__(self) -> None:
        super().__init__("1NN")

    def fit(self, X, y):
        self.__y = y
        self.__spatial = KDTree(X)
    
    def pred(self, X):
        indices = self.__spatial.query(X, k=1, return_distance=False)
        return self.__y[indices.ravel()].reshape(-1,1)

def mistake(y_pred, y_target):

    """
    
    Provides an implementation for the mistake function.

    @param y_pred: The predicted labels.
    @param y_target: The target labels.
    
    """
    Ierror = y_pred != y_target

    return np.sum(Ierror) / y_pred.shape[0]


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

    y = X[:, 0]

    return X, y # return the data and labels.

def train_test_split(data):

    """
    
    Implement a function to split the data into training and testing data.

    @param data: The data.
    
    @return: The training and testing data.

    """

    return data[:int(len(data) * 0.8)], data[int(len(data) * 0.8):] # return the training and testing data.

models = np.array([Winnow(), LeastSquare(), OneNearestNeighbor(), Perceptron()])

M = np.zeros((len(models), 100)) # create an array to store the errors by dimension.
E_S = 50

for n in range(1, 101):

    generalization_error = float('inf') # initialize the generalization error.

    # Approximate 2^n by maclaurin expansion up to x^2.
    # MC: We use this to sample the population \mathcal{X} to approximate expectation
    # TODO analyse the bias and try to incorporate importance weighting.
    N = int(n * np.log(2))
    L = int(1 + N)

    print("RUNNING: n = %0.3d / 100"%(n), end='\r')

    cnts = np.zeros((len(models)))

    for _ in range(E_S):
        masked = np.zeros(len(models), dtype=bool)
        m = 1
        while m <= 100:

            # No need for train test spilt, as we have direct access to D
            # we can simply drawn from D for any novel test points.
            X_test, y_test = sample_data(L, n)
            X, y = sample_data(m, n) # sample S_m

            # no need to proceed no classifier is pending
            if np.prod(masked) == 1:
                break
                
            for i, model in enumerate(models):
                if masked[i]:
                    continue

                model.fit(X, y)

                y_pred = model.pred(X_test) # predict the labels using the winnow algorithm.

                generalization_error = mistake(y_pred, y_test) # calculate the generalization error.

                if generalization_error <= 0.1: # if the generalization error is less than or equal to 0.1, allocate the value to the array.
                    
                    # mark this classifier as ok
                    masked[i] = True
                    M[i, n - 1] += m # store the generalization error.
                    cnts[i] += 1
                    break
                
            m += 1 # increment the number of samples.
    M[:, n - 1] /= np.maximum(cnts, 1)
print()

M[M == 0] = np.nan

for i in range(M.shape[0]):
    model = models[i]
    name = model.get_name()

    plt.plot(range(100), M[i,:], label=name) # plot the errors by dimension.

plt.xlabel("Dimensions (n)") # set the x-axis label.
plt.ylabel("Sample Size (m)") # set the y-axis label.
plt.legend()
plt.savefig("Part-3-images/plot_complexity.png") # save the plot as a png file.

print("SUCCESS!")