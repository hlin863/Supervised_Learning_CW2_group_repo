import numpy as np # import the numpy library

def cross_validation(x, y, p):

    """
    
    @param: x is the 16 x 16 matrix representing the image of the digit.
    @param: y is the digit.
    @param: p is the degree of folding for the cross validation.

    @return: the list containing train and test data for each fold. 
    
    """

    # determine the number of entries
    n_x = len(x)

    random_indexes = np.arange(n_x) # create an array of indexes from 0 to n_x - 1

    np.random.shuffle(random_indexes) # shuffle the indexes

    result = [] # initialise the result as an empty list.

    for test_indices in np.array_split(random_indexes, p): # iterate through the indexes in p folds

        train_indices = np.setdiff1d(random_indexes, test_indices) # get the training indices

        x = np.array(x) # convert the x to a numpy array

        y = np.array(y) # convert the y to a numpy array

        x_train, x_test = x[train_indices], x[test_indices] # get the training and test data

        y_train, y_test = y[train_indices], y[test_indices] # get the training and test labels

        result.append([x_train, y_train, x_test, y_test]) # append the training and test data to the result list.

    return result # return the result as the list containing train and test data for each fold.

    