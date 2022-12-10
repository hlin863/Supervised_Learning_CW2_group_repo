import numpy as np # import the numpy library

def confusion_matrix_function(confusion_matrix):

    """
    
    This function aims to normalise the confusion matrix values

    @param: confusion_matrix is the confusion matrix to be printed.
    
    @return: the normalised confusion matrix.
    
    
    """

    return np.nan_to_num(confusion_matrix / confusion_matrix.sum(axis=1)[:,np.newaxis])