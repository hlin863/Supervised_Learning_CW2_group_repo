##############################################################################################
### This file contains a function to spli the data into train and test sets for question 1 ###
##############################################################################################

def split_data(data):

    """
    
    This function splits the data into train and test sets: 80% for training and 20% for testing.

    @param data: The data to be split

    @return: The train and test sets
    
    """
    train = data[:int(len(data)*0.8)] # 80% of the data for training
    test = data[int(len(data)*0.8):]  # 20% of the data is used for testing
    
    return train, test # Return the train and test sets