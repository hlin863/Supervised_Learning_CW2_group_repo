#########################################################################
### Load zipcombo data using the load_data function from load_data.py ###
#########################################################################

import numpy as np # import the numpy library

# import the load_data function from load_data.py
from load_data import load_data

# import the train_test_split function from split_data.py
from split_data import split_data

import random # import the random library to help with schuffling the data. 

from q6_multi_perceptron import MultiClassPerceptronOvO # import the multiclassperceptron class from multiclassperceptron.py

from cross_validation import cross_validation # import the cross_validation function from cross_validation.py

# load the data
data = load_data()

data = data.split("\n") # split the data into lines

processed_data = []

for line in data: # loop through the lines
    line = line.split(" ") # split the line into columns

    # remove the empty strings
    line = [x for x in line if x != '']

    processed_data.append(line) # append the line to the processed_data list


for i in range(len(processed_data)): # loop through the lines

    processed_data[i] = [float(x) for x in processed_data[i]] # convert the columns to floats

labels = [] # create an empty list to store the digits
matrixs = [] # create an empty list to store the matrix value of the digits representing an image.


for i in range(len(processed_data) - 1): # loop through the lines to remove the first column

    digit = processed_data[i][0] # get the digit
    matrix = processed_data[i][1:] # remove the first column

    labels.append(digit) # append the digit to the labels list

    matrixs.append(matrix) # append the matrix to the matrixs list

# reshuffle the labels
for d in range(1, 8): # do 20 runs on the dataset.

    train_scores = np.empty((20)) # create an empty array to store the training scores

    test_scores = np.empty((20)) # initialise the test scores as an empty list.

    for runs in range(20):

        # join digit and matrix
        joined = list(zip(labels, matrixs))

        # sort the joined list
        joined.sort(key=lambda x: random.random())

        # split the joined list
        labels_random, matrixs_random = zip(*joined)

        # convert the labels and matrixs to lists
        labels_random = list(labels_random)

        matrixs_random = list(matrixs_random)

        X_train, X_test = split_data(matrixs_random) # split the data into training and testing data

        y_train, y_test = split_data(labels_random) # split the labels into training and testing labels

        cross_validation_data = cross_validation(X_train, y_train, 5) # perform 5 fold cross validation on the training data

        train_scores = np.empty((5)) # create an empty array to store the training scores

        test_scores = np.empty((5)) # initialise the test scores as an empty list.
    
        for index, (X_train, y_train, X_test, y_test) in enumerate(cross_validation_data): # loop through the cross validation data

            model = MultiClassPerceptronOvO(10) # create a multi class perceptron model

            model.polynomial_fitting(d) # fit the model to the data with the polynomial kernel of degree d. 

            train_score = model.train(X_train, y_train)[0] # train the model

            test_score, _ = model.test(X_test, y_test) # test the model

            train_scores[index] = train_score # store the training score

            test_scores[index] = test_score # store the test score

        print("Average training score for degree " + str(d) + " is " + str(np.mean(train_scores)) + " with cross-validation training. ") # print the average training score under cross validation
        print("Average test score for degree " + str(d) + " is " + str(np.mean(test_scores)) + " with cross-validation training. ") # print the average test score under cross validation

        print("Average training standard deviation for degree " + str(d) + " is " + str(np.std(train_scores)) + " with cross-validation training. ") # print the average training score under cross validation
        print("Average test standard deviation for degree " + str(d) + " is " + str(np.std(test_scores)) + " with cross-validation training. ") # print the average test score under cross validation

print("SUCCESS") # print success if the code runs without errors