#########################################################################
### Load zipcombo data using the load_data function from load_data.py ###
#########################################################################

import numpy as np # import the numpy library

# import the load_data function from load_data.py
from load_data import load_data

# import the train_test_split function from split_data.py
from split_data import split_data

import random # import the random library to help with schuffling the data. 

from multiclassperceptron import MultiClassPerceptron # import the multiclassperceptron class from multiclassperceptron.py

from cross_validation import cross_validation # import the cross_validation function from cross_validation.py

import pandas as pd # import the pandas library to help with the confusion matrix

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

test_errors = np.empty((20)) # initialise the test scores as an empty list to store the test errors for each run.

best_degrees = np.empty((20)) # initialise the best degrees as an empty list to store the best degree for each run.

# reshuffle the labels
for runs in range(20): # perform 20 runs

    train_score_list = np.empty((8)) # create an empty array to store the training scores
    test_score_list = np.empty((8)) # initialise the test scores as an empty list.
    train_std_list = np.empty((8)) # create an empty array to store the training standard deviations
    test_std_list = np.empty((8)) # initialise the test standard deviations as an empty list.

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

    for d in range(8): # iterate through the different polynomial degrees.

        cross_validation_data = cross_validation(X_train, y_train, 5) # perform 5 fold cross validation on the training data

        train_scores = np.empty((5)) # create an empty array to store the training scores

        test_scores = np.empty((5)) # initialise the test scores as an empty list.
    
        for index, (X_train, y_train, X_test, y_test) in enumerate(cross_validation_data): # loop through the cross validation data

            model = MultiClassPerceptron(10) # create a multi class perceptron model

            model.polynomial_fitting(d) # fit the model to the data with the polynomial kernel of degree d. 

            train_score = model.train(X_train, y_train)[0] # train the model

            test_score, _, _ = model.test(X_test, y_test) # test the model

            train_scores[index] = train_score # store the training score

            test_scores[index] = test_score # store the test score
        
        train_score_list[d] = np.mean(train_scores) # store the average training score
        test_score_list[d] = np.mean(test_scores) # store the average test score

        train_std_list[d] = np.std(train_scores) # store the standard deviation of the training scores
        test_std_list[d] = np.std(test_scores) # store the standard deviation of the test scores

    # selects the degree with the highest test score from test_score_list. 
    best_degree = np.argmax(test_score_list) # determines the "best" parameter d for retrain on the full training and testing set.

    model = MultiClassPerceptron(10) # create a multi class perceptron model

    model.polynomial_fitting(best_degree) # fit the model to the data with the polynomial kernel of degree d.

    best_train_score = model.train(X_train, y_train)[0] # train the model

    best_test_score, _, _ = model.test(X_test, y_test) # test the model

    best_test_error = 1 - best_test_score # calculate the test error on the remaining 20% test data. 

    test_errors[runs] = best_test_error # store the test error

    best_degrees[runs] = best_degree # store the best degree

print("The average test error is: ", np.mean(test_errors)) # print the average test error

print("The test error standard deviation is: ", np.std(test_errors)) # print the test error standard deviation

print("The average best degree is: ", np.mean(best_degrees)) # print the average best degree

print("The best degree standard deviation is: ", np.std(best_degrees)) # print the best degree standard deviation

# convert the test errors to a pandas dataframe
test_errors_df = pd.DataFrame(test_errors)

# convert the best degrees to a pandas dataframe
best_degrees_df = pd.DataFrame(best_degrees)

# save the test errors to a csv file
test_errors_df.to_csv("test_errors.csv")

# save the best degrees to a csv file
best_degrees_df.to_csv("best_degrees.csv")

print("SUCCESS") # print success if the code runs without errors