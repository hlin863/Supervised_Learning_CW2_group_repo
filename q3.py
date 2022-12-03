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

from confusion import confusion_matrix_function # import the confusion_matrix_function from confusion.py

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

confusion_matrices = np.empty((20, 10, 10))

confusion_matrice_stds = np.empty((20, 10, 10)) # initialise a variable to store the list of confusion matrices' standard deviations

# reshuffle the labels
for runs in range(20): # do 20 runs on the dataset.

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

    mean_test_scores = np.empty((7)) # initialise an empty array to store the mean test scores

    misclassifications = np.zeros((len(matrixs_random[0]))) # initialise an array to store the misclassifications

    for d in range(1, 8): # iterate through each polynomial degree

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

        mean_test_score = np.mean(test_scores) # get the mean test score

        mean_test_scores[d - 1] = mean_test_score # store the mean test score
    
    argmax = np.argmax(mean_test_scores) # get the index of the maximum mean test score

    degrees = [1, 2, 3, 4, 5, 6, 7] # create a list of the degrees

    best_degree = degrees[argmax] # get the best degree

    model = MultiClassPerceptron(10) # create a multi class perceptron model

    model.polynomial_fitting(best_degree) # fit the model to the data with the polynomial kernel of degree d.

    model.train(X_train, y_train) # train the model

    test_score, confusion_matrix, _ = model.test(X_test, y_test) # test the model

    data_score, _, misclassification = model.test(matrixs_random, labels_random) # test the model on the whole dataset

    confusion_matrices[runs] = confusion_matrix_function(confusion_matrix)

    confusion_matrix_std = np.std(confusion_matrices, axis=0) # calculate the standard deviation of the confusion matrices

    confusion_matrice_stds[runs] = confusion_matrix_std

    # displays the confusion matrix
    print("The confusion matrix for run " + str(runs) + " is: ")

    print("[")

    for i in range(len(confusion_matrix)):

        row = ""

        for j in range(len(confusion_matrix[i])):
            
            row = row + (str(confusion_matrix[i][j]) + "+/-" + str(confusion_matrix_std[i][j]))

        print("[" + row + "]")

    print("]")

print("SUCCESS") # print success if the code runs without errors