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
    
        model = MultiClassPerceptronOvO(10) # create a multi class perceptron model

        model.gaussian_fitting(d) # fit the model to the data with the polynomial kernel of degree d. 

        train_score = model.train(X_train, y_train) # train the model

        test_score, _ = model.test(X_test, y_test) # test the model

        train_scores[runs] = train_score[0] # append the training score to the training scores list

        test_scores[runs] = test_score # append the test score to the test scores list

    print("The average training score for d = " + str(d) + " is " + str(np.mean(train_scores))) # print the average training score

    print("The standard deviation of the training score for d = " + str(d) + " is " + str(np.std(train_scores))) # print the standard deviation of the training score

    print("The average test score for d = " + str(d) + " is " + str(np.mean(test_scores))) # print the average test score

    print("The standard deviation of the test score for d = " + str(d) + " is " + str(np.std(test_scores))) # print the standard deviation of the test score

print("SUCCESS") # print success if the code runs without errors