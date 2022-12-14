#########################################################################
### Load zipcombo data using the load_data function from load_data.py ###
#########################################################################

#########################################################################
### EXPERIMENTAL SETUP                                                ###
### DO NOT ASSESS THIS FILE                                           ###
#########################################################################

import numpy as np # import the numpy library

import pandas as pd # import the pandas library

# initialise a variable as the array to sore the kernel widths. 
KERNEL_WIDTHS = [2 ** k for k in range(-15, -8)]

# import the load_data function from load_data.py
from load_data import load_data

# import the train_test_split function from split_data.py
from split_data import split_data

import random # import the random library to help with schuffling the data. 

from multiclassperceptron import MultiClassPerceptron # import the multiclassperceptron class from multiclassperceptron.py
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

NUM_CLASSES = 10

TRAIN_TEST_SPLIT = 0.8

WIDTHS = [2 ** k for k in range(-15, 10)]
RUNS = 20

data_summary = []

for width in WIDTHS: # loop through the kernel widths

  train_scores = np.empty((RUNS)) # create an empty array to store the train accuracy
  test_scores = np.empty((RUNS)) # create an empty array to store the test accuracy

  for run in range(RUNS): # loop through the runs

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

    # Initialise model
    model = MultiClassPerceptron(NUM_CLASSES)
    model.set_gaussian_kernel(width)

    # Train model and evaluate on test set
    train_accuracy = model.train(X_train, y_train)[-1]
    test_accuracy, _, _ = model.test(X_test, y_test)

    train_scores[run] = train_accuracy
    test_scores[run] = test_accuracy

    print(f"Gaussian kernel width: {width}. Run: {run}. Train accuracy: {train_accuracy}. Test accuracy: {test_accuracy}.")

  # Calculate mean and std for train and test accuracy
  train_error_mean, train_error_std = 1 - np.mean(train_scores), np.std(train_scores)
  test_error_mean, test_error_std = 1 - np.mean(test_scores), np.std(test_scores)

  data_summary.append([width, train_error_mean, train_error_std, test_error_mean, test_error_std]) # adds the data to the summary
  
# Create a DataFrame of the summary data
summary = pd.DataFrame(data_summary, columns = ('width', 'train_error_mean', 'train_error_std', 'test_error_mean', 'test_error_std'))

# Save the dataframe as an excel file
summary.to_excel("summary.xlsx")