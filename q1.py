#########################################################################
### Load zipcombo data using the load_data function from load_data.py ###
#########################################################################

import numpy as np # import the numpy library

# import the load_data function from load_data.py
from load_data import load_data

# import the train_test_split function from split_data.py
from split_data import split_data

# load the data
data = load_data()

data = data.split("\n") # split the data into lines

processed_data = []

for line in data: # loop through the lines
    line = line.split(" ") # split the line into columns

    # remove the empty strings
    line = [x for x in line if x != '']

    processed_data.append(line) # append the line to the processed_data list

print(processed_data[0]) # print the first line

for i in range(len(processed_data)): # loop through the lines to convert the data to floats
    
    processed_data[i] = list(np.array(processed_data[i], dtype=np.float)) # convert the data to floats

print(processed_data[0]) # print the first line