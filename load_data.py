#################
### Libraries ###
#################
import os # import the libary to load the file.

#############################################################
### This file loads the data from /Datasets/zipcombo.data ###
#############################################################

def load_data():

    """
    
    This function loads the data from the zipcombo.data file.
    
    @return: the dataset from zipcombo.dat
    
    """

    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

    # add the subfolder Datasets to the location path
    __location__ = os.path.join(__location__, 'Datasets')

    # add the filename to the location path
    __location__ = os.path.join(__location__, 'zipcombo.dat')

    # open the file
    f = open(__location__, "r")

    # read the file
    data = f.read()

    # Close the file
    f.close()

    return data # Return the data