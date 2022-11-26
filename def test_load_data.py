import load_data 
import unittest
import os

class TestLoadData(unittest.TestCase):

    def test_load_data(self):

        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

        # add the subfolder Datasets to the location path
        __location__ = os.path.join(__location__, 'Datasets')

        # add the filename to the location path
        __location__ = os.path.join(__location__, 'zipcombo.dat')

        # open the file
        f = open(__location__, "r")

        data = f.read() # read the file

        f.close() # Close the file

        if self.assertEqual(load_data.load_data(), data): # Compare the data

            print("The data is the same")


if __name__ == '__main__':

    test = TestLoadData()

    test.test_load_data()
