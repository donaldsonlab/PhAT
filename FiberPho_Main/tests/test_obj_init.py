import io
import sys
import unittest
import pandas as pd
import pickle
# from ..FiberPho_Main.FiberClass import fiberObj
sys.path.append("../FiberPho_Main/")
import FiberClass as fc

""" A script to ensure fiber objects can be created.

Methods
----------
setUp(): Unittest method
    Creates an object used for testing with sample data

test_init():
    Tests for object initialization
"""

""" 
TODO:
Get multiple sample sets to test multiple object creations
"""

# Test for object initialization case
class test_obj_init(unittest.TestCase):
    # Initialize object with dummy params and sample fiber data
    def setUp(self): # setUp and tearDown are functions of the unittest framework
        # Sample data file paths
        test_data = "/Users/yobae/Desktop/CS Stuff/fiberobj_panel/Yobe_GUI/fiberpho_gui/tests/src/files/Sample_Data_1.csv"
        test_data_2 = "/Users/yobae/Desktop/CS Stuff/fiberobj_panel/Yobe_GUI/fiberpho_gui/tests/src/files/Sample_Data_2.csv"
        test_data_3 = "/Users/yobae/Desktop/CS Stuff/fiberobj_panel/Yobe_GUI/fiberpho_gui/tests/src/files/Sample_Data_3.csv"
        test_data_4 = "/Users/yobae/Desktop/CS Stuff/fiberobj_panel/Yobe_GUI/fiberpho_gui/tests/src/files/Sample_Data_4.csv"
        # Read into dataframe
        df = pd.read_csv(test_data)
        df_2 = pd.read_csv(test_data_2) 
        df_3 = pd.read_csv(test_data_3) 
        df_4 = pd.read_csv(test_data_4) 


        self.test_obj = fc.fiberObj(df, 'Test_Object', 1, 2, '02/11', '2:27',
                            -1, 0, 'Sample_Data_1')
        self.test_obj_2 = fc.fiberObj(df_2, 'Test_Object_2', 2, 2, '04/11', '2:22',
                            -1, 0, 'Sample_Data_2')
        self.test_obj_3 = fc.fiberObj(df_3, 'Test_Object_3', 1, 1, '05/11', '2:23',
                            -1, 0, 'Sample_Data_3')
        self.test_obj_4 = fc.fiberObj(df_4, 'Test_Object_4', 2, 1, '06/11', '2:24',
                            -1, 0, 'Sample_Data_4')
        
        
    # Test for object initialization
    def test_init(self):
        # Error to relay if test case fails
        message = "Given object is not an instance of fiberObj"
        # Check for object instance
        self.assertIsInstance(self.test_obj, fc.fiberObj, message)
        self.assertIsInstance(self.test_obj_2, fc.fiberObj, message)
        self.assertIsInstance(self.test_obj_3, fc.fiberObj, message)
        self.assertIsInstance(self.test_obj_4, fc.fiberObj, message)
        
        
    # Test for pickled object uploads
    def test_pkl(self):
        # Pickle file path
        infile_1 = "/Users/yobae/Desktop/CS Stuff/fiberobj_panel/Yobe_GUI/fiberpho_gui/tests/src/pickles/Pkl_Sample_1.pickle"
        infile_2 = "/Users/yobae/Desktop/CS Stuff/fiberobj_panel/Yobe_GUI/fiberpho_gui/tests/src/pickles/Pkl_Sample_2.pickle"
        message = "Pickle cannot be uploaded"
        
        pkl_1 = open(infile_1, 'rb')
        obj_1 = pickle.load(pkl_1)
        pkl_1.close()
        
        pkl_2 = open(infile_2, 'rb')
        obj_2 = pickle.load(pkl_2)
        pkl_2.close()
        # Check if pickle has been read succesfully
        self.assertIsInstance(obj_1, fc.fiberObj, message)
        self.assertIsInstance(obj_2, fc.fiberObj, message)
         
        
if __name__ == '__main__':
    unittest.main()