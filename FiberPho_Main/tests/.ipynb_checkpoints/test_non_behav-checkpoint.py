import io
import os
import sys
import unittest
import pandas as pd
import plotly
import pickle
import random
from plotly.subplots import make_subplots
# from ..FiberPho_Main.FiberClass import FiberClass as fc
# sys.path.append(".fixtures/")
import FiberClass_Test as fc

""" A script to test FiberClass methods that don't require a behavior file

Methods
----------
setUp(): Unittest method
    Creates an object used for testing with sample data

test_raw_sig():
    Test for behavior import
    
test_norm_sig():
    Test for normalized signal traces
    
    
-----    
TODO:
Modify normalize sig function to actually check if signals 
have been normalized! Update ---- Sort of got the idea? Check w Kathleen
-----  
"""

# Tests for non-behavior methods
class test_non_behav(unittest.TestCase):
    # Setup for initializing object w dummy params/data
    def setUp(self):
        # Sample data file paths
        test_data = "/Users/yobae/Desktop/CS Stuff/fiberobj_panel/Yobe_GUI/fiberpho_gui/tests/src/Sample_Data_1.csv"
        test_data_2 = "/Users/yobae/Desktop/CS Stuff/fiberobj_panel/Yobe_GUI/fiberpho_gui/tests/src/Sample_Data_2.csv"
        test_data_3 = "/Users/yobae/Desktop/CS Stuff/fiberobj_panel/Yobe_GUI/fiberpho_gui/tests/src/Sample_Data_3.csv"
        test_data_4 = "/Users/yobae/Desktop/CS Stuff/fiberobj_panel/Yobe_GUI/fiberpho_gui/tests/src/Sample_Data_4.csv"
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

    
    # Test for raw signal trace
    def test_raw_sig(self):
        # Error to relay if test case fails
        msg = "Raw Signal Trace failed to return a plot"
        # Test Case
        # This checks if the function will return a plot correctly
        self.assertIsInstance(self.test_obj.raw_signal_trace(), plotly.graph_objs._figure.Figure, msg)
        self.assertIsInstance(self.test_obj_2.raw_signal_trace(), plotly.graph_objs._figure.Figure, msg) 
        self.assertIsInstance(self.test_obj_3.raw_signal_trace(), plotly.graph_objs._figure.Figure, msg) 
        self.assertIsInstance(self.test_obj_4.raw_signal_trace(), plotly.graph_objs._figure.Figure, msg) 
        # test_fig = self.test_obj.raw_signal_trace()
        # test_fig.show()
        # self.assertEqual(self.test_obj.raw_signal_trace(), test_fig, msg) # This checks if the function call is equivalent to the figure from an earlier function call
        
    # Test for normalized signals
    def test_norm_sig(self):
        msg = "Normalize Signal method failed to return a plot"
        # Example Signal/Reference to pass
        signals = ["Raw_Green", "Raw_Red", "Raw_Isosbestic"]
        refs = ["Raw_Green", "Raw_Red", "Raw_Isosbestic"]
        # Test Case
        # This will check if the function will return a figure object without error
        # These tests should prove sufficient bc it will not produce a graph if an error is raised
        self.assertIsInstance(self.test_obj.normalize_a_signal(random.choice(signals), 
                                                               random.choice(refs)), plotly.graph_objs._figure.Figure, msg)
        
        self.assertIsInstance(self.test_obj_2.normalize_a_signal(random.choice(signals), 
                                                                 random.choice(refs)), plotly.graph_objs._figure.Figure, msg)
        
        self.assertIsInstance(self.test_obj_3.normalize_a_signal(random.choice(signals), 
                                                                 random.choice(refs)), plotly.graph_objs._figure.Figure, msg)
        
        self.assertIsInstance(self.test_obj_4.normalize_a_signal(random.choice(signals), 
                                                                 random.choice(refs)), plotly.graph_objs._figure.Figure, msg)
        
        # Test if exception is properly raised
        self.assertRaises(KeyError, lambda: self.test_obj_4.normalize_a_signal("Raw_Poop", "Raw_Pee"))
        self.assertRaises(KeyError, lambda: self.test_obj_4.normalize_a_signal('', ''))
        
if __name__ == '__main__':
    unittest.main()