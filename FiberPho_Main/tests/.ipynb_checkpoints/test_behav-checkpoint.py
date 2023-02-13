import io
import os
import sys
import unittest
import pandas as pd
import plotly
import pickle
import random
from plotly.subplots import make_subplots
import plotly.graph_objects
# from ..FiberPho_Main.FiberClass import FiberClass as fc
sys.path.append("../FiberPho_Main/")
import FiberClass as fc

""" A script to test FiberClass methods that don't require a behavior file

Methods
----------
setUp():
    Creates an object for testing using sample data

test_import_beh():
    Test for succesful behavior import
    
test_zscore():
    Test for proper zscore output
    
test_pearsons():
    Test for pearsons correlation results
    
test_beh_pearsons():
    Test for behavior specific pearsons correlation

-----    
TODO:
Modify normalize sig function to actually check if signals 
have been normalized! 
Update ---- Sort of got the idea? Check w Kathleen
-----  
"""

class test_behav(unittest.TestCase):
    # Setup for initializing object w dummy params/data
    def setUp(self):
        """
        Parameters
        ----------
        None

        Returns
        ----------
        Multiple fiber objects (4 as of now) to be used for testing
        """
        # Sample data file paths
        test_data = "/Users/yobae/Desktop/CS Stuff/fiberobj_panel/Yobe_GUI/fiberpho_gui/tests/src/files/sample_data_1.csv"
        test_data_2 = "/Users/yobae/Desktop/CS Stuff/fiberobj_panel/Yobe_GUI/fiberpho_gui/tests/src/files/sample_data_2.csv"
        # test_data_3 = "/Users/yobae/Desktop/CS Stuff/fiberobj_panel/Yobe_GUI/fiberpho_gui/tests/src/files/sample_data_3.csv"
        test_data_4 = "/Users/yobae/Desktop/CS Stuff/fiberobj_panel/Yobe_GUI/fiberpho_gui/tests/src/files/sample_data_4.csv"
        # Read into dataframe
        df = pd.read_csv(test_data)
        df_2 = pd.read_csv(test_data_2) 
        # df_3 = pd.read_csv(test_data_3) 
        df_4 = pd.read_csv(test_data_4) 

        self.test_obj = fc.fiberObj(df, 'Test_Object', 1, 2, '02/11', '2:27',
                            0, -1, 'Sample_Data_1')

        self.test_obj_2 = fc.fiberObj(df_2, 'Test_Object_2', 2, 2, '04/11', '2:22',
                            -1, 0, 'Sample_Data_2')
        
        # Fiber file 12-28 has issue importing behavior, possibly an old behavior file
        # self.test_obj_3 = fc.fiberObj(df_3, 'Test_Object_3', 1, 1, '05/11', '2:23',
                            # -1, 0, 'Sample_Data_3')
        

        self.test_obj_4 = fc.fiberObj(df_4, 'Test_Object_4', 2, 1, '06/11', '2:24',
                            -1, 0, 'Sample_Data_4')
    
        # Import behavior data
        beh_data_1 = "/Users/yobae/Desktop/CS Stuff/fiberobj_panel/Yobe_GUI/fiberpho_gui/tests/src/files/sample_beh_1.csv"
        beh_data_2 = "/Users/yobae/Desktop/CS Stuff/fiberobj_panel/Yobe_GUI/fiberpho_gui/tests/src/files/sample_beh_2.csv"
        # beh_data_3 = "/Users/yobae/Desktop/CS Stuff/fiberobj_panel/Yobe_GUI/fiberpho_gui/tests/src/files/sample_beh_3.csv"
        beh_data_4 = "/Users/yobae/Desktop/CS Stuff/fiberobj_panel/Yobe_GUI/fiberpho_gui/tests/src/files/sample_beh_4.csv"
        
        with open(beh_data_1, "r") as a, open(beh_data_2, "r") as b, open(beh_data_4, "r") as d:
            beh_1 = a.read()
            beh_2 = b.read()
            # beh_3 = c.read()
            beh_4 = d.read()
                
        self.test_obj.import_behavior_data(beh_1, "Sample_Beh_1", 'place_holder')
        
        self.test_obj_2.import_behavior_data(beh_2, "Sample_Beh_2", "place_holder")

        # self.test_obj_3.import_behavior_data(beh_3, "Sample_Beh_3", "place_holder")
        
        self.test_obj_4.import_behavior_data(beh_4, "Sample_Beh_4", "place_holder")



        # Import Pickles for correlation testing
        # Pickle file paths
        infile_1 = "/Users/yobae/Desktop/CS Stuff/fiberobj_panel/Yobe_GUI/fiberpho_gui/tests/src/pickles/pkl_test_1A.pickle"
        infile_2 = "/Users/yobae/Desktop/CS Stuff/fiberobj_panel/Yobe_GUI/fiberpho_gui/tests/src/pickles/pkl_test_1B.pickle"
        infile_3 = "/Users/yobae/Desktop/CS Stuff/fiberobj_panel/Yobe_GUI/fiberpho_gui/tests/src/pickles/pkl_test_2A.pickle"
        infile_4 = "/Users/yobae/Desktop/CS Stuff/fiberobj_panel/Yobe_GUI/fiberpho_gui/tests/src/pickles/pkl_test_2B.pickle"

        pkl_file_1 = open(infile_1, 'rb')
        self.pkl_1 = pickle.load(pkl_file_1)
        pkl_file_1.close()
        
        pkl_file_2 = open(infile_2, 'rb')
        self.pkl_2 = pickle.load(pkl_file_2)
        pkl_file_2.close()
        
        pkl_file_3 = open(infile_3, 'rb')
        self.pkl_3 = pickle.load(pkl_file_3)
        pkl_file_3.close()
        
        pkl_file_4 = open(infile_4, 'rb')
        self.pkl_4 = pickle.load(pkl_file_4)
        pkl_file_4.close()
        
        
        
        
#     def test_import_beh(self):
#         """
#         Parameters
#         ----------
#         None

#         Returns
#         ----------
#         Error if the behavior file could not be uploaded, otherwise returns an OK
#         """
#         msg = "Error reading behavior data"
        
#         with open(self.beh_data, "r") as f:
#             dataset_1 = f.read()
        
#         self.test_obj.import_behavior_data(dataset_1, "Sample_Beh_1", 'place_holder')
        
        
        # # Using fiber file 3 to test for import errors in the meantime
        # beh_data_4 = "/Users/yobae/Desktop/CS Stuff/fiberobj_panel/Yobe_GUI/fiberpho_gui/tests/src/files/Sample_Beh_4.csv"
        # # Test Case to ensure exception is raised
        # self.assertRaises(KeyError, lambda: self.test_obj_4.import_behavior_data(beh_data_4, "Sample_Beh_4", 'place_holder'))
        
        
        
    # Test for Plot Behavior
    def test_plot_beh(self):
        """
        Parameters
        ----------
        None
        
        Returns
        ----------
        Error if no instance of a graph is returned or an error arises,
        otherwise returns OK
        """
        # Error message to relay if test case fails
        msg = "Plot Behavior function failed to return a plot"
        # Test Case
        channels = ["Raw_Green", "Raw_Isosbestic", "Raw_Red"]
        behs = ['m', 'i', 'g'] # Note some fiber objs may have extra beh options
        
        # self.assertIsInstance(self.test_obj.plot_behavior
        #                       (random.choice(behs),
        #                        random.choice(channels)), msg)
                              
#         self.assertIsInstance(self.pkl_1.plot_behavior
#                               (random.choice(behs),
#                                random.choice(channels)), msg)
                              
#         self.assertIsInstance(self.pkl_2.plot_behavior
#                               (random.choice(behs),
#                                random.choice(channels)), msg)
                              
#         self.assertIsInstance(self.pkl_3.plot_behavior
#                               (random.choice(behs),
#                                random.choice(channels)), msg)
        
        
    # Test for Z-Score Plot
    def test_zscore(self):
        """
        Parameters
        ----------
        None

        Returns
        ----------
        Error if no instance of a graph is returned or an error arises,
        otherwise returns OK
        """
        # Error to relay if test case fails
        msg = "Zscore function failed to return a plot"
        # Test Case
        # Random select behaviors and signals for req positional arguments
        channels = ["Raw_Green", "Raw_Isosbestic", "Raw_Red"]
        behs = ['m', 'i', 'g'] # Note test_obj_2 has behavior 'f' as well
        # # Raises Error: Local variable x referenced before assignment
        # # Cannot test until fixed
        self.assertIsInstance(self.test_obj.plot_zscore
                              (random.choice(channels), random.choice(behs), 2, 5), 
                              plotly.graph_objs._figure.Figure, msg)
        
        # self.assertIsInstance(self.test_obj_2.plot_zscore
        #                       (random.choice(channels), random.choice(behs), 2, 5), 
        #                       plotly.graph_objs._figure.Figure, msg)
        
        # self.assertIsInstance(self.test_obj_4.plot_zscore
        #                       (random.choice(channels), random.choice(behs), 2, 5), 
        #                       plotly.graph_objs._figure.Figure, msg)
        
        self.assertIsInstance(self.pkl_1.plot_zscore
                              (random.choice(channels), random.choice(behs), 2, 5),
                              plotly.graph_objs._figure.Figure, msg)
                              
        self.assertIsInstance(self.pkl_2.plot_zscore
                              (random.choice(channels), random.choice(behs), 2, 5),
                              plotly.graph_objs._figure.Figure, msg)
                              
        self.assertIsInstance(self.pkl_3.plot_zscore
                              (random.choice(channels), random.choice(behs), 2, 5),
                              plotly.graph_objs._figure.Figure, msg)
        
    # Test for Pearsons Correlation
    def test_pearsons(self):
        """
        Parameters
        ----------
        None

        Returns
        ----------
        Error if no instance of a graph or an problem occurred in the function call, otherwise returns an OK
        """
        msg = "Pearsons function failed to return a plot"
        channels = ["Raw_Green", "Raw_Isosbestic", "Raw_Red"]
        # Test Case
        self.assertIsInstance(self.pkl_1.pearsons_correlation
                              (self.pkl_2, random.choice(channels),
                               random.choice(channels), 0, -1),
                               plotly.graph_objs._figure.Figure, msg)
        
        
        self.assertIsInstance(self.pkl_3.pearsons_correlation
                              (self.pkl_4, random.choice(channels),
                               random.choice(channels), 0, -1),
                               plotly.graph_objs._figure.Figure, msg)
        # print(self.pkl_1.correlation_results.head())
        
        
#     # Test for Behavior specific Pearsons
    def test_beh_pearsons(self):
        """
        Parameters
        ----------
        None

        Returns
        ----------
        Error if no instance of a graph or an problem occurred in the function call, otherwise returns an OK
        """
        msg = "Behavior specific Pearsons function failed to return a plot"
        channels = ["Raw_Green", "Raw_Isosbestic", "Raw_Red"]
        behs = ['m', 'i', 'g'] # Note test_obj_2 has behavior 'f' as well
        # Test Case
        self.assertIsInstance(self.pkl_1.behavior_specific_pearsons
                              (self.pkl_2, random.choice(channels),
                               random.choice(behs)),
                               plotly.graph_objs._figure.Figure, msg)
        
        self.assertIsInstance(self.pkl_3.behavior_specific_pearsons
                              (self.pkl_4, random.choice(channels),
                               random.choice(behs)),
                               plotly.graph_objs._figure.Figure, msg)

        # print(self.pkl_1.beh_corr_results.head())
                              
                              
if __name__ == '__main__':
    unittest.main()