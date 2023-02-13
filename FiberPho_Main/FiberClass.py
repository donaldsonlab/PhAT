from os import error
import io
import sys
import argparse
import pandas as pd
import numpy as np
import csv
import pickle
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from plotly.subplots import make_subplots
from pathlib import Path
import panel as pn
from statistics import mean
import matplotlib.pyplot as plt
import scipy.stats as ss
import re

pn.extension('terminal')

class fiberObj:
    
    """
    A class to represent a fiber object for fiber photometry and behavior analysis.

    Attributes
    ----------
    obj_name : str
        Name of the fiber object
    
    fiber_num : int
        Fiber number used in photometry data (ranges from 0 - 2 as of now)
        
    animal_num : int
        The animal number used in the experiment
        
    exp_date : Date-string (MM/DD)
        Date of the captured photometry recording
    
    exp_time : Time (Hr/Min)
        Time of the captured photometry recording
        
    start_time : int
        Time to exclude from beginning of recording
    
    stop_time : int
        Time to stop at from start of recording 
        
    file_name : str
        File name of the uploaded photometry data
    
    beh_file : Dataframe
        Stores a pandas dataframe of the behavior recording
        
    beh_filename : str
        Name of the behavior dataset
        
    behaviors : set
        Stores unique behaviors of fiber object
    
    channels : set
        Stores the signals used in photometry data
    
    fpho_data_dict : dict
        Stores photometry data into a dictionary
        
    fpho_data_df : Dataframe
        Uses fpho_data_dict to convert photometry data into a pandas dataframe for use
        
    color_dict : dict
        stores translated channel labels
    
    z_score_results : Dataframe
        stores results of Z-Score computations
    
    correlation_results : Dataframe
        stores results of Pearsons computations
    
    beh_corr_results : Dataframe
        stores results of behavior specific Pearsons computations
        
    frame_rate : List ???
        calculates frame rate of captured data
        
    """
    def __init__(self, file, obj, fiber_num, animal, exp_date,
                 exp_start_time, start_time, stop_time, filename):
        """
        Constructs all the necessary attributes for the FiberPho object. Holds all 
        the data from a fiber experiment as well as some results from analysis. 
        Takes in a fiber photometry file (.csv) and parses it into a dataframe 
        (fpho_data_df) with 9 columns: time_iso, time_green, time_red, 
        green_iso, green_green, green_red, red_iso, red_green, red_red.

        Parameters
        ----------
        obj_name : str
            name of the fiber object

        fiber_num : int
            fiber number to analyze (range: 0-20)

        animal_num : int
            the animal number used in the experiment

        exp_date : Date-string (MM/DD), optional
            date of the captured photometry recording

        exp_time : Time (Hr/Min), optional
            time of the captured photometry recording

        start_time : int
            time to exclude from beginning of recording

        stop_time : int
            time to stop at from start of recording 

        file_name : str
            file name of the uploaded photometry file

        Returns
        ----------
        class object : fiberObj
            Initialized object of type fiberObj
        """
        self.obj_name = obj
        self.fiber_num = fiber_num
        self.animal_num = animal
        self.exp_date = exp_date
        self.exp_start_time = exp_start_time
        self.start_time = start_time #looking for better names
        self.stop_time = stop_time #looking for better names
        self.file_name = filename
        self.beh_filename = 'N/A'
        self.behaviors = set()
        self.channels = set()
        self.version = 1
        self.color_dict = {'Raw_Green' : 'LawnGreen', 'Raw_Red': 'Red', 'Raw_Isosbestic': 'Cyan',
                           'Normalized_Green': 'MediumSeaGreen', 'Normalized_Red': 'Dark_Red',
                           'Normalized_Isosbestic':'DeepSkyBlue'}
        self.z_score_results = pd.DataFrame(columns = ['Object Name', 'Behavior', 'Channel', 'delta Z_score',
                                                       'Max value', 'Time of max value',
                                                       'Min value', 'Time of min value',
                                                       'Average value before event', 'Average value after event',
                                                       'Time before', 'Time after', 'Number of events',
                                                       'Baseline', 'Normalization type'])
        self.correlation_results = pd.DataFrame(columns = ['Object Name', 'Channel', 'Obj2', 'Obj2 Channel',
                                                           'start_time', 'end_time', 
                                                           'R Score', 'p score'])
        self.beh_corr_results = pd.DataFrame(columns = ['Object 1 Name', 'Object 1 Channel', 'Object 2 Name',
                                                        'Object 2 Channel', 'Behavior', 'Number of Events', 
                                                        'R Score', 'p score'])
        file['Timestamp'] = (file['Timestamp'] - file['Timestamp'][0])
        
        self.frame_rate = (file['Timestamp'].iloc[-1]
                            - file['Timestamp'][0])/len(file['Timestamp'])
        if start_time == 0:
            self.start_idx = 0
        else:
            self.start_idx = np.searchsorted(file['Timestamp'], start_time)
        
        if stop_time == -1:
            self.stop_idx = len(file['Timestamp'])
        else:
            self.stop_idx = np.searchsorted(file['Timestamp'], stop_time) 
        
        time_slice = file.iloc[self.start_idx : self.stop_idx]

        if fiber_num is not None:
            self.npm__init__(time_slice)
        
        else:
            self.csv__init__(time_slice)
         
        
    def npm__init__(self, time_slice):
        data_dict = {}
        #Check for green ROI
        try: 
            test_green = time_slice.columns.str.endswith('G')
        except:
            green_ROI = False
            print('no green ROI found')
        else:
            green_ROI = True
            green_col = np.where(test_green)[0][self.fiber_num - 1]
            
        #Check for red ROI   
        try: 
            test_red = time_slice.columns.str.endswith('R')
        except:
            red_ROI = False
            print('no red ROI found')
        else:
            red_ROI = True
            red_col = np.where(test_red)[0][self.fiber_num - 1]
        
        time_slice.columns = time_slice.columns.str.replace('Flags', 'LedState')

        led_states = time_slice['LedState'][2:8].unique()
        npm_dict = {'Green' : {2, 10, 18, 34, 66, 130, 258, 514},
                    'Isosbestic':{1, 9, 17, 33, 65, 129, 257, 513},
                    'Red': {4, 12, 20, 36, 68, 132, 260, 516}}
        
        for color in led_states:
            if color in npm_dict['Green']:
                data_dict['time_Green'] =  time_slice[
                    time_slice['LedState'] == color][
                        'Timestamp'].values.tolist()
            elif color in npm_dict['Red']:
                data_dict['time_Red'] =  time_slice[
                    time_slice['LedState'] == color][
                    'Timestamp'].values.tolist()
            elif color in npm_dict['Isosbestic']:
                data_dict['time_Isosbestic'] =  time_slice[
                    time_slice['LedState'] == color][
                    'Timestamp'].values.tolist()
                
            if green_ROI: 
                if color in npm_dict['Green'] :
                    data_dict['Raw_Green'] = time_slice[
                        time_slice['LedState'] == color].iloc[:, green_col].values.tolist()
                    self.channels.add('Raw_Green')
                elif color in npm_dict['Isosbestic']:
                    data_dict['Raw_Isosbestic'] = time_slice[
                        time_slice['LedState'] == color].iloc[:, green_col].values.tolist()
                    self.channels.add('Raw_Isosbestic')
            
            if red_ROI: 
                if color in npm_dict['Red']:
                    data_dict['Raw_Red'] = time_slice[
                        time_slice['LedState'] == color].iloc[:, red_col].values.tolist() 
                    self.channels.add('Raw_Red')
            
        shortest_list = min([len(data_dict[ls]) for ls in data_dict])
        
        for ls in data_dict:
            data_dict[ls] = data_dict[ls][:shortest_list-1]
        self.fpho_data_df = pd.DataFrame.from_dict(data_dict)
        time_cols = [col for col in self.fpho_data_df.columns if 'time' in col]
        self.fpho_data_df.insert(0, 'time', self.fpho_data_df[time_cols].mean(axis = 1))

    
    def csv__init__(self, time_slice): 
        data_dict = {}
         #Check for green ROI
        try: 
            data_dict['Raw_Green'] = time_slice['Green'].values.tolist()
            data_dict['time'] =  time_slice['Timestamp'].values.tolist()
            self.channels.add('Raw_Green')
        except:
            print('no green data found')
        
        try: 
            data_dict['Raw_Red'] = time_slice['Red'].values.tolist()
            data_dict['time'] =  time_slice['Timestamp'].values.tolist()
            self.channels.add('Raw_Red')
        except:
            print('no red data found')    
        
        try: 
            data_dict['Raw_Isosbestic'] = time_slice['Isosbestic'].values.tolist()
            data_dict['time'] =  time_slice['Timestamp'].values.tolist()
            self.channels.add('Raw_Isosbestic')
        except:
            print('no isosbestic data found')
                       
        shortest_list = min([len(data_dict[ls]) for ls in data_dict])
        
        for ls in data_dict:
            data_dict[ls] = data_dict[ls][:shortest_list-1]
        
        self.fpho_data_df = pd.DataFrame.from_dict(data_dict)
        
    
#### Helper Functions ####
    def fit_exp(self, values, a, b, c, d, e):
        """
        Transforms data into an exponential function
        of the form 
        ..math:: 
            y = A * exp(-B * X) + C * exp(-D * x) + E

        Parameters
        ----------
        values : list
            data

        a, b, c, d, e : int/float
            estimates for the parameter values of A, B, C, D and E
        """
        values = np.array(values)

        return a * np.exp(-b * values) + c * np.exp(-d * values) + e

#### End Helper Functions #### 
    

    
##### Class Functions #####
    def combine_objs(self, obj2, new_obj_name, combine_type, time_adj):
        """
        Combines all data from two different objectss into a new object.

        Parameters
        ----------
        obj2 : fiberObj
            object that will be concatenated to the end of the first objects dataframe

        new_obj_name : str
            name for the combined obj that will be created

        combine_type : str
            One of four ways to adjust the time to combine the two dataframes

        time_adj : float
            time adjustment for obj2 in different ways
            depending on the combine_type selected

        Returns
        ----------
        class object : fiberObj
        """
        #first check for compatibility
        if self.version != obj2.version:
            print('error')
            return "an error"
        
        if self.channels != obj2.channels:
            print('error')
            return "an error" 
        
        if abs(self.frame_rate - obj2.frame_rate) > 1:
            print('error')
            return "an error"
        
        if self.behaviors != obj2.behaviors:
            print('Warning: the behaviors in obj1 and different than the behaviors in obj2')
        
        self.obj_name = new_obj_name
        
        if self.fiber_num != obj2.fiber_num:
            self.fiber_num = [self.fiber_num, obj2.fiber_num]
            
        if self.animal_num != obj2.animal_num:
            self.animal_num = [self.animal_num, obj2.animal_num]

        if self.exp_date != obj2.exp_date:
            self.exp_date = self.exp_date + ', ' + obj2.exp_date
        
        if self.file_name != obj2.file_name:
            self.file_name = self.file_name + ', ' + obj2.file_name

        if self.beh_filename != obj2.beh_filename:
            self.beh_filename = self.beh_filename + ', ' + obj2.beh_filename
        self.z_score_results = pd.DataFrame(columns = ['Object Name', 'Behavior', 'Channel', 'delta Z_score',
                                                       'Max Z_score', 'Max Z_score Time',
                                                       'Min Z_score', 'Min Z_score Time',
                                                       'Average Z_score Before', 'Average Z_score After',
                                                       'Time Before', 'Time After', 'Number of events',
                                                       'Z_score Baseline'])
        self.correlation_results = pd.DataFrame(columns = ['Object Name', 'Channel', 'Obj2', 'Obj2 Channel',
                                                           'start_time', 'end_time', 
                                                           'R Score', 'p score'])
        self.beh_corr_results = pd.DataFrame(columns = ['Object Name', 'Channel', 'Obj2', 'Obj2 Channel',
                                                        'Behavior', 'Number of Events', 
                                                        'R Score', 'p score'])

        # Decide what to do with start_time, stop_time, start_idx and stop_idx. Do I even want to keep them as variables??? idk
                  # also have to look for all the other hundreds of variables and decide what to do
        # self.start_time = start_time #looking for better names
        # self.stop_time = stop_time #looking for better names
        # self.start_idx = 0
        # self.stop_idx = len(file['Timestamp'])
        
        ## Do a bunch of shit to combine the frames
        time_cols = [col for col in self.fpho_data_df.columns if 'time' in col]
        if combine_type == 'Obj2 starts immediately after Obj1':
                obj2.fpho_data_df[time_cols] = (obj2.fpho_data_df[time_cols] -
                                                obj2.fpho_data_df[time_cols[0]][0])
                obj2.fpho_data_df[time_cols] = (obj2.fpho_data_df[time_cols] +
                                                self.fpho_data_df[time_cols[0]].iloc[-1] +
                                                self.frame_rate)
        elif combine_type == 'Use Obj2 current start time':
            if obj2.fpho_data_df[time_cols[0]][0] < self.fpho_data_df[time_cols[0]].iloc[-1]:
                print(obj2.obj_name + ' starts before ' + self.obj_name +
                      ' ends. Choose a different stitching method' )
                return 
                
        elif combine_type == 'Use x secs for Obj2s start time':
            obj2.fpho_data_df[time_cols] = (obj2.fpho_data_df[time_cols] -
                                                obj2.fpho_data_df[time_cols[0]][0])
            obj2.fpho_data_df[time_cols] = (obj2.fpho_data_df[time_cols] + time_adj)
            
        elif combine_type == 'Obj2 starts x secs after Obj1 ends':
            obj2.fpho_data_df[time_cols] = (obj2.fpho_data_df[time_cols] -
                                                obj2.fpho_data_df[time_cols[0]][0])
            obj2.fpho_data_df[time_cols] = (obj2.fpho_data_df[time_cols] +
                                            self.fpho_data_df[time_cols[0]].iloc[-1] +
                                            time_adj)
            
                                                 
                                                  
        #self.fpho_data_df = pd.DataFrame.from_dict(data_dict)
        
        self_beh_cols = self.fpho_data_df.select_dtypes('object').columns
        obj2_beh_cols = obj2.fpho_data_df.select_dtypes('object').columns
        for col in self_beh_cols:
            if col not in obj2_beh_cols:
                obj2.fpho_data_df[col] = ""
        for col in obj2_beh_cols:
            if col not in self_beh_cols:
                self.fpho_data_df[col] = ""        
        self.fpho_data_df = pd.concat([self.fpho_data_df, obj2.fpho_data_df])
              
        return 
    
    #Signal Trace function
    def raw_signal_trace(self):
        """
        Creates and displays graphs of a fiber object's signals.

        Parameters
        ----------
        None

        Returns
        -------
        fig : plotly.graph_objects.Scatter
            Plot of raw signal traces
        """

        fig = make_subplots(rows = 1, cols = 1, shared_xaxes = True,
                            vertical_spacing = 0.02, x_title = "Time (s)",
                            y_title = "Fluorescence (au)")
        for channel in self.channels:
            try:
                time = self.fpho_data_df['time' + channel[3:]]
            except KeyError:
                try:
                    time = self.fpho_data_df['time' + channel[9:]]
                except KeyError:
                    time = self.fpho_data_df['time']
            fig.add_trace(
                go.Scatter(
                    x = time,
                    y = self.fpho_data_df[channel],
                    mode = "lines",
                    line = go.scatter.Line(color = self.color_dict[channel]),
                    name = channel,
                    text = channel,
                    showlegend = True
                ), row = 1, col = 1
            )
        fig.update_layout(
            title = self.obj_name + ' Data'
        )
        return fig
        # fig.write_html(self.obj_name+'raw_sig.html', auto_open = True)

    #Plot fitted exp function
    def normalize_a_signal(self, signal, reference,
                           biexp_thres, linfit_type):
        """
        Creates a plot normalizing 1 fiber data to an
        exponential of the form y=A*exp(-B*X)+C*exp(-D*x)

        Parameters
        ----------
        signal : string
                user channel selection
        reference : string
                user reference trace selection
        Returns
        --------
        output_filename_f1GreenNormExp.png
        & output_filename_f1RedNormExp.png: png files
        containing the normalized plot for each fluorophore
        """
        # Get coefficients for normalized fit using first guesses
        # for the coefficients - B and D (the second and fourth
        # inputs for p0) must be negative, while A and C (the
        # first and third inputs for p0) must be positive
        
        try:
            sig_time = self.fpho_data_df['time' + signal[3:]]
        except KeyError:
            sig_time = self.fpho_data_df['time']
        
        sig = self.fpho_data_df[signal]
        popt, pcov = curve_fit(self.fit_exp, sig_time, sig/(1.5*(max(sig))),
                               p0 = (1.0, 0, 1.0, 0, 0),
                               bounds = (0, np.inf))

        AS = popt[0]  # A value
        BS = popt[1]  # B value
        CS = popt[2]  # C value
        DS = popt[3]  # D value
        ES = popt[4]  # E value
        
        # Generate fit line using calculated coefficients
        fitSig = 1.5*(max(sig))*self.fit_exp(sig_time, AS, BS, CS, DS, ES)
        sigRsquare = np.corrcoef(sig, fitSig)[0,1] ** 2
        
        if sigRsquare < biexp_thres:

            AS = 0
            BS = 0
            CS = 0
            DS = 0
            ES = np.median(sig)
            fitSig = self.fit_exp(sig_time, AS, BS, CS, DS, ES)
        
        normed_sig = [(k / j) for k,j in zip(sig, fitSig)]
        self.fpho_data_df.loc[:, signal + ' expfit'] = fitSig
        self.sig_fit_coefficients = ['A= ' + str(AS), 'B= ' + str(BS), 'C= ' 
                                     + str(CS), 'D= ' + str(DS), 'E= ' + str(ES)]
        self.fpho_data_df.loc[:, signal + ' normed to exp']=normed_sig
                
        
        if reference is not None:  
            try:
                ref_time = self.fpho_data_df['time' + reference[3:]]
            except KeyError:
                ref_time = self.fpho_data_df['time']
            ref = self.fpho_data_df[reference]
            popt, pcov = curve_fit(self.fit_exp, ref_time, ref/(1.5*(max(ref))),
                                   p0=(1.0, 0, 1.0, 0, 0), bounds = (0,np.inf))

            AR = popt[0]  # A value
            BR = popt[1]  # B value
            CR = popt[2]  # C value
            DR = popt[3]  # D value
            ER = popt[4]  # E value     

            # Generate fit line using calculated coefficients

            fitRef = 1.5*(max(ref))*self.fit_exp(ref_time, AR, BR, CR, DR, ER)
            refRsquare = np.corrcoef(ref, fitRef)[0,1] ** 2

            if refRsquare < biexp_thres:
                AR = 0
                BR = 0
                CR = 0
                DR = 0
                ER = np.median(ref)
                fitRef = self.fit_exp(ref_time, AR, BR, CR, DR, ER)

            normed_ref = [(k / j) for k,j in zip(ref, fitRef)]      
                  
            if linfit_type == 'Least squares':
                results = ss.linregress(normed_ref, normed_sig, alternative = 'greater')
                AL = results.slope
                BL = results.intercept
            
            else:
                sig_q75, sig_q25 = np.percentile(normed_sig, [75 ,25])
                sig_IQR = sig_q75 - sig_q25
                ref_q75, ref_q25 = np.percentile(normed_ref, [75 ,25])
                ref_IQR = ref_q75 - ref_q25

                AL = sig_IQR/ref_IQR
                BL = np.median(normed_sig) - AL * np.median(normed_ref)


            adjusted_ref=[AL * j + BL for j in normed_ref]
            normed_to_ref=[(k / j) for k,j in zip(normed_sig, adjusted_ref)]
                
            linR = np.corrcoef(adjusted_ref, normed_sig)[0,1]

            # below saves all the variables we generated to the df #
            #  data frame inside the obj ex. self 
            # and assign all the long stuff to that
            # assign the AS, BS,.. etc and AR, BR, etc to lists called self.sig_fit_coefficients, self.ref_fit_coefficients and self.sig_to_ref_coefficients

            self.fpho_data_df.loc[:, reference + ' expfit']=fitRef
            self.ref_fit_coefficients = ['A= ' + str(AR), 'B= ' + str(BR), 'C= ' +
                                         str(CR), 'D= ' + str(DR), 'E= ' + str(ER)]
            self.fpho_data_df.loc[:, reference + ' normed to exp']=normed_ref
            self.fpho_data_df.loc[:,reference + ' fitted to ' + signal]=adjusted_ref
            self.sig_to_ref_coefficients = ['A= ' + str(AL), 'B= ' + str(BL)]
        
        else: 
            normed_to_ref = normed_sig
            
        self.fpho_data_df.loc[:,'Normalized_' + signal[4:]] = normed_to_ref
        self.channels.add('Normalized_' + signal[4:])
        if reference is not None:
            fig = make_subplots(rows = 3, cols = 2, x_title = 'Time(s)', y_title = 'Flourescence (au)',
                        subplot_titles=("Biexponential Fitted to Signal (R^2 = " + str(sigRsquare) + ")",
                                        "Signal Normalized to Biexponential",
                                        "Biexponential Fitted to Ref (R^2 = " + str(refRsquare) + ")", 
                                        "Reference Normalized to Biexponential",
                                        "Reference Linearly Fitted to Signal(R = " + str(linR) + ")",
                                        "Final Normalized Signal"),
                        shared_xaxes = True, vertical_spacing = 0.1)
            fig.add_trace(
                go.Scatter(
                x = sig_time,
                y = sig,
                mode = "lines",
                line = go.scatter.Line(color="rgba(0, 255, 0, 1)"),
                name ='Signal:' + signal,
                text = 'Signal',
                showlegend = True),
                row = 1, col = 1
                )
            fig.add_trace(
                go.Scatter(
                x = sig_time,
                y = self.fpho_data_df[signal + ' expfit'],
                mode = "lines",
                line = go.scatter.Line(color="Purple"),
                name = 'Biexponential fitted to Signal',
                text = 'Biexponential fitted to Signal',
                showlegend = True),
                row = 1, col = 1
                )
            fig.add_trace(
                go.Scatter(
                x = sig_time,
                y = self.fpho_data_df[signal + ' normed to exp'],
                mode = "lines",
                line = go.scatter.Line(color="rgba(0, 255, 0, 1)"),
                name = 'Signal Normalized to Biexponential',
                text = 'Signal Normalized to Biexponential',
                showlegend = True),
                row = 1, col = 2
                )
            fig.add_trace(
                go.Scatter(
                x = ref_time,
                y = ref,
                mode = "lines",
                line = go.scatter.Line(color="Cyan"),
                name = 'Reference:' + reference,
                text = 'Reference',
                showlegend = True),
                row = 2, col = 1
                )
            fig.add_trace(
                go.Scatter(
                x = ref_time,
                y = self.fpho_data_df[reference + ' expfit'],
                mode = "lines",
                line = go.scatter.Line(color="Purple"),
                name = 'Biexponential fit to Reference',
                text = 'Biexponential fit to Reference',
                showlegend = True),
                row = 2, col = 1
                )
            fig.add_trace(
                go.Scatter(
                x = ref_time,
                y = self.fpho_data_df[reference + ' normed to exp'],
                mode = "lines",
                line = go.scatter.Line(color="Cyan"),
                name = 'Reference Normalized to Biexponential',
                text = 'Reference Normalized to Biexponential',
                showlegend = True),
                row = 2, col = 2
                )
            fig.add_trace(
                go.Scatter(
                x = ref_time,
                y = self.fpho_data_df[reference + ' fitted to ' + signal],
                mode = "lines",
                line = go.scatter.Line(color="Cyan"),
                name = 'Reference linearly scaled to signal',
                text = 'Reference linearly scaled to signal',
                showlegend = True),
                row = 3, col = 1  
                )
            fig.add_trace(
                go.Scatter(
                x = sig_time,
                y = self.fpho_data_df[signal + ' normed to exp'],
                mode = "lines",
                line = go.scatter.Line(color="rgba(0, 255, 0, 0.5)"),
                name = 'Signal Normalized to Biexponential',
                text = 'Signal Normalized to Biexponential',
                showlegend = True),
                row = 3, col = 1
                )
            fig.add_trace(
                go.Scatter(
                x = sig_time,
                y = self.fpho_data_df['Normalized_' + signal[4:]],
                mode="lines",
                line = go.scatter.Line(color = "Hot Pink"), 
                name = 'Final Normalized Signal',
                text = 'Final Normalized Signal',
                showlegend = True), 
                row = 3, col = 2
                )
        else:
            fig = make_subplots(rows = 1, cols = 2, 
                                subplot_titles=("Biexponential Fitted to Signal(R^2 = " 
                                        + str(sigRsquare) + ")",
                                        "Signal Normalized to Biexponential"),
                        shared_xaxes = True, vertical_spacing = 0.1,
                        x_title = "Time (s)", y_title = "Fluorescence (au)")
            fig.add_trace(
                go.Scatter(
                x = sig_time,
                y = sig,
                mode = "lines",
                line = go.scatter.Line(color="rgba(0, 255, 0, 1)"),
                name ='Signal:' + signal,
                text = 'Signal',
                showlegend = True),
                row = 1, col = 1
                )
            fig.add_trace(
                go.Scatter(
                x = sig_time,
                y = self.fpho_data_df[signal + ' expfit'],
                mode = "lines",
                line = go.scatter.Line(color="Purple"),
                name = 'Biexponential fitted to Signal',
                text = 'Biexponential fitted to Signal',
                showlegend = True),
                row = 1, col = 1
                )
            fig.add_trace(
                go.Scatter(
                x = sig_time,
                y = self.fpho_data_df[signal + ' normed to exp'],
                mode = "lines",
                line = go.scatter.Line(color="rgba(0, 255, 0, 1)"),
                name = 'Signal Normalized to Biexponential',
                text = 'Signal Normalized to Biexponential',
                showlegend = True),
                row = 1, col = 2
                )
            
        fig.update_layout(
            title = "Normalizing " + signal + ' for ' + 
            self.obj_name + ' using ' + linfit_type
        )
        return fig
    

    # ----------------------------------------------------- # 
    # Behavior Functions
    # ----------------------------------------------------- # 

    def import_behavior_data(self, beh_data, filename):
        """
        Takes a file name and returns a dataframe of parsed data

        Parameters
        ----------
        file : string
            String that contains the entire csv file

        Returns
        --------
        behaviorData : pandas dataframe
            contains - Time(total msec), Time(sec), 
            Subject, Behavior, Status
        """
        beh_data = beh_data.sort_values('Time')

        unique_behaviors = beh_data['Behavior'].unique()
        for beh in unique_behaviors:
            if beh in self.fpho_data_df.columns:
                print(beh + ' is already in ' + self.obj_name + ' and cannot be added again.')
            else:
                self.behaviors.add(beh)
                idx_of_beh = [i for i in range(len(beh_data['Behavior']
                             )) if beh_data.loc[i, 'Behavior'] == beh]             
                j = 0
                self.fpho_data_df[beh] = ' '
                while j < len(idx_of_beh):
                    if beh_data.loc[(idx_of_beh[j]), 'Status']=='POINT': 
                        point_idx=self.fpho_data_df['time'].searchsorted(
                            beh_data.loc[idx_of_beh[j],'Time'])
                        self.fpho_data_df.loc[point_idx, beh]='S'
                        j = j + 1
                    elif (beh_data.loc[(idx_of_beh[j]), 'Status']=='START' and 
                          beh_data.loc[(idx_of_beh[j + 1]), 'Status']=='STOP'):
                        startIdx = self.fpho_data_df['time'].searchsorted(
                            beh_data.loc[idx_of_beh[j],'Time'])
                        endIdx = self.fpho_data_df['time'].searchsorted(
                            beh_data.loc[idx_of_beh[j + 1],'Time'])
                        if endIdx < len(self.fpho_data_df['time']) and startIdx > 0:
                            self.fpho_data_df.loc[endIdx, beh] = 'E'
                            self.fpho_data_df.loc[startIdx, beh] = 'S'
                            self.fpho_data_df.loc[startIdx+1 : endIdx-1, beh] = 'O'
                        j = j + 2
                    else: 
                        print("\nStart and stops for state behavior:" 
                              + beh + " are not paired correctly.\n")
                        sys.exit()
        if self.beh_filename == 'N/A':
            self.beh_filename = filename
        else:
            self.beh_filename = self.beh_filename + ", " + filename
        return

    def plot_behavior(self, behaviors, channels):
        """
        Plots behavior specific signals
        
        Parameters
        ----------
        behaviors : list
            user selected behaviors
        
        channels : list
            user selected channel
            
        Returns
        ----------
        fig : plotly.graph_objects.Scatter
            Scatter plot of select behavior signals
        """
        
        fig = make_subplots(rows = len(channels), cols = 1,
                            subplot_titles = [channel for channel in channels],
                            shared_xaxes = True,
                            x_title = "Time (s)",
                            y_title = "Fluorescence (au)")
        
        for i, channel in enumerate(channels):
            try:
                time = self.fpho_data_df['time' + channel[3:]]
            except KeyError:
                try:
                    time = self.fpho_data_df['time' + channel[9:]]
                except KeyError:
                    time = self.fpho_data_df['time']
            fig.add_trace(
                go.Scatter(
                x = time,
                y = self.fpho_data_df[channel],
                mode = "lines",
                line = go.scatter.Line(color = "Green"),
                name = channel,
                showlegend = False), row = i + 1, col = 1
                )
            colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
                      '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
            j = 0
            behaviorname = ""
            for j, beh in enumerate(behaviors):
                behaviorname = behaviorname + " " + beh
                temp_beh_string = ''.join([key for key in self.fpho_data_df[beh]])
                pattern = re.compile(r'S[O]+E')
                bouts = pattern.finditer(temp_beh_string)
                for bout in bouts:
                    start_time = time.at[bout.start()]
                    end_time = time.at[bout.end()]
                    fig.add_vrect(x0 = start_time, x1 = end_time, 
                                opacity = 0.75,
                                layer = "below",
                                line_width = 1, 
                                fillcolor = colors[j % 10],
                                row = i + 1, col = 1,
                                name = beh
                                )
                S = re.compile(r'S')
                starts = S.finditer(temp_beh_string)
                for start in starts:
                    start_time = time.at[start.start()]
                    fig.add_vline(x = start_time, 
                                layer = "below",
                                line_width = 3, 
                                line_color = colors[j % 10],
                                row = i + 1, col = 1,
                                name = beh
                                )
                
                fig.add_annotation(xref = "x domain", yref = "y domain",
                    x = 1, 
                    y = (j + 1) / len(self.behaviors),
                    text = beh,
                    bgcolor = colors[j % 10],
                    showarrow = False,
                    row = i + 1, col = 1
                    )
        return fig
        
    
    def plot_zscore(self, channel, beh, time_before, time_after,
                    baseline = 0, base_option = 'Each event', show_first = 0,
                    show_last = -1, show_every = 1,
                    save_csv = False, percent_bool = False):
        
        """
        Takes a dataframe and creates plot of z-scores for
        each time a select behavior occurs with the avg
        z-score and SEM. Stores results in dataframe
    
        Parameters
        ----------
        channel : string
            user selected channels
        
        beh : string
            user selected behaviors
        
        time_before : int
            timestamps to include before event
            
        time_after : int
            timestamps to include after start of event
        
        baseline : list, optional
            baseline window start and end times [0, 1] respectively
        
        base_option : int, optional
            baseline parameter options - start of sample, before events, end of sample
        
        show_first : int, optional
            show traces from event number [int]
        
        show_last : int, optional
            show traces up to event number [int]
            
        show_every : int, optional
            show one in every [int] traces
        
        percent_bool
        
        save_csv
        
        Returns
        ----------
        fig : plotly.graph_objects.Scatter
            Plot of z-scores for select behaviors
        """
        try:
            full_time = self.fpho_data_df['time' + channel[3:]]
            time_name = 'time' + channel[3:]
        except KeyError:
            try:
                full_time = self.fpho_data_df['time' + channel[9:]]
                time_name = 'time' + channel[9:]
            except KeyError:
                full_time = self.fpho_data_df['time']
                time_name = 'time'
        # Finds all times where behavior starts, turns into list
        beh_times = list(self.fpho_data_df[(
            self.fpho_data_df[beh]=='S')][time_name])
        if len(beh_times) < 2:
            print('There is 1 or less bouts of ' + beh + ' in ' + self.obj_name)
            return
        # Initialize figure
        fig = make_subplots(rows = 1, cols = 2,
                            subplot_titles = ('Full trace with events',
                                              'Average'
                                             )
                           )
        # Adds trace
        fig.add_trace(
            # Scatter plot
            go.Scatter(
            # X = all times
            # Y = all values at that channel
            x = full_time,
            y = self.fpho_data_df[channel],
            mode = "lines",
            line = go.scatter.Line(color="Green"),
            name = channel,
            showlegend = False), 
            row = 1, col =1
            )

        # Initialize array of zscore sums
        Zscore_data = pd.DataFrame()
        # Initialize events counter to 0
        n_events = 0
        
        if base_option == 'Each event':
            base_mean = None
            base_std = None
            zscore_baseline = 'Each event'
        
        elif base_option == 'Start of Sample':
            # idx = np.where((start_event_time > baseline[0]) & (start_event_time < baseline[1]))
            # Find baseline start/end index
            # Start event time is the first occurrence of event, this option will be for a baseline at the beginning of the trace
            base_start_idx = full_time.searchsorted(
                baseline[0])
            base_end_idx = full_time.searchsorted(
                baseline[1])
            # Calc mean and std for values within window
            base_mean = np.nanmean(self.fpho_data_df.loc[
                base_start_idx:base_end_idx, channel]) 
            base_std = np.nanstd(self.fpho_data_df.loc[
                base_start_idx:base_end_idx, channel])
            zscore_baseline = 'From ' + str(baseline[0]) + ' to ' + str(baseline[1])
            
        
        elif base_option == 'End of Sample':
            # Indexes for finding baseline at end of sample
            start = max(baseline)
            end = min(baseline)
            end_time = full_time.iloc[-1]
            base_start_idx = full_time.searchsorted(
                end_time - start)
            base_end_idx = full_time.searchsorted(
                end_time - end)
            # Calculates mean and standard deviation
            base_mean = np.nanmean(self.fpho_data_df.loc[
                base_start_idx:base_end_idx, channel])
            base_std = np.nanstd(self.fpho_data_df.loc[
                base_start_idx:base_end_idx, channel])
            zscore_baseline = 'From ' + str(end_time - start) + ' to ' + str(end_time - end)

        # Loops over all start times for this behavior
        #  time = actual time
        for time in beh_times:
            # Calculates indices for baseline window before each event
            if base_option == 'Before Events':
                start = max(baseline)
                end = min(baseline)
                zscore_baseline = 'From ' + str(start) + ' to ' + str(end) + ' before each event'
                base_start_idx = full_time.searchsorted(
                    time - start)
                base_end_idx = full_time.searchsorted(
                    time - end)
                base_mean = np.nanmean(self.fpho_data_df.loc[
                    base_start_idx:base_end_idx, channel])
                base_std = np.nanstd(self.fpho_data_df.loc[
                    base_start_idx:base_end_idx, channel])

            # time - time_Before = start_time for this event trace, time is the actual event start, time before is secs input before event start
            # Finds time in our data that is closest to time - time_before
            # start_idx = index of that time
            start_idx = full_time.searchsorted(
                time - time_before)
            # time + time_after = end_time for this event trace, time is the actual event start, time after is secs input after event start
            # end_idx = index of that time
            end_idx = full_time.searchsorted(
                time + time_after)
            
            # Edge case: If indexes are within bounds
            if (start_idx > 0 and 
                end_idx < len(full_time) - 1):
                # Finds usable events
                n_events = n_events + 1
                # Tempy stores channel values for this event trace
                trace = self.fpho_data_df.loc[
                    start_idx : end_idx, channel].values.tolist()
                if percent_bool:
                    if base_option == 'Each event':
                        base_mean = np.nanmean(trace)
                        norm_type = 'Percent'
                    this_Zscore=[((i / base_mean)-1)*100 for i in trace]
                else:
                    this_Zscore = self.zscore(trace, base_mean, base_std)
                    norm_type = 'Z-score'
                # Adds each trace to a dict
                Zscore_data['event' + str(n_events)] = this_Zscore 
                
                if show_first == 0:
                    show_first = 1
                if show_last == -1:
                    show_last = len(beh_times)
                events_to_show = np.arange(show_first, show_last, show_every)
                if n_events in events_to_show:
                    # Times for this event trace
                    time_clip = full_time[start_idx : end_idx]
                    # Trace color (First event blue, last event red)
                    trace_color = 'rgb(' + str(
                        int((n_events+1) * 255/(len(beh_times)))) + ', 0, 255)'
                    # Adds a vertical line for each event time
                    fig.add_vline(x = time, line_dash = "dot", row = 1, col = 1)
                    # Adds trace for each event
                    fig.add_trace(
                        # Scatter plot
                        go.Scatter( 
                        # Times starting at user input start time, ending at user input end time
                        x = time_clip - time,
                        y = this_Zscore, 
                        mode = "lines",
                        line = dict(color = trace_color, width = 2),
                        name = 'Event:' + str(n_events),
                        text = 'Event:' + str(n_events),
                        showlegend=True), 
                        row = 1, col = 2
                        )

        avg_Zscore = Zscore_data.mean(axis=1).to_list()
        sem_Zscore = Zscore_data.sem(axis=1).to_list()
        graph_time = np.linspace(-time_before, time_after,
                                 num = len(avg_Zscore)).tolist()
        Zscore_data.insert(0, 'time', graph_time)
        Zscore_data.insert(1, 'Average', avg_Zscore)
        Zscore_data.insert(2, 'SEM', sem_Zscore)
        Zscore_data.insert(3, 'SEM Upper Bound', Zscore_data['Average'] + Zscore_data['SEM'])
        Zscore_data.insert(4, 'SEM Lower Bound', Zscore_data['Average'] - Zscore_data['SEM'])
        upper_bound = Zscore_data['SEM Upper Bound'].to_list()
        lower_bound = Zscore_data['SEM Lower Bound'].to_list()
        zero_idx = np.searchsorted(graph_time, 0)
        fig.add_vline(x = 0, line_dash = "dot", row = 1, col = 2)
        fig.add_trace(
            # Scatter plot
            go.Scatter( 
            # Times for baseline window
            x = graph_time + graph_time[::-1],
            # Y = SEM
            y = upper_bound + lower_bound[::-1],
            fill= 'toself',
            fillcolor =  'rgba(255, 255, 255, 0.8)',
            line=dict(color='rgba(255,255,255,0)'),
            name = 'SEM',
            text = 'SEM',
            showlegend = True),
            row = 1, col = 2
            )
        # Adds trace
        fig.add_trace(
            # Scatter plot
            go.Scatter( 
            # Times for baseline window
            x = graph_time,
            # Y = Zscore average of all event traces
            y = avg_Zscore,
            mode = "lines",
            line = dict(color = "Black", width = 5),
            name = 'average',
            text = 'average',
            showlegend = True),
            row = 1, col = 2
            )
        #fig.update_yaxes(range = [.994, 1.004])

        fig.update_layout(
            title = 'Z-score of ' + beh + ' for ' 
                    + self.obj_name + ' in channel ' + channel
            )   
        fig.update_xaxes(title_text = 'Time (s)')
        fig.update_yaxes(title_text = 'Fluorescence (au)', col = 1, row = 1)
        fig.update_yaxes(title_text = norm_type, col = 2, row = 1)

        results = {'Object Name': self.obj_name, 'Behavior': beh, 'Channel' : channel,
                   'Max value' : max(avg_Zscore),
                   'Time of max ' : graph_time[np.argmax(avg_Zscore)], 
                   'Min value': min(avg_Zscore), 
                   'Time of min' : graph_time[np.argmin(avg_Zscore)],
                   'delta Z_score' : max(avg_Zscore) - min(avg_Zscore), 
                   'Average value before event' : np.mean(avg_Zscore[:zero_idx]),
                   'Average value after event' : np.mean(avg_Zscore[zero_idx:]),
                   'Time before':time_before, 'Time after':time_after,
                   'Number of events' : n_events, 'Baseline' : zscore_baseline,
                   'Normalization type' : norm_type}
        self.z_score_results = self.z_score_results.concat(results, ignore_index = True)
        if save_csv:
            Zscore_data.to_csv(self.obj_name + '_' + channel + '_' + beh +
                               '_Baseline_' + zscore_baseline + '.csv') 
        return fig
        
        
    # Zscore calc helper
    def zscore(self, ls, mean = None, std = None):
        """
        Helper function to calculate z-scores

        Parameters
        ----------
        ls : list
            list of trace signals

        mean : int/float, optional
            baseline mean value

        std : int/float, optional
            baseline standard deviation value

        Returns
        ----------
        new_ls : list
            list of calculated z-scores per event
        """
        # Default Params, no arguments passed
        if mean is None and std is None:
            mean = np.nanmean(ls)
            std = np.nanstd(ls)
        # Calculates zscore per event in list  
        new_ls = [(i - mean) / std for i in ls]
        return new_ls
        
        
        
        
        
         #return the pearsons correlation coefficient and r value between 2 full channels and plots the signals overlaid and their scatter plot
    def pearsons_correlation(self, obj2, channel1, channel2, start_time, end_time):
        """
        Takes in user chosen objects and channels then returns the 
        Pearson’s correlation coefficient and plots the signals. 

        Parameters
        ----------
        obj2 : fiber object
            second object for correlation analysis
        
        channel1 : string
            first object's selected signal for analysis
        
        channel2 : string
            second object's selected signal for analysis
        
        start_time : int
            starting timestamp of data
        
        end_time : int
            ending timestamp of data
            
        Returns
        ----------
        fig : plotly.graph_objects.Scatter
            Plot of signals based on correlation results
        """
        try:
            time1 = self.fpho_data_df['time' + channel1[3:]]
        except KeyError:
            try:
                time1 = self.fpho_data_df['time' + channel1[9:]]
            except KeyError:
                time1 = self.fpho_data_df['time']
        try:
            time2 = obj2.fpho_data_df['time' + channel2[3:]]
        except KeyError:
            try:
                time2 = obj2.fpho_data_df['time' + channel2[9:]]
            except KeyError:
                time2 = obj2.fpho_data_df['time']
        #find start
        if np.round(self.frame_rate) != np.round(self.frame_rate):
            print('These traces have different frame rates\n')
            
        start_idx1 = np.searchsorted(time1, start_time)
        start_idx2 = np.searchsorted(time2, start_time)

        #find end
        if end_time == -1:
            end_idx1 = len(time1)-1
            end_idx2 = len(time2)-1
        else:
            end_idx1 = np.searchsorted(time1, end_time)-1
            end_idx2 = np.searchsorted(time2, end_time) -1 

        if start_time - time1[start_idx1] < -1/self.frame_rate:
            print('Trace 1 starts at ' + str(np.round(time1[start_idx1])) + 's after your start time ' + str(start_time) + '\n')
        if end_time - time1[end_idx1] > 1/self.frame_rate:
            print('Trace 1 ends at ' + str(np.round(time1[end_idx1])) + 's before your end time ' + str(end_time) + 's\n')

        
        if start_time - time2[start_idx2] < -1/obj2.frame_rate:
            print('Trace 2 starts at ' + str(np.round(time2[start_idx2])) + 's after your start time ' + str(start_time) + 's\n')
        if end_time - time2[end_idx2] > 1/obj2.frame_rate:
            print('Trace 2 ends at ' + str(np.round(time2[end_idx2])) + 's before your end time ' + str(end_time) + 's\n')

        
        sig1 = self.fpho_data_df.loc[start_idx1:end_idx1, channel1]
        sig2 = obj2.fpho_data_df.loc[start_idx2:end_idx2, channel2]
        crop_time1 = time1[start_idx1:end_idx1]
        crop_time2 = time2[start_idx2:end_idx2]

        fig = make_subplots(rows = 1, cols = 2)
        #creates a scatter plot
        fig.add_trace(
            go.Scattergl(
            x = sig1,
            y = sig2,
            mode = "markers",
            name ='correlation',
            showlegend = False),
            row = 1, col = 2
            )
        #plots sig1
        fig.add_trace(
            go.Scattergl(
            x = crop_time1,
            y = sig1,
            mode = "lines",
            name = 'sig1',
            showlegend = False),
            row = 1, col = 1
            )
        #plots sig2
        fig.add_trace(
            go.Scattergl(
            x = crop_time2,
            y = sig2,
            mode = "lines",
            name = "sig2",
            showlegend = False),
            row = 1, col = 1
            )
    
        #make traces the same length for correlation 
        shorter_len = min(len(sig1), len(sig2)) 
        #calculates the pearsons R  
        [r, p] = ss.pearsonr(sig1, sig2)
        res = ss.linregress(sig1, sig2)
                #plots sig2
        fig.add_trace(
            go.Scattergl(
            x = sig1,
            y = res.intercept + res.slope*sig1,
            mode = "lines",
            line = dict(color = "darkgray", width = 5),
            name = "best fit",
            showlegend = False),
            row = 1, col = 2
            )
        results = {'Object Name': self.obj_name, 'Channel' : channel1, 'Obj2': obj2.obj_name, 'Obj2 Channel': channel2, 'start_time' : start_time,
                   'end_time' : end_time, 'R Score' : str(r), 'p score': str(p)}
        self.correlation_results = self.correlation_results.concat(results, ignore_index = True)
        fig.update_layout(
            title = 'Correlation between ' + self.obj_name + ' and ' 
                  + obj2.obj_name + ' is, ' + str(r) + ' p = ' + str(p)
            )
        fig.update_xaxes(title_text = self.obj_name + " " + channel1, col = 2, row = 1)
        fig.update_yaxes(title_text = obj2.obj_name + " " + channel2, col = 2, row = 1)
        fig.update_xaxes(title_text = 'Time (s)', col = 1, row = 1)
        fig.update_yaxes(title_text = 'Fluorescence (au)', col = 1, row = 1)
        return fig
        

    def behavior_specific_pearsons(self, obj2, channel1, channel2, beh):
        """
        Takes in user chosen objects, channels and behaviors to calculate 
        the behavior specific Pearson’s correlation and plot the signals. 

        Parameters
        ----------
        obj2 : fiber object
            second object for correlation analysis
        
        channel : string
            user selected signals for analysis
        
        beh : string
            user selected behaviors for analysis 
            
        Returns
        --------
        fig : plotly.graph_objects.Scatter
            Plot of signals based on behavior specific correlations
        """
        
        # behaviorSlice=df.loc[:,beh]
        behaviorSlice1 = self.fpho_data_df[self.fpho_data_df[beh] != ' ']
        behaviorSlice2 = obj2.fpho_data_df[self.fpho_data_df[beh] != ' ']
        try:
            time = behaviorSlice1['time' + channel1[3:]]
        except KeyError:
            try:
                time = behaviorSlice1['time' + channel1[9:]]
            except KeyError:
                time = behaviorSlice1['time']
        sig1 = behaviorSlice1[channel1]
        sig2 = behaviorSlice2[channel2]
        fig = make_subplots(rows = 1, cols = 2)
        fig.add_trace(
            go.Scattergl(
            x = sig1,
            y = sig2,
            mode = "markers",
            name = beh,
            showlegend = False), 
            row = 1, col = 2
            )
        fig.add_trace(
            go.Scatter(
            x = time,
            y = sig1,
            mode = "lines",
            line = go.scatter.Line(color = 'rgb(255,100,150)'),
            name = channel1,
            showlegend = False),
            row = 1, col = 1
            )
        fig.add_trace(
            go.Scatter(
            x = time,
            y = sig2,
            mode = "lines",
            line = go.scatter.Line(color = 'rgba(100,0,200, .6)'),
            showlegend = False),
            row = 1, col = 1
            )

        [r, p] = ss.pearsonr(sig1, sig2) # 'r' and 'p' were both being referenced before assignment

        fig.update_layout(
                title = 'Correlation between ' + self.obj_name + ' and ' 
                  + obj2.obj_name + ' during ' + beh + ' is, ' + str(r) + ' p = ' + str(p)
                )
        fig.update_xaxes(title_text = self.obj_name + " " + channel1, col = 2, row = 1)
        fig.update_yaxes(title_text = obj2.obj_name + " " + channel2, col = 2, row = 1)
        fig.update_xaxes(title_text = 'Time (s)', col = 1, row = 1)
        fig.update_yaxes(title_text = 'Fluorescence (au)', col = 1, row = 1)

        results = {'Object 1 Name': self.obj_name, 'Object 1 Channel': channel1, 'Object 2 Name': obj2.obj_name, 'Object 2 Channel': channel2,
                   'Behavior' : beh, 'Number of Events': self.fpho_data_df[beh].value_counts()['S'],  'R Score' : r, 'p score': p}
        self.beh_corr_results = self.beh_corr_results.concat(results, ignore_index = True)       
        
        return fig
    
##### End Class Functions #####

def lick_to_boris(beh_file, time_unit, beh_false, time_between_bouts):
    """
    Converts lickometer data to a BORIS file that is readable by the GUI

    Parameters
    ----------
    beh_file : file
        uploaded lickometer file

    Returns
    ----------
    boris : Dataframe
        converted data for download
    """
    boris_df = pd.DataFrame(columns = ['Time', 'Behavior', 'Status'])
    conversion_dict = {'milliseconds':1/1000,'seconds':1,'minutes':60}
    conversion_to_sec = conversion_dict[time_unit]
    behaviors = list(beh_file.columns)
    behaviors.remove('Time')
    print(behaviors)
    for beh in behaviors:
        trimmed = beh_file[beh_file[beh] != beh_false]
        starts = [(trimmed.iloc[0]['Time'] - beh_file.iloc[0]['Time']) * conversion_to_sec]
        stops = []
        diffs = np.diff(trimmed.index)

        for i, v in enumerate(diffs):
            if v > (time_between_bouts / conversion_to_sec):
                stops.concat((trimmed.iloc[i]['Time'] 
                              - beh_file.iloc[0]['Time']) * conversion_to_sec)
                if i+1 < len(diffs):
                    starts.concat((trimmed.iloc[i+1]['Time'] 
                                   - beh_file.iloc[0]['Time']) * conversion_to_sec)
        stops.concat((trimmed.iloc[-1]['Time'] 
                      - beh_file.iloc[0]['Time']) * conversion_to_sec)

        time = starts + stops
        time.sort()
        status = ['START'] * len(time)
        half = len(time) / 2
        status[1::2] = ['STOP'] * int(half)
        behavior = [beh] * len(time)
        beh_df = pd.DataFrame(data = {'Time': time, 'Behavior': behavior, 'Status' : status})
        boris_df = pd.concat([boris_df, beh_df])
    boris_df.sort_values(by = 'Time')
    return boris_df
