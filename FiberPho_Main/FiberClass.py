import sys
import re
import pandas as pd
import numpy as np
import panel as pn
import scipy.stats as ss
import scipy.signal as sci_sig
import scipy.integrate as integrate
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from plotly.subplots import make_subplots



pn.extension('terminal')

class FiberObj:
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
    filename : str
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
    PETS_results : Dataframe
        stores results of peri event time series computations
    correlation_results : Dataframe
        stores results of Pearsons computations
    beh_corr_results : Dataframe
        stores results of behavior specific Pearsons computations
    frame_rate : float
        calculates frame rate of captured data
    """
    def __init__(self, file, obj_name, fiber_num, animal_num, exp_date,
                 exp_time, start_time, stop_time, filename):
        """
        Constructs all the necessary attributes for the FiberPho object.
        Takes in a fiber photometry file (.csv) and parses it into a dataframe
        (fpho_data_df) with up to 7 columns: time, time_iso, time_green, time_red,
        Raw_Red, Raw_Green, Raw_Isosestic.

        Parameters
        ----------
        file : Datafrmae
            pandas dataframe created directly from your photometry data .csv file
        obj_name : str
            name of the fiber object
        fiber_num : int
            fiber number to analyze (range: 0-20)
        animal_num : int, optional
            Id number for the animal used in the experiment
        exp_date : Date-string (MM/DD), optional
            date of the captured photometry recording
        exp_time : Time (Hr/Min), optional
            time of the captured photometry recording
        start_time : int
            time to exclude from beginning of recording
        stop_time : int
            time to stop at from start of recording
        filename : str
            file name of the uploaded photometry file

        Returns
        ----------
        class object : fiberObj
            Initialized object of type fiberObj
        """

        self.obj_name = obj_name
        self.fiber_num = fiber_num
        self.animal_num = animal_num
        self.exp_date = exp_date
        self.exp_time = exp_time
        self.start_time = start_time #looking for better names
        self.stop_time = stop_time #looking for better names
        self.filename = filename
        self.beh_filename = 'N/A'
        self.behaviors = set()
        self.channels = set()
        self.sig_fit_coefficients = ''
        self.ref_fit_coefficients = ''
        self.sig_to_ref_coefficients = ''
        self.version = 4 #variable names have changed since version 1 and 2
        #out of date objs can be updated by updating attribute variable names
        #file_name -> filename, startIdx -> start_idx, endIdx -> end_idx
        #From versions 3-4 a peak_results DataFrarme was added. 
        #This will only impact the peak finding module.
        #You can update this by adding a the Adding the DataFrame below
        self.color_dict = {'Raw_Green' : 'LawnGreen', 'Raw_Red': 'Red',
                           'Raw_Isosbestic': 'Cyan',
                           'Normalized_Green': 'MediumSeaGreen',
                           'Normalized_Red': 'Dark_Red',
                           'Normalized_Isosbestic':'DeepSkyBlue'}
        self.PETS_results = pd.DataFrame(columns = ['Object Name', 'Behavior',
                                                    'Channel', 'range',
                                                    'Max value',
                                                    'Time of max',
                                                    'Min value',
                                                    'Time of min',
                                                    'AUC'
                                                    'Average value before event',
                                                    'Average value after event',
                                                    'Time before', 'Time after',
                                                    'Number of events', 'Baseline',
                                                    'Normalization type'])
        self.peak_results = pd.DataFrame(columns = ['Object Name',
                                                    'Channel',
                                                    'Start time',
                                                    'End time',
                                                    'Width range entered'
                                                    'Number of peaks',
                                                    'Average peak amplitude',
                                                    'Average trace amplitude'
                                                    'Frequency of peaks (peaks/s)'])
        self.correlation_results = pd.DataFrame(columns = ['Object Name', 'Channel',
                                                           'Obj2', 'Obj2 Channel',
                                                           'start_time', 'end_time',
                                                           'R Score', 'p score'])
        self.beh_corr_results = pd.DataFrame(columns = ['Object 1 Name',
                                                        'Object 1 Channel',
                                                        'Object 2 Name',
                                                        'Object 2 Channel',
                                                        'Behavior',
                                                        'Number of Events',
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
        """
        Called in __init__ if fiber_num is not None. Takes a slice of
        the photometry dataframe based on start_time and stop_time.
        Parses the time_slice data so it can be reorganzed for the
        fpho_data_df attribute.

        Parameters
        ----------
        time_slice : Dataframe
            A slice of the photometry data csv file from the
            start_time to the stop_time

        Returns
        ----------
        None
        """

        data_dict = {}
        #Check for green ROI
        try:
            test_green = time_slice.columns.str.endswith('G')
        except:
            green_roi = False
            print('no green ROI found')
        else:
            green_roi = True
            green_col = np.where(test_green)[0][self.fiber_num - 1]
        #Check for red ROI
        try:
            test_red = time_slice.columns.str.endswith('R')
        except:
            red_roi = False
            print('no red ROI found')
        else:
            red_roi = True
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

            if green_roi:
                if color in npm_dict['Green'] :
                    data_dict['Raw_Green'] = time_slice[
                        time_slice['LedState'] == color].iloc[:, green_col].values.tolist()
                    self.channels.add('Raw_Green')
                elif color in npm_dict['Isosbestic']:
                    data_dict['Raw_Isosbestic'] = time_slice[
                        time_slice['LedState'] == color].iloc[:, green_col].values.tolist()
                    self.channels.add('Raw_Isosbestic')

            if red_roi:
                if color in npm_dict['Red']:
                    data_dict['Raw_Red'] = time_slice[
                        time_slice['LedState'] == color].iloc[:, red_col].values.tolist()
                    self.channels.add('Raw_Red')

        shortest_list = min((len(data_dict[ls]) for ls in data_dict))

        for ls in data_dict:
            data_dict[ls] = data_dict[ls][:shortest_list-1]
        self.fpho_data_df = pd.DataFrame.from_dict(data_dict)
        time_cols = [col for col in self.fpho_data_df.columns if 'time' in col]
        self.fpho_data_df.insert(0, 'time', self.fpho_data_df[time_cols].mean(axis = 1))

    def csv__init__(self, time_slice):
        """
        Called in __init__ if fiber_num is None. Takes a slice of
        the photometry dataframe based on start_time and stop_time.
        Parses the time_slice data so it can be reorganzed for the
        fpho_data_df attribute.

        Parameters
        ----------
        time_slice : Dataframe
            A slice of the photometry data csv file from the
            start_time to the stop_time

        Returns
        ----------
        None
        """

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

        shortest_list = min(len(channels) for channels in data_dict.values())
        for channel in data_dict:
            data_dict[channel] = data_dict[channel][:shortest_list-1]
        self.fpho_data_df = pd.DataFrame.from_dict(data_dict)

#### Helper Functions ####
    def fit_exp(self, time, a, b, c, d, e):
        """
        Creates np.array of an exponential function
        of the form
        ..math::
            y = A * exp(-B * X) + C * exp(-D * x) + E

        Parameters
        ----------
        time : list
            time values of you photometry data
        a, b, c, d, e : float
            Parameter values of A, B, C, D and E found using ss.curve_fit
            to fit your photometry data to a biexponential.

        Returns
        ----------
        np.array
            y values of the biexponential fit for you photometry data
        """
        time = np.array(time)
        return a * np.exp(-b * time) + c * np.exp(-d * time) + e

    def fit_lin(self, data, a, b):
        """
        Linearly transforms data using the coefficients a and b.

        Parameters
        ----------
        time : list
            time values of you photometry data
        a : float
            slope coefficient used to linearly transform data
        b: float
            intercept coefficient used to linearly transform data

        Returns
        ----------
        np.array
            data linearly transformed
        """
        data = np.array(data)

        return a * data + b
    def save_data_to_csv(self, df, csv_name):
        """
        Takes in a DataFrame and a name
        then saves the DataFrame as a csv file
        with the name and appends a suffix if
        necessary to avoid overwriting.

        Parameters
        ----------
        time : list
            time values of you photometry data
        a : float
            slope coefficient used to linearly transform data
        b: float
            intercept coefficient used to linearly transform data

        Returns
        ----------
        np.array
            data linearly transformed
        """
        not_saved = True
        try:
            df.to_csv(csv_name + '.csv', mode = 'x')
            not_saved = False
        except:
            copy = 1
        while not_saved:
            try:
                df.to_csv(csv_name + '(' + str(copy) + ').csv')
                print(csv_name + '(' + str(copy) + ').csv saved')
                not_saved = False
            except:
                copy = copy + 1
        return

#### End Helper Functions ####
##### Class Functions #####
    def combine_objs(self, obj2, new_obj_name, combine_type, time_adj):
        """
        Combines attributes from two different objects to create a new object.

        Parameters
        ----------
        self : FiberObj
            The new object that starts identical to object 1
        obj2 : FiberObj
            object that will be concatenated to the first objects
        new_obj_name : str
            name for the combined obj that will be created
        combine_type : str
            One of four ways to adjust the time to combine the two dataframes
        time_adj : float
            time adjustment a value used in different ways
            depending on the combine_type selected

        Returns
        ----------
        class object : fiberObj
            a new fiber object that contains data from two existing objects
        """
        #first check for compatibility
        if self.version != obj2.version:
            print('These files cannot be combined do to version incompatibilities')
            return
        #find and compare raw data channels
        raw_obj2_channels = set()
        raw_obj1_channels = set()
        for channel in self.channels:
            if "Normalized" not in channel:
                raw_obj1_channels.add(channel)
        for channel in self.channels:
            if "Normalized" not in channel:
                raw_obj2_channels.add(channel)
        if raw_obj1_channels != raw_obj2_channels:
            print("""These files cannot be combined because
            they do not have the same raw data channels""")
            return

        if abs(self.frame_rate - obj2.frame_rate) > 1:
            print('error')
            return

        if self.behaviors != obj2.behaviors:
            print('Warning: the behaviors in obj1 and different than the behaviors in obj2')

        self.obj_name = new_obj_name

        if self.fiber_num != obj2.fiber_num:
            self.fiber_num = [self.fiber_num, obj2.fiber_num]

        if self.animal_num != obj2.animal_num:
            self.animal_num = [self.animal_num, obj2.animal_num]

        if self.exp_date != obj2.exp_date:
            self.exp_date = self.exp_date + ', ' + obj2.exp_date

        if self.filename != obj2.filename:
            self.filename = self.filename + ', ' + obj2.filename

        if self.beh_filename != obj2.beh_filename:
            self.beh_filename = self.beh_filename + ', ' + obj2.beh_filename
        #self.start_time and start_idx will be the start time and idx from the first obj
        self.stop_time = obj2.stop_time
        self.stop_idx = obj2.stop_idx
        self.PETS_results = pd.DataFrame(columns = ['Object Name', 'Behavior',
                                                    'Channel', 'range',
                                                    'Max value',
                                                    'Time of max',
                                                    'Min value',
                                                    'Time of min',
                                                    'AUC'
                                                    'Average value before event',
                                                    'Average value after event',
                                                    'Time before', 'Time after',
                                                    'Number of events', 'Baseline',
                                                    'Normalization type'])
        self.correlation_results = pd.DataFrame(columns = ['Object Name', 'Channel',
                                                           'Obj2', 'Obj2 Channel',
                                                           'start_time', 'end_time',
                                                           'R Score', 'p score'])
        self.beh_corr_results = pd.DataFrame(columns = ['Object Name', 'Channel',
                                                        'Obj2', 'Obj2 Channel',
                                                        'Behavior',
                                                        'Number of Events',
                                                        'R Score', 'p score'])

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
        obj2_temp_df = obj2.fpho_data_df
        for col in self_beh_cols:
            if col not in obj2_beh_cols:
                obj2_temp_df[col] = ""
        for col in obj2_beh_cols:
            if col not in self_beh_cols:
                self.fpho_data_df[col] = ""
        self.fpho_data_df = pd.concat([self.fpho_data_df, obj2_temp_df], join = 'inner')
        return

    #Signal Trace function
    def plot_traces(self):
        """
        Creates and displays graphs of a fiber object's signals.

        Parameters
        ----------
        None

        Returns
        -------
        fig : plotly.graph_objects.Scatter
            Plot of all traces in the object
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
        Normalizes signal using user specified normalization technique.
        Adds normalized traces to the fpho_data_df attribute of the object.
        Creates a plots to visualize the normalization process.
        Creates a plot normalizing 1 fiber data to an
        exponential of the form y=A*exp(-B*X)+C*exp(-D*x)

        Parameters
        ----------
        signal : string
            the key of the channel to normalize
        reference : string
            the key of the channel to be used as a reference or None
        biexp_thres : float
            value used to reject poor biexponential fits
        linfit_type : string
            one of two options used to determine of to define
            coefficients for the linear fit
        Returns
        --------
        fig : plotly.graph_objects.Scatter
            plot with a different panel for each step in the
            normalization process
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
                               bounds =  ([0, 0, 0, 0, 0], [np.inf,10,np.inf,100,1]))
        a_sig= popt[0]  # A value
        b_sig= popt[1]  # B value
        c_sig = popt[2]  # C value
        d_sig = popt[3]  # D value
        e_sig = popt[4]  # E value

        # Generate fit line using calculated coefficients
        fit_sig = 1.5*(max(sig))*self.fit_exp(sig_time, a_sig, b_sig, c_sig, d_sig, e_sig)
        sig_r_square = np.corrcoef(sig, fit_sig)[0,1] ** 2

        if sig_r_square < biexp_thres:

            a_sig= 0
            b_sig= 0
            c_sig = 0
            d_sig = 0
            e_sig = np.median(sig)
            fit_sig = self.fit_exp(sig_time, a_sig, b_sig, c_sig, d_sig, e_sig)

        normed_sig = [(k / j) for k,j in zip(sig, fit_sig)]
        self.fpho_data_df.loc[:, signal + ' expfit'] = fit_sig
        self.sig_fit_coefficients = ['A= ' + str(a_sig), 'B= ' + str(b_sig), 'C= '
                                     + str(c_sig), 'D= ' + str(d_sig), 'E= ' + str(e_sig)]
        self.fpho_data_df.loc[:, signal + ' normed to exp']=normed_sig

        if reference is not None:
            try:
                ref_time = self.fpho_data_df['time' + reference[3:]]
            except KeyError:
                ref_time = self.fpho_data_df['time']
            ref = self.fpho_data_df[reference]
            popt, pcov = curve_fit(self.fit_exp, ref_time, ref/(1.5*(max(ref))),
                                   p0=(1.0, 0, 1.0, 0, 0),
                                   bounds = ([0, 0, 0, 0, 0], [np.inf,10,np.inf,100,1]))
            a_ref = popt[0]  # A value
            b_ref = popt[1]  # B value
            c_ref = popt[2]  # C value
            d_ref = popt[3]  # D value
            e_ref = popt[4]  # E value

            # Generate fit line using calculated coefficients

            fit_ref = 1.5*(max(ref))*self.fit_exp(ref_time, a_ref, b_ref, c_ref, d_ref, e_ref)
            ref_r_square = np.corrcoef(ref, fit_ref)[0,1] ** 2

            if ref_r_square < biexp_thres:
                a_ref = 0
                b_ref = 0
                c_ref = 0
                d_ref = 0
                e_ref = np.median(ref)
                fit_ref = self.fit_exp(ref_time, a_ref, b_ref, c_ref, d_ref, e_ref)

            normed_ref = [(k / j) for k,j in zip(ref, fit_ref)]

            if linfit_type == 'Least squares':
                popt, pcov = curve_fit(self.fit_lin, normed_sig, normed_ref,
                                       bounds = ([0, -1], [100, 1]))
                a_lin = popt[0]
                b_lin = popt[1]

            else:
                sig_q75, sig_q25 = np.percentile(normed_sig, [75 ,25])
                sig_iqr = sig_q75 - sig_q25
                ref_q75, ref_q25 = np.percentile(normed_ref, [75 ,25])
                ref_iqr = ref_q75 - ref_q25

                a_lin = sig_iqr/ref_iqr
                b_lin = np.median(normed_sig) - a_lin * np.median(normed_ref)

            adjusted_ref=[a_lin * j + b_lin for j in normed_ref]
            normed_to_ref=[(k / j) for k,j in zip(normed_sig, adjusted_ref)]

            lin_r = np.corrcoef(adjusted_ref, normed_sig)[0,1]

            # below saves all the variables we generated to the df
            #  data frame inside the obj ex. self
            # and assign all the long stuff to that
            # assign the a_sig, b_sig,.. etc and a_ref, b_ref, etc to lists called
            #self.sig_fit_coefficients, self.ref_fit_coefficients and self.sig_to_ref_coefficients
            self.fpho_data_df.loc[:, reference + ' expfit']=fit_ref
            self.ref_fit_coefficients = ['A= ' + str(a_ref), 'B= ' + str(b_ref), 'C= ' +
                                         str(c_ref), 'D= ' + str(d_ref), 'E= ' + str(e_ref)]
            self.fpho_data_df.loc[:, reference + ' normed to exp']=normed_ref
            self.fpho_data_df.loc[:,reference + ' fitted to ' + signal]=adjusted_ref
            self.sig_to_ref_coefficients = ['A= ' + str(a_lin), 'B= ' + str(b_lin)]

        else:
            normed_to_ref = normed_sig

        self.fpho_data_df.loc[:,'Normalized_' + signal[4:]] = normed_to_ref
        self.channels.add('Normalized_' + signal[4:])
        if reference is not None:
            fig = make_subplots(rows = 3, cols = 2, x_title = 'Time(s)',
                                y_title = 'Flourescence (au)',
                                subplot_titles=("Biexponential Fitted to Signal (R^2 = " +
                                                str(np.round(sig_r_square, 3)) + ")",
                                                "Signal Normalized to Biexponential",
                                                "Biexponential Fitted to Ref (R^2 = " +
                                                str(np.round(ref_r_square, 3)) + ")",
                                                "Reference Normalized to Biexponential",
                                                "Reference Linearly Fitted to Signal(R = " +
                                                str(np.round(lin_r, 3)) + ")",
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
                showlegend = False),
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
                showlegend = False),
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
                showlegend = False),
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
                showlegend = False),
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
                showlegend = False),
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
                showlegend = False),
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
                showlegend = False),
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
                showlegend = False),
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
                showlegend = False),
                row = 3, col = 2
                )
        else:
            fig = make_subplots(rows = 1, cols = 2,
                                subplot_titles=(
                                    "Biexponential Fitted to Signal(R^2 = "
                                    + str(sig_r_square) + ")",
                                    "Signal Normalized to Biexponential"
                                ),
                                shared_xaxes = True, vertical_spacing = 0.1,
                                x_title = "Time (s)",
                                y_title = "Fluorescence (au)")
            fig.add_trace(
                go.Scatter(
                x = sig_time,
                y = sig,
                mode = "lines",
                line = go.scatter.Line(color="rgba(0, 255, 0, 1)"),
                name ='Signal:' + signal,
                text = 'Signal',
                showlegend = False),
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
                showlegend = False),
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
                showlegend = False),
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
        Takes a .csv file with behavior data and adds it to the
        fpho_data_df dataframe

        Parameters
        ----------
        beh_data : Dataframe
            Dataframe created from the behavior data .csv file
        filename : string
            name of behavior data .csv file

        Returns
        --------
        None
        """

        beh_data = beh_data.sort_values(by = 'Time', kind = 'stable')
        unique_behaviors = beh_data['Behavior'].unique()
        for beh in unique_behaviors:
            if beh in self.fpho_data_df.columns:
                print(beh + ' is already in ' + self.obj_name +
                      ' and cannot be added again.')
            else:
                self.behaviors.add(beh)
                idx_of_beh = beh_data.index[beh_data['Behavior'] == beh].tolist()
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
                        start_idx = self.fpho_data_df['time'].searchsorted(
                            beh_data.loc[idx_of_beh[j],'Time'])
                        end_idx = self.fpho_data_df['time'].searchsorted(
                            beh_data.loc[idx_of_beh[j + 1],'Time'])
                        if end_idx < len(self.fpho_data_df['time']) and start_idx > 0:
                            self.fpho_data_df.loc[end_idx, beh] = 'E'
                            self.fpho_data_df.loc[start_idx, beh] = 'S'
                            self.fpho_data_df.loc[start_idx+1 : end_idx-1, beh] = 'O'
                        j = j + 2
                    else:
                        print("\nStart and stops for state behavior:"
                             + beh + " are not paired correctly.\n")
                        j = j + 1
        if self.beh_filename == 'N/A':
            self.beh_filename = filename
        else:
            self.beh_filename = self.beh_filename + ", " + filename
        return

    def plot_behavior(self, behaviors, channels):
        """
        Creates a plot for one or more channels with behavior data
        overlaid as colored rectangles.

        Parameters
        ----------
        behaviors : list
            user selected behaviors
        channels : list
            user selected channel

        Returns
        ----------
        fig : list
            list of plotly.graph_objects.Scatter
            Scatter plot of photometry signal with
            behavior data overlaid as colored rectangles
        """
        colors = ['#02F2C2', '#0D85FD', '#C221FD', '#FF4CD2', '#FB0E59',
                  '#FD8059', '#FFF159', '#69FF70', '#A162FF', '#626562',
                  '#07FFDA', '#30A3FF', '#FF5AFF', '#FF94E3', '#FF2E99',
                  '#FFBB9D', '#FFFAA2', '#8BFF8A', '#CD83FF', '#777777',
                  '#019980', '#08529B', '#801CF5', '#BB3280', '#BD0B35',
                  '#C35035', '#D4C931', '#42A542', '#693DBB', '#393939']

        behavior_colors = {behavior: colors[i%30] for i, behavior in enumerate(behaviors)}
        try:
            time_key = 'time' + (channels[0][3:] if channels[0].startswith('data') else channels[0][9:])
            time = self.fpho_data_df[time_key]
        except KeyError:
            time = self.fpho_data_df['time']
        start_lines = []
        bout_rects = []
        beh_labels = []
        for i, beh in enumerate(behaviors):
                behavior_color = behavior_colors[beh]
                temp_beh_string = ''.join(list(self.fpho_data_df[beh]))
                #Find starts and create lines to add to graphs
                S = re.compile(r'S')
                starts = S.finditer(temp_beh_string)
                for start in starts:
                    start_time = time.at[start.start()]
                    vertical_line = dict(
                        type="line",
                        xref="x",
                        yref="paper",
                        x0=start_time,
                        x1=start_time,
                        y0=0,
                        y1=1,
                        line=dict(width=2, color=behavior_color),
                        layer="below"
                    )
                    start_lines.append(vertical_line)
                #Find bouts and create rectangles to add to graphs
                pattern = re.compile(r'S[O]+E')
                bouts = pattern.finditer(temp_beh_string)
                for bout in bouts:
                    start_time = time.at[bout.start()]
                    end_time = time.at[bout.end()]
                    rectangle = dict(
                        type="rect",
                        xref="x",
                        yref="paper",
                        x0=start_time,
                        y0=0,
                        x1=end_time,
                        y1=1,
                        fillcolor=behavior_color,
                        opacity=0.5,
                        line=dict(width=0),
                        layer="below")
                    bout_rects.append(rectangle)
                #create an annotation for each behavior
                behavior_color = behavior_colors[beh]
                annotation = dict(
                    xref="paper",
                    yref="paper",
                    x=1,
                    y=1 - i/len(behaviors),
                    text=beh,
                    bgcolor=behavior_color,
                    showarrow = False
                )
                beh_labels.append(annotation)

        figs = []
        for i, channel in enumerate(channels):
            fig = go.Figure()
            fig.update_layout(shapes = bout_rects + start_lines, annotations = beh_labels, title = self.obj_name + ' ' + channel)
            fig.update_xaxes(title_text = 'Time (s)', showgrid = False)
            fig.update_yaxes(title_text = 'Fluorescence (au)')
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=self.fpho_data_df[channel],
                    mode="lines",
                    line=go.scatter.Line(color="Black"),
                    name=channel,
                    showlegend=False)
            )
            fig.show()
            figs.append(fig)
        return figs

    def plot_PETS(self, channel, beh, time_before, time_after,
                    baseline = 0, base_option = 'Each event', show_first = 0,
                    show_last = -1, show_every = 1,
                    save_csv = False, percent_bool = False):

        """
        Takes a dataframe and creates plot of the photometry data around
        the start of each occurance of a behavior as well as average and
        SEM of the data at all occurances. Stores results in dataframe.

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
        percent_bool: bool
            if true percent will be calculated as opposed to z-score
        save_csv: bool
            if true the dateframe used to create the graph will be saved in a csv file

        Returns
        ----------
        fig : plotly.graph_objects.Scatter
            PETS plot for select behaviors
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
                                             ))
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

        # Initialize array of time series sums
        PETS_dict = {}
        PETS_time_dict = {}
        # Initialize events counter to 0
        n_events = 0

        if base_option == 'Each event':
            base_mean = None
            base_std = None
            PETS_baseline = 'Each event'

        elif base_option == 'Start of Sample':
            # idx = np.where((start_event_time > baseline[0]) & (start_event_time < baseline[1]))
            # Find baseline start/end index
            # Start event time is the first occurrence of event,
            #this option will be for a baseline at the beginning of the trace
            base_start_idx = full_time.searchsorted(
                baseline[0])
            base_end_idx = full_time.searchsorted(
                baseline[1])
            # Calc mean and std for values within window
            base_mean = np.nanmean(self.fpho_data_df.loc[
                base_start_idx:base_end_idx, channel])
            base_std = np.nanstd(self.fpho_data_df.loc[
                base_start_idx:base_end_idx, channel])
            PETS_baseline = 'From ' + str(baseline[0]) + ' to ' + str(baseline[1])

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
            PETS_baseline = 'From ' + str(end_time - start) + ' to ' + str(end_time - end)

        # Loops over all start times for this behavior
        #  time = actual time
        for time in beh_times:
            # Calculates indices for baseline window before each event
            if base_option == 'Before Events':
                start = max(baseline)
                end = min(baseline)
                PETS_baseline = 'From ' + str(start) + ' to ' + str(end) + ' before each event'
                base_start_idx = full_time.searchsorted(
                    time - start)
                base_end_idx = full_time.searchsorted(
                    time - end)
                base_mean = np.nanmean(self.fpho_data_df.loc[
                    base_start_idx:base_end_idx, channel])
                base_std = np.nanstd(self.fpho_data_df.loc[
                    base_start_idx:base_end_idx, channel])

            # time - time_Before = start_time for this event trace
            #time is the actual event start, time before is secs input before event start
            # Finds time in our data that is closest to time - time_before
            # start_idx = index of that time
            start_idx = full_time.searchsorted(
                time - time_before)
            # time + time_after = end_time for this event trace
            #time is the actual event start, time after is secs input after event start
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
                time_trace = full_time.iloc[start_idx:end_idx].subtract(
                    full_time.iloc[start_idx] + time_before).tolist()
                if percent_bool:
                    if base_option == 'Each event':
                        base_mean = np.nanmean(trace)
                        norm_type = 'Percent'
                    this_time_series=[((i / base_mean)-1)*100 for i in trace]
                else:
                    this_time_series = self.zscore(trace, base_mean, base_std)
                    norm_type = 'Z-score'

                # Aligns time series and adds each trace to a dict
                if n_events > 1:
                    PETS_dict['event' + str(n_events)] = this_time_series
                    PETS_time_dict['event' + str(n_events)] = time_trace
                #plot requested traces
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
                        x = time_trace,
                        y = this_time_series,
                        mode = "lines",
                        line = dict(color = trace_color, width = 2),
                        name = 'Event:' + str(n_events),
                        text = 'Event:' + str(n_events),
                        showlegend=True),
                        row = 1, col = 2
                        )
        #drop values that aren't in all traces and add to a DataFrame
        PETS_data = pd.DataFrame()
        min_length, min_key = min((len(lists), key) for key, lists in PETS_dict.items())
        for key in PETS_dict:
            target_times = PETS_time_dict[min_key]
            closest_times = [min(PETS_time_dict[key], key=lambda x: abs(x - time)) for time in target_times]
            PETS_data[key] = [PETS_dict[key][PETS_time_dict[key].index(time)] for time in closest_times]

        average = PETS_data.mean(axis=1).to_list()
        sem = PETS_data.sem(axis=1).to_list()
        graph_time = target_times
        PETS_data.insert(0, 'time', graph_time)
        PETS_data.insert(1, 'Average', average)
        PETS_data.insert(2, 'SEM', sem)
        PETS_data.insert(3, 'SEM Upper Bound', PETS_data['Average'] + PETS_data['SEM'])
        PETS_data.insert(4, 'SEM Lower Bound', PETS_data['Average'] - PETS_data['SEM'])
        upper_bound = PETS_data['SEM Upper Bound'].to_list()
        lower_bound = PETS_data['SEM Lower Bound'].to_list()
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
            # Y = Average of all event traces
            y = average,
            mode = "lines",
            line = dict(color = "Black", width = 5),
            name = 'average',
            text = 'average',
            showlegend = True),
            row = 1, col = 2
            )
        #fig.update_yaxes(range = [.994, 1.004])

        fig.update_layout(
            title = 'PETS plot of ' + beh + ' for '
                    + self.obj_name + ' in channel ' + channel
            )
        fig.update_xaxes(title_text = 'Time (s)')
        fig.update_yaxes(title_text = 'Fluorescence (au)', col = 1, row = 1)
        fig.update_yaxes(title_text = norm_type, col = 2, row = 1)

        results = pd.DataFrame({'Object Name': [self.obj_name],
                               'Behavior': [beh], 'Channel' : [channel],
                               'Max value' : [max(average)],
                               'Time of max ' : [graph_time[np.argmax(average)]],
                               'Min value': [min(average)],
                               'Time of min' : [graph_time[np.argmin(average)]],
                               'AUC' : [integrate.simpson(average[zero_idx:],graph_time[zero_idx:])],
                               'range' : [max(average) - min(average)],
                               'Average value before event' : [np.mean(average[:zero_idx])],
                               'Average value after event' : [np.mean(average[zero_idx:])],
                               'Time before' : [time_before], 'Time after': [time_after],
                               'Number of events' : [n_events], 'Baseline' : [PETS_baseline],
                               'Normalization type' : [norm_type]})
        self.PETS_results = pd.concat([self.PETS_results, results])
        if save_csv:
            csv_name = self.obj_name + '_' + channel + '_' + beh + '_Baseline_' + PETS_baseline
            self.save_data_to_csv(PETS_data, csv_name)
        return fig

    # Zscore calc helper
    def zscore(self, data, mean = None, std = None):
        """
        Helper function to calculate z-scores

        Parameters
        ----------
        data : list
            time series data for one event
        mean : int/float, optional
            baseline mean value
        std : int/float, optional
            baseline standard deviation value

        Returns
        ----------
        zscored_data : list
            zscore values of time series for an event
        """
        # Default Params, no arguments passed
        if mean is None and std is None:
            mean = np.nanmean(data)
            std = np.nanstd(data)
        # Calculates zscore per event in list
        zscored_data = [(i - mean) / std for i in data]
        return zscored_data

    def peak_finding(self, channel, start_time, end_time, peak_widths, save_data):
        start_size = peak_widths[0]/self.frame_rate
        stop_size = peak_widths[1]/self.frame_rate
        try:
            time = self.fpho_data_df['time' + channel[3:]]
        except KeyError:
            try:
                time = self.fpho_data_df['time' + channel[9:]]
            except KeyError:
                time = self.fpho_data_df['time']
        start_idx = np.searchsorted(time, start_time)
        if end_time == -1:
            end_idx = len(time)-1
        else:
            end_idx = np.searchsorted(time, end_time)
        time = time[start_idx:end_idx].reset_index(drop = True)
        data = self.fpho_data_df.loc[start_idx:end_idx, channel].reset_index(drop = True)
        peaks = sci_sig.find_peaks_cwt(data, np.arange(start_size, stop_size))

        #clean up the peaks a bit
        min_distance_between_peaks = min(peaks[i+1] - peaks[i] for i in range(len(peaks) - 1))
        half_min = int(min_distance_between_peaks/2)
        adj_peaks = []
        for peak in peaks:
            real_peak = data.loc[peak-half_min:peak+half_min].idxmax()
            adj_peaks.append(real_peak)
        peak_times = [time[idx] for idx in adj_peaks]
        peak_heights = [data[idx] for idx in adj_peaks]
        peak_data = pd.DataFrame({'Time':peak_times, 'Peak Height': peak_heights})

        fig = go.Figure()
        #plots data
        fig.add_trace(
            go.Scattergl(
            x = time,
            y = data,
            mode = "lines",
            line = dict(color = 'green', width = 2),
            name = channel,
            showlegend = True)
        )
        #plots peaks
        fig.add_trace(
            go.Scattergl(
            x = peak_times,
            y = peak_heights,
            mode = "markers",
            marker = dict(color = 'black', size=5, symbol="cross"),
            name = "peaks",
            showlegend = True)
        )
        fig.show()
        results = pd.DataFrame({'Object Name': [self.obj_name],
                       'Channel' : [channel],
                       'Start time' : [time.iloc[0]],
                       'End time' : [time.iloc[-1]],
                       'Width range entered': [peak_widths],
                       'Number of peaks' : [len(peaks)],
                       'Average peak amplitude' : [np.mean(peak_heights)],
                       'Average trace amplitude' : [np.mean(data)],
                       'Frequency of peaks (peaks/s)' : [len(peaks)/(time.iloc[-1]-time.iloc[0])]})
        try:
            self.peak_results = pd.concat([self.peak_results, results])
        except AttributeError:
            self.peak_results = results
        if save_data:
            csv_name = self.obj_name + '_' + channel + 'from' + str(start_time) + 'to' + str(end_time)
            self.save_data_to_csv(peak_data, csv_name)
        return fig

    def pearsons_correlation(self, obj2, channel1, channel2, start_time, end_time):
        """
        Calculates the pearsons correlation coefficient for and plot the channels
        specified by the user. Store results in the self.correlation_results and
        obj2.correlation_results dataframes.

        Parameters
        ----------
        obj2 : fiber object
            second object for correlation analysis
        channel1 : string
            key of the signal selected for the first object
        channel2 : string
            key of the signal selected for the first object
        start_time : int
            starting timestamp of data
        end_time : int
            ending timestamp of data

        Returns
        ----------
        fig : plotly.graph_objects.Scatter
            Plots of both signal against time and the signals against eachother
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
            print('Trace 1 starts at ' + str(np.round(time1[start_idx1])) +
                  's after your start time ' + str(start_time) + '\n')
        if end_time - time1[end_idx1] > 1/self.frame_rate:
            print('Trace 1 ends at ' + str(np.round(time1[end_idx1])) +
                  's before your end time ' + str(end_time) + 's\n')

        if start_time - time2[start_idx2] < -1/obj2.frame_rate:
            print('Trace 2 starts at ' + str(np.round(time2[start_idx2])) +
                  's after your start time ' + str(start_time) + 's\n')
        if end_time - time2[end_idx2] > 1/obj2.frame_rate:
            print('Trace 2 ends at ' + str(np.round(time2[end_idx2])) +
                  's before your end time ' + str(end_time) + 's\n')

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
        results = pd.DataFrame({'Object Name': [self.obj_name], 'Channel' : [channel1],
                                'Obj2': [obj2.obj_name], 'Obj2 Channel': [channel2],
                                'start_time' : [start_time], 'end_time' : [end_time],
                                'R Score' : [str(r)], 'p score': [str(p)]})
        self.correlation_results = pd.concat([self.correlation_results, results])
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
        Calculates the pearsons correlation coefficient for and plots the
        channels specified by the user during times that a specific behavior
        is occuring. Store results in the self.beh_corr_results and
        obj2.beh_corr_results dataframes.

        Parameters
        ----------
        obj2 : fiber object
            second object for correlation analysis
        channel1 : string
            key of the signal selected for the first object
        channel2 : string
            key of the signal selected for the first object
        beh : str
            key of the behavior

        Returns
        ----------
        fig : plotly.graph_objects.Scatter
            Plots of both signal against time and the signals against eachother
        """

        behavior_slice1 = self.fpho_data_df[self.fpho_data_df[beh] != ' ']
        behavior_slice2 = obj2.fpho_data_df[self.fpho_data_df[beh] != ' ']
        try:
            time = behavior_slice1['time' + channel1[3:]]
        except KeyError:
            try:
                time = behavior_slice1['time' + channel1[9:]]
            except KeyError:
                time = behavior_slice1['time']
        sig1 = behavior_slice1[channel1]
        sig2 = behavior_slice2[channel2]
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

        [r, p] = ss.pearsonr(sig1, sig2)

        fig.update_layout(
                title = 'Correlation between ' + self.obj_name + ' and '
                  + obj2.obj_name + ' during ' + beh + ' is, ' + str(r) + ' p = ' + str(p)
                )
        fig.update_xaxes(title_text = self.obj_name + " " + channel1, col = 2, row = 1)
        fig.update_yaxes(title_text = obj2.obj_name + " " + channel2, col = 2, row = 1)
        fig.update_xaxes(title_text = 'Time (s)', col = 1, row = 1)
        fig.update_yaxes(title_text = 'Fluorescence (au)', col = 1, row = 1)

        results = pd.DataFrame({'Object 1 Name': [self.obj_name],
                   'Object 1 Channel': [channel1],
                   'Object 2 Name': [obj2.obj_name],
                   'Object 2 Channel': [channel2],
                   'Behavior' : [beh],
                   'Number of Events': [self.fpho_data_df[beh].value_counts()['S']],
                   'R Score' : [r], 'p score': [p]})
        self.beh_corr_results = pd.concat([self.beh_corr_results, results])

        return fig

##### End Class Functions #####

def alt_to_boris(beh_file, time_unit, beh_false, time_between_bouts):
    """
    Converts alternative format behavior data to a BORIS file format.

    Parameters
    ----------
    beh_file : Dataframe
        Pandas dataframe created from the .csv behavior file
    time_unit : str
        One of three options that specifies the time unit used
        in the .csv behavior file
    beh_false : str
        value in the .csv file that signifies when a behavior
        is not occuring
    time_between_bouts : float
        The minumum time between behavioral epochs
        that qualifies as two different bouts

    Returns
    ----------
    boris_df : Dataframe
        A dataframe of behavior data in the boris format
    """
    boris_df = pd.DataFrame(columns = ['Time', 'Behavior', 'Status'])
    conversion_dict = {'milliseconds':1/1000,'seconds':1,'minutes':60}
    conversion_to_sec = conversion_dict[time_unit]
    behaviors = list(beh_file.columns)
    behaviors.remove('Time')
    for beh in behaviors:
        trimmed = beh_file[beh_file[beh] != beh_false]
        starts = [(trimmed.iloc[0]['Time'] - beh_file.iloc[0]['Time']) * conversion_to_sec]
        stops = []
        diffs = np.diff(trimmed.index)

        for i, v in enumerate(diffs):
            if v > (time_between_bouts / conversion_to_sec):
                stops.append((trimmed.iloc[i]['Time']
                              - beh_file.iloc[0]['Time']) * conversion_to_sec)
                if i < len(diffs):
                    starts.append((trimmed.iloc[i+1]['Time']
                                   - beh_file.iloc[0]['Time']) * conversion_to_sec)
        stops.append((trimmed.iloc[-1]['Time']
                      - beh_file.iloc[0]['Time']) * conversion_to_sec)

        starts_df = pd.DataFrame(data = {'Time': starts, 'Behavior': [beh]*len(starts), 'Status' : ['START']*len(starts)})
        stops_df = pd.DataFrame(data = {'Time': stops, 'Behavior': [beh]*len(stops), 'Status' : ['STOP']*len(stops)})
        beh_df =  pd.concat([starts_df, stops_df])
        beh_df = beh_df.sort_values(by = 'Time', kind = 'stable')
        boris_df = pd.concat([boris_df, beh_df])
    boris_df = boris_df.sort_values(by = 'Time', kind = 'stable')
    boris_df.reset_index(inplace=True, drop=True)
    return boris_df