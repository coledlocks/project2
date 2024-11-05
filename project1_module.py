#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
project1_module.py (ECG Visualization)

This module contains all the required functions to 
analyze, extract, and store ecg data to determine a normal heart beat
from an arrhythmic heart beat. 

Jackie Berk and Cole Drozdek

Sources from the ECG ReadMe to get and understand the dataset:
    
Taddei A, Distante G, Emdin M, Pisani P, Moody GB, Zeelenberg C, Marchesi C. The European 
ST-T Database: standard for evaluating systems for the analysis of ST-T changes in ambulatory 
electrocardiography. European Heart Journal 13: 1164-1172 (1992).

Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. 
(2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for
complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.
"""

# importing packages to use
import numpy as np
from matplotlib import pyplot as plt

# %% PART 1

# defining a function to read the npz file
def load_data(input_file):
    """
    This function loads data from the npz file,
    extracts the list of fields, and corresponds them
    to the same variable name.Then, these new variables are returned
    to the function to be used in the code.

    Parameters
    ----------
    input_file : str
        Contains variable names to be extracted.

    Returns
    -------
    ecg_voltage : float, array of voltages
        Contains voltage values from dataset.
    fs : int
        The sampling frequency or number of samples recorded per second (Hz).
    label_samples : int, array of samples
         Samples that are emphasized to indicate that an event occurred.
    label_symbols : str, array of symbols
        Represent differing events that occurred and correspond the label_samples.
    subject_id : str
        Indication for which subject was used.
    electrode : str
        Indication of which ecg electrode is recording data.
    units : str
        Recorded voltage is measured in volts (V).

    """
    # loading dataset
    data = np.load(input_file)
    
    # looping through and printing variable data
    for variable in data.files:
        print(variable)
    
    # assigning variables
    ecg_voltage = data['ecg_voltage']
    fs = data['fs']
    label_samples = data['label_samples']
    label_symbols = data['label_symbols']
    subject_id = data['subject_id']
    electrode = data['electrode']
    units = data['units']
    return ecg_voltage, fs, label_samples, label_symbols, subject_id, electrode, units # returns ecg_voltage, fs, label_samples, label_symbols, subject_id, electrode, units in function
# %% PART 2

def plot_raw_data(signal_voltage, signal_time, units="V", title=''):
    """
    This function plots the ecg signal, measured in volts, 
    over time, measured in seconds. 

    Parameters
    ----------
    signal_voltage : float, array of voltages
        Contains voltages that make the ecg signal.
    signal_time : float, array of times
        Contains time values that correspond to signal_voltage to create the ecg signal.
    units : str, optional
        The unit used to describe the voltage. The default is "V".
    title : str, optional
        The name used to label the ecg plot. The default is ''.

    Returns
    -------
    None.

    """
    plt.figure(1, clear=True)
    plt.plot(signal_time, signal_voltage, label='ecg')
    plt.xlabel('time (secs)')
    plt.ylabel(units)
    plt.title(title)

# %% PART 3


def plot_events(label_samples, label_symbols, signal_time, signal_voltage):
    """
    This function isolates normal sample readings 
    from arrhythmic sample readings. A plot is generated
    with different colored markings to differentiate the
    two types of heartbeats. 

    Parameters
    ----------
    label_samples : int, array of samples
        Samples that are emphasized to indicate that an event occurred.
    label_symbols : str, array of symbols
        Represent differing events that occurred and correspond the label_samples.
    signal_time : float, array of times
        Contains time values that correspond to signal_voltage to create the ecg signal.
    signal_voltage : float, array of voltages
        Contains voltages that make the ecg signal.

    Returns
    -------
    None.

    """
    symbols = np.unique(label_symbols) # seeing how many types of heart arrythmias there are
    label_samples_v = np.array([]) # start with empty array for arrhythmic samples
    label_samples_n = np.array([]) # start with empty array for normal samples
    for symbol_index in range(len(label_symbols)): # find position of symbols
        if label_symbols[symbol_index] == symbols[1]: # in label_symbols, find "v" symbols
           label_samples_v = np.append(label_samples_v,label_samples[symbol_index])  # add samples that correspond to "v" symbol to "v" sample array
        else:
            label_samples_n = np.append(label_samples_n,label_samples[symbol_index]) # if not "v" symbol, add the sample with "n" symbol to "n" sample array
    for v_index in range(len(label_samples_v)):
        x = int(label_samples_v[v_index]) # x used as a placeholder for cleaner code
        # the if statement makes it so the label only appears a singular time
        if v_index == 0:                  
            plt.plot(signal_time[x], signal_voltage[x], 'o', color='orange', markersize=5, label='arrythmia')
        else:
            plt.plot(signal_time[x], signal_voltage[x], 'o', color='orange', markersize=5)
    for n_index in range(len(label_samples_n)):
        x = int(label_samples_n[n_index]) # x used as a placeholder for cleaner code
        # the if statement makes it so the label only appears a singular time
        if n_index == 0:                    
            plt.plot(signal_time[x], signal_voltage[x], 'o', color='green', markersize=5, label='normal')
        else:
            plt.plot(signal_time[x], signal_voltage[x], 'o', color='green', markersize=5)
    plt.xlim(2709, 2711.5) # show two normal heartbeats and one arrhythmic
    plt.legend()
    plt.show()
    plt.savefig('proj1_figure1', format='png', dpi=300)

# %% PART 4
def extract_trials(signal_voltage, trial_start_samples, trial_sample_count):
    """
    This function stores and extracts ecg voltages as trials 0.5 seconds
    before and after a normal and arrhythmic heartbeat occurred. Additionally,
    these one trial of each heartbeat type is plotted.

    Parameters
    ----------
    signal_voltage : float, array of voltages
        Contains voltages that make the ecg signal.
    trial_start_samples : float
        Sample numbers indicating when a trial should start.
    trial_sample_count : int
        Indicates how many trials should be within a sample.

    Returns
    -------
    trials : float, 2D array (trial_count x trial_sample_count)
        Stores ecg voltages 0.5 seconds before and after a heartbeat.

    """
    # start with array of zeros
    trials = np.zeros((len(trial_start_samples), trial_sample_count), dtype=float)
    for trial_index in range(len(trial_start_samples)):
        if (trial_start_samples[trial_index]) < 0: # check if starting sample is negative
            trials[trial_index,:] = trials[trial_index,:] # if negative, zeros remain in that row of trials array
        elif (trial_start_samples[trial_index] + trial_sample_count) > len(signal_voltage): # check is starting sample is greater than length of signal
            trials[trial_index,:] = trials[trial_index,:] # if so, zeros remain in that row of trials array
        # if neither condition applies, valid start and end samples can be extracted and stored into trials 2D array
        else:
            trials[trial_index,:] = signal_voltage[int(trial_start_samples[trial_index]) : int(trial_start_samples[trial_index] + trial_sample_count)]
    return trials # return trials in function

# %% PART 5

def plot_mean_and_std_trials(signal_voltage, label_samples, label_symbols, trial_duration_seconds, fs, units, title):
    """
    This function calculates and graphs
    the average and standard deviation of all trials separately
    for both a normal and arrhythmic heartbeat. 

    Parameters
    ----------
    signal_voltage : float, array of voltages
        Contains voltages that make the ecg signal.
    label_samples : int, array of samples
        Samples that are emphasized to indicate that an event occurred.
    label_symbols : str, array of symbols
        Represent differing events that occurred and correspond the label_samples.
    trial_duration_seconds : float
        The intended trial durations measured in seconds.
    fs : int
        The sampling frequency or number of samples recorded per second (Hz).
    units : str
        Recorded voltage is measured in V.
    title : str
        The name used to describe the plot.

    Returns
    -------
    symbols : str, array of symbols
        The defined label symbols contained in this ecg dataset.
    trial_time : float, 1D array
        The corresponding times of each sample plotted across the mean of trials.
    mean_trial_signal : float, 2D array (symbols x trial_time)
        Contains the mean voltage signal at a specific time and corresponding trial type. 
       
    """
    symbols = np.unique(label_symbols) # get symbols "n" and "v"
    trial_sample_count = 250
    label_samples_n = np.array([]) # start with empty array for "n" samples
    label_samples_v = np.array([]) # start with empty array for "v" samples
    for sample_index in range(len(label_symbols)):
        if label_symbols[sample_index] == symbols[1]: # check if the sample matches with the symbol "v"
            label_samples_v = np.append(label_samples_v, label_samples[sample_index]) # if so, add value to the samples "v" array
        else:
            label_samples_n = np.append(label_samples_n, label_samples[sample_index]) # if not, add value to the samples "n" array
    
    # centering the data
    start = int(fs * 0.5)
    label_samples_v_start = label_samples_v - start
    label_samples_n_start = label_samples_n - start

    # calling the functions to extract the trials at the heartbeats
    trials_n = extract_trials(signal_voltage, label_samples_n_start, trial_sample_count)
    trials_v = extract_trials(signal_voltage, label_samples_v_start, trial_sample_count)
    
    # calculating the mean and standard deviations across all trials for both event types
    mean_n = np.mean(trials_n, axis=0)
    mean_v = np.mean(trials_v, axis=0)
    std_n = np.std(trials_n, axis=0)
    std_v = np.std(trials_v, axis=0)
    
    trial_time = (np.arange(trial_sample_count) - start)/fs # creates time arrangement
    
    plt.figure(2, clear=True)
   
    # plotting mean and standard deviation values
    plt.fill_between(trial_time, mean_v-std_v, mean_v+std_v, color='lightblue', alpha=0.5, label='v mean +/- std')
    plt.fill_between(trial_time, mean_n-std_n, mean_n+std_n, color='orange', alpha=0.5, label='n mean +/- std')
    
    # plotting one of each trial
    plt.plot(trial_time, trials_n[1000], color='orange', label='n trial')
    plt.plot(trial_time, trials_v[40], label='v trial')
    
    # plot annotation
    plt.title(title)
    plt.xlabel('time (secs)')
    plt.ylabel(units)
    plt.legend()
    plt.show()
    plt.savefig('proj1_figure2', format='png', dpi=300)
    
    # creating 1D mean value arrays for each symbol
    mean_trial_signal_v = np.array([])
    mean_trial_signal_n = np.array([])
    mean_trial_signal_v = np.append(mean_trial_signal_v, mean_v)
    mean_trial_signal_n = np.append(mean_trial_signal_n, mean_n)
    
    # combining 1D arrays of the means for each symbol
    mean_trial_signal = np.vstack((mean_trial_signal_v, mean_trial_signal_n))
    
    return symbols, trial_time, mean_trial_signal # return symbols, trial_time, mean_trial_signal in function

# %% PART 6

def save_means(symbols, trial_time, mean_trial_signal, out_filename='ecg_means.npz'): 
        """
        This function will save the calculated means
        across all trials for each event type.

        Parameters
        ----------
        symbols : str, array of symbols
            The defined label symbols contained in this ecg dataset.
        trial_time : float, array of times
            The corresponding times of each sample plotted across the mean of trials.
        mean_trial_signal : float, 2D array (symbols x trial_time)
            Contains the mean voltage signal at a specific time and corresponding trial type.  
        out_filename : str
            Variable of file for where data will be saved to. 

        Returns
        -------
        None.

        """
        # save data as npz file
        np.savez(out_filename, symbols=symbols, trial_time=trial_time, mean_trial_signal=mean_trial_signal)


