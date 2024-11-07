#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
project2_module.py
Created on Tue Oct 29 10:52:25 2024

this module contains a host of functions that use the previous projects data as templates
of normal and arrhythmic heart beats and compare them to raw ECG data.
it uses those matches to predict where beats occur and eventually detect where
normal beats and/or arrhythmic beats occur in new datasets.

@authors: Cole Drozdek, Matthew Bishop-Gylys
"""
# importing packages
import numpy as np

#%% PART 1: load data

# defining a function to read the npz file
def load_means(input_file):
    """
    this function loads in data from an npz file, prints the names of all the fields identified,
    extracts data and compiles them into arrays of the same name, which are then returned.

    Parameters
    ----------
    input_file : str
        contains field names and variables to be extracted under field names.

    Returns
    -------
    symbols : 1D array of strings, size:(n,), where n is the number of samples
        an array of the symbols that differentiate between a normal and arrhythmic heart beat.
    trial_time : 1D array of floats, size:(n,), where n is the time each trial occurs
        times when the trials were sampled.
    mean_trial_signal : 2D array of floats, size:(n, t), where n is which symbol is represented and t is the trial time of occurence.
        contains the mean voltage signal at a specific time and corresponding trial type.

    """
    # load in the npz file we saved in project 1
    # loading dataset
    data = np.load(input_file)
    
    # looping through and printing variable data
    for variable in data.files:
        print(variable)
    
    # assigning variables
    symbols = data['symbols']
    trial_time = data['trial_time']
    mean_trial_signal = data['mean_trial_signal']
    
    # return statement
    return symbols, trial_time, mean_trial_signal


#%% PART 2: resampling data

def decimate_data(original_signal, decimation_factor):
    """
    this function receives an array and a factor to decimate the array by.

    Parameters
    ----------
    original_signal : array of floats
        array to be decimated.
    decimation_factor : int
        factor by how much the array will be decimated by.

    Returns
    -------
    decimated_signal : array of floats
        a shortened version of the original_signal array that has been decimated by the inputted decimation factor.

    """
    # Extract every nth sample from the original signal where n is the decimation factor
    decimated_signal = original_signal[::decimation_factor]
    
    # return statement
    return decimated_signal
    
# %% PART 3: normalizing signal

def normalize_template(trial_mean):
    """
    this function normalizes beats to compare the shapes of the beats independently of size

    Parameters
    ----------
    trial_mean : 1D array of floats, size:(n,), where n is the mean of a beat event
        an array comprised of the mean signals of a heart beat.

    Returns
    -------
    template : 1D array of floats, size:(n,), where n is the normalized values of the trial_mean array
        an array that has been normalized from the inputted array.

    """
    # calculating the demeaned signal
    demeaned_signal = trial_mean - np.mean(trial_mean)
    
    # calculating energy of the demeaned signal
    energy = np.sum(demeaned_signal ** 2)
    template = demeaned_signal / energy
    
    # return statement
    return template

#%% PART 4: template match
def get_template_match(signal_voltage, template):
    """
    this function uses discrete cross-correlation to see how well a produced template matches a raw signal

    Parameters
    ----------
    signal_voltage : 1D array of floats, size:(n,), where n is the voltage samples 
        contains voltages that make up the ecg signal
    template : 1D array of floats, size:(n,), where n is the normalized values of heart beats 
        an array that has been normalized from average heart beats

    Returns
    -------
    template_match : 1D array of floats, size:(n,), where n is a list of cross-correlated values
        the discrete cross-correlation of the raw signal and the provided template

    """
    # Reverse the template for cross-correlation
    reversed_template = template[::-1]
    
    # Compute the cross-correlation using convolution
    template_match = np.convolve(signal_voltage, reversed_template, mode='same')
    
    # return statement
    return template_match   

# %% PART 5
def predict_beat_times(template_match, threshold=None):
    """
    this function uses a threshold (provided or defined by the function) to predict the times when heartbeats occur

    Parameters
    ----------
    template_match : 1D array of floats, size:(n,), where n is a list of cross-correlated values
        the discrete cross-correlation from the previous section.
    threshold : float, (optional)
        a threshold value to predict where a normal heartbeat would occur.

    Returns
    -------
    beat_samples : 1D array of ints, size:(n,), where n is the values directly after the threshold is exceeded
        instances where the template_match exceeded the given threshold value.

    """
    # if statement catching the optional threshold statement
    if threshold is None:
        threshold = 0.5 * np.max(template_match)
    
    above_threshold = template_match > threshold
    
    # Find where the signal crosses the threshold from below
    crossings = (above_threshold[1:] & (~above_threshold[:-1]))
    
    # The indices where crossings occur are the beat samples
    beat_samples = np.where(crossings)[0] + 1  # Add 1 because of the shift
    
    # return statement
    return beat_samples

# %% PART 6
def run_beat_detection(trial_mean, signal_voltage, threshold):
    """
    this function acts as parent function that runs all the previously defined functions.

    Parameters
    ----------
    trial_mean : 1D array of floats, size:(n,), where n is the mean of a beat event
        an array comprised of the mean signals of a heart beat.
    signal_voltage : 1D array of floats, size:(n,), where n is the voltage samples 
        contains voltages that make up the ecg signal
    threshold : float
        a threshold value to predict where a normal heartbeat would occur

    Returns
    -------
    beat_samples : 1D array of ints, size:(n,), where n is the values directly after the threshold is exceeded
        instances where the template_match exceeded the given threshold value.
    template_match : 1D array of floats, size:(n,), where n is a list of cross-correlated values
        the discrete cross-correlation from the previous section.

    """
    # Normalize the template
    template = normalize_template(trial_mean)
    
    # Cross-correlate the template with the signal voltage
    template_match = get_template_match(signal_voltage, template)
    
    # Predict beat times using the threshold
    beat_samples = predict_beat_times(template_match, threshold=threshold)
    
    # return statement
    return beat_samples, template_match
