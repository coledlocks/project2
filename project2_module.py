#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
project2_module.py
Created on Tue Oct 29 10:52:25 2024

@author: coledlocks
"""
# importing packages
import numpy as np

#%% PART 1: load data

# defining a function to read the npz file
def load_means(input_file):
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
    
    return symbols, trial_time, mean_trial_signal # returns symbols, trial_time, mean_trial_signal


#%% PART 2: resampling data

def decimate_data(original_signal, decimation_factor):
    # Extract every nth sample from the original signal where n is the decimation factor
    decimated_signal = original_signal[::decimation_factor]
    return decimated_signal
    
# %% PART 3: normalizing signal

def normalize_template(trial_mean):
    # calculating the demeaned signal
    demeaned_signal = trial_mean - np.mean(trial_mean)
    # calculating energy of the demeaned signal
    energy = np.sum(demeaned_signal ** 2)
    template = demeaned_signal / energy
    
    return template

#%% PART 4: match data to template

# def get_template_match(signal_voltage, template):
#     # folding the template
#     folded_template = np.array([])
#     for fold_index in range(len(template)):
#         folded_template = np.append(folded_template, -1*(template[fold_index]))
#     # convolving the folded signal with the voltage to get the correlation
#     template_match = np.convolve(signal_voltage, folded_template, mode='same')
#     return template_match

#%% PART 4: template match
def get_template_match(signal_voltage, template):
    # Reverse the template for cross-correlation
    reversed_template = template[::-1]
    
    # Compute the cross-correlation using convolution
    template_match = np.convolve(signal_voltage, reversed_template, mode='same')
    
    return template_match   
# %% PART 5
def predict_beat_times(template_match, threshold=None):
    if threshold is None:
        threshold = 0.5 * np.max(template_match)
    
    above_threshold = template_match > threshold
    
    # Find where the signal crosses the threshold from below
    crossings = (above_threshold[1:] & (~above_threshold[:-1]))
    
    # The indices where crossings occur are the beat samples
    beat_samples = np.where(crossings)[0] + 1  # Add 1 because of the shift
    
    return beat_samples, threshold
# %% PART 6
def run_beat_detection(trial_mean, signal_voltage, threshold):
    # Normalize the template
    template = normalize_template(trial_mean)
    
    # Cross-correlate the template with the signal voltage
    template_match = get_template_match(signal_voltage, template)
    
    # Predict beat times using the threshold
    beat_samples, threshold_used = predict_beat_times(template_match, threshold=threshold)
    
    return beat_samples, template_match