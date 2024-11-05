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
    
# %% PART 3: normalize signal

def normalize_template(trial_mean):
    demeaned_signal = trial_mean - np.mean(trial_mean)
    energy = np.sum(demeaned_signal ** 2)
    template = demeaned_signal / energy
    
    return template










