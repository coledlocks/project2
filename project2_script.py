#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
project2_script.py

Created on Tue Oct 29 10:57:28 2024

@author: coledlocks
"""
# importing packages
import project2_module as p2m
import project1_module as p1m
import numpy as np
import matplotlib.pyplot as plt

#%% PART 1: Load data

ecg_voltage, fs, label_samples, label_symbols, subject_id, electrode, units = p1m.load_data('ecg_e0103_half2.npz')
symbols, trial_time, mean_trial_signal = p2m.load_means('ecg_means.npz')

#%% PART 2: Resampling data

# Compute sampling rates
sampling_rate_mean_trial = 1 / np.mean(np.diff(trial_time))
sampling_rate_ecg = fs

# Determine which signal has the higher sampling rate
if sampling_rate_mean_trial > sampling_rate_ecg:
    original_signal = mean_trial_signal
    original_time = trial_time

    # Decimate mean_trial_signal
    decimation_factor = int(sampling_rate_mean_trial / sampling_rate_ecg)
    decimated_signal = p2m.decimate_data(mean_trial_signal, decimation_factor)
    decimated_time = p2m.decimate_data(trial_time, decimation_factor)
    
else:
    # Decimate ecg_voltage
    decimation_factor = int(sampling_rate_ecg / sampling_rate_mean_trial)
    # Create time vector for ecg_voltage
    ecg_time = np.arange(len(ecg_voltage)) / fs
    original_signal = ecg_voltage
    original_time = ecg_time
    decimated_signal = p2m.decimate_data(ecg_voltage, decimation_factor)
    decimated_time = p2m.decimate_data(np.arange(len(ecg_voltage)) / fs, decimation_factor)
    # Adjust label_samples
    decimated_label_samples = label_samples / decimation_factor
    
# Plot the original and decimated signals for the selected lead
plt.figure(1)
plt.plot(original_time, original_signal, label='Original Signal')
plt.plot(decimated_time, decimated_signal, linestyle='--', label='Decimated Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (units)')
plt.title('Original and Decimated Signal Comparison')
plt.legend()
plt.xlim([0.5, 0.51])
plt.ylim([-0.1,0.1])

# %% 
normal_index = np.where(symbols == 'N')[0][0]

# Extract the mean signal for the normal beat
trial_mean = mean_trial_signal[normal_index]

# Normalize the template
template = p2m.normalize_template(trial_mean)
print("Units of the template: 1 / V")

# %%