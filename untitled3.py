# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 20:55:43 2024

@author: Matthew
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
    
# plotting the original and decimated signals for the selected lead
plt.figure(1, clear=True)
plt.plot(original_time, original_signal, label='Original Signal')
plt.plot(decimated_time, decimated_signal, linestyle='--', label='Decimated Signal')

# annotating figure
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (a.u.)')
plt.title('Original and Decimated Signal Comparison')
plt.legend()
plt.xlim([0.5, 0.51])
plt.ylim([-0.1,0.1])
plt.savefig(f'original_and_decimated_signal_comparison_subject_{subject_id}.png')

# %% PART 3
normal_index = np.where(symbols == 'V')[0][0]

# Extract the mean signal for the normal beat
trial_mean = mean_trial_signal[normal_index]

# Normalize the template
template = p2m.normalize_template(trial_mean)
print("Units of the template: 1 / V")

#%% PART 4: Template Matching

# Call the get_template_match function
template_match = p2m.get_template_match(ecg_voltage, template)

# Create time vector for ecg_voltage
ecg_time = np.arange(len(ecg_voltage)) / fs

# Plotting the cross-correlation
plt.figure(2, clear=True)
plt.plot(ecg_time, template_match)
plt.xlim(200,201)
plt.xlabel('Time (s)')
plt.ylabel('Cross-correlation (dimensionless)')
plt.title('Template Matching using Cross-Correlation')
plt.grid(True)
plt.savefig(f'cross_correlation_raw_and_template_data_subject_{subject_id}.png')
#%% PART 5: Detect Beats

# Use the default threshold
beat_samples, threshold = p2m.predict_beat_times(template_match)

# Convert beat samples to times
beat_times = beat_samples / fs

plt.figure(3, clear=True)
plt.plot(ecg_time, template_match, label='Cross-correlation')
plt.plot(beat_times, template_match[beat_samples], 'v', label='Detected Beats')
plt.xlabel('Time (s)')
plt.ylabel('Cross-correlation (dimensionless)')
plt.title('Template Matching with Detected Beats')
plt.legend()
plt.xlim(10, 12)  
plt.savefig(f'detected_normal_beats_subject_{subject_id}.png')
#%% PART 6: Detect Arrhythmic Beats
# Extract the mean signal for the normal beat
normal_trial_mean = mean_trial_signal[normal_index]

# Set threshold for normal beats
normal_threshold = 0.5 * np.max(ecg_voltage)

# Run beat detection for normal beats
normal_beat_samples, normal_template_match = p2m.run_beat_detection(normal_trial_mean, ecg_voltage, normal_threshold)
normal_beat_times = normal_beat_samples / fs

# Find the index of the arrhythmic beat in the symbols array
arrhythmia_index = np.where(symbols == 'N')[0][0]
arrhythmia_trial_mean = mean_trial_signal[arrhythmia_index]

# Set threshold for arrhythmic beats
arrhythmia_threshold = 0.5 * np.max(ecg_voltage)  # Adjust as needed

# Run beat detection for arrhythmic beats
arrhythmia_beat_samples, arrhythmia_template_match = p2m.run_beat_detection(arrhythmia_trial_mean, ecg_voltage, arrhythmia_threshold)
arrhythmia_beat_times = arrhythmia_beat_samples / fs

# plot
plt.figure()
# ecg signal
plt.plot(ecg_time, ecg_voltage, label='ECG Signal', color='blue', alpha=0.6)

# normal template match
plt.plot(ecg_time, normal_template_match, label='Normal Template Match', color='green', alpha=0.7)

# arrhythmic template match
plt.plot(ecg_time, arrhythmia_template_match, label='Arrhythmic Template Match', color='red', alpha=0.7)

# detected normal beats
plt.plot(normal_beat_times, normal_template_match[normal_beat_samples], 'v', color='green', markersize=8, label='Detected Normal Beats')

# detected arrhythmic beats
plt.plot(arrhythmia_beat_times, arrhythmia_template_match[arrhythmia_beat_samples], '^', color='red', markersize=8, label='Detected Arrhythmic Beats')

# Plot human-annotated events (from Project 1)
label_times = label_samples / fs
unique_symbols = np.unique(label_symbols)

# Use different markers for annotations
annotation_markers = {'N': 'o', 'V': 'o', 'S': '_', 'F': '+'} 

for symbol in unique_symbols:
    symbol_indices = np.where(label_symbols == symbol)[0]
    symbol_times = label_times[symbol_indices]
    marker = annotation_markers.get(symbol)
    plt.plot(symbol_times, np.zeros_like(symbol_times), marker, markersize=10, label=f'Annotation: {symbol}')

# Labels and Legend
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (V) / Cross-correlation')
plt.title(f'ECG Signal with Detected Beats and Annotations for Subject {subject_id}')
plt.legend(loc='upper right')
plt.grid(True)
plt.xlim(2919.5, 2923.5)

# Save the figure with an informative name containing the subject ID
plt.savefig(f'beat_detection_results_subject_{subject_id}.png')