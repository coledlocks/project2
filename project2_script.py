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

#%% PART 1: load data

ecg_voltage, fs, label_samples, label_symbols, subject_id, electrode, units = p1m.load_data('ecg_e0103_half2.npz')
symbols, trial_time, mean_trail_signal = p2m.load_means('ecg_means.npz')

