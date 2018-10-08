#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 17:19:06 2018

@author: jwouters
"""
import numpy as np
import matplotlib.pyplot as plt

from hybridizer.io import Recording, Phy
from hybridizer.spikes import SpikeTrain

plt.close('all')

# files and folders
rec_fn = '/media/jwouters/DATA/UMC_UTRECHT/thesis_Julien/c2/data1.raw'
prb_fn = '/home/jwouters/spyking-circus/probes/hd_semg_126_c2.prb'
phy_path = '/media/jwouters/DATA/UMC_UTRECHT/thesis_Julien/c2/data1/data1.GUI'

fs = 4096
dtype = 'float32'
order = 'F' # data was generated in matlab
save_output = True

# create objects
recording = Recording(rec_fn, prb_fn, fs, dtype, order=order)
sorting_info = Phy(phy_path)

# find good cluster
good_cluster = sorting_info.get_good_clusters()[0] # pick the first one
spike_times = sorting_info.get_cluster_activation(good_cluster)

# the spikeTrain object ties together the recording object with the activation
spikeTrain = SpikeTrain(recording,spike_times)

# calculate template and show
spikeTrain.calculate_template(realign=True)

plt.figure()
plt.plot(spikeTrain.template.data.T)

plt.figure()
plt.plot(spikeTrain.template.PC.T)

spikeTrain.subtract_train(plot=False)

if save_output: recording.save_raw(suffix="subtracted")

spikeTrain.insert_train(spatial_map="reverse")

if save_output: recording.save_raw(suffix="hybrid")

# visualize data: compare original to subtracted to hybrid
