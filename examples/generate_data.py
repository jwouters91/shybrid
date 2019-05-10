#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 09:42:38 2018

@author: Jasper Wouters

Running this script generates toy example testing data that can be used with
the supplied graphical user interface. This script should be run from within
the examples folder.

"""

import os
import shutil

import yaml

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

# parameters
fs = 25000
duration = 0.5 # in seconds
spike_size = 50 # in samples
nb_channels = 30
nb_channels_spike = 10 # number of channels on which a spike is measured
data_dtype = 'float32'
nb_spikes = 5

# plot toy spike templates?
plot = False

# generate zero data matrix
#data = np.zeros((nb_channels, int(fs*duration)), dtype=data_dtype)
data = np.random.randn(nb_channels, int(fs*duration)) * 0.01
data = data.astype(data_dtype)

# construct 2 clusters with each 3 "spikes"
# CLUSTER 0
cluster_0 = np.ones((nb_spikes,2), dtype=np.int) * 0
cluster_0[0,1] = 3000
cluster_0[1,1] = 5000
cluster_0[2,1] = 7000
cluster_0[3,1] = 9000
cluster_0[4,1] = 12400

cluster_0_channel_template = np.linspace(-1, 0, spike_size)
cluster_0_channel_scaling = np.linspace(0.5, 1.5, nb_channels_spike)
cluster_0_spike_scaling = np.array([0.88, 1.05, 0.9, 1.01, 1])

cluster_0_template = np.dot(cluster_0_channel_scaling[:,None],
                            cluster_0_channel_template[None,:])

if plot:
    plt.figure()
    plt.plot(cluster_0_template.T + np.linspace(0, -1, 10))
    plt.title('template cluster 0')

# CLUSTER 1
cluster_1 = np.ones((nb_spikes,2), dtype=np.int) * 1
cluster_1[0,1] = 4000
cluster_1[1,1] = 6000
cluster_1[2,1] = 8000
cluster_1[3,1] = 9010
cluster_1[4,1] = 10000

cluster_1_channel_template = np.linspace(0, -1, spike_size)
cluster_1_channel_scaling = np.linspace(1.5, 0.5, nb_channels_spike)
cluster_1_spike_scaling = np.array([1.3, 0.73, 1.12, 0.8, 1])

cluster_1_template = np.dot(cluster_1_channel_scaling[:,None],
                            cluster_1_channel_template[None,:])

if plot:
    plt.figure()
    plt.plot(cluster_1_template.T + np.linspace(0, 1, 10))
    plt.title('template cluster 1')

# Insert "spikes" for every cluster
for idx in range(nb_spikes):
    # cluster 0
    data[:10,cluster_0[idx,1]:(cluster_0[idx,1]+spike_size)] = data[:10,cluster_0[idx,1]:(cluster_0[idx,1]+spike_size)] + cluster_0_template * cluster_0_spike_scaling[idx]
    data[-10:,cluster_1[idx,1]:(cluster_1[idx,1]+spike_size)] = data[-10:,cluster_1[idx,1]:(cluster_1[idx,1]+spike_size)] + cluster_1_template * cluster_1_spike_scaling[idx]

# export data
data_fn = "simulated_data.bin"
directory = os.path.expanduser('~/hybrid_test')

try:
    os.makedirs(directory)
except FileExistsError:
    pass
except Exception as e:
    raise e

data_full_fn = os.path.join(directory, data_fn)
data.tofile(data_full_fn)

# generate cluster csv file
clusters = np.concatenate((cluster_0, cluster_1), axis=0)

clusters_fn = "simulated_clusters.csv"
clusters_full_fn = os.path.join(directory, clusters_fn)
np.savetxt(clusters_full_fn, clusters, delimiter=',', fmt='%i')

# copy probe file
prb_fn = 'example_probe.prb'
prb_full_fn = os.path.join(directory, prb_fn)
shutil.copy(prb_fn, prb_full_fn)

# generate parameter yaml file
config = {}
config['data'] = {}
config['clusters'] = {}

config['data']['probe'] = prb_full_fn
config['data']['fs'] = fs
config['data']['dtype'] = data_dtype
config['data']['order'] = 'C'           
config['clusters']['csv'] = clusters_full_fn

config_fn, _ = os.path.splitext(data_fn)
config_full_fn = os.path.join(directory, config_fn)
config_full_fn += '.yml'

with open(config_full_fn, 'w') as outfile:
    yaml.dump(config, outfile, default_flow_style=False)

print('# Following files have been generated:\n- {}\n- {}\n- {}\n- {}'.\
          format(data_full_fn, clusters_full_fn,
                 prb_full_fn, config_full_fn))
