#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 13:52:29 2018

@author: jwouters
"""

import numpy as np
import scipy.sparse.linalg as la

class SpikeTrain:
    """ Spike train class grouping spikes and recording

    Args:
        recording (Recording): recording object related to this spike train

        spike_times (ndarray): array containing starting times of every spike
    """

    def __init__(self, recording, spike_times):
        self.recording = recording
        self.spikes = spike_times

    def calculate_template(self, window_size=100):
        """ Calculate a template for this spike train for the given discrete
        window size
        """
        self.template = Template(self, window_size)

    def get_nb_spikes(self):
        """ Return the number of spikes in the spike train
        """
        return self.spikes.size


class Template:
    """ Template class

    Args:
        spike_train (SpikeTrain): spike train

        window_size (int): window size used to determine te template

        realign (boolean, optional): realign spikes based on template matched
        filter
    """

    def __init__(self, spike_train, window_size, realign=False):
        self.window_size = window_size

        # build spike tensor
        spike_tensor = np.empty((spike_train.get_nb_spikes(),
                                 spike_train.recording.get_nb_good_channels(),
                                 self.window_size))

        # working channels
        channels = spike_train.recording.probe.channels

        for spike_idx in range(spike_train.get_nb_spikes()):
            start = int(spike_train.spikes[spike_idx] - window_size/2)
            end = int(start + self.window_size)
            spike_tensor[spike_idx] = spike_train.recording.data[channels,
                                                                 start:end]

        self.data = np.median(spike_tensor, axis=0)

        # calculate first PC for fitting
        spike_matrix = spike_tensor.reshape((spike_tensor.shape[0],
                                             spike_tensor.shape[1]*
                                             spike_tensor.shape[2]))

        spike_cov = np.cov(spike_matrix.T)
        _, PCs = la.eigs(spike_cov, k=1) # makes use of spare linalg

        self.PC = PCs[:,0].reshape((spike_tensor.shape[1],
                                    spike_tensor.shape[2]))
        # TODO implement realignment if deemed necessary
