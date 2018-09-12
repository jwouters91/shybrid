#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 10:16:20 2018

@author: jwouters
"""

import os
import csv

import numpy as np


class Recording:
    """ Recording in binary format. The raw recording is memory-mapped on
    initialization.

    Note: it is assumed that the given binary representation of the data
    has the different channels as rows. Depending on the software used to
    capture the data, the matrix is either stored in a binary file as row-major
    (e.g. python) or column-major (e.g. matlab). To take this difference into
    account the order argument should be carefully checked.

    Args:
        fn (str): path to the binary recording

        nb_channels (int): number of recording channels

        sampling_rate (int): sampling rate of the provided data 

        dtype (str): datatype of given binary recording

        mode (str): load mode (see numpy.memmap), 'r' is (default)

        order (str): 'C' (row-major) or 'F' (column-major) type of binary

    Attributes:
        data (ndarray): memory-mapped numpy array where every row is a channel

        sampling_rate (int): sampling rate of the data
    """

    def __init__(self, fn, nb_channels, sampling_rate, dtype,
                 mode='r', order='C'):
        # load actual data
        # don't reshape yet, 'cause we need to know the actual size first
        self.data = np.memmap(fn, dtype, mode)

        # check if nb_channels fits the data
        assert self.data.size / nb_channels == self.data.size // nb_channels,\
               "the given nb_channels does not match the given data"

        # reshape the data assuming
        self.data = self.data.reshape((nb_channels,
                                       self.data.size // nb_channels),
                                       order=order)
        self.sampling_rate = sampling_rate

    def isWritable(self):
        """ Return whether or not this Recording is writable 
        """
        return self.data.flags.writeable

    def flush(self):
        """ Flush changes for the case the memmap is read+write
        """
        self.data.flush()

    def unload(self):
        """ Unload the recording data
        """
        if self.isWritable():
            self.flush()

        # remove reference
        self.data = None


class Phy:
    """ Phy class that exposes spike sorting results and manual curation
    information

    Args:
        root (string): path to phy folder
    """

    _SPIKE_CLUSTERS  = 'spike_clusters.npy'
    _SPIKE_TIMES = 'spike_times.npy'
    _CLUSTER_GROUPS = 'cluster_groups.csv'

    def __init__(self, root):
        self.root = root

    def get_good_clusters(self):
        """ Return all clusters labeled good
        """
        with open(self._get_path_to(Phy._CLUSTER_GROUPS), 'r') as csvfile:
             csv_reader = csv.reader(csvfile, delimiter='\t')
             groups = []
             labels = []
             next(csv_reader) # skip first line
             for group, label in csv_reader:
                 groups.append(int(group))
                 labels.append(label)

        groups = np.array(groups)
        labels = np.array(labels)

        return groups[labels == 'good']

    def get_cluster_activation(self, cluster_id, start=0, end=None):
        """ Return an array with discrete activation times for the given
        cluster id

        Note: given start and end are expected in samples (not real time)
        """
        spike_times = self._get_spike_times()
        spike_cluster_mask = self._get_spike_cluster_mask(cluster_id)

        spike_times = spike_times[spike_cluster_mask]

        # apply bounds if any
        if start > 0:
            spike_times = spike_times[spike_times >= start]
        if end is not None and end > start:
            spike_times = spike_times[spike_times < end]

        # sort result (in-place)
        spike_times.sort()

        return spike_times

    """ Private methods section
    """
    def _get_path_to(self, fn):
        """ Return the full path for the given file name
        """
        return os.path.join(self.root, fn)

    def _get_spike_cluster_mask(self, cluster_id):
        """ Return the spike cluster mask for the given cluster_id
        """
        spike_clusters = np.load(self._get_path_to(Phy._SPIKE_CLUSTERS))
        # choice for boolean mask over indices for clarity
        return spike_clusters == cluster_id

    def _get_spike_times(self):
        """ Return all spike times
        """
        return np.load(self._get_path_to(Phy._SPIKE_TIMES))
