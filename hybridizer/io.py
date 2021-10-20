#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHYBRID
Copyright (C) 2018  Jasper Wouters

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import os
import csv

import numpy as np

from hybridizer.spikes import SpikeTrain
from hybridizer.hybrid import HybridCluster
from hybridizer.probes import Probe

import yaml

class Recording:
    """ Recording in binary format. The raw recording is memory-mapped on
    initialization.

    Note: it is assumed that the given binary representation of the data
    has the different channels as rows. Depending on the software used to
    capture the data, the matrix is either stored in a binary file as row-major
    (e.g. python) or column-major (e.g. matlab). To take this difference into
    account the order argument should be carefully checked.

    Note: for performance reasons it is assumed that the recording has been
    preprocessed accordingly. This package does not provide support for 
    preprocessing for now.

    Parameters
    ----------
        fn (str): path to the binary recording

        probe_fn (string): path to probe file

        sampling_rate (int): sampling rate of the provided data 

        dtype (str): datatype of given binary recording

        mode (str): load mode (see numpy.memmap), 'c' is (default)

        order (str): 'C' (row-major) or 'F' (column-major) type of binary
    """
    # only signed datatypes are accepted for now, this is also the most natural
    # representation for extracellular recordings.
    supported_dtypes = ('float32', 'float64', 'double', 'int16', 'int32')

    def __init__(self, fn, probe_fn, sampling_rate, dtype,
                 mode='r+', order='C'):
        # check if the given datatype is supported
        if dtype not in self.supported_dtypes:
            raise TypeError('The given data type ({}) is not supported. '
                            'Only the following signed types are supported: {}'
                                .format(dtype,
                                        self.supported_dtypes))

        # keep track of parameter for dump
        self._fn = fn
        self._dtype = dtype

        # load the probe file
        self.probe = Probe(probe_fn)

        # load actual data
        # don't reshape yet, 'cause we need to know the actual size first
        self.data = np.memmap(fn, dtype, mode)

        # NOTE: we don't remove the bad channels here for performance reasons,
        # selecting the data would load the entire array into memory
        nb_channels = self.probe.total_nb_channels
        # check if nb_channels fits the data
        assert self.data.size / nb_channels == self.data.size // nb_channels,\
               "the given nb_channels does not match the given data"

        # reshape the data assuming
        self.data = self.data.reshape((nb_channels,
                                       self.data.size // nb_channels),
                                       order=order)

        self.sampling_rate = sampling_rate

    def get_nb_good_channels(self):
        """ Return the number of working channels
        """
        return self.probe.channels.size

    def get_duration(self):
        """ Return the number of samples recorded per channel
        """
        return self.data.shape[1]

    def is_writable(self):
        """ Return whether or not this Recording is writable 
        """
        return self.data.flags.writeable

    def flush(self):
        """ Flush changes for the case the memmap is read+write
        """
        self.data.flush()

    def save_raw(self, full_fn):
        """  Save the data in raw format in the original data folder
        """
        # save transpose such that file is compatible with spyking circus
        self.data.astype(self._dtype).T.tofile(full_fn)

    def save_npy(self):
        """ Save the data in npy format in the original data folder
        """
        fn, _ = os.path.splitext(self._fn)
        np.save(fn, self.data)

    def get_good_chunk(self, start, end):
        """ Return a chunk of data with only good channels
        """
        return self.data[self.probe.channels,start:end]

    def count_spikes(self, C=5, dur=30):
        """ Count the number of spikes on every channel

        Parameters
        ----------
          C (float) : Factor multiplied with channel standard deviation for
          determining the spike detection threshold

          dur (float) : Duration in seconds over which to calculate the
          spike detection rate
        """
        # calculate a simple spike threshold based on the standard deviation
        channel_thresholds = np.std(self.data[self.probe.channels, :int(dur*self.sampling_rate)], axis=1) * C

        # detect negative peaks in the signal, resulting in a binary signal          
        detections = self.data[self.probe.channels, :int(dur*self.sampling_rate)] < (-1 * channel_thresholds[:,np.newaxis])
        detections = detections.astype(np.int8)

        # extract the number of threshold crossings (per channel)
        # by detecting rising edges
        detections = np.diff(detections)
        detections = np.count_nonzero(detections, axis=1)
        detections = detections / 2 # divide by two because both the rising and falling edge are counted otherwise

        # convert to spike rates
        detections = detections / dur

        return detections

    def get_signal_power(self, dur=30):
        """ Return the signal power
        """
        return self.data[self.probe.channels, :int(dur*self.sampling_rate)].var()

    def get_signal_power_for_channel(self, channel, dur=30):
        """ Return the signal power
        """
        # inspired by MAD assuming zero-median signal and then correcting to std (x1.5), all of this assumes normally distributed data
        return (np.median(np.abs(self.data[self.probe.channels, :int(dur*self.sampling_rate)][channel]))*1.5)**2

class SpikeClusters:
    """ Class modeling a collection of spike clusters
    """
    def __init__(self):
        """ This class wraps a dictionary
        """
        self.__mem__ = {}

    def __getitem__(self, arg):
        """ implement square brackets syntax
        """
        return self.__mem__[arg]

    def __setitem__(self, key, value):
        """ implement square brackets syntax
        """
        self.__mem__[key] = value

    def keys(self):
        """ Return the keys ordered ascending
        """
        return np.sort(list(self.__mem__.keys()))

    def fromCSV(self, fn, recording, delimiter=','):
        """ First column of CSV must contain integer cluster indices, second
        column of CSV must contain integer spike times.
        """
        cluster_info = np.loadtxt(fn, delimiter=delimiter, dtype=np.int)

        for cluster in set(cluster_info[:,0]):
            cluster_mask = np.where(cluster_info[:,0] == cluster)[0]

            cluster_spike_times = cluster_info[cluster_mask,1]
            cluster_spike_times.sort()

            spike_train = SpikeTrain(recording, cluster_spike_times,
                                     template_jitter=np.zeros(cluster_spike_times.size))

            self[cluster] = HybridCluster(cluster, spike_train)

    def dumpCSV(self, fn, delimiter=','):
        """ Dump the spike clusters in a CSV-file
        """
        dump = np.empty((0,2))

        # loop over different moved clusters
        for gt_idx, cluster_idx in enumerate(self.keys()):
            cluster = self[cluster_idx]

            if cluster.is_hybrid():
                gt_spikes = cluster.get_actual_spike_train().spikes
                tmp_dump = np.ones((gt_spikes.size,2))

                # renumber clusters for dump
                tmp_dump[:,0] *= gt_idx
                tmp_dump[:,1] = gt_spikes
                dump = np.concatenate((dump, tmp_dump), axis=0)

        # dump
        np.savetxt(fn, dump, delimiter=delimiter, fmt='%i')

    def fromPhy(self, phy, recording):
        """ Load cluster information using a phy object
        """
        for cluster in phy.get_good_clusters():
            cluster_spike_times = phy.get_cluster_activation(cluster)

            spike_train = SpikeTrain(recording, cluster_spike_times,
                                     template_jitter=np.zeros(cluster_spike_times.size))

            self[cluster] = HybridCluster(cluster, spike_train)

    def forget_recording(self):
        """ Forget about current recording in every cluster
        """
        for cluster_idx in self.keys():
            self[cluster_idx].forget_recording()

    def add_recording(self, recording):
        """ Add recording back to all clusters
        """
        for cluster_idx in self.keys():
            self[cluster_idx].add_recording(recording)

    def add_empty_cluster(self, idx):
        """ Add an empty cluster to the collection of clusters
        """
        self[idx] = HybridCluster(idx, None)

    def remove_cluster(self, idx):
        """ Remove a cluster from the the collection of clusters
        """
        del self.__mem__[idx]


class Phy:
    """ Phy class that exposes spike sorting results and manual curation
    information

    Parameters
    ----------
        root (string): path to phy folder
    """

    _SPIKE_CLUSTERS  = 'spike_clusters.npy'
    _SPIKE_TEMPLATES = 'spike_templates.npy'
    _SPIKE_TIMES = 'spike_times.npy'
    _CLUSTER_GROUPS = 'cluster_groups.csv'
    _CLUSTER_GROUPS_V2 = 'cluster_group.tsv'

    def __init__(self, root):
        self.root = root

    def get_good_clusters(self):
        """ Return all clusters labeled good
        """
        V1_exists = os.path.isfile(self._get_path_to(Phy._CLUSTER_GROUPS))
        V2_exists = os.path.isfile(self._get_path_to(Phy._CLUSTER_GROUPS_V2))

        if V1_exists:
            cluster_groups_fn = self._get_path_to(Phy._CLUSTER_GROUPS)
        elif V2_exists:
            cluster_groups_fn = self._get_path_to(Phy._CLUSTER_GROUPS_V2)
        else:
            raise FileNotFoundError("Phy cluster groups could not be found")

        with open(cluster_groups_fn, 'r') as csvfile:
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

    def get_cluster_and_times(self, curated=True):
        """
        Returns
        -------
        cluster_spikes (ndarray) : (nspikes, 2) numpy array containing in the
        first column the cluster and in the second column the time of all
        spikes
        """
        # spike_clusters only exists after curation
        if curated:
            spike_clusters = np.load(self._get_path_to(Phy._SPIKE_CLUSTERS)).flatten()
        else:
            spike_clusters = np.load(self._get_path_to(Phy._SPIKE_TEMPLATES)).flatten()

        spike_times = np.load(self._get_path_to(Phy._SPIKE_TIMES)).flatten()

        return np.concatenate((spike_clusters[:,np.newaxis],
                               spike_times[:,np.newaxis]),
                              axis=1)

    """ Private methods section
    """
    def _get_path_to(self, fn):
        """ Return the full path for the given file name
        """
        return os.path.join(self.root, fn)

    def _get_spike_cluster_mask(self, cluster_id):
        """ Return the spike cluster mask for the given cluster_id
        """
        spike_clusters = np.load(self._get_path_to(Phy._SPIKE_CLUSTERS)).flatten()
        # choice for boolean mask over indices for clarity
        return spike_clusters == cluster_id

    def _get_spike_times(self):
        """ Return all spike times
        """
        return np.load(self._get_path_to(Phy._SPIKE_TIMES)).flatten()

def get_params(binary_fn):
    """ Return yaml parameters given the binary recording full file name
    """
    base_fn, _ = os.path.splitext(binary_fn)
    params_fn = base_fn + '.yml'

    if os.path.isfile(params_fn):
        with open(params_fn, 'r') as f:
            params = yaml.safe_load(f)
    else:
        raise FileNotFoundError('the parameters file "{}" could not be found'.format(params_fn))

    return params
