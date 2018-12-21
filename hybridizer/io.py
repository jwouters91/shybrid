#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 10:16:20 2018

@author: jwouters
"""

import os
import csv

import numpy as np

import matplotlib.pyplot as plt

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

    Args:
        fn (str): path to the binary recording

        probe_fn (string): path to probe file

        sampling_rate (int): sampling rate of the provided data 

        dtype (str): datatype of given binary recording

        mode (str): load mode (see numpy.memmap), 'c' is (default)

        order (str): 'C' (row-major) or 'F' (column-major) type of binary

    Attributes:
        data (ndarray): memory-mapped numpy array where every row is a channel

        sampling_rate (int): sampling rate of the data
    """

    def __init__(self, fn, probe_fn, sampling_rate, dtype,
                 mode='c', order='C'):
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
        self.data.tofile(full_fn)

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

        Args:
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


class SpikeClusters:
    """ Class modeling a collection of spike clusters
    """
    def __init__(self):
        """ This class wraps a dictionary
        """
        self.__mem__ = {}

    def __getitem__(self, arg):
        return self.__mem__[arg]

    def __setitem__(self, key, value):
        self.__mem__[key] = value

    def keys(self):
        """ Return the keys ordered ascending
        """
        return np.sort(list(self.__mem__.keys()))

    def fromCSV(self, fn, delimiter=','):
        """ First column of CSV must contain integer cluster indices, second
        column of CSV must contain integer spike times.
        """
        cluster_info = np.loadtxt(fn, delimiter=delimiter, dtype=np.int)

        for cluster in set(cluster_info[:,0]):
            cluster_mask = np.where(cluster_info[:,0] == cluster)[0]

            cluster_spike_times = cluster_info[cluster_mask,1]
            cluster_spike_times.sort()

            self[cluster] = cluster_spike_times

    def dumpCSV(self, fn, delimiter=','):
        """ Dump the spike clusters in a CSV-file
        """
        dump = np.empty((0,2))

        # loop over different moved clusters
        for idx, cluster in enumerate(self.keys()):
            gt_tmp = self[cluster]
            tmp_dump = np.ones((gt_tmp.size,2))

            # renumber clusters for dump
            tmp_dump[:,0] *= idx
            tmp_dump[:,1] = gt_tmp
            dump = np.concatenate((dump, tmp_dump), axis=0)

        # dump
        np.savetxt(fn, dump, delimiter=delimiter, fmt='%i')

    def fromPhy(self, phy):
        """ Load cluster information using a phy object
        """
        for cluster in phy.get_good_clusters():
            self[cluster] = phy.get_cluster_activation(cluster)

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


class Probe:
    """ Class exposing the probe file. Supporting only single-shank 
    with shank id 1.

    Args:
        probe_fn (string): full path and filename to the probe file

    Attributes:
        channels (ndarray): array containing the good channels

        total_nb_channels (int): total number of channels on the probe
    """

    def __init__(self, probe_fn):
        variables = {}
        # execute the probe file
        exec(open(probe_fn).read(), variables)

        # extract channels from probe
        self.channels = variables['channel_groups'][1]['channels']
        self.channels = np.array(self.channels)

        # extract total number of channels
        self.total_nb_channels = variables['total_nb_channels']

        # extract geometry
        self.geometry = variables['channel_groups'][1]['geometry']

        # assuming rectangular probes with equal x spacing and equal y spacing
        self.x_between = self.get_x_between()
        self.y_between = self.get_y_between()

    def get_min_geometry(self):
        """ Return the minimum geometry value for each dimension in a single
        ndarray
        """
        return np.array(list(self.geometry.values())).min(axis=0)

    def get_max_geometry(self):
        """ Return the minimum geometry value for each dimension in a single
        ndarray
        """
        return np.array(list(self.geometry.values())).max(axis=0)

    def get_x_between(self):
        """ Return the electrode pitch in the x direction
        """
        X = 0

        x_locs = np.array(list(self.geometry.values()))[:,X]

        # TODO refactor
        self.x_max = x_locs.max()
        self.x_min = x_locs.min()

        # init at the maximum possible difference (possibly zero)
        x_between = self.get_max_geometry()[X] - self.get_min_geometry()[X]

        # choose x between as the shortest non-zero difference
        for x_tmp_1 in x_locs:
            for x_tmp_2 in x_locs:
                x_diff = abs(x_tmp_1 - x_tmp_2)
                if x_diff > 0 and x_diff < x_between:
                    x_between = x_diff

        if x_between == 0:
            x_between = 1

        return x_between

    def get_y_between(self):
        """ Return the electrode pitch in the y direction
        """
        Y = 1

        y_locs = np.array(list(self.geometry.values()))[:,Y]

        # TODO refactor
        self.y_max = y_locs.max()
        self.y_min = y_locs.min()

        # init at the maximum possible difference (possibly zero)
        y_between = self.get_max_geometry()[Y] - self.get_min_geometry()[Y]

        # choose x between as the shortest non-zero difference
        for y_tmp_1 in y_locs:
            for y_tmp_2 in y_locs:
                y_diff = abs(y_tmp_1 - y_tmp_2)
                if y_diff > 0 and y_diff < y_between:
                    y_between = y_diff

        if y_between == 0:
            y_between = 1

        return y_between


class RectangularProbe:
    """ A model for probes that have a rectangular electrode grid. Automatic
    probe graph generation supported.

    Attributes:
        channels (ndarray) : array containing the channels

        origin (Channel) : the origin channel of the probe
    """

    def __init__(self):
        self.channels = np.array([], dtype=Channel)
        self.origin = None

    def fill_channels(self, channel_geometries):
        """ Fill the channels attribute with channel objects generated from
        the given channels geometries dictionary

        Args:
            channel_geometries (dict) : dictionary containing the coordinates
            for every channel
        """
        # extract channels
        channels = channel_geometries.keys()

        # loop over the channels
        for channel in channels:
            x, y = channel_geometries[channel]
            channel_obj = Channel(x, y, channel=channel)
            self.channels = np.append(self.channels, channel_obj)

    def connect_channels(self):
        """ Calculate the channel graph from its channels and add broken/
        missing channels to the channels list
        """
        # extract the in-between-channels geometry assuming a rectangular grid
        self._calculate_interchannel_stats()

        # add broken / missing channels
        x = self.x_min
        y = self.y_min

        more_x = True
        more_y = True

        while more_y:
            while more_x:
                # Find channel with x,y coordinate
                tmp_channel = self.get_channel_from_xy(x, y)

                # Insert stub channel if missing
                if tmp_channel is None:
                    broken_channel = Channel(x, y)
                    broken_channel.set_broken(True)

                    self.channels = np.append(self.channels, broken_channel)

                # update x
                x = x + self.x_between
                if x > self.x_max:
                    x = self.x_min
                    # exit while loop
                    more_x = False

            # update y
            y = y + self.y_between
            # reenable inner loop
            more_x = True
            if y > self.y_max:
                # exit while loop
                more_y = False

        # for every channel start looking for its neighbors
        for channel in self.channels:
            for candidate in self.channels:
                rel = self._is_neighbor(channel, candidate)

                if rel == 'left':
                    channel.left = candidate
                elif rel == 'upper':
                    channel.upper = candidate
                elif rel == 'right':
                    channel.right = candidate
                elif rel == 'lower':
                    channel.lower = candidate

        # set the origin channel
        self.origin = self.get_channel_from_xy(self.x_min, self.y_min)

    def _calculate_interchannel_stats(self):
        """ Calculate the interchannel statistics
        """        
        self.x_between = -1
        self.y_between = -1

        self.x_min = self.channels[0].x
        self.x_max = self.channels[0].x
        self.y_min = self.channels[0].y
        self.y_max = self.channels[0].y

        for channel_i in self.channels:
            # extract min/max
            if channel_i.x < self.x_min:
                self.x_min = channel_i.x

            if channel_i.x > self.x_max:
                self.x_max = channel_i.x

            if channel_i.y < self.y_min:
                self.y_min = channel_i.y

            if channel_i.y > self.y_max:
                self.y_max = channel_i.y

            # double loop over channels for comparison
            for channel_j in self.channels:
                # compare interchannel distance for the current pair
                tmp_x_between = abs(channel_i.x - channel_j.x)
                tmp_y_between = abs(channel_i.y - channel_j.y)

                # assign if smaller but non-zero or if the initial is detected
                # and non-zero
                if tmp_x_between < self.x_between or self.x_between < 0:
                    if tmp_x_between > 0:
                        self.x_between = tmp_x_between

                if tmp_y_between < self.y_between or self.y_between < 0:
                    if tmp_y_between > 0:
                        self.y_between = tmp_y_between

        self.nb_cols = int((self.x_max - self.x_min) / self.x_between + 1)
        self.nb_rows = int((self.y_max - self.y_min) / self.y_between + 1)

    def _is_neighbor(self, channel, candidate):
        """ Check whether the candidate is a neighbor of the given channel
        and return the relation or None
        """
        x_diff = channel.x - candidate.x
        y_diff = channel.y - candidate.y

        if abs(x_diff) == self.x_between and y_diff == 0:
            if x_diff > 0:
                return 'left'
            else:
                return 'right'

        if abs(y_diff) == self.y_between and x_diff == 0:
            if y_diff > 0:
                return 'lower'
            else:
                return 'upper'

        # no neighbor == return None
        return None

    def validate_probe_graph(self):
        """ If the graph is valid it should be possible to back reference from
        every neighbor
        """
        error_found = False

        # loop over all channels and try to reference back from a neighbor
        for channel in self.channels:
            if not channel.left is None:
                if not channel.left.right is channel:
                    print('broken graph left {} {}'.format(channel.x, channel.y))
                    error_found = True

            if not channel.right is None:
                if not channel.right.left is channel:
                    print('broken graph right {} {}'.format(channel.x, channel.y))
                    error_found = True

            if not channel.upper is None:
                if not channel.upper.lower is channel:
                    print('broken graph upper {} {}'.format(channel.x, channel.y))
                    error_found = True

            if not channel.lower is None:
                if not channel.lower.upper is channel:
                    print('broken graph lower {} {}'.format(channel.x, channel.y))
                    error_found = True

        if not error_found:
            print('-- probe graph is found to be healthy')

    def get_channel_from_xy(self, x, y):
        """ Return the channel on the probe with the given x, y coordinate,
        or None if no channel was found.
        """
        for channel in self.channels:
            if channel.x == x and channel.y == y:
                return channel

        # if no match is found return None
        return None

    def sort_given_channel_idx(self, channel_idxs):
        """ Sort the given channel idxs from left to right (fastest changing)
        and bottom to top
        """
        output_idxs = np.array([]) # initialise return variable
        traversing = True # true as long as the probe has not been traversed

        # start traversing from the origin of the probe
        current_channel = self.origin
        next_row = self.origin.upper

        # traverse the probe
        while traversing:
            if not current_channel.is_broken():
                # extract the index of the current channel
                channel_idx = current_channel.channel
                if channel_idx in channel_idxs:
                    output_idxs = np.append(output_idxs, current_channel.channel)

            # update current channel
            current_channel = current_channel.right
            
            # if the next channel is None go to the next row
            if current_channel is None:
                current_channel = next_row

                # if this next row turns out to be None we return the result
                if current_channel is None:
                    # the traversing doesn't have to changed explicitly because
                    # the return statement breaks the while loop
                    return output_idxs

                # update next row
                next_row = current_channel.upper

    def get_channels_from_zone(self, nb_channels, x_reach, x_offset):
        """ Return a list channels idxs generated from the given geometrical
        zone. A zone is always created from the origin.

        Broken channels are represented by Nones
        """
        output_idxs = np.array([], dtype=np.int)

        if x_offset >= x_reach:
            raise Exception('Given offset has to smaller than the given reach')

        if x_reach > self.nb_cols:
            raise Exception('Given reach cannot exceed the probe dimensions')

        # go to start
        current_channel = self.origin
        next_row = current_channel.upper

        # apply offset
        if x_offset > 0:
            # walk to right
            for _ in range(x_offset):
                    current_channel = current_channel.right

        # walk right to left in this row
        walk_right = x_reach - x_offset

        while not output_idxs.size == nb_channels:
            while walk_right:
                output_idxs = np.append(output_idxs, current_channel.channel)

                current_channel = current_channel.right
                walk_right -= 1

            # update to next_row
            current_channel = next_row

            if not current_channel is None:
                next_row = current_channel.upper
                walk_right = x_reach

            # no more channels left
            else:
                # complete with None till desired size is reach
                output_idxs = np.append(output_idxs, None)

        # return the raw data channels for this zone
        return output_idxs


class Channel:
    """ A channel class with neighbor support

    Args:
        x (float) : x-coordinate

        y (float) : y-coordinate

        channel (int) : corresponding channel id in the recording data

    Attributes:
        x (float) : x-coordinate

        y (float) : y-coordinate

        channel (int) : corresponding raw data channel index

        left (Channel) : left neighboring channel (negative x direction)

        right (Channel) : right neighboring channel (positive x direction)

        lower (Channel) : lower neighboring channel (negative y direction)

        upper (Channel) : upper neighboring channel (positive x direction)

        broken (bool) : boolean indicating broken channels
    """

    def __init__(self, x, y, channel=None, broken=False):
        self.x = x
        self.y = y
        self.channel = channel

        self.left = None
        self.right = None
        self.lower = None
        self.upper = None

        self.broken = broken

    def set_broken(self, broken):
        """ Set the broken attribute to the given boolean broken
        """
        self.broken = broken

    def is_broken(self):
        """ Return whether this channel is broken or not
        """
        return self.broken
