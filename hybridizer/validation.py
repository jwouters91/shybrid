#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 10:24:26 2019

@author: Jasper Wouters
"""

import numpy as np


def validate_from_phy(gt_csv_fn, phy_sorting_path, comparison_window=30):
    """ Compare spike sorting results (in phy format) with a hybrid ground
    truth. The results of this comparison are printed in the terminal.

    Parameters
    ----------
    gt_csv_fn (str) : Full filename of hybrid ground truth csv file.

    phy_sorting_path (str) : Full path to the folder containing the spike
    sorting results in the phy format.

    comparison_window (int, optional) : A discrete window that is placed around
    every sorted spike, to account for an offset between the spike sorting
    time stamps and the ground truth.
    """
    pass

def validate_from_csv(gt_csv_fn, csv_sorting_fn, comparison_window=30):
    """ Compare spike sorting results (in csv format) with a hybird ground
    truth. The results of this comparison are printed in the terminal.

    Parameters
    ----------
    gt_csv_fn (str) : Full filename of hybrid ground truth csv file.

    csv_sorting_fn (str) : Full filename of the csv file containing the spike
    sorting results.

    comparison_window (int, optional) : A discrete window that is placed around
    every sorted spike, to account for an offset between the spike sorting
    time stamps and the ground truth.
    """
    pass


""" UTILS
"""
class Segments:
    """ Class to model binary (windowed) event trains. The standard constructor
    returns a blank object.
    """
    def __init__(self, segments=set(), nb_events=0):
        self.segments = segments
        self.nb_events = nb_events

    @classmethod
    def from_events(cls, events, window=5):
        """ Return a Segments object created from the given discrete times.

        Parameters
        ----------
        events (array_like) : List containing discrete time integer events.

        window (int, optional) : Size of window to be applied with the event at
        its center. The window has to be greater than 0.

        Returns
        -------
        segments (Segments) : Segments object created from the given events.
        """
        assert(window > 0)

        # working memory
        tmp_list = []

        half_window = window // 2
        # iterate over all events and extent with the given window
        for event in events:
            start = event - half_window
            time_window = list(range(int(start), int(start+window)))
            tmp_list.extend(time_window)

        segments = cls()

        # turning the working memory into a set for O(1) retrieval rates(?)
        segments.segments = set(tmp_list)
        segments.nb_events = len(events)

        return segments

    def contains(self, events):
        """ Return a boolean array with the same length as events to indicate
        whether or not a given event coincides with a segment of the binary
        train.

        Parameters
        ----------
        events (array_like) : List containing integer events.

        Returns
        -------
        contains (list) : Boolean list.
        """
        contains = [False] * len(events)

        for idx, event in enumerate(events):
            if event in self.segments:
                contains[idx] = True

        return contains

    def merge(self, segments):
        """ Return a segments object that is merged from self and the given
        segments object.

        Parameters
        ----------
        segments (Segments) : A segments object to merge with this object.

        Returns
        -------
        merged_segments (Segments) : The merged segments object
        """
        merged_segments = self.segments.union(segments.segments)
        merged_nb_events = self.nb_events + segments.nb_events

        return Segments(segments=merged_segments, nb_events=merged_nb_events)


class SegmentsCollection:
    """ Model for a collection of segments.

    Parameters
    ----------
    sorting_results (SortingResults) : A sorting results object.
    """
    def __init__(self, sorting_results, window=5):
        self.collection = {}

        for cluster, events in sorting_results:
            self.collection[cluster] = Segments.from_events(events,
                                                            window=window)

    def merge_segments(self, to_merge):
        """ Return a segments object containing the merged segments from all
        clusters provided in to_merge.

        Parameters
        ----------
        to_merge (array_like) : Array containing the indices of the clusters
        that are to be merged.

        Returns
        -------
        merged (Segments) : Resulting merged segments
        """
        merged = Segments()

        for cluster in to_merge:
            merged = merged.merge(self.collection[cluster])

        return merged


class SortingResults:
    """ Model for spike sorting results.

    Parameters
    ----------
    cluster_events (ndarray) : (n, 2) containing in its row individual spikes,
    the first column contains the cluster ID and the second column contains the
    discrete spike time
    """
    # lambda helpers to process numpy constructor input
    _events_from_cluster = lambda arr, cls: arr[np.isin(arr[:,0], cls), 1]
    _all_clusters = lambda arr: np.unique(arr[:,0])

    def __init__(self, cluster_events):
        self.clusters = {}

        derived_clusters = self._all_clusters(cluster_events)
        for cluster in derived_clusters:
            self.clusters[cluster] = self._events_from_cluster(cluster_events,
                                                               cluster)

    def __iter__(self):
        # use an iterator on the list of cluster idxs contained in the dict
        cluster_idxs = list(self.clusters.keys())
        self._key_iterator = cluster_idxs.__iter__()

        return self

    def __next__(self):
        # return the next cluster information using an iterator on the keys
        cluster_idx = self._key_iterator.__next__()
        return cluster_idx, self.clusters[cluster_idx]
