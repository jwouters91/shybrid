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

import numpy as np

from hybridizer.io import Phy

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
    gt = SortingResults.fromCSV(gt_csv_fn)
    sorting = SortingResults(Phy(phy_sorting_path).get_cluster_and_times(curated=False))

    validate(gt, sorting, comparison_window=comparison_window)

def validate_from_csv(gt_csv_fn, sorting_csv_fn, comparison_window=30):
    """ Compare spike sorting results (in csv format) with a hybird ground
    truth. The results of this comparison are printed in the terminal.

    Parameters
    ----------
    gt_csv_fn (str) : Full filename of hybrid ground truth csv file.

    sorting_csv_fn (str) : Full filename of the csv file containing the spike
    sorting results.

    comparison_window (int, optional) : A discrete window that is placed around
    every sorted spike, to account for an offset between the spike sorting
    time stamps and the ground truth.
    """
    gt = SortingResults.fromCSV(gt_csv_fn)
    sorting = SortingResults.fromCSV(sorting_csv_fn)

    validate(gt, sorting, comparison_window=comparison_window)

def validate(gt, sorting, comparison_window=30):
    """ Compare spike sorting results with a hybird ground
    truth. The results of this comparison are printed in the terminal.

    Parameters
    ----------
    gt (SortingResults) : Sorting result object containing the ground truth.

    sorting (SortingResults) : Sorting result object containing the sorting
    results.

    comparison_window (int, optional) : A discrete window that is placed around
    every sorted spike, to account for an offset between the spike sorting
    time stamps and the ground truth.
    """
    sorting_segments = SegmentsCollection(sorting, window=comparison_window)

    _print_header()

    for gt_cluster, gt_events in gt:
        best_matching_clusters = sorting_segments.auto_merge(gt_events)
        merged_segment = sorting_segments.merge(best_matching_clusters)

        precision, recall = merged_segment.calculate_precision_recall(gt_events)

        _print_values(gt_cluster, precision, recall, best_matching_clusters)


""" UTILS
"""
def _print_header():
    print('gt_cluster;F1;precision;recall;nb_clusters;clusters')

def _print_values(gt_cluster, precision, recall, matching_clusters):
    print('{:d};{:.3f};{:.3f};{:.3f};{:d};{}'.format(int(gt_cluster),
                                                     F1(precision, recall),
                                                     precision,
                                                     recall,
                                                     matching_clusters.size,
                                                     matching_clusters.tolist()))

# F1 metric lambda helper function
F1 = lambda prec, rec: 2 * prec * rec / (prec + rec) if prec + rec != 0 else 0

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

    def calculate_precision_recall(self, gt_events):
        """ Return the precision and recall for a given set of ground truth
        events.

        Parameters
        ----------
        gt_events (ndarray) : array containing ground truth events

        Returns
        -------
        precision (float) : the fraction of detections that are true detections
        (true positives) / (true positives + false positives)

        recall (float) : the fraction of all positives that are retrieved
        (true positives) / (true positives + false negatives)
        """
        contains_result = self.contains(gt_events)

        # count the number of true detections
        nb_true_positives = np.where(contains_result)[0].size

        nb_positives = gt_events.size
        nb_detections = self.nb_events

        # Due to the windowing applied on the segments we might get more
        # TP than detections. This is solved by the following line of code
        nb_true_positives = min(nb_true_positives, nb_detections)

        precision = nb_true_positives / nb_detections
        recall = nb_true_positives / nb_positives

        return precision, recall

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

    window (int, optional) : Segments window (see Segments.from_events)
    """
    def __init__(self, sorting_results, window=5):
        self.collection = {}

        for cluster, events in sorting_results:
            self.collection[cluster] = Segments.from_events(events,
                                                            window=window)

    def merge(self, to_merge):
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

    def rank_clusters(self, gt_events):
        """ Rank clusters by precision (single-unitness).

        Parameters
        ----------
        gt_events (ndarray) : array containing ground truth events

        Returns
        -------
        ranked_cluster (ndarray) : (nclusters, 2) array containing in the first
        column the cluster id and in the second column the precision for that
        cluster
        """
        clusters = list(self.collection)
        ranked_clusters = np.zeros((len(clusters), 2))

        for idx, cluster in enumerate(clusters):
            segments = self.collection[cluster]

            prec, _ = segments.calculate_precision_recall(gt_events)

            ranked_clusters[idx, 0] = cluster
            ranked_clusters[idx, 1] = prec

        sorted_idxs = np.argsort(ranked_clusters[:,1])

        return ranked_clusters[sorted_idxs[::-1]]

    def auto_merge(self, gt_events):
        """ Automatically merge clusters if combining ranked clusters leads to
        increased F1-score.

        Parameters
        ----------
        gt_events (ndarray) : array containing ground truth events

        Returns
        -------
        merged_clusters (ndarray) : array containing cluster numbers that are
        merge according to the increasing F1-metric.
        """
        ranked_clusters = self.rank_clusters(gt_events)

        clusters_to_merge = np.array([ranked_clusters[0,0]])

        merged_segments = self.merge(clusters_to_merge)
        prec, rec = merged_segments.calculate_precision_recall(gt_events)

        accumulated_F1 = F1(prec, rec)

        for next_cluster in ranked_clusters[1:,0]:
            clusters_to_merge = np.concatenate((clusters_to_merge,
                                                np.array([next_cluster])))

            merged_segments = self.merge(clusters_to_merge)
            prec, rec = merged_segments.calculate_precision_recall(gt_events)

            if F1(prec, rec) > accumulated_F1:
                accumulated_F1 = F1(prec, rec)
            else:
                return clusters_to_merge[:-1]

        return clusters_to_merge


class SortingResults:
    """ Model for spike sorting results (or hybrid GT).

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

        derived_clusters = SortingResults._all_clusters(cluster_events)
        for cluster in derived_clusters:
            self.clusters[cluster] = SortingResults.\
                _events_from_cluster(cluster_events, cluster)

    @classmethod
    def fromCSV(cls, csv_fn, delimiter=','):
        """ Create a Sorting Results object from a csv file with two columns,
        where the first column contains the cluster index and the second
        column contains the discrete time of a spike.

        Parameters
        ----------
        csv_fn (str) : Full filename of the csv file containing the sorting
        results.

        delimiter (str, optional) : The delimiter used in the csv file.

        Returns
        -------
        sorting_results (SortingResults) : The corresponding SortingResults
        object
        """
        cluster_events = np.loadtxt(csv_fn, delimiter=delimiter)

        return cls(cluster_events)

    def __iter__(self):
        # use an iterator on the list of cluster idxs contained in the dict
        cluster_idxs = list(self.clusters)
        self._key_iterator = cluster_idxs.__iter__()

        return self

    def __next__(self):
        # return the next cluster information using an iterator on the keys
        cluster_idx = self._key_iterator.__next__()
        return cluster_idx, self.clusters[cluster_idx]
