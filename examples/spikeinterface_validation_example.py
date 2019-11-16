#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 13:04:03 2019

@author: Jasper Wouters
"""

import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import spikeinterface.comparison as sc

# full filenames to both the hybrid recording and ground truth
recording_fn = '/path/to/recording.bin'
gt_fn = '/path/to/hybrid_GT.csv'

# create extractor object for both the recording data and ground truth labels
recording_ex = se.SHYBRIDRecordingExtractor(recording_fn)
sorting_ex = se.SHYBRIDSortingExtractor(gt_fn)

# perform spike sorting (e.g., using spyking circus)
sc_params = ss.SpykingcircusSorter.default_params()
sorting_sc = ss.run_spykingcircus(recording=recording_ex, **sc_params,
                                  output_folder='tmp_sc')

# calculate spike sorting performance
# note: exhaustive_gt is set to False, because the hybrid approach generates
#       partial ground truth only
comparison = sc.compare_sorter_to_ground_truth(sorting_ex, sorting_sc, exhaustive_gt=False)
print(comparison.get_performance())
