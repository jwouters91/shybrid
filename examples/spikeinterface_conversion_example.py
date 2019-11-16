#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 13:04:03 2019

@author: Jasper Wouters
"""

import os

import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import spikeinterface.comparison as sc

""" 
This example explains how to convert a Recording and Sorting Extractor into
SHYBRID data. The example itself starts from SHYBRID data in the first
place, making this example of little practical use. Nonetheless, the
write_recording and write_sorting methods can be used with any type of
extractor, expanding the range of input types that can be transformed into
hybrid ground truth data.
"""

# full filenames to both the hybrid recording and ground truth
recording_fn = '/path/to/recording.bin'
gt_fn = '/path/to/hybrid_GT.csv'

# create extractor object for both the recording data and ground truth labels
recording_ex = se.SHYBRIDRecordingExtractor(recording_fn)
sorting_ex = se.SHYBRIDSortingExtractor(gt_fn)

# Now we will write create SHYBRID compatible data from a recording and sorting
# extractor
output_folder = '/path/to/spikeinterface_conversion'

se.SHYBRIDSortingExtractor.write_sorting(sorting_ex,output_folder)
initial_sorting_fn = os.path.join(output_folder, 'initial_sorting.csv')

se.SHYBRIDRecordingExtractor.write_recording(recording_ex, output_folder,
                                             initial_sorting_fn)

# Let's try to load those generated files and apply sorting for validation
# purposes
recording_fn_conv = os.path.join(output_folder, 'recording.bin')
sorting_fn_conv = os.path.join(output_folder, 'initial_sorting.csv') # which is effectively the ground truth by construction in this example

recording_ex_conv = se.SHYBRIDRecordingExtractor(recording_fn_conv)
sorting_ex_conv = se.SHYBRIDSortingExtractor(sorting_fn_conv)

# perform spike sorting (e.g., using spyking circus)
sc_params = ss.SpykingcircusSorter.default_params()
sorting_sc_conv = ss.run_spykingcircus(recording=recording_ex_conv, **sc_params,
                                       output_folder='tmp_sc')

# calculate spike sorting performance
# note: exhaustive_gt is set to False, because the hybrid approach generates
#       partial ground truth only
comparison = sc.compare_sorter_to_ground_truth(sorting_ex_conv, sorting_sc_conv, exhaustive_gt=False)
print(comparison.get_performance())
