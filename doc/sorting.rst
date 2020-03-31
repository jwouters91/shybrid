Sorting and comparing sorting results
=====================================
Option 1: SpikeInterface
------------------------
Eventually, after you've generated hybrid ground truth data, you will want to sort this data and compare the sorting results against the ground truth labels. Luckily, these tasks are easy because of the SHYBRID SpikeInterface integration. Please consider the code example below, which implements an entire spike sorting and ground truth validation in just a couple of lines of code.

.. code-block:: python

  import spikeinterface.extractors as se
  import spikeinterface.sorters as ss
  import spikeinterface.comparison as sc

  # full filenames to both the hybrid recording and ground truth
  recording_fn = '/path/to/recording.bin'
  gt_fn = '/path/to/hybrid_GT.csv'

  # create extractor object for both the recording data and ground truth labels
  recording_ex = se.SHYBRIDRecordingExtractor(recording_fn)
  sorting_ex = se.SHYBRIDSortingExtractor(gt_fn)

  # perform spike sorting (e.g., using SpyKING CIRCUS)
  sc_params = ss.SpykingcircusSorter.default_params()
  sorting_sc = ss.run_spykingcircus(recording=recording_ex, **sc_params,
	                            output_folder='tmp_sc')

  # calculate spike sorting performance
  # note: exhaustive_gt is set to False, because the hybrid approach generates
  #       partial ground truth only
  comparison = sc.compare_sorter_to_ground_truth(sorting_ex, sorting_sc, exhaustive_gt=False)
  print(comparison.get_performance())


Option 2: Using the shybrid validation api
------------------------------------------
After you have sorted the hybrid recording, either by using your sorting pipeline of choice or SpikeInterface, the sorting results can also be compared to the ground truth through the SHYBRID validation API. Compared to SpikeInterface, our API implements an automatic cluster merging, which might give more realistic sorting results for spike sorting software that has been tuned toward overclustering (i.e., splitting a single unit cluster into multiple clusters). Please consider the following code example which shows you how to use our validation API.

.. code-block:: python

  import os

  from hybridizer.validation import validate_from_phy, validate_from_csv

  root = '/path/to/hybrid'
  phy_folder = 'phy_results_folder'
  hybrid_gt = 'hybrid_GT.csv'

  comparison_window = 10

  # non-curated spike sorting results
  phy_results = os.path.join(root, phy_folder)
  # ground truth
  hybrid_gt = os.path.join(root, hybrid_gt)

  print('compare from phy')
  validate_from_phy(hybrid_gt, phy_results,
                    comparison_window=comparison_window)

  print('\ncompare from csv (to itself in this example)')
  validate_from_csv(hybrid_gt, hybrid_gt,
                    comparison_window=comparison_window)

.. note::
  You can easily convert sorting results to the CSV format by using SpikeInterface's *write_sorting* method, as was shown in :doc:`loading`.

