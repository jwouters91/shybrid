Use your own data
=================
The main idea behind SHYBRID is to provide a tool that can be used to transform your own extracellular recordings into ground truth recordings that can be used in the context of spike sorting. In this section we will show you how to use your own data (extracellular recording and initial sorting) with SHYBRID.

Native support
--------------
SHYBRID provides native support for a limited number of file formats only. We will first go over these formats and show you further down how you can easily convert between other file formats and the native SHYBRID formats using `SpikeInterface <https://spikeinterface.readthedocs.io/en/latest/>`_.

As already discussed in :doc:`quickstart`, a SHYBRID *project* consist of four files as summarized in the table below:

+-------------------------+------------------+
| SHYBRID file            | native format    |
+=========================+==================+
| extracellular recording | .bin, .raw, .dat |
+-------------------------+------------------+
| initial spike sorting   | .csv, phy-format |
+-------------------------+------------------+
| probe file              | .prb             |
+-------------------------+------------------+
| parameter file          | .yml             | 
+-------------------------+------------------+

Extracellular recording
~~~~~~~~~~~~~~~~~~~~~~~
Only binary extracellular recordings are supported natively. SHYBRID assumes that the recording has been high-pass filtered (as is a typical prerequisite in the context of spike sorting). Please note that only binary recordings that encode their samples in a signed data type are supported.

Initial spike sorting
~~~~~~~~~~~~~~~~~~~~~
The initial spike sorting is natively supported in two formats:

1. As a CSV-file where the first column contains the neuron ID of the spike and where the second column contains the spike times in samples. For example, consider the following initial sorting example that contains two clusters (neuron 0 and neuron 1), where cluster 0 contains six spike times and cluster 1 contains eight spikes:

   .. code-block:: python
     :caption: *initial-sorting.csv*

     0,6088484
     0,6417026
     0,6839332
     0,1605072
     0,6893813
     0,2860165
     1,7599880
     1,6430392
     1,3161103
     1,1214103
     1,3112478
     1,7447323
     1,6454460
     1,7652192

2. In the phy sorting format. This format is commonly used by various spike sorting packages such as KiloSort and SpyKING CIRCUS. Essentially, the phy format consists of a folder with multiple saved numpy arrays that contain information related to the spike sorting results.

  .. note::
    When using the phy format for the initial spike sorting, only clusters that have been labeled as *good* (e.g., during manual curation) are loaded by SHYBRID.

Probe file
~~~~~~~~~~
The probe file that is required by SHYBRID, has to be supplied in the phy format. An example probe file for a neural probe with 16 recording channels is shown below.

  .. code-block:: python
    :caption: *probe.prb*

    total_nb_channels = 16
    radius            = 100

    channel_groups = {
      1: {
          'channels': list(range(16)),
          'geometry': {
            0: [0, 0],
            1: [0, 50],
            2: [0, 100],
            3: [0, 150],
            4: [0, 200],
            5: [0, 250],
            6: [0, 300],
            7: [0, 350],
            8: [0, 400],
            9: [0, 450],
            10: [0, 500],
            11: [0, 550],
            12: [0, 600],
            13: [0, 650],
            14: [0, 700],
            15: [0, 750]
          },
          'graph' : []
          }
      }

Parameter file
~~~~~~~~~~~~~~
Finally a YAML parameter file has to be supplied that contains the required recording meta information and links together all of the above files. Below, an example parameters file is shown:

  .. note::
    The parameter file has to have the same file name as the binary recording, i.e, *recording-name.yml*.

  .. code-block:: yaml
    :caption: *recording-name.yml* (with CSV initial sorting)

    ---
    data:
      fs: 25000
      dtype: float32
      order: C
      probe: /path/to/probe/probe.prb

    clusters:
      csv: /path/to/clusters/initial-sorting.csv
    ...

As can be seen from this example parameter file, the following parameters related to the data have to be provided:

- *fs*: represents the sampling frequency in Hz
- *dtype*: represents the datatype that was used to encode the signal samples of the binary recording files
- *order*: contains the order in which the data matrix has been serialized (F: by stacking matrix columns, or C: by stacking matrix rows (`more info <https://en.wikipedia.org/wiki/Row-_and_column-major_order>`_))
- *probe*: contains the path to the probe file

When the initial spike sorting is provided in the CSV format, the path to the CSV file has to be passed as shown in the above example. However, when the initial sorting is given in the phy format, consider the example below:

  .. code-block:: yaml
    :caption: *recording-name.yml* (with phy format initial sorting)

    ---
    data:
      fs: 25000
      dtype: float32
      order: C
      probe: /path/to/probe/probe.prb

    clusters:
      phy: /path/to/phy-initial-sorting
    ...


Data conversion
---------------
Although the binary recording and phy format can be considered as *de facto* standards in spike sorting, many other formats exist. To improve the compatibility of SHYBRID in terms of input data, we implemented SHYBRID extractors for the `SpikeInterface project <https://spikeinterface.readthedocs.io/en/latest/>`_. Besides improved input data compatibility with all technologies that are supported by SpikeInterface, the SpikeInterface integration also enables a straightforward spike sorting of the SHYBRID data and ground truth validation of those sorting results, as we will show in :doc:`sorting`.

This section is not meant to be a thorough introduction for SpikeInterface. We will only provide a minimal code example, to give you an idea of how easy it is to convert your data into SHYBRID compatible data. If you want to learn more about SpikeInterface, we would like to point you to their extensive `tutorials section <https://spikeinterface.readthedocs.io/en/latest/modules/index.html>`_.

  .. code-block:: python
    :caption: *data conversion python code example*

    import spikeinterface.extractors as se

    # create a recording and initial spike sorting extractor
    # for example when SpyKING CIRCUS was used for the initial sorting
    sc_path = '/path/to/sc-initial-sorting'
    sorting_ex = se.SpykingCircusSortingExtractor(sc_path)
    recording_ex = se.SpykingCircusRecordingExtractor(sc_path)

    # define a SHYBRID project folder
    project_folder = '/path/to/my-shybrid-project'

    # create SHYBRID compatible data from a recording and sorting extractor
    se.SHYBRIDSortingExtractor.write_sorting(sorting_ex, project_folder)
    initial_sorting_fn = os.path.join(project_folder, 'initial_sorting.csv')
    se.SHYBRIDRecordingExtractor.write_recording(recording_ex, project_folder,
                                                 initial_sorting_fn)

The result of executing the above code example is that all files required by SHYBRID have been generated, based on the SpyKING CIRCUS project. After executing this script, you can immediately load this data into SHYBRID. Although this example is minimal, often nothing more is needed to convert the data from your workflow into a SHYBRID project. Note that SpikeInterface can also be used to perform the high-pass filtering (by adding one line of code) and can also be used to perform a curation of the initial sorting. If you want to know more about these features, we recommend you to consult the `SpikeInterface manual <https://spikeinterface.readthedocs.io/en/latest/>`_.




