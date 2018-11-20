# SHY BRIDE README
## Hybrid data

## Installation instructions
1. Install miniconda (Python 3.x version) for your operaring system. Please follow the official conda.io [instructions](https://conda.io/docs/user-guide/install/index.html#regular-installation).
2. Clone this GIT project on your local machine:
``` 
git clone https://gitlab.esat.kuleuven.be/Jasper.Wouters/shy.bride.git
```
3. Create a conda environment for SHY BRIDE:
```
conda create -n shybride --file requirements
```
Or use requirements_strict to enforce the exact package versions used for testing.
4. Activate the environment:
```
conda activate shybride
```
5. Install shy bride package
```
python setup.py install
```
6. Run and have fun
```
shybride.py
```

We kept the extension on the executable, such that it can also be executed from a windows command line (no shebang support on windows). Keep in mind that the program is only accessible from within the shybride conda environment (i.e., reactivate the environment after ,e.g., a reboot).

## Generating hybrid data using SHY BRIDE

### Data required
To generate hybrid ground truth spiking data the following files are required by the tool:

* recording data in binary format (e.g. recording.bin)
* probe file describing the recording probe geometry
* single-unit cluster information either in
	* csv format, or
	* phy format
* yaml-file having the same name as the recording (e.g. recording.yml) containing:
	* binary format meta information
	* path to probe file
	* path to cluster information

An example yaml file (recording.yml) is given below (all parameters shown are mandatory):

``` 
---
# parameters used by SHY BRIDE
data:
  fs: 25000
  dtype: float32
  order: C
  probe: /path/to/probe/probe.prb

clusters:
  csv: /path/to/clusters/clusters.csv
...
```
An example that reads single-unit cluster information directly from phy is given below:

``` 
---
# parameters used by SHY BRIDE (using phy clusters)
data:
  fs: 25000
  dtype: float32
  order: C
  probe: /path/to/probe/probe.prb

clusters:
  phy: /path/to/phy-data
...
```

### Exporting hybrid data
Note: the exported binary is vectorized using the C-style formatting (row-major), this independent of the format of the original data.
