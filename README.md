# SHY BRIDE README
## Installation instructions
1. Install miniconda (Python 3.x version) for your operaring system. Please follow the official conda.io [instructions](https://conda.io/docs/user-guide/install/index.html#regular-installation).
2. Clone this GIT project on your local machine:
` git clone https://gitlab.esat.kuleuven.be/Jasper.Wouters/Hybridizer.git `
3. Create a conda environment for SHY BRIDE:
` conda create -n shybride --file requirements `
Or use requirements_strict to enforce the exact package versions used for testing.
4. Activate the environment:
` conda activate shybride `
5. Install shy bride package
` python setup.py install `
6. Run and have fun
` shybride.py `
We kept the extension on the executable, such that it can also be executed from a windows command line (no shebang support on windows). Keep in mind that the program is only accessible from within the shybride conda environment (i.e., reactivate the environment after ,e.g., a reboot).

## Generating hybrid data using SHY BRIDE

### exporting data
The exported binary is vectorized using the C-style formatting (row-major), this independent of the format of the original data.
