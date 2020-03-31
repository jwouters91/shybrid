Installation Procedure
======================
Installing SHYBRID is as simple as installing a python package. On this page we will guide you through the different installation steps.

1. Install Miniconda or another Python distribution (Python 3.x version) for your operating system. Miniconda installation instructions can be found on the official `Conda website <https://conda.io/projects/conda/en/latest/user-guide/install/>`_.

The remaining SHYBRID installation commands have to executed in the (Anaconda) terminal.

2. Optional, yet recommended, create a virtual environment for SHYBRID::

   >> conda create -n shybrid_env python=3

   and activate the newly created environment::

   >> conda activate shybrid_env

3. Install SHYBRID using pip::

   >> pip install shybrid

4. To launch SHYBRID execute the following command::

   >> shybrid

If you decided to install SHYBRID in a virtual environment, keep in mind that the program is only accessible from within the shybrid conda environment. This implies that you have to reactivate this environment, e.g., after a PC reboot.
