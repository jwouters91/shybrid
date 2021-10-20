#!/usr/bin/env python

from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='shybrid',
      version='0.4.3',
      description='A graphical tool for generating hybrid ground-truth spiking data for evaluating spike sorting performance',
      author='Jasper Wouters',
      author_email='jasper.wouters@esat.kuleuven.be',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/jwouters91/shybrid',
      packages=['hybridizer', 'hybridizer.ui'],
      entry_points={
          'gui_scripts': [
              'shybrid = hybridizer.shybrid:main'
          ]
      },
      classifiers=[
          'Programming Language :: Python :: 3',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Operating System :: OS Independent',
      ],
      python_requires='>=3.6',
      install_requires=['numpy', 'scipy', 'PyQt5==5.13', 'PyYAML', 'matplotlib>=3.1.2']
     )
