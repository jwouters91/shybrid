#!/usr/bin/env python

from distutils.core import setup

setup(name='hybridizer',
      version='0.1',
      description='Hybrid Data Generator',
      author='Jasper Wouters',
      author_email='jasper.wouters@esat.kuleuven.be',
      packages=['hybridizer', 'hybridizer.ui'],
      scripts=['shybride.py'],
     )
