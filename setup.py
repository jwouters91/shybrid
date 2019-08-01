#!/usr/bin/env python

from distutils.core import setup

setup(name='hybridizer',
      version='0.2',
      description='Hybrid Data Generator',
      author='Jasper Wouters',
      author_email='jasper.wouters@esat.kuleuven.be',
      packages=['hybridizer', 'hybridizer.ui'],
      scripts=['shybrid.py'],
     )
