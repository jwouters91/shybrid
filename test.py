#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 14:30:11 2018

@author: jwouters
"""

import unittest
import tempfile

import numpy as np

from hybridizer.io import Recording

class TestIO(unittest.TestCase):
    
    def test_load_raw(self):
        nb_channels = 5
        dtype = "float32"
        sampling_rate = 1

        # generate test data with 5 channels and 20 samples per channel      
        data_gen = np.random.randn(5,20).astype(dtype)
        fn = tempfile.mktemp()
        data_gen.tofile(fn)

        # load data
        self.recording = Recording(fn, nb_channels, sampling_rate, dtype)

        # compare gen and loaded
        np.testing.assert_array_equal(data_gen, self.recording.data)
        self.assertEqual(sampling_rate, self.recording.sampling_rate)

        self.assertFalse(self.recording.isWritable())


if __name__ == "__main__":
    unittest.main()
        