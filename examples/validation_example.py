#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 16:43:45 2019

@author: Jasper Wouters
"""

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
