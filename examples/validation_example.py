#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 16:43:45 2019

@author: Jasper Wouters
"""

import os

from hybridizer.validation import validate_from_phy, validate_from_csv

root = '/media/jwouters/DATA/KU_LEUVEN/Paper_shybride/hybrid_sc'
# non-curated spike sorting results
phy_ks = os.path.join(root, 'kilosort_more_clusters')
# ground truths
hybrid_gt = os.path.join(root, 'hybrid_GT.csv')
comparison_window = 10

print('test phy')
validate_from_phy(hybrid_gt, phy_ks,
                  comparison_window=comparison_window)

print('\ntest csv')
validate_from_csv(hybrid_gt,hybrid_gt,
                  comparison_window=comparison_window)
