#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 15:45:15 2019

@author: Jasper Wouters

SHYBRID
Copyright (C) 2018  Jasper Wouters

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

import os
import numpy as np

from PyQt5.QtCore import QThread, pyqtSignal

from hybridizer.io import SpikeClusters
from hybridizer.spikes import SpikeTrain


class TemplateWorker(QThread):
    template_ready = pyqtSignal(SpikeTrain)

    def __init__(self):
        QThread.__init__(self)

    def run(self):
        # calculate template
        self.spike_train.calculate_template(window_size=self.window_size,
                                           zf_frac=self.zf_frac)
        # calculate fit factors
        self.spike_train.fit_spikes()

        self.template_ready.emit(self.spike_train)


class ActivationWorker(QThread):
    activation_ready = pyqtSignal(np.ndarray, float)

    def __init__(self):
        QThread.__init__(self)

    def run(self):
        activations = self.recording.count_spikes(C=6).astype(np.float)
        # clip activations for visual purposes
        pct = np.percentile(activations, 90)
        activations[activations > pct] = pct

        sig_power = self.recording.get_signal_power()

        self.activation_ready.emit(activations, sig_power)

class MoveWorker(QThread):
    move_ready = pyqtSignal(SpikeTrain, SpikeClusters)

    def __init__(self):
        QThread.__init__(self)

        self.recording = None
        self.spike_train = None
        self.generated_GT = None

        self.current_cluster = None
        self.energy_LB = None
        self.energy_UB = None
        self.window_size = None
        self.dump_path = None

    def run(self):
        self.spike_train.subtract_train()

        # re-insert the shifted template for the selected energy interval
        sorted_idxs = self.spike_train.get_energy_sorted_idxs().copy()
        sorted_spikes = self.spike_train.get_energy_sorted_spikes().copy()

        if self.energy_LB is None:
            l_idx = None
        else:
            l_idx = self.energy_LB+1 # exclusive

        if self.energy_UB is None:
            u_idx = None
        else:
            u_idx = self.energy_UB # also exclusive, but handled by python

        sorted_spikes_slice = sorted_spikes[l_idx:u_idx]
        sorted_idxs = sorted_idxs[l_idx:u_idx]

        assert(sorted_spikes_slice.shape == sorted_idxs.shape)

        print('# {} spikes considered for migration'.format(int(sorted_spikes_slice.size)))

        # add fixed temporal offset to avoid residual correlation
        time_shift = int(2*self.window_size)
        sorted_spikes_insert = sorted_spikes_slice + time_shift

        # insertion shifted template
        sorted_template_fit = self.spike_train._template_fitting[sorted_idxs]

        assert(sorted_spikes_slice.shape == sorted_template_fit.shape)

        inserted_spikes = self.spike_train.insert_given_train(sorted_spikes_insert,
                                                             self.spike_train.template.shifted_template,
                                                             sorted_template_fit)

        self.spike_train.update(inserted_spikes)
        self.spike_train.fit_spikes() 

        self.generated_GT[self.current_cluster] = inserted_spikes

        print('# {} spikes migrated'.format(int(inserted_spikes.size)))

        self.flush()

        self.move_ready.emit(self.spike_train, self.generated_GT)

    def flush(self):
        """ Flush recording and update GT
        """
        csv_path = os.path.join(self.dump_path, 'hybrid_GT.csv')
        self.recording.flush()
        print('# updated ground truth in {}'.format(csv_path))
        self.generated_GT.dumpCSV(csv_path)
