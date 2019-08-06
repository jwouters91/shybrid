#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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

import numpy as np

from PyQt5.QtCore import QThread, pyqtSignal

from hybridizer.spikes import SpikeTrain
from hybridizer.hybrid import Insert, Subtract


class TemplateWorker(QThread):
    template_ready = pyqtSignal(SpikeTrain)

    def __init__(self):
        QThread.__init__(self)

    def run(self):
        if not self.is_hybrid:
            # calculate template
            self.spike_train.calculate_template(self.window_size,
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
    move_ready = pyqtSignal()

    def __init__(self):
        QThread.__init__(self)

    def run(self):
        spike_train = self.cluster.get_actual_spike_train()
        train_subtraction = Subtract(spike_train)
        self.cluster.apply_operator(train_subtraction)

        sorted_spikes = spike_train.get_energy_sorted_spikes().copy()

        if self.energy_LB is None:
            l_idx = None
        else:
            l_idx = self.energy_LB+1 # exclusive

        if self.energy_UB is None:
            u_idx = None
        else:
            u_idx = self.energy_UB # also exclusive, but handled by python

        sorted_spikes = sorted_spikes[l_idx:u_idx]
        print('# {} spikes considered for migration'.format(int(sorted_spikes.size)))

        # add fixed temporal offset to avoid residual correlation
        time_shift = int(2*spike_train.template.window_size)
        sorted_spikes += time_shift

        # look for out of bounds spikes (conservative bound used here)
        within_bounds = sorted_spikes <  (spike_train.recording.get_duration() - spike_train.template.window_size)
        sorted_spikes = sorted_spikes[within_bounds]

        # insertion shifted template
        sorted_template_fit = spike_train.get_energy_sorted_fittings().copy()
        sorted_template_fit = sorted_template_fit[l_idx:u_idx]
        sorted_template_fit = sorted_template_fit[within_bounds]

        assert(sorted_spikes.shape == sorted_template_fit.shape)

        insert_train = SpikeTrain(spike_train.recording, sorted_spikes,
                                  template=self.shifted_template,
                                  template_fitting=sorted_template_fit)

        if self.target_PSNR is not None:
            insert_train.set_target_PSNR(self.target_PSNR)

        train_insertion = Insert(insert_train)
        self.cluster.apply_operator(train_insertion)

        print('# {} spikes migrated'.format(int(sorted_spikes.size)))

        self.flush()

        self.move_ready.emit()

    def flush(self):
        """ Flush recording and update GT
        """
        self.cluster.get_actual_spike_train().recording.flush()
