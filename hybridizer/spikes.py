#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 13:52:29 2018

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

import numpy as np

class SpikeTrain:
    """ Spike train class grouping spikes and recording

    Args:
        recording (Recording): recording object related to this spike train

        spike_times (ndarray): array containing times of every spike
    """

    def __init__(self, recording, spike_times, template=None,
                 template_fitting=None):
        self.recording = recording
        self.spikes = spike_times
        self.template = template

        if template_fitting is not None:
            self._template_fitting = template_fitting
            self._fitting_energy = template_fitting**2

        self._energy_sorted_idxs = None
        self._PSNR = None

    def calculate_template(self, window_size, from_import=False, zf_frac=0.03):
        """ Calculate a template for this spike train for the given discrete
        window size
        """
        self.template = Template(from_import=from_import, zf_frac=zf_frac)
        self.template.calculate_from_spike_train(self, window_size)

    def get_nb_spikes(self):
        """ Return the number of spikes in the spike train
        """
        return self.spikes.size

    def fit_spikes(self):
        """ Calculate spike fits
        """
        self._template_fitting = np.zeros(self.spikes.shape)
        self._fitting_energy = np.zeros(self.spikes.shape)

        for idx, spike in enumerate(self.spikes):
            start, end = self.get_spike_start_end(spike)
            chunk = self.recording.get_good_chunk(start, end)

            # fit template to spike
            temp_fit = self.template._fit_template(chunk)

            self._template_fitting[idx] = temp_fit
            # fitting energy only based on template
            self._fitting_energy[idx] = temp_fit**2

        # important to reset the energy sorted idxs on a new fit
        self._energy_sorted_idxs = None

    def subtract(self):
        """ Subtract spike train from the recording.
        """
        channels = self.recording.probe.channels

        for idx, spike in enumerate(self.spikes):
            start, end = self.get_spike_start_end(spike)

            temp_fit = self._template_fitting[idx]

            if temp_fit == 0:
                continue

            fitted_waveform = self.template.get_fitted_waveform(temp_fit)

            # subtract
            self.recording.data[channels, start:end] = \
                 self.recording.data[channels, start:end] - fitted_waveform

    def insert(self):
        """ Insert spike train in the recording
        """
        channels = self.recording.probe.channels

        for idx, spike in enumerate(self.spikes):
            start, end = self.get_spike_start_end(spike)

            temp_fit = self._template_fitting[idx]

            if temp_fit == 0:
                continue

            fitted_waveform = self.template.get_fitted_waveform(temp_fit)

            # insert
            self.recording.data[channels, start:end] = \
                 self.recording.data[channels, start:end] + fitted_waveform

    def get_spike_start_end(self, spike):
        """ Get start and end for the given spike
        """
        start = int(spike - self.template.window_size / 2)
        end = int(start + self.template.window_size)

        return start, end

    def get_energy_sorted_spike_time(self, spike_idx):
        """ Return the spike time from the energy sorted spike train for the
        given spike_idx

        This method requires that the energies have been initialized already
        """
        return self.get_energy_sorted_spikes()[spike_idx]

    def get_energy_sorted_idxs(self):
        """ Return energy sorted idxs.

        This method requires that the fitting factors have been initialized.
        """
        if self._energy_sorted_idxs is None:
            self._energy_sorted_idxs = np.argsort(self._fitting_energy)

        return self._energy_sorted_idxs

    def get_energy_sorted_spikes(self):
        """ Return energy sorted spikes

        This method requires that the energies have been initialized already
        """
        return self.spikes[self.get_energy_sorted_idxs()]

    def get_energy_sorted_fittings(self):
        return self._template_fitting[self.get_energy_sorted_idxs()]

    def get_automatic_energy_bounds(self, C=0.75):
        """ Return the lower and upper index obtained from robust
        statistics estimated on the logarithm of the fitting energy.

        Parameters
        ----------
        C (float) : Factor that determines the width of the interval that is
        considered. The default value is a safe conservative choice, that will
        lead to good spikes not being reinserted during hybridization.

        Returns
        -------
        lower_idx (int) : lower bound index in the energy sorted domain
        that corresponds to Q1 - C * IQR.

        upper_idx (int) : upper bound index in the energy sorted domain
        that corresponds to Q3 + C * IQR.
        """
        # log space is considered to excluded values close to zero
        pcts = np.percentile(np.log10(self._fitting_energy), [25.0, 75.0])

        Q1 = pcts[0]
        Q3 = pcts[1]

        IQR = Q3 - Q1

        lower_bound = Q1 - C * IQR
        upper_bound = Q3 + C * IQR

        energy_sorted_energy = np.log10(self._fitting_energy)[self.get_energy_sorted_idxs()]

        try:
            lower_idx = np.where(energy_sorted_energy < lower_bound)[0].max()
        except ValueError:
            # this exception is thrown if the boolean condition returns an
            # empty array
            lower_idx = None

        try:
            upper_idx = np.where(energy_sorted_energy > upper_bound)[0].min()
        except ValueError:
            upper_idx = None

        return lower_idx, upper_idx

    def get_automatic_move(self):
        """ Return random template move along y-axis

        Returns
        -------
        move (int) : number of moves along the y-axis
        """
        max_channel = self.template.get_max_channel_idx()

        probe = self.recording.probe
        max_channel_input_space = probe.channels[max_channel]
        max_channel_geo = probe.geometry[max_channel_input_space]

        nb_positions = (probe.y_max - probe.y_min) / probe.y_between + 1
        current_position = (max_channel_geo[1] - probe.y_min) / probe.y_between

        random_shift = current_position

        while self._no_valid_shift(random_shift, current_position):
            random_shift = np.random.randint(0, nb_positions)

        return random_shift - current_position

    def _no_valid_shift(self, random, current):
        """ Helper function to calculate random shift
        """
        if random == current or random == current + 1 or random == current - 1:
            return True
        else:
            return False

    def get_PSNR(self):
        """ Calculate the peak-signal-to-noise ratio
        """
        if self._PSNR is None:
            max_channel = self.template.get_max_channel_idx()

            PS = np.abs(self.template.get_template_data()[max_channel]).max()
            PS = PS**2

            N = self.recording.get_signal_power_for_channel(max_channel)

            self._PSNR = 10*np.log10(PS / N)

        return self._PSNR

    def set_target_PSNR(self, target_PSNR):
        """ Set the target PSNR by scaling the template
        """
        desired_scaling = np.sqrt(10**(target_PSNR/10))
        actual_scaling = np.sqrt(10**(self.get_PSNR()/10))
        scaling = desired_scaling / actual_scaling

        # scale template
        self.template._data *= scaling

        # PSNR altered, so reset the PSNR
        self._PSNR = None

class Template:
    """ Template class
    """
    def __init__(self, data=None, from_import=False, zf_frac=None):
        if data is not None:
            self._data = data
            self.window_size = self._data.shape[1]
            self._calculate_template_energy()

        if from_import:
            self.imported=True
        else:
            self.imported=False

        self._zf_frac = zf_frac

    def calculate_from_spike_train(self, spike_train, window_size):
        """ Calculate a spike template using the given spike train
        """
        self.window_size = window_size

        # build spike tensor
        spike_tensor = np.empty((spike_train.get_nb_spikes(),
                                 spike_train.recording.get_nb_good_channels(),
                                 self.window_size))

        for spike_idx in range(spike_train.get_nb_spikes()):
            # TODO use get_spike_start_end
            start = int(spike_train.spikes[spike_idx] - window_size/2)
            end = int(start + self.window_size)

            # boundary checks
            if (start < 0) or (end > spike_train.recording.data.shape[1]):
                continue

            spike_tensor[spike_idx] =\
                spike_train.recording.get_good_chunk(start,
                                                     end)

        self._data = np.median(spike_tensor, axis=0)
        self._calculate_template_energy()

    def _calculate_template_energy(self):
        """ Calculate the template energy in every channel
        """
        DC_corrected_temp = self._data - self._data.mean(axis=1)[:,np.newaxis]
        energy = DC_corrected_temp**2
        self._energy = np.sum(energy, axis=1)

    def get_template_data(self):
        """ Return the template data containing the waveform
        """
        tmp_data = self._data.copy()

        if self._zf_frac is not None:
            tmp_data[self._energy<self._zf_frac*self._energy.max(),:] = 0

        return tmp_data

    def update_zf_frac(self, zf_frac):
        """ Update the zero force fraction
        """
        self._zf_frac = zf_frac

    def get_shifted_template(self, spike_train, x_shift, y_shift):
        """ Calculate shifted template
        """
        # initialize shifted template
        shifted_template = np.zeros(spike_train.template._data.shape)

        # extract geometrical information
        x_between = spike_train.recording.probe.x_between
        y_between = spike_train.recording.probe.y_between

        for idx, channel in enumerate(spike_train.recording.probe.channels):
            geo = spike_train.recording.probe.geometry[channel]

            # find location that projects on this channel (that's why minus)
            geo_x = geo[0] - x_shift * x_between
            geo_y = geo[1] - y_shift * y_between

            # if this location is not located on the probe we can continue
            # this prevents extrapolation from happening
            if geo_x < spike_train.recording.probe.x_min:
                continue
            if geo_x > spike_train.recording.probe.x_max:
                continue
            if geo_y < spike_train.recording.probe.y_min:
                continue
            if geo_y > spike_train.recording.probe.y_max:
                continue

            # initialize interpolated waveform
            interpolated_waveform = np.zeros(spike_train.template._data[0].shape)

            interpolation_count = 0
            interpolation_needed = True

            # loop over channels to find the projection channel
            for jdx, project_channel in enumerate(spike_train.recording.probe.channels):
                project_geo = spike_train.recording.probe.geometry[project_channel]

                if geo_x == project_geo[0] and geo_y == project_geo[1]:
                    shifted_template[idx] = spike_train.template._data[jdx]
                    interpolation_needed = False
                else:
                    if abs(geo_x - project_geo[0]) <= x_between and abs(geo_y - project_geo[1]) <= y_between:
                        interpolated_waveform += spike_train.template._data[jdx]
                        interpolation_count += 1

            if interpolation_needed and interpolation_count > 0:
                shifted_template[idx] = interpolated_waveform / interpolation_count

        return Template(data=shifted_template, zf_frac=self._zf_frac)

    def _fit_template(self, chunk):
        """ Fit the given chunk to the template
        """
        # make use of zero forcing while fitting -> load template through api
        template_flat = self.get_template_data().flatten()
        chunk_flat = chunk.flatten()

        # boundary cases set to zero (i.e., they don't get subtracted)
        if template_flat.size != chunk_flat.size:
            fit = 0
        else:
            # calculate fit according to yger et al. 2018
            fit = np.dot(chunk_flat, template_flat)
            fit /= np.linalg.norm(template_flat)**2

        return fit

    def get_fitted_waveform(self, template_fit):
        """ Return fitted waveform
        """
        return template_fit * self.get_template_data()

    def get_max_channel_idx(self):
        """ Return the index of the maximum peak energy channel in the good
        channels space (i.e., the space obtained from slicing the the original
        data with the given channels array in the probe file).

        Returns
        -------
        channel_idx (int) : index indication the channel with the highest
        template peak energy
        """
        return np.argmax(np.max(np.abs(self.get_template_data()), axis=1))
