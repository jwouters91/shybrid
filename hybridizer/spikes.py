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
import scipy.signal as sg

class SpikeTrain:
    """ Spike train class grouping spikes and recording

    Parameters
    ----------
        recording (Recording): recording object related to this spike train

        spike_times (ndarray): array containing times of every spike

        template (Template): optional template, if already known

        template_fitting (ndarray): optional template fitting (same size as spike_times)

        template_jitter (ndarray): optional template jitter (same size as spike_times), default is random jitter

        upsampling_factor (int) : template upsampling factor, default is 10
    """

    def __init__(self, recording, spike_times, template=None,
                 template_fitting=None, template_jitter=None,
                 upsampling_factor=10):
        self.recording = recording
        self.spikes = spike_times
        self.template = template

        self._template_fitting = template_fitting

        # if no jitter vector is given, a random jitter vector is constructed
        if template_jitter is None:
            if upsampling_factor % 2 == 0:
                low = -upsampling_factor // 2 + 1
            else:
                low = -upsampling_factor // 2

            high = upsampling_factor // 2 + 1 # exclusive

            self._template_jitter = np.random.randint(low, high=high,
                                                      size=spike_times.size)
        else:
            self._template_jitter = template_jitter

        self._upsampling_factor = upsampling_factor

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

        for idx, spike in enumerate(self.spikes):
            start, end = self.get_spike_start_end(spike)
            chunk = self.recording.get_good_chunk(start, end)

            # fit template to spike
            temp_fit = self.template._fit_template(chunk)

            self._template_fitting[idx] = temp_fit

        self._template_fitting[self._template_fitting < 0] = 0

        # important to reset the energy sorted idxs on a new fit
        self._energy_sorted_idxs = None

    def subtract(self):
        """ Subtract spike train from the recording.
        """
        channels = self.recording.probe.channels

        for idx, spike in enumerate(self.spikes):
            start, end = self.get_spike_start_end(spike)

            temp_fit = self._template_fitting[idx]
            jitter = self._template_jitter[idx]

            if temp_fit == 0:
                continue

            fitted_waveform = self.template.get_fitted_waveform(temp_fit,
                                                                jitter,
                                                                self._upsampling_factor)

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
            jitter = self._template_jitter[idx]

            if temp_fit == 0:
                continue

            fitted_waveform = self.template.get_fitted_waveform(temp_fit,
                                                                jitter,
                                                                self._upsampling_factor)

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
            self._energy_sorted_idxs = np.argsort(self._template_fitting)

        return self._energy_sorted_idxs

    def get_energy_sorted_spikes(self):
        """ Return energy sorted spikes

        This method requires that the energies have been initialized already
        """
        return self.spikes[self.get_energy_sorted_idxs()]

    def get_energy_sorted_fittings(self):
        return self._template_fitting[self.get_energy_sorted_idxs()]

    def get_automatic_energy_bounds(self, C=0.5):
        """ Return the lower and upper index obtained from robust
        statistics estimated on the logarithm of the template fitting.

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
        pcts = np.percentile(np.log10(self._template_fitting), [25.0, 75.0])

        Q1 = pcts[0]
        Q3 = pcts[1]

        IQR = Q3 - Q1

        lower_bound = Q1 - C * IQR
        upper_bound = Q3 + C * IQR

        energy_sorted_energy = np.log10(self._template_fitting)[self.get_energy_sorted_idxs()]

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

    def forget_recording(self):
        """ Forget about the current recording associated with self
        """
        self.recording = None

    def add_recording(self, recording):
        """ Add the given recording object to self
        """
        self.recording = recording

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

            # skip off probe locations apart from the edges
            if geo_x < spike_train.recording.probe.x_min - x_between:
                continue
            if geo_x > spike_train.recording.probe.x_max + x_between:
                continue
            if geo_y < spike_train.recording.probe.y_min - y_between:
                continue
            if geo_y > spike_train.recording.probe.y_max + y_between:
                continue

            # determine if extrapolation is needed
            if geo_x < spike_train.recording.probe.x_min or \
                geo_x > spike_train.recording.probe.x_max or \
                    geo_y < spike_train.recording.probe.y_min or \
                        geo_y > spike_train.recording.probe.y_max:
                extrapolate = True
            else:
                extrapolate = False

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
                if extrapolate:
                    interpolation_count *= 2
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

    def get_fitted_waveform(self, template_fit, jitter, upsampling_factor):
        """ Return fitted waveform
        """
        return template_fit * self.get_jittered(jitter, upsampling_factor)

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

    def get_jittered(self, jitter, upsampling_factor):
        """ return a jittered template
        """
        assert jitter < upsampling_factor, "jitter should be less than upsampling factor"

        template_data = self.get_template_data()
        duration = template_data.shape[1]

        # extend data with a zero at the left
        template_data_ext = np.concatenate((np.zeros((template_data.shape[0],1)),
                                            template_data),
                                           axis=1)

        idxs = np.arange(duration) * upsampling_factor + jitter + upsampling_factor
        idxs = idxs.astype(np.int)

        upsampled_template = sg.resample_poly(template_data_ext, upsampling_factor,
                                              1, axis=1)

        return upsampled_template[:,idxs]

    def check_template_edges_to_zero(self, tol=0.1):
        """ Check whether the mean edges of the template are sufficiently close
        to zero, sufficiently close is determined as the given tolerance (tol)
        times the mean absolute maximum of the template.
        """
        active_channels = self.get_template_data().sum(axis=1) != 0
        active_slice = np.abs(self.get_template_data()[active_channels])

        template_max = active_slice.max(axis=1).mean()

        left_boundaries = np.abs(active_slice[:,0])
        left_check = left_boundaries.mean() < (tol * template_max)

        right_boundaries = np.abs(active_slice[:,-1])
        right_check = right_boundaries.mean() < (tol * template_max)

        return left_check and right_check
