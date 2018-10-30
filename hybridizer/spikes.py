#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 13:52:29 2018

@author: jwouters
"""

import numpy as np
import scipy.sparse.linalg as la

class SpikeTrain:
    """ Spike train class grouping spikes and recording

    Args:
        recording (Recording): recording object related to this spike train

        spike_times (ndarray): array containing starting times of every spike
    """

    def __init__(self, recording, spike_times):
        self.recording = recording
        self.spikes = spike_times
        self.permutation= None

        # init this cache, sorting only once on first request
        self._energy_sorted_idxs = None

    def calculate_template(self, window_size=100, realign=False):
        """ Calculate a template for this spike train for the given discrete
        window size
        """
        self.template = Template(self, window_size, realign=realign)

    def get_nb_spikes(self):
        """ Return the number of spikes in the spike train
        """
        return self.spikes.size

    def fit_spikes(self, use_PC=False):
        """ Calculate spike fits
        """
        self._template_fitting = np.zeros(self.spikes.shape)
        if use_PC: self._residual_fitting = np.zeros(self.spikes.shape)
        self._fitting_energy = np.zeros(self.spikes.shape)

        for idx, spike in enumerate(self.spikes):
            start, end = self.get_spike_start_end(spike)
            chunk = self.recording.get_good_chunk(start, end)

            # fit template to spike
            if use_PC:
                temp_fit, res_fit = self.template.fit(chunk)
                self._residual_fitting[idx] = res_fit
            else:
                temp_fit = self.template._fit_template(chunk)

            self._template_fitting[idx] = temp_fit

            # fitting energy only based on template
            self._fitting_energy[idx] = temp_fit**2

    def subtract_train(self, use_PC=False):
        """ Subtract spike train from the recording.
        """
        channels = self.recording.probe.channels
        for idx, spike in enumerate(self.spikes):
            start, end = self.get_spike_start_end(spike)

            temp_fit = self._template_fitting[idx]
            if use_PC:
                res_fit = self._residual_fitting[idx]
                fitted_waveform = self.template.get_fitted_waveform(temp_fit,
                                                                    res_fit)
            else:
                fitted_waveform = self.template.get_fitted_waveform(temp_fit)

            # subtract
            self.recording.data[channels, start:end] = \
                 self.recording.data[channels, start:end] - fitted_waveform

    def insert_train(self, use_fit=True, permutate_channels=True,
                     spatial_map='random'):
        """ Re-insert spike train with the desired randomness
        """
        if permutate_channels and self.permutation is None:
            if spatial_map == 'random':
                self.permutation = \
                    np.random.permutation(self.template.data.shape[0])
            elif spatial_map == 'reverse':
                self.permutation = np.arange(self.template.data.shape[0])
                self.permutation = np.flip(self.permutation, 0)
        else:
            self.permutation = np.arange(self.template.data.shape[0])

        # good channels
        channels = self.recording.probe.channels

        for spike_idx in range(self.get_nb_spikes()):
            if use_fit:
                insert_waveform = \
                    self.template.get_fitted_waveform(self._template_fitting[spike_idx],
                                                      self._residual_fitting[spike_idx])
                insert_waveform = insert_waveform[self.permutation]
            else:
                insert_waveform = self.template.data[self.permutation]

            start, end = self.get_spike_start_end(self.spikes[spike_idx])

            self.recording.data[channels, start:end] = \
                 self.recording.data[channels, start:end] + insert_waveform

    def insert_given_train(self, spike_times, template, template_fitting,
                           residual=None, residual_fitting=None):
        """ Insert given spike train

        Returns:
            inserted_spikes (ndarray): spike that were actually inserted
        """
        channels = self.recording.probe.channels

        inserted_spikes = np.array([])
        for idx, spike_t in enumerate(spike_times):
            temp_fit = template_fitting[idx]
            if residual is not None: res_fit = residual_fitting[idx]

            insert_spike = temp_fit * template

            if residual is not None:
                insert_spike += res_fit * residual_fitting 

            start, end = self.get_spike_start_end(spike_t)

            if end > self.recording.get_duration():
                continue

            self.recording.data[channels, start:end] = \
                 self.recording.data[channels, start:end] + insert_spike

            inserted_spikes = np.append(inserted_spikes, spike_t)

        inserted_spikes.sort()
        return inserted_spikes

    def update(self, spike_times):
        """ Update spike times and trigger resets
        """
        self.spikes = spike_times

        self._template_fitting = None
        self._residual_fitting = None
        self._fitting_energy = None
        self._energy_sorted_idxs = None

        self.calculate_template(window_size=self.template.window_size)

    def get_spike_start_end(self, spike):
        """ Get start and end for the given spike
        """
        start = int(spike - self.template.window_size / 2)
        end = int(start + self.template.window_size)

        return start, end

    def realign(self, offsets):
        """ Realign spikes according to the given offsets
        """
        self.spikes = self.spikes + offsets

    def retrieve_energy_sorted_spike_time(self, spike_idx):
        """ Return the spike time from the energy sorted spike train for the
        given spike_idx

        This method requires that the energies have been initialized already
        """
        return self.retrieve_energy_sorted_spikes()[spike_idx]

    def retrieve_energy_sorted_spikes(self):
        """ Return energy sorted spikes

        This method requires that the energies have been initialized already
        """
        if self._energy_sorted_idxs is None:
            self._energy_sorted_idxs = np.argsort(self._fitting_energy)

        return self.spikes[self._energy_sorted_idxs]

    def get_energy_sorted_idxs(self):
        """ Return energy sorted indices

        This method requires that the energies have been initialized already
        """
        if self._energy_sorted_idxs is None:
            self._energy_sorted_idxs = np.argsort(self._fitting_energy)

        return self._energy_sorted_idxs


class Template:
    """ Template class

    Args:
        spike_train (SpikeTrain): spike train

        window_size (int): window size used to determine te template

        realign (boolean, optional): realign spikes based on template matched
        filter. Realign will also affect the spike train given to this
        function.
    """

    def __init__(self, spike_train, window_size, realign=False,
                 force_zero=True, calculate_PC=False):
        self.window_size = window_size

        # build spike tensor
        spike_tensor = np.empty((spike_train.get_nb_spikes(),
                                 spike_train.recording.get_nb_good_channels(),
                                 self.window_size))

        for spike_idx in range(spike_train.get_nb_spikes()):
            start = int(spike_train.spikes[spike_idx] - window_size/2)
            end = int(start + self.window_size)
            spike_tensor[spike_idx] =\
                spike_train.recording.get_good_chunk(start,
                                                     end)

        self.data = np.median(spike_tensor, axis=0)

        # set every channel to zero that has less than 1% of max channel energy
        if force_zero:
            energy = self.data**2
            energy = np.sum(energy, axis=1)
            self.data[energy<0.01*energy.max(),:] = 0

        if realign:
            offsets = np.zeros(spike_train.get_nb_spikes())
            for spike_idx in range(spike_train.get_nb_spikes()):
                start = int(spike_train.spikes[spike_idx] - window_size/2)
                end = int(start + self.window_size)

                chunk =  spike_train.recording.get_good_chunk(start,
                                                              end)

                # align on shorter window to avoid boundary effects
                sub_start = int((self.window_size/2) - (self.window_size * 0.1))
                sub_end = int(sub_start + self.window_size * 0.2)

                sub_template = self.data[:,sub_start:sub_end]
                sub_chunk = chunk[:,sub_start:sub_end]

                # find most dominant signal minima in sub template
                template_min = np.min(sub_template, axis=1)
                sorted_idxs = np.argsort(template_min)
                sorted_idxs = sorted_idxs[:int(0.1*sub_template.shape[0])]

                # template dips
                template_dips = np.argmin(sub_template[sorted_idxs], axis=1)
                chunk_dips = np.argmin(sub_chunk[sorted_idxs], axis=1)

                offset = np.round(np.mean(chunk_dips - template_dips))

                offsets[spike_idx] = offset

            # save offsets for debugging purposes
            self.offsets = offsets

            # correct spike train
            spike_train.realign(offsets)

            # rebuild spike tensor
            spike_tensor = np.empty((spike_train.get_nb_spikes(),
                                     spike_train.recording.get_nb_good_channels(),
                                     self.window_size))

            for spike_idx in range(spike_train.get_nb_spikes()):
                start = int(spike_train.spikes[spike_idx] - window_size/2)
                end = int(start + self.window_size)
                spike_tensor[spike_idx] =\
                    spike_train.recording.get_good_chunk(start,
                                                         end)

        if calculate_PC:
            self.calculate_PC(spike_train, spike_train.spikes)

    def calculate_PC(self, spike_train, selected_spikes):
        """ Calculate principle component using only the given spikes
        """
        # build spike tensor
        spike_tensor = np.empty((selected_spikes.size,
                                 self.data.shape[0],
                                 self.window_size))

        for idx, spike in enumerate(selected_spikes):
            start = int(spike - self.window_size/2)
            end = int(start + self.window_size)
            spike_tensor[idx] =\
                spike_train.recording.get_good_chunk(start,
                                                     end)
        # project snippets into space orthogonal to template
        for idx, spike in enumerate(selected_spikes):
            fit = self._fit_template(spike_tensor[idx])
            spike_tensor[idx] -= fit * self.data

        # calculate first PC for fitting
        spike_matrix = spike_tensor.reshape((spike_tensor.shape[0],
                                             spike_tensor.shape[1]*
                                             spike_tensor.shape[2]))

        spike_cov = np.cov(spike_matrix.T)
        _, PCs = la.eigs(spike_cov, k=1) # makes use of spare linalg

        self.PC = PCs[:,0].reshape((spike_tensor.shape[1],
                                    spike_tensor.shape[2]))
        self.PC = np.real(self.PC)

    def calculate_shifted_template(self, spike_train, x_shift, y_shift,
                                   shifted_PC=False):
        """ Calculate shifted template
        """
        shifted_template = np.zeros(spike_train.template.data.shape)
        if shifted_PC: shifted_PC = np.zeros(spike_train.template.PC.shape)

        # TODO extract from data
        x_between = 1
        y_between = 1

        for idx, channel in enumerate(spike_train.recording.probe.channels):
            geo = spike_train.recording.probe.geometry[channel]

            # find location that projects on this channel (that's why minus)
            geo_x = geo[0] - x_shift * x_between
            geo_y = geo[1] - y_shift * y_between

            interpolated_waveform = np.zeros(spike_train.template.data[0].shape)
            if shifted_PC: interpolated_PC = np.zeros(spike_train.template.PC[0].shape)

            interpolation_count = 0
            interpolation_needed = True
            for jdx, project_channel in enumerate(spike_train.recording.probe.channels):
                project_geo = spike_train.recording.probe.geometry[project_channel]

                if geo_x == project_geo[0] and geo_y == project_geo[1]:
                    shifted_template[idx] = spike_train.template.data[jdx]
                    if shifted_PC: shifted_PC[idx] = spike_train.template.PC[jdx]
                    interpolation_needed = False
                else:
                    if abs(geo_x - project_geo[0]) <= x_between and abs(geo_y - project_geo[1]) <= y_between:
                        interpolated_waveform += spike_train.template.data[jdx]
                        if shifted_PC: interpolated_PC += spike_train.template.PC[jdx]
                        interpolation_count += 1

            if interpolation_needed and interpolation_count > 0:
                shifted_template[idx] = interpolated_waveform / interpolation_count
                if shifted_PC: shifted_PC[idx] = interpolated_PC / interpolation_count

        # set those shifted waveforms as class variables for reuse
        self.shifted_template = shifted_template
        if shifted_PC: self.shifted_PC = shifted_PC

    def _fit_template(self, chunk):
        """ Fit the given chunk to the template
        """
        template_flat = self.data.flatten()
        chunk_flat = chunk.flatten()

        # calculate fit according to yger et al. 2018
        fit = np.dot(chunk_flat, template_flat)
        fit /= np.linalg.norm(template_flat)**2

        return fit

    def _fit_residual(self, residual_chunk):
        """ Fit residual chunk to principal component
        """
        PC_flat = self.PC.flatten()
        chunk_flat = residual_chunk.flatten()

        # calculate fit according to yger et al. 2018
        fit = np.dot(chunk_flat, PC_flat)
        fit /= np.linalg.norm(PC_flat)**2

        return fit

    def fit(self, chunk):
        """ Fit both template and residual
        """
        # work on copy because ndarray is pass by reference
        chunk = chunk.copy()

        template_fit = self._fit_template(chunk)
        chunk -= template_fit * self.data

        residual_fit = self._fit_residual(chunk)

        return template_fit, residual_fit

    def get_fitted_waveform(self, template_fit, residual_fit=None):
        """ Return fitted waveform
        """
        if residual_fit is None:
            return template_fit * self.data
        else:
            return template_fit * self.data + residual_fit * self.PC
