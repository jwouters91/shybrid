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

    def calculate_template(self, window_size=100):
        """ Calculate a template for this spike train for the given discrete
        window size
        """
        self.template = Template(self, window_size)

    def get_nb_spikes(self):
        """ Return the number of spikes in the spike train
        """
        return self.spikes.size

    def subtract_train(self):
        """ Subtract spike train from the recording.
        """
        # keep track of the fitting factor for later insertion
        self._template_fitting = np.zeros(self.spikes.shape)
        self._residual_fitting = np.zeros(self.spikes.shape)

        channels = self.recording.probe.channels

        for idx, spike in enumerate(self.spikes):
            start, end = self.get_spike_start_end(spike)
            chunk = self.recording.get_good_chunk(start, end)

            # fit template to spike
            temp_fit, res_fit = self.template.fit(chunk)

            self._template_fitting[idx] = temp_fit
            self._residual_fitting[idx] = res_fit

            fitted_waveform = self.template.get_fitted_waveform(temp_fit,
                                                                res_fit)

            # subtract
            self.recording.data[channels, start:end] = \
                 self.recording.data[channels, start:end] - fitted_waveform

    def insert_train(self, use_fit=True, permutate_channels=True):
        """ Re-insert spike train with the desired randomness
        """
        pass

    def get_spike_start_end(self, spike):
        """ Get start and end for the given spike
        """
        start = int(spike - self.template.window_size / 2)
        end = int(start + self.template.window_size)

        return start, end


class Template:
    """ Template class

    Args:
        spike_train (SpikeTrain): spike train

        window_size (int): window size used to determine te template

        realign (boolean, optional): realign spikes based on template matched
        filter
    """

    def __init__(self, spike_train, window_size, realign=False):
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

        # TODO implement realignment based on template if deemed necessary

        # project snippets into space orthogonal to template
        for spike_idx in range(spike_train.get_nb_spikes()):
            fit = self._fit_template(spike_tensor[spike_idx])
            spike_tensor[spike_idx] -= fit * self.data

        # calculate first PC for fitting
        spike_matrix = spike_tensor.reshape((spike_tensor.shape[0],
                                             spike_tensor.shape[1]*
                                             spike_tensor.shape[2]))

        spike_cov = np.cov(spike_matrix.T)
        _, PCs = la.eigs(spike_cov, k=1) # makes use of spare linalg

        self.PC = PCs[:,0].reshape((spike_tensor.shape[1],
                                    spike_tensor.shape[2]))

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

    def get_fitted_waveform(self, template_fit, residual_fit):
        """ Return fitted waveform
        """
        return template_fit * self.data + residual_fit * self.PC
