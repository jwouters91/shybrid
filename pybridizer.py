#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 13:36:01 2018

@author: jwouters
"""

import sys
import os

import yaml
from PyQt5 import QtWidgets
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from hybridizer.ui import design
from hybridizer.io import Recording, Phy
from hybridizer.spikes import SpikeTrain

CHOOSE_CLUSTER = 'select cluster'
TEMPLATE_COLOR = '#1F77B4'
SPIKE_COLOR = '#E44A3D'
MOVE_COLOR = 'g--'

class Pybridizer(QtWidgets.QMainWindow, design.Ui_Pybridizer):
    def __init__(self, parent=None):
        super(self.__class__, self).__init__(parent)
        self.setupUi(self)

        # helper variables
        self._select_path = os.path.expanduser('~')
        self.GUI_enabled = True

        # connect listeners
        self.btnDataSelect.clicked.connect(self.select_data)
        self.listClusterSelect.activated.connect(self.select_cluster)
        self.btnDraw.clicked.connect(lambda: self.draw(calcTemp=True))
        # don't recalculate template
        self.radioTemplate.clicked.connect(lambda: self.draw(calcTemp=False))
        self.radioFit.clicked.connect(self.template_fit)
        self.btnLeftSpike.clicked.connect(self.lower_spike)
        self.btnRightSpike.clicked.connect(self.increase_spike)
        self.horizontalSlider.valueChanged.connect(self.slide_spike)
        self.radioMove.clicked.connect(self.move_template)
        self.btnMoveLeft.clicked.connect(self.move_left)
        self.btnMoveRight.clicked.connect(self.move_right)
        self.btnMoveUp.clicked.connect(self.move_up)
        self.btnMoveDown.clicked.connect(self.move_down)
        self.btnReset.clicked.connect(self.move_template)

        #set up plotting area
        canvas = FigureCanvas(plt.figure())
        #canvas.figure.patch.set_facecolor(self.palette().color(10).getRgbF())
        #pickcid needed later to disconnect and reconnect when selecting for removal
        #self.__pickcid = canvas.mpl_connect('pick_event',self.pickHandler)
        self.axes = canvas.figure.add_subplot(111)
        self.axes.spines['top'].set_visible(False)
        self.axes.spines['right'].set_visible(False)
        self.axes.spines['bottom'].set_visible(False)
        self.axes.spines['left'].set_visible(False)
        self.axes.yaxis.set_ticks([])
        self.axes.xaxis.set_ticks([])
        canvas.figure.subplots_adjust(0,0,1,1,0,0)

        # set size policy for the plot object
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                           QtWidgets.QSizePolicy.Expanding)
        canvas.setSizePolicy(sizePolicy)

        # add canvas widget to GUI
        self.plotCanvas.addWidget(canvas)

    def select_data(self):
        raw_fn, _ = QtWidgets.QFileDialog.\
            getOpenFileName(self,
                            'select raw recording',
                            directory=self._select_path,
                            filter='raw recording (*.raw *.bin)')

        if raw_fn != "": # empty string is returned if cancel is pushed
            # update select path for user convenience
            self._select_path, _ = os.path.split(raw_fn)

            # parse config file
            config_fn, _ = os.path.splitext(raw_fn)
            config_fn = config_fn + '.yaml'

            if os.path.isfile(config_fn):
                with open(config_fn, 'r') as f:
                    config = yaml.load(f)

                data_conf = config['data']

                # initialise program objects
                self.recording = Recording(raw_fn, data_conf['probe'],
                                           data_conf['fs'], data_conf['dtype'],
                                           order=data_conf['order'])
                self.phy = Phy(config['phy'])
            else:
                # throw message box if config file does not exist
                QtWidgets.QMessageBox.critical(self, 'config file not found',
                                               "The program expected to find {}, but couldn't.".\
                                               format(config_fn))
                return

            # update dropdown
            self.fill_cluster_dropdown()

    def fill_cluster_dropdown(self):
        good_clusters = self.phy.get_good_clusters()
        good_clusters = [CHOOSE_CLUSTER] + np.sort(good_clusters).astype('str').tolist()
        self.listClusterSelect.clear()
        self.listClusterSelect.addItems(good_clusters)

    def select_cluster(self):
        label = self.listClusterSelect.currentText()
        if label != CHOOSE_CLUSTER:            
            self._current_cluster = int(label)
            self.btnDraw.setEnabled(True)

    def draw(self, calcTemp=True):
        if self.fieldWindowSize.text() == '':
             QtWidgets.QMessageBox.critical(self, 'no window size',
                                            'Please provide the desired spike window size')
        else:
            if calcTemp:
                self.disable_GUI()

                # TODO provide more check to validate that the given value is a double
                # TODO bis clear canvas
                self.spike_times = self.phy.get_cluster_activation(self._current_cluster)
                self.spikeTrain = SpikeTrain(self.recording, self.spike_times)

                # calculate template and show
                self._window_samples = int(np.ceil(float(self.fieldWindowSize.text()) / 1000 * self.recording.sampling_rate))
                self.spikeTrain.calculate_template(realign=True, window_size=self._window_samples)

                self.enable_GUI()

            # draw template
            self.axes.clear()
            self._template_scaling = self.plot_multichannel(self.spikeTrain.template.data)
            self.axes.autoscale(False)

            self.plotTitle.setText('Cluster {}'.format(self._current_cluster))

            # enable options for further use
            self.radioTemplate.setChecked(True)
            self.radioFit.setEnabled(True)
            self.radioMove.setEnabled(True)
            self.radioTemplate.setEnabled(True)

            self.template_fit_enabled(False)
            self.move_template_enabled(False)

    def template_fit(self):
        # plot first spike fitted on template
        self._current_spike = 0
        self.render_current_spike()

        # activate and set slider
        self.horizontalSlider.setMaximum(self.spikeTrain.spikes.size-1)
        self.template_fit_enabled(True)
        self.move_template_enabled(False)

    def template_fit_enabled(self, enabled):
        self.btnLeftSpike.setEnabled(enabled)
        self.btnRightSpike.setEnabled(enabled)
        self.horizontalSlider.setEnabled(enabled)
        self.horizontalSlider.valueChanged.disconnect()
        self.horizontalSlider.setValue(0)
        self.horizontalSlider.valueChanged.connect(self.slide_spike)

        if enabled == True:
            self.labelSpike.setText('1/{} '.format(int(self.spikeTrain.spikes.size)))
        else:
            self.labelSpike.setText('')

        self.btnExport.setEnabled(False)
        self.btnReset.setEnabled(False)

    def render_current_spike(self):
        self.axes.lines = []

        spike_time = self.spikeTrain.spikes[self._current_spike]
        start, end = self.spikeTrain.get_spike_start_end(spike_time)
        spike = self.recording.get_good_chunk(start,end)

        # init fit factors
        self.spikeTrain.subtract_train(fitOnly=True)
        fit = self.spikeTrain._template_fitting[self._current_spike]

        self.plot_multichannel(spike, color=SPIKE_COLOR, scaling=self._template_scaling/fit)
        self.plot_multichannel(self.spikeTrain.template.data)

    def move_template(self):
        self.template_fit_enabled(False)
        self.move_template_enabled(True)

        self.x_shift = 0
        self.y_shift = 0

        self.render_shifted_template()

    def move_left(self):
        self.x_shift -= 1
        self.render_shifted_template()
        self.btnReset.setEnabled(True)
        self.btnExport.setEnabled(True)

    def move_right(self):
        self.x_shift += 1
        self.render_shifted_template()
        self.btnReset.setEnabled(True)
        self.btnExport.setEnabled(True)

    def move_up(self):
        self.y_shift += 1
        self.render_shifted_template()
        self.btnReset.setEnabled(True)
        self.btnExport.setEnabled(True)

    def move_down(self):
        self.y_shift -= 1
        self.render_shifted_template()
        self.btnReset.setEnabled(True)
        self.btnExport.setEnabled(True)

    def move_template_enabled(self, enabled):
        self.btnMoveLeft.setEnabled(enabled)
        self.btnMoveUp.setEnabled(enabled)
        self.btnMoveRight.setEnabled(enabled)
        self.btnMoveDown.setEnabled(enabled)

        self.btnExport.setEnabled(False)
        self.btnReset.setEnabled(False)

    def calculate_shifted_template(self):
        # init template
        shifted_template = np.zeros(self.spikeTrain.template.data.shape)

        # TODO extract from data
        x_between = 1
        y_between = 1

        for idx, channel in enumerate(self.recording.probe.channels):
            geo = self.recording.probe.geometry[channel]

            # find location that projects on this channel (that's why minus)
            geo_x = geo[0] - self.x_shift * x_between
            geo_y = geo[1] - self.y_shift * y_between

            interpolated_waveform = np.zeros(self.spikeTrain.template.data[0].shape)
            interpolation_count = 0
            interpolation_needed = True
            for jdx, project_channel in enumerate(self.recording.probe.channels):
                project_geo = self.recording.probe.geometry[project_channel]

                if geo_x == project_geo[0] and geo_y == project_geo[1]:
                    shifted_template[idx] = self.spikeTrain.template.data[jdx]
                    interpolation_needed = False
                else:
                    if abs(geo_x - project_geo[0]) <= x_between and abs(geo_y - project_geo[1]) <= y_between:
                        interpolated_waveform += self.spikeTrain.template.data[jdx]
                        interpolation_count += 1

            if interpolation_needed and interpolation_count > 0:
                shifted_template[idx] = interpolated_waveform / interpolation_count

        return shifted_template

    def render_shifted_template(self):
        self.axes.lines = []
        #self.plot_multichannel(self.spikeTrain.template.data)

        # calculate template permutation
        shifted_template = self.calculate_shifted_template()
        self.plot_multichannel(shifted_template, color=MOVE_COLOR)

    def plot_multichannel(self, data, color=TEMPLATE_COLOR, scaling=None):
        """ Plot multichannel data on the figure canvas
        """
        min_geo = self.recording.probe.get_min_geometry()
        max_geo = self.recording.probe.get_max_geometry()

        # probe info
        x_bias = -min_geo[0]
        y_bias = -min_geo[1]

        x_range = max_geo[0] - min_geo[0]
        y_range = max_geo[1] - min_geo[1]

        # TODO extract from data
        x_between = 1
        y_between = 1

        # data info
        min_dat = data.min()
        max_dat = data.max()

        dat_range = 0.5 * (max_dat - min_dat)
        
        if dat_range == 0:
            # everything zero
            dat_range = 1

        if scaling is None:
            scaling = y_between / (dat_range * y_range)

        # loop over channels
        for idx, channel in enumerate(self.recording.probe.channels):
            geo = self.recording.probe.geometry[channel]

            x_start = (geo[0] - x_bias) / x_range
            x_end = x_start + 0.9*(x_between/x_range)
            time = np.linspace(x_start, x_end, num=data.shape[1])

            y_start = (geo[1] - y_bias) / y_range
            signal = data[idx] * scaling
            signal += y_start

            self.axes.plot(time, signal, color)

        # draw
        self.axes.figure.canvas.draw()

        return scaling

    def lower_spike(self):
        if self._current_spike > 0:
            self.horizontalSlider.setValue(self._current_spike-1)
            # spike is rendered by slider listener

    def increase_spike(self):
        if self._current_spike < self.spikeTrain.spikes.size - 1:
            self.horizontalSlider.setValue(self._current_spike+1)
             # spike is rendered by slider listener

    def slide_spike(self):
        self._current_spike = self.horizontalSlider.value()
        self.labelSpike.setText('{}/{} '.format(int(self._current_spike)+1,
                                                int(self.spikeTrain.spikes.size)))
        self.render_current_spike()

    def build_GUI_status_dict(self):
        # empty previous one or init
        self.GUI_status = {}

        # fill GUI status dictionary
        self.GUI_status['btnDataSelect'] = self.btnDataSelect.isEnabled()
        self.GUI_status['listClusterSelect'] = self.listClusterSelect.isEnabled()
        self.GUI_status['fieldWindowSize'] = self.fieldWindowSize.isEnabled()
        self.GUI_status['btnDraw'] = self.btnDraw.isEnabled()
        self.GUI_status['radioTemplate'] = self.radioTemplate.isEnabled()
        self.GUI_status['radioFit'] = self.radioFit.isEnabled()
        self.GUI_status['btnLeftSpike'] = self.btnLeftSpike.isEnabled()
        self.GUI_status['btnRightSpike'] = self.btnRightSpike.isEnabled()
        self.GUI_status['radioMove'] = self.radioMove.isEnabled()
        self.GUI_status['btnMoveLeft'] = self.btnMoveLeft.isEnabled()
        self.GUI_status['btnMoveUp'] = self.btnMoveUp.isEnabled()
        self.GUI_status['btnMoveRight'] = self.btnMoveRight.isEnabled()
        self.GUI_status['btnMoveDown'] = self.btnMoveDown.isEnabled()
        self.GUI_status['btnReset'] = self.btnReset.isEnabled()
        self.GUI_status['btnExport'] = self.btnExport.isEnabled()
        self.GUI_status['horizontalSlider'] = self.horizontalSlider.isEnabled()

    def disable_GUI(self):
        if self.GUI_enabled:
            # prevent losing status dict
            self.GUI_enabled = False

            # build dict
            self.build_GUI_status_dict()

            # disable GUI
            self.btnDataSelect.setEnabled(False)
            self.listClusterSelect.setEnabled(False)
            self.fieldWindowSize.setEnabled(False)
            self.btnDraw.setEnabled(False)
            self.radioTemplate.setEnabled(False)
            self.radioFit.setEnabled(False)
            self.btnLeftSpike.setEnabled(False)
            self.btnRightSpike.setEnabled(False)
            self.radioMove.setEnabled(False)
            self.btnMoveLeft.setEnabled(False)
            self.btnMoveUp.setEnabled(False)
            self.btnMoveRight.setEnabled(False)
            self.btnMoveDown.setEnabled(False)
            self.btnReset.setEnabled(False)
            self.btnExport.setEnabled(False)
            self.horizontalSlider.setEnabled(False)

            # for redraw
            self.repaint()

    def enable_GUI(self):
        self.btnDataSelect.setEnabled(self.GUI_status['btnDataSelect'])
        self.listClusterSelect.setEnabled(self.GUI_status['listClusterSelect'])
        self.fieldWindowSize.setEnabled(self.GUI_status['fieldWindowSize'])
        self.btnDraw.setEnabled(self.GUI_status['btnDraw'])
        self.radioTemplate.setEnabled(self.GUI_status['radioTemplate'])
        self.radioFit.setEnabled(self.GUI_status['radioFit'])
        self.btnLeftSpike.setEnabled(self.GUI_status['btnLeftSpike'])
        self.btnRightSpike.setEnabled(self.GUI_status['btnRightSpike'])
        self.radioMove.setEnabled(self.GUI_status['radioMove'])
        self.btnMoveLeft.setEnabled(self.GUI_status['btnMoveLeft'])
        self.btnMoveUp.setEnabled(self.GUI_status['btnMoveUp'])
        self.btnMoveRight.setEnabled(self.GUI_status['btnMoveRight'])
        self.btnMoveDown.setEnabled(self.GUI_status['btnMoveDown'])
        self.btnReset.setEnabled(self.GUI_status['btnReset'])
        self.btnExport.setEnabled(self.GUI_status['btnExport'])
        self.horizontalSlider.setEnabled(self.GUI_status['horizontalSlider'])

        self.GUI_enabled = True


def main():
    app = QtWidgets.QApplication(sys.argv)

    form = Pybridizer()
    form.show()
    app.exec_()

if __name__ == '__main__':
    main()
