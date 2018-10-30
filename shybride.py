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
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from hybridizer.ui import design
from hybridizer.io import Recording, SpikeClusters, Phy
from hybridizer.spikes import SpikeTrain

CHOOSE_CLUSTER = 'select cluster'
TEMPLATE_COLOR = '#1F77B4'
SPIKE_COLOR = 'salmon'
MOVE_COLOR = '#1F77B4'
ENERGY_COLOR = 'salmon'
INACTIVE_COLOR = 'gray'
FLAT_COLOR = 'lightgray'
MARKER_COLOR = 'darkslategray'

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
        self.btnDraw.clicked.connect(lambda: self.draw_template(calcTemp=True))
        # don't recalculate template
        self.radioTemplate.clicked.connect(lambda: self.draw_template(calcTemp=False))
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
        self.checkBoxLower.clicked.connect(lambda: self.set_energy_lb(draw=True))
        self.checkBoxUpper.clicked.connect(lambda: self.set_energy_ub(draw=True))
        self.btnMove.clicked.connect(self.execute_move)
        self.btnExport.clicked.connect(self.export_data)

        self.btnResetZoom.clicked.connect(self.reset_view_plot)
        self.btnZoom.clicked.connect(self.zoom_plot)
        self.btnPan.clicked.connect(self.pan_plot)

        self.btnZoom.setCheckable(True)
        self.btnPan.setCheckable(True)

        #set up plotting area
        canvas = FigureCanvas(plt.figure())
        self.toolbar = NavigationToolbar(canvas, self)
        self.toolbar.hide()

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


    """
    Methods related to data loading and cluster selection
    """

    def select_data(self):
        """ Open the selected data and load recording and clusters
        """
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

            # assuming that params file is correct (TODO add explicit checks)
            if os.path.isfile(config_fn):
                # reset/init file related state
                self.reset_file_session_variables()

                with open(config_fn, 'r') as f:
                    config = yaml.load(f)

                data_conf = config['data']
                data_clus = config['clusters']

                # initialise program objects
                self.recording = Recording(raw_fn, data_conf['probe'],
                                           data_conf['fs'], data_conf['dtype'],
                                           order=data_conf['order'])

                for clus_mode in data_clus.keys():
                    # load cluster information from phy
                    if clus_mode == 'phy':
                        phy = Phy(data_clus[clus_mode])
                        self.clusters = SpikeClusters()
                        self.clusters.fromPhy(phy)
                    # load cluster information from csv
                    elif clus_mode == 'csv':
                        self.clusters = SpikeClusters()
                        self.clusters.fromCSV(data_clus[clus_mode])

                self.fill_cluster_dropdown()

                # init generated GT
                self.generated_GT = SpikeClusters()

                self.reset_GUI_initial()

            else:
                # throw message box if config file does not exist
                QtWidgets.QMessageBox.critical(self, 'config file not found',
                                               "The program expected to find {}, but couldn't.".\
                                               format(config_fn))

    def fill_cluster_dropdown(self):
        """ Fill the cluster select dropdown menu
        """
        good_clusters = self.clusters.keys().tolist()
        self.good_clusters = [CHOOSE_CLUSTER] + np.sort(good_clusters).astype('str').tolist()
        self.listClusterSelect.clear()
        self.listClusterSelect.addItems(self.good_clusters)

    def select_cluster(self):
        """ Select cluster from dropdown menu
        """
        label = self.listClusterSelect.currentText()
        if label != CHOOSE_CLUSTER:            
            self._current_cluster = int(label)
            self.btnDraw.setEnabled(True)


    """
    Methods related to the template only view
    """

    def draw_template(self, calcTemp=True):
        """ (Calculate and) draw template
        """
        # check if given window converts to a float
        try:
            float(self.fieldWindowSize.text())
            window_is_float = True
        except:
            window_is_float = False

        # if no given window size or invalid window size, raise error message
        if self.fieldWindowSize.text() == '':
             QtWidgets.QMessageBox.critical(self, 'No window size',
                                            'Please provide the desired spike window size.')
        elif not window_is_float:
             QtWidgets.QMessageBox.critical(self, 'Invalid window size',
                                            'Please provide a valid spike window size.')
        else:
            if calcTemp:
                self.disable_GUI()

                if self._current_cluster in self.generated_GT.keys():
                    spike_times = self.generated_GT[self._current_cluster]
                else:
                    # use external
                    spike_times = self.clusters[self._current_cluster]
                self.spikeTrain = SpikeTrain(self.recording, spike_times)

                # calculate template
                self._window_samples = int(np.ceil(float(self.fieldWindowSize.text()) / 1000 * self.recording.sampling_rate))
                self.spikeTrain.calculate_template(window_size=self._window_samples)

                # init fit factors
                self.spikeTrain.fit_spikes()

                self.reset_energy_bounds()

                self.enable_GUI()

            # draw template
            self.axes.clear()
            self._template_scaling = self.plot_multichannel(self.spikeTrain.template.data)
            self.axes.autoscale(False)

            # clear removes callbacks
            self.axes.callbacks.connect('ylim_changed', self.lim_changed)
            self.axes.callbacks.connect('xlim_changed', self.lim_changed)

            # enable options for further use
            self.radioTemplate.setChecked(True)
            self.radioTemplate.setEnabled(True)
            self.radioFit.setEnabled(True)

            if self._current_cluster in self.generated_GT.keys():
                self.radioMove.setEnabled(False)
                self.plotTitle.setText('Cluster {} [ALREADY MOVED]'.format(self._current_cluster))
            else:
                self.radioMove.setEnabled(True)
                self.plotTitle.setText('Cluster {}'.format(self._current_cluster))

            self.set_template_fit_enabled(False)
            self.set_move_template_enabled(False)


    """ 
    Methods related to the template fit view
    """

    def template_fit(self):
        """ Switch to template fit plot
        """
        # plot first spike fitted on template
        self._current_spike = 0
        self.render_current_spike()

        # activate and set slider
        self.horizontalSlider.setMaximum(self.spikeTrain.spikes.size-1)
        self.set_template_fit_enabled(True)
        self.set_move_template_enabled(False)

    def set_template_fit_enabled(self, enabled):
        """ Enable or disabled the template fit part of the GUI
        """
        self.btnLeftSpike.setEnabled(enabled)
        self.btnRightSpike.setEnabled(enabled)
        self.horizontalSlider.setEnabled(enabled)
        self.horizontalSlider.valueChanged.disconnect()
        self.horizontalSlider.setValue(0)
        self.horizontalSlider.valueChanged.connect(self.slide_spike)

        # value used for replotting when canvas is being moved (TODO implement
        # solution with less lag (see actual replotting code))
        self.template_fit_active = enabled

        if self._current_cluster in self.generated_GT.keys():
            self.checkBoxLower.setEnabled(False)
            self.checkBoxUpper.setEnabled(False)
        else:
            self.checkBoxLower.setEnabled(enabled)
            self.checkBoxUpper.setEnabled(enabled)

        if enabled == True:
            self.labelSpike.setText('1/{} '.format(int(self.spikeTrain.spikes.size)))
        else:
            self.labelSpike.setText('')

    def render_current_spike(self):
        """ Function that actually plots the current spike and the template
        """
        self.clear_canvas()

        # plot energy bar first
        self.plot_energy()

        # sort using fitting energy
        spike_time = self.spikeTrain.retrieve_energy_sorted_spike_time(self._current_spike)
        start, end = self.spikeTrain.get_spike_start_end(spike_time)
        spike = self.recording.get_good_chunk(start,end)

        # apply sorting also to fitting factors
        sorted_idxs = self.spikeTrain.get_energy_sorted_idxs()
        fit = self.spikeTrain._template_fitting[sorted_idxs][self._current_spike]

        self.plot_multichannel(spike, color=SPIKE_COLOR, scaling=self._template_scaling/fit)
        self.plot_multichannel(self.spikeTrain.template.data)

    def lower_spike(self):
        """ Lower current spike index
        """
        if self._current_spike > 0:
            self.horizontalSlider.setValue(self._current_spike-1)
            # spike is rendered by slider listener

    def increase_spike(self):
        """ Increase current spike index
        """
        if self._current_spike < self.spikeTrain.spikes.size - 1:
            self.horizontalSlider.setValue(self._current_spike+1)
             # spike is rendered by slider listener

    def slide_spike(self):
        """ Change current spike index using the slider
        """
        self._current_spike = self.horizontalSlider.value()
        self.labelSpike.setText('{}/{} '.format(int(self._current_spike)+1,
                                                int(self.spikeTrain.spikes.size)))
        self.render_current_spike()

    def set_energy_lb(self, draw=True):
        """ Set the energy lower bound
        """
        if self.checkBoxLower.isChecked():
            self._energy_LB = self._current_spike
            self.labelLB.setText('{}'.format(int(self._energy_LB+1)))
        else:
            self._energy_LB = None
            self.labelLB.setText('')

        if draw:
            self.render_current_spike()

    def set_energy_ub(self, draw=True):
        """ Set the energy upper bound
        """
        if self.checkBoxUpper.isChecked():
            self._energy_UB = self._current_spike
            self.labelUB.setText('{}'.format(int(self._energy_UB+1)))
        else:
            self._energy_UB = None
            self.labelUB.setText('')

        if draw:
            self.render_current_spike()


    """ 
    Methods related to the move template view
    """

    def move_template(self):
        """ Switch to move template mode in GUI
        """
        self.set_template_fit_enabled(False)
        self.set_move_template_enabled(True)

        self.x_shift = 0
        self.y_shift = 0

        self.render_shifted_template()

    def move_left(self):
        """ Move template left
        """
        self.x_shift -= 1
        self.render_shifted_template()
        self.btnReset.setEnabled(True)
        self.btnMove.setEnabled(True)

    def move_right(self):
        """ Move template right
        """
        self.x_shift += 1
        self.render_shifted_template()
        self.btnReset.setEnabled(True)
        self.btnMove.setEnabled(True)

    def move_up(self):
        """ Move template up
        """
        self.y_shift += 1
        self.render_shifted_template()
        self.btnReset.setEnabled(True)
        self.btnMove.setEnabled(True)

    def move_down(self):
        """ Move template down
        """
        self.y_shift -= 1
        self.render_shifted_template()
        self.btnReset.setEnabled(True)
        self.btnMove.setEnabled(True)

    def set_move_template_enabled(self, enabled):
        """ Enable/disable move template GUI elements
        """
        self.btnMoveLeft.setEnabled(enabled)
        self.btnMoveUp.setEnabled(enabled)
        self.btnMoveRight.setEnabled(enabled)
        self.btnMoveDown.setEnabled(enabled)

        self.btnMove.setEnabled(False)
        self.btnReset.setEnabled(False)

    def render_shifted_template(self):
        """ Calculate and render the shifted template
        """
        self.clear_canvas()

        self.spikeTrain.template.calculate_shifted_template(self.spikeTrain,
                                                            self.x_shift,
                                                            self.y_shift)
        self.plot_multichannel(self.spikeTrain.template.shifted_template,
                               color=MOVE_COLOR,
                               scaling=self._template_scaling)

    def execute_move(self):
        """ Move shifted template in the data
        """
        # subtract train: all spikes are removed, so not only the onces within
        # the bounds
        self.spikeTrain.subtract_train()

        # re-insert the shifted template for the selected energy interval
        sorted_idxs = self.spikeTrain.get_energy_sorted_idxs().copy()
        sorted_spikes = self.spikeTrain.retrieve_energy_sorted_spikes().copy()

        if self._energy_LB is None:
            l_idx = None
        else:
            l_idx = self._energy_LB+1 # exclusive

        if self._energy_UB is None:
            u_idx = None
        else:
            u_idx = self._energy_UB # also exclusive, but handled by python

        sorted_spikes_slice = sorted_spikes[l_idx:u_idx]
        sorted_idxs = sorted_idxs[l_idx:u_idx]

        assert(sorted_spikes_slice.shape == sorted_idxs.shape)

        print('# {} spikes considered for migration'.format(int(sorted_spikes_slice.size)))

        # add fixed temporal offset to avoid residual correlation
        time_shift = int(2*self._window_samples)
        sorted_spikes_insert = sorted_spikes_slice + time_shift

        # insertion shifted template instead of random permutation
        sorted_template_fit = self.spikeTrain._template_fitting[sorted_idxs]

        assert(sorted_spikes_slice.shape == sorted_template_fit.shape)

        inserted_spikes = self.spikeTrain.insert_given_train(sorted_spikes_insert,
                                                             self.spikeTrain.template.shifted_template,
                                                             sorted_template_fit)

        print('# {} spikes migrated'.format(int(inserted_spikes.size)))

        # keep track
        self.generated_GT[self._current_cluster] = inserted_spikes

        self.disable_GUI()
        self.spikeTrain.update(inserted_spikes)
        self.spikeTrain.fit_spikes()        
        self.enable_GUI()

        self.reset_energy_bounds()

        # enable export
        self.btnExport.setEnabled(True)

        self.radioTemplate.setChecked(True)
        self.draw_template(calcTemp=False)

        # diable choice in cluster list
        idx = np.where(np.array(self.good_clusters) == str(int(self._current_cluster)))[0][0]
        self.listClusterSelect.model().item(idx).setEnabled(False)


    """
    Methods related to exporting data
    """

    def export_data(self):
        """ Export data
        """
        export_path, _ = \
            QtWidgets.QFileDialog.getSaveFileName(self,
                                                  'save hybrid recording',
                                                  directory=self._select_path,
                                                  filter='raw recording (*.raw *.bin)')

        if export_path != '':
            # generate GT path
            csv_path, _ = os.path.splitext(export_path)
            csv_path = '{}_GT.csv'.format(csv_path)

            self.disable_GUI()
            print('# exporting {}'.format(export_path))
            self.recording.save_raw(export_path)
            print('# exporting {}'.format(csv_path))
            self.generated_GT.dumpCSV(csv_path)
            self.enable_GUI()


    """
    Methods related to plotting on the GUI canvas
    """

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

            if not signal.min() == signal.max():
                self.axes.plot(time, signal, color)
            else: # if all zeros / flat channel
                self.axes.plot(time, signal, FLAT_COLOR)
        # draw
        self.axes.figure.canvas.draw()

        return scaling

    def plot_energy(self):
        """ Plot the energy bar at the bottom of the canvas
        """
        sorted_idxs = self.spikeTrain.get_energy_sorted_idxs()
        energy = self.spikeTrain._fitting_energy.copy()

        energy = energy[sorted_idxs]

        energy = energy - energy.min()
        energy = energy / energy.max()

        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()

        y_range = (ylim[1] - ylim[0]) / 15
        energy = energy * y_range + ylim[0]
        bottom = np.zeros(energy.shape) + ylim[0]

        x_range = xlim[1] - xlim[0]
        xpct = x_range * 0.005

        x = np.linspace(xlim[0]+xpct, xlim[1]-xpct, num=energy.size)

        self.axes.plot(x, energy, ENERGY_COLOR, zorder=1)
        self.fill = self.axes.fill_between(x, energy, bottom,
                                           color=ENERGY_COLOR)

        # draw lower bound graphics
        if self._energy_LB is not None:
            self.axes.plot(x[:(self._energy_LB+1)],
                           energy[:(self._energy_LB+1)],
                           INACTIVE_COLOR, zorder=1)
            self.fill_LB = self.axes.fill_between(x[:(self._energy_LB+1)],
                                                  energy[:(self._energy_LB+1)],
                                                  bottom[:(self._energy_LB+1)],
                                                  color=INACTIVE_COLOR)

        # draw upper bound graphics
        if self._energy_UB is not None:
            UB_idx = self._energy_UB - energy.size

            self.axes.plot(x[UB_idx:], energy[UB_idx:], INACTIVE_COLOR, zorder=1)
            self.fill_UB = self.axes.fill_between(x[UB_idx:],
                                                  energy[UB_idx:],
                                                  bottom[UB_idx:],
                                                  color=INACTIVE_COLOR)

        self.vline = self.axes.vlines(x[self._current_spike], ylim[0],
                                      energy[self._current_spike],
                                      color=MARKER_COLOR,
                                      linewidth=3, zorder=2)
        self.axes.plot(x[self._current_spike], energy[self._current_spike],
                       color=MARKER_COLOR, marker='|')

        # draw
        self.axes.figure.canvas.draw()

    def clear_canvas(self, redraw=False):
        """ Clear the plotting canvas without explicitly clearing to keep the
        bounds
        """
        # clear regular lines
        self.axes.lines = []

        # clear the non-line objects we are keeping track of, if they exist
        try:
            self.vline.remove()
            self.fill.remove()
        except:
            # ignore, objects don't exist
            pass 

        # try separate for LB
        try:
            self.fill_LB.remove()
        except:
            pass

        # try separate for UB
        try:
            self.fill_UB.remove()
        except:
            pass

        # removing the variables is not sufficient to erase visually
        if redraw:
            self.axes.figure.canvas.draw()


    def reset_view_plot(self):
        """ Reset view on canvas
        """
        self.toolbar.home()

    def zoom_plot(self):
        """ Zoom on canvas
        """
        self.toolbar.zoom()

        # bring up pan button if necessary
        if self.btnPan.isChecked():
            self.btnPan.setChecked(False)

    def pan_plot(self):
        """ Drag canvas
        """
        self.toolbar.pan()

        # bring up zoom button if necessary
        if self.btnZoom.isChecked():
            self.btnZoom.setChecked(False)

    def lim_changed(self, ax):
        """ Callback for keeping the energy bar tidy when zooming / dragging
        """
        if self.template_fit_active:
            self.render_current_spike()


    """
    Housekeeping methods
    """

    def reset_file_session_variables(self):
        """ Reset file session variables (variables related to certain file)
        """
        self.recording = None
        self.clusters = None
        self.generated_GT = None

        self._current_cluster = None
        self.spikeTrain = None
        self._window_samples = None
        self._template_scaling = None

        self._current_spike = None

    def reset_GUI_initial(self):
        """ Reset GUI to initial enabled state
        """
        self.btnDraw.setEnabled(False)

        self.radioTemplate.setChecked(True)
        self.radioTemplate.setEnabled(False)
        self.radioFit.setEnabled(False)
        self.radioMove.setEnabled(False)

        self.set_template_fit_enabled(False)
        self.set_move_template_enabled(False)

        self.btnExport.setEnabled(False)

        self.reset_energy_bounds()
        self.clear_canvas(redraw=True)

    def reset_energy_bounds(self):
        """ Reset energy bounds and reset the related labels
        """
        self.checkBoxLower.setChecked(False)
        self.set_energy_lb(draw=False)
        self.checkBoxUpper.setChecked(False)
        self.set_energy_ub(draw=False)

    def build_GUI_status_dict(self):
        """ Build GUI status dictionary to keep track of the GUI state
        """
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
        self.GUI_status['checkBoxLower'] = self.checkBoxLower.isEnabled()
        self.GUI_status['checkBoxUpper'] = self.checkBoxUpper.isEnabled()
        self.GUI_status['btnMove'] = self.btnMove.isEnabled()
        self.GUI_status['btnResetZoom'] = self.btnResetZoom.isEnabled()
        self.GUI_status['btnZoom'] = self.btnZoom.isEnabled()
        self.GUI_status['btnPan'] = self.btnPan.isEnabled()

    def disable_GUI(self):
        """ Disable GUI
        """
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
            self.checkBoxLower.setEnabled(False)
            self.checkBoxUpper.setEnabled(False)
            self.btnMove.setEnabled(False)
            self.btnResetZoom.setEnabled(False)
            self.btnZoom.setEnabled(False)
            self.btnPan.setEnabled(False)

            # force repainting of entire GUI
            self.repaint()

    def enable_GUI(self):
        """ Enable GUI
        """
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
        self.btnResetZoom.setEnabled(self.GUI_status['btnResetZoom'])
        self.btnZoom.setEnabled(self.GUI_status['btnZoom'])
        self.btnPan.setEnabled(self.GUI_status['btnPan'])

        self.GUI_enabled = True


def main():
    """ Main program
    """
    app = QtWidgets.QApplication(sys.argv)

    form = Pybridizer()
    form.show()
    app.exec_()


if __name__ == '__main__':
    main()
