#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 13:36:01 2018

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

import sys
import os
import yaml
import pickle

from PyQt5 import QtWidgets, QtCore

import numpy as np

import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from hybridizer.ui import design, import_template, insert_template, select_auto
from hybridizer.io import Recording, SpikeClusters, Phy
from hybridizer.probes import RectangularProbe
from hybridizer.spikes import SpikeTrain, Template
from hybridizer.threads import TemplateWorker, ActivationWorker, MoveWorker
from hybridizer.hybrid import Insert


class ShyBride(QtWidgets.QMainWindow, design.Ui_ShyBride):
    """ Spike HYBRIDizer for Extracellular recordings
    """
    # Application constants
    CHOOSE_CLUSTER = 'select cluster'

    TEMPLATE_COLOR = '#1F77B4'
    SPIKE_COLOR = 'salmon'
    ENERGY_COLOR = 'salmon'
    INACTIVE_COLOR = 'gray'
    FLAT_COLOR = 'lightgray'
    MARKER_COLOR = 'darkslategray'
#    COLOR_MAP = 'winter'
    COLOR_MAP = 'rainbow'

    HISTORY_DUMP = ".history.dump"

    def __init__(self, parent=None):
        # setup UI
        super(self.__class__, self).__init__(parent)
        self.setupUi(self)

        self.init_variables()
        self.connect_listeners()

        # create plotting widgets
        self.create_plotting_area()
        self.create_color_bar()

        self.init_multi_threading()

        self.print_legal()

    def connect_listeners(self):
        """ Connect listener functions to GUI signals
        """
        self.btnDataSelect.clicked.connect(self.select_data)
        self.listClusterSelect.activated.connect(self.select_cluster)
        self.btnDraw.clicked.connect(lambda: self.draw_template(calcTemp=True))
        self.btnMagic.clicked.connect(self.auto_hybrid)

        self.radioTemplate.clicked.connect(lambda: self.draw_template(calcTemp=False))
        self.zeroForceFraction.valueChanged.connect(self.update_zero_force_fraction)

        self.radioFit.clicked.connect(self.template_fit)
        self.btnLeftSpike.clicked.connect(self.lower_spike)
        self.checkBoxLower.clicked.connect(lambda: self.set_energy_lb(draw=True))
        self.checkBoxUpper.clicked.connect(lambda: self.set_energy_ub(draw=True))
        self.btnRightSpike.clicked.connect(self.increase_spike)
        self.horizontalSlider.valueChanged.connect(self.slide_spike)

        self.radioMove.clicked.connect(self.move_template)
        self.btnMoveLeft.clicked.connect(self.move_left)
        self.btnMoveRight.clicked.connect(self.move_right)
        self.btnMoveUp.clicked.connect(self.move_up)
        self.btnMoveDown.clicked.connect(self.move_down)
        self.btnReset.clicked.connect(self.move_template)
        self.checkHeatMap.clicked.connect(self.toggle_heat_map)
        self.checkCustomSNR.clicked.connect(self.toggle_custom_snr)
        self.btnMove.clicked.connect(self.execute_move)

        self.btnResetZoom.clicked.connect(self.reset_view_plot)
        self.btnZoom.clicked.connect(self.zoom_plot)
        self.btnPan.clicked.connect(self.pan_plot)
        self.btnSave.clicked.connect(self.save_plot)

        self.btnTemplateExport.clicked.connect(self.export_template)
        self.btnTemplateImport.clicked.connect(self.import_template)

        self.btnUndo.clicked.connect(self.undo_move)

    def create_plotting_area(self):
        """ create and set up main plotting area
        """
        # activate toggle mode on zoom and pan button
        self.btnZoom.setCheckable(True)
        self.btnPan.setCheckable(True)

        # create plotting canvas
        canvas = FigureCanvas(plt.figure())
        self.toolbar = NavigationToolbar(canvas, self)
        self.toolbar.hide()

        # set up
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

        # add canvas widget to UI
        self.plotCanvas.addWidget(canvas)

    def create_color_bar(self):
        """ Create and set up color bar plotting area
        """
        canvas_color = FigureCanvas(plt.figure(figsize=(0,0.2)))

        self.axes_color = canvas_color.figure.add_subplot(111)
        self.axes_color.spines['top'].set_visible(False)
        self.axes_color.spines['right'].set_visible(False)
        self.axes_color.spines['bottom'].set_visible(False)
        self.axes_color.spines['left'].set_visible(False)
        self.axes_color.yaxis.set_ticks([])
        self.axes_color.xaxis.set_ticks([])
        canvas_color.figure.subplots_adjust(0,0,1,1,0,0)

        sizePolicyColor = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum,
                                                QtWidgets.QSizePolicy.Fixed)
        canvas_color.setSizePolicy(sizePolicyColor)

        # add canvas widget to GUI
        bg_color = self.palette().color(10).getRgbF()
        self.axes_color.set_facecolor(bg_color)
        self.colorBar.addWidget(canvas_color)

    def init_multi_threading(self):
        """ Initialize multi threading workers
        """
        self.templateWorker = TemplateWorker()
        self.templateWorker.template_ready.connect(
                lambda spike_train: self.draw_template(calcTemp=False,
                                                       spike_train=spike_train))

        self.activationWorker = ActivationWorker()
        self.activationWorker.activation_ready.connect(
                lambda act, sig_pow : self.render_shifted_template(activations=act,
                                                                   sig_power=sig_pow))

        self.moveWorker = MoveWorker()
        self.moveWorker.move_ready.connect(self.move_finished)

        # create event loop for use in the automatic hybridization magic
        self.magicEventLoop = QtCore.QEventLoop()

    def init_variables(self):
        """ Initialize helper variables
        """
        # path that is shown when the user opens a file browser dialog
        self._select_path = os.path.expanduser('~')
        # tracks the state of the GUI
        self.GUI_enabled = True

    def print_legal(self):
        """ print license related information
        """
        note = 'SHYBRID  Copyright (C) 2019  Jasper Wouters\n'\
               'This program comes with ABSOLUTELY NO WARRANTY. '\
               'This is free software, and you are welcome to redistribute it '\
               'under certain conditions.'
        print(note)


    """
    Methods related to data loading and cluster selection
    """
    def select_data(self):
        """ Open the selected data and load recording and clusters
        """
        # show memory mapping related warning message
        warn_message = 'This tool will alter the provided data directly. '\
            'Make sure to keep a copy of your original recording data.'
        QtWidgets.QMessageBox.information(self, 'data warning', warn_message)

        # open file browsing dialog
        raw_fn, _ = QtWidgets.QFileDialog.\
            getOpenFileName(self,
                            'select raw recording',
                            directory=self._select_path,
                            filter='raw recording (*.raw *.bin *.dat)')

        if raw_fn != "": # empty string is returned if cancel is pushed
            # update select path for user convenience
            self._select_path, _ = os.path.split(raw_fn)

            # try to load paramers file
            config_fn, _ = os.path.splitext(raw_fn)
            config_fn = config_fn + '.yml'

            if os.path.isfile(config_fn):
                # reset/init file related state
                self.reset_file_session_variables()

                with open(config_fn, 'r') as f:
                    config = yaml.safe_load(f)

                # process config parameters
                rec_params = config['data']

                try:
                    # initialise recording objects
                    self.recording = Recording(raw_fn, rec_params['probe'],
                                               rec_params['fs'],
                                               rec_params['dtype'],
                                               order=rec_params['order'])
                except TypeError as e:
                    # throw message box if provided type is not supported
                    QtWidgets.QMessageBox.critical(self, 'type error', str(e))

                    self.reset_GUI_initial(data_loaded=False)
                    self.clear_cluster_dropdown()

                    return

                except Exception as e:
                    # throw message box for unexpected errors
                    QtWidgets.QMessageBox.critical(self, 'unexpected error',
                                                   str(e))

                    self.reset_GUI_initial(data_loaded=False)
                    self.clear_cluster_dropdown()

                    return

                # process prior spike sorting results
                clus_params = config['clusters']

                try:
                    if os.path.isfile(self._get_dump_fn()):
                        self.load_history()
                        print('# found and reusing cluster dump')
                    else:
                        for clus_mode in clus_params.keys():
                            # load cluster information from phy
                            if clus_mode == 'phy':
                                phy = Phy(clus_params[clus_mode])
                                self.clusters = SpikeClusters()
                                self.clusters.fromPhy(phy, self.recording)
                                info_msg = 'The supplied prior spike sorting'\
                                    'information contains 0 clusters marked'\
                                    'as good.'
                            # load cluster information from csv
                            elif clus_mode == 'csv':
                                self.clusters = SpikeClusters()
                                self.clusters.fromCSV(clus_params[clus_mode],
                                                      self.recording)
                                info_msg = 'The supplied prior spike sorting'\
                                    'information contains 0 clusters.'
    
                            if len(self.clusters.keys()) == 0:
                                QtWidgets.QMessageBox.information(self,
                                                                  'no clusters found',
                                                                  info_msg)
                except Exception as e:
                    # throw message box for unexpected errors
                    QtWidgets.QMessageBox.critical(self, 'unexpected error',
                                                   str(e))

                    self.reset_GUI_initial(data_loaded=False)
                    self.clear_cluster_dropdown()

                    return

                self.fill_cluster_dropdown()

                # init probe model used for template import
                # TODO integrate with regular probe model
                self.connected_probe = RectangularProbe()
                self.connected_probe.fill_channels(self.recording.probe.geometry)
                self.connected_probe.connect_channels()

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

        if len(good_clusters) > 0:
            self._import_counter = np.array(good_clusters).max()
        else:
            self._import_counter = 0

        self.good_clusters = [self.CHOOSE_CLUSTER] + np.sort(good_clusters).astype('str').tolist()

        self.listClusterSelect.clear()
        self.listClusterSelect.addItems(self.good_clusters)

        # apply appropriate color
        for cluster in good_clusters:
            self.paint_cluster_list(cluster_idx=cluster)


    def clear_cluster_dropdown(self):
        """ Clear the cluster dropdown menu
        """
        self.good_clusters = [self.CHOOSE_CLUSTER]

        self.listClusterSelect.clear()
        self.listClusterSelect.addItems(self.good_clusters)

    def select_cluster(self):
        """ Select cluster from dropdown menu
        """
        label = self.listClusterSelect.currentText()
        if label != self.CHOOSE_CLUSTER:            
            self._current_cluster = int(label)
            self.btnDraw.setEnabled(True)

    def is_window_size_valid(self):
        """ validate the window size
        """
        try:
            window_size = float(self.fieldWindowSize.text())
            window_is_float = True
        except:
            window_is_float = False

        # if no given window size or invalid window size, raise error message
        if self.fieldWindowSize.text() == '':
             QtWidgets.QMessageBox.critical(self, 'No window size',
                                            'Please provide the desired spike window size.')
             return False

        elif not window_is_float:
             QtWidgets.QMessageBox.critical(self, 'Invalid window size',
                                            'Please provide a valid spike window size.')
             return False

        elif window_size <= 0:
            QtWidgets.QMessageBox.critical(self, 'Invalid window size',
                                           'Please provide a strictly positive window size.')
            return False

        return True


    """ Auto hybrid
    """
    def auto_hybrid(self):
        """ Show import template dialog
        """
        # build on the fly
        self.build_cluster_select_dialog()

    def auto_hybrid_run(self, clusters):
        """ Automatically generate ground truth
        """
        if self.is_window_size_valid():
            for cluster in clusters:
                self._current_cluster = cluster

                self.draw_template()
                self.radioTemplate.setChecked(True)
                # wait for multithreaded work to finish
                self.magicEventLoop.exec()

                self._energy_LB, self._energy_UB =\
                    self.spikeTrain.get_automatic_energy_bounds()

                self.x_shift = 0
                self.y_shift = self.spikeTrain.get_automatic_move()

                self.render_shifted_template()
                self.radioMove.setChecked(True)
                # wait for multithreaded work to finish
                if self.activations is None:
                    self.magicEventLoop.exec()

                self.execute_move()
                self.magicEventLoop.exec()

    def build_cluster_select_dialog(self):
        """ Build the graphical template import dialog
        """
        self.clusterSelectContainer = QtWidgets.QDialog(self)
        # constructor does nothing
        self.clusterSelectDialog = select_auto.Ui_AutoSelect()
        self.clusterSelectDialog.setupUi(self.clusterSelectContainer)

        self.clusterSelectDialog.listWidget.addItems(self.good_clusters[1:])

        for idx in range(self.clusterSelectDialog.listWidget.count()):
            self.clusterSelectDialog.listWidget.item(idx).setSelected(True)

        # prevent dialog resize
        self.clusterSelectContainer.setFixedSize(self.clusterSelectContainer.size())

        # connect listener
        self.clusterSelectDialog.buttonBox.accepted.connect(self.accept_cluster_select)

        self.clusterSelectContainer.exec()

    def accept_cluster_select(self):
        selected_clusters = self.clusterSelectDialog.listWidget.selectedItems()

        clusters_int = []
        for idx in range(len(selected_clusters)):
            clusters_int.append(int(selected_clusters[idx].text()))

        self.auto_hybrid_run(clusters_int)


    """
    Methods related to the template only view
    """
    # TODO split this function over different functions, too much functionality
    def draw_template(self, calcTemp=True, template=None, spike_train=None):
        """ (Calculate and) draw template
        """
        # TODO move to API
        self.paint_cluster_list()

        # if no given window size or invalid window size, raise error message
        if self.is_window_size_valid():
            if calcTemp and template is None:
                self.disable_GUI(msg='estimating template')

                self.spikeTrain = self.clusters[self._current_cluster].get_actual_spike_train()

                # calculate template
                self._window_samples = int(np.ceil(float(self.fieldWindowSize.text()) / 1000 * self.recording.sampling_rate))

                # transfer data to worker
                self.templateWorker.spike_train = self.spikeTrain
                self.templateWorker.window_size = self._window_samples
                self.templateWorker.zf_frac = self.convert_zero_force_fraction()
                self.templateWorker.is_hybrid = self.cluster_is_hybrid()

                self.templateWorker.start()

                # return and wait for worker thread to finish
                return

            elif spike_train is not None:
                window = spike_train.template.get_template_data().shape[1]
                if self._window_samples != window:
                    # throw warning
                    QtWidgets.QMessageBox.information(self, 'window size warning',
                                                      'The template window size for a hybrid cluster can not be altered.')

                    # change window size field
                    real_window = window / self.recording.sampling_rate * 1000
                    self.fieldWindowSize.setText(str(real_window))

                # finish multi threaded job
                self.spikeTrain = spike_train
                self.reset_energy_bounds()

                self.enable_GUI()

            # use provided template instead of data derived one
            elif template is not None:
                imported_template = Template(data=template, from_import=True)
                self.spikeTrain = SpikeTrain(self.recording, np.array([]),
                                             template=imported_template)

            elif not self.import_mode_active():
                self.spikeTrain = self.clusters[self._current_cluster].get_actual_spike_train()

            # spike train updated
            self.enable_disable_undo()

            print('# PSNR for current spike train = {}'.format(self.spikeTrain.get_PSNR()))

            # draw template
            self.axes.clear()
            self._template_scaling = self.plot_multichannel(np.zeros(self.spikeTrain.template.get_template_data().shape))
            self._template_scaling = self.plot_multichannel(self.spikeTrain.template.get_template_data())
            self.axes.autoscale(False)

            # add callbacks
            self.axes.callbacks.connect('ylim_changed', self.lim_changed)
            self.axes.callbacks.connect('xlim_changed', self.lim_changed)

            # check template to zero
            if not self.spikeTrain.template.check_template_edges_to_zero():
                QtWidgets.QMessageBox.information(self, 'window size warning',
                                                  'Consider increasing the window size, because the template edges are not reaching close to zero. '\
                                                  'This message can be ignored when working with noisy templates.')

            # enable options for further use
            self.radioTemplate.setChecked(True)
            self.radioTemplate.setEnabled(True)
            if self.spikeTrain.template.imported:
                self.radioFit.setEnabled(False)
                self.set_display_template_enabled(False)
            else:
                self.radioFit.setEnabled(True)
                self.set_display_template_enabled(True)

            self.btnTemplateExport.setEnabled(True)

            self.radioMove.setEnabled(True) # we opted for eternal program flow

            if self.import_mode_active():
                self.plotTitle.setText('Imported template')
            elif self.cluster_is_hybrid():
                self.plotTitle.setText('Cluster {} (SNR: {:.2f} dB) [ALREADY MOVED]'.format(self._current_cluster, self.spikeTrain.get_PSNR()))
            else:
                self.plotTitle.setText('Cluster {} (SNR: {:.2f} dB)'.format(self._current_cluster, self.spikeTrain.get_PSNR()))

            self.set_template_fit_enabled(False)
            self.set_move_template_enabled(False)

            self.magicEventLoop.exit()

    def set_display_template_enabled(self, enabled):
        """ Enable or disable the display template part of the GUI
        """
        # overwrite enabled if cluster is hybrid
        if self.cluster_is_hybrid():
            enabled = False

        self.zeroForceFraction.setEnabled(enabled)
        self.zeroForceFraction.lineEdit().deselect()
        self.zeroForceLabel.setEnabled(enabled)

    def update_zero_force_fraction(self):
        """ Update zero force fraction
        """
        # update the zero force fraction
        zf_frac = self.convert_zero_force_fraction()
        self.spikeTrain.template.update_zf_frac(zf_frac)

        # recalculate the fitting factors
        self.spikeTrain.fit_spikes()

        # reset the fitting factor bounds
        self.reset_energy_bounds()

        self.draw_template(calcTemp=False)

    def convert_zero_force_fraction(self):
        """ Return the zero force fraction in the backend format
        """
        return self.zeroForceFraction.value() / 100

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
        self.set_display_template_enabled(False)
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

        self.checkBoxLower.setEnabled(enabled)
        self.checkBoxUpper.setEnabled(enabled)

        if enabled == True:
            self.labelSpike.setText('1/{} '.format(int(self.spikeTrain.spikes.size)))
            # labelfit is set in energy plotting routine
        else:
            self.labelSpike.setText('')
            self.labelFit.setText('')

    def render_current_spike(self):
        """ Function that actually plots the current spike and the template
        """
        self.clear_canvas()

        # plot energy bar first
        self.plot_energy()

        # sort using fitting energy
        spike_time = self.spikeTrain.get_energy_sorted_spike_time(self._current_spike)
        start, end = self.spikeTrain.get_spike_start_end(spike_time)
        spike = self.recording.get_good_chunk(start,end)

        # apply sorting also to fitting factors
        sorted_idxs = self.spikeTrain.get_energy_sorted_idxs()
        fit = self.spikeTrain._template_fitting[sorted_idxs][self._current_spike]

        if fit > 0:
            scaling = self._template_scaling/fit
        else:
            scaling = 0.0

        self.plot_multichannel(spike, color=self.SPIKE_COLOR,
                               scaling=scaling)
        self.plot_multichannel(self.spikeTrain.template.get_template_data())

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
        self.set_display_template_enabled(False)
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
        self.checkHeatMap.setEnabled(enabled)
        self.checkCustomSNR.setEnabled(enabled)

        try:
            template_imported = self.spikeTrain.template.imported
        except:
            template_imported = False

        # enable SNR spinbox only if the checkbox is checked
        if template_imported:
            self.checkCustomSNR.setEnabled(False)
            self.spinSNR.setEnabled(False)
        elif self.checkCustomSNR.isChecked():
            self.spinSNR.setEnabled(enabled)
        else:
            self.spinSNR.setEnabled(False)

        if enabled and template_imported:
            self.btnMove.setEnabled(True)
        else:
            self.btnMove.setEnabled(False)

        self.btnReset.setEnabled(False)

        if not enabled:
            self.show_color_bar(False)

    def render_shifted_template(self, activations=None, sig_power=None):
        """ Calculate and render the shifted template
        """
        self.clear_canvas()

        self.shifted_template =\
            self.spikeTrain.template.get_shifted_template(self.spikeTrain,
                                                          self.x_shift,
                                                          self.y_shift)

        # act on worker thread results
        if activations is not None:
            self.activations = activations
            self.sig_power = sig_power

            self.enable_GUI()
            self.magicEventLoop.exit()

        if self.activations is None:
            self.disable_GUI(msg='calculating spiking activity')

            self.activationWorker.recording = self.recording
            self.activationWorker.start()

            # return and wait for thread to finish
            return

        if self.checkHeatMap.isChecked():
            norm_activations = self.activations / self.activations.max()
            self.plot_multichannel(self.shifted_template.get_template_data(),
                                   scaling=self._template_scaling,
                                   activations=norm_activations)
        else:
            self.plot_multichannel(self.shifted_template.get_template_data(),
                                   color=self.TEMPLATE_COLOR,
                                   scaling=self._template_scaling)

        self.show_color_bar(self.checkHeatMap.isChecked())

    def toggle_custom_snr(self):
        """ Toggle the custom SNR spinbox
        """
        if self.checkCustomSNR.isChecked():
            self.spinSNR.setEnabled(True)
        else:
            self.spinSNR.setEnabled(False)

    def toggle_heat_map(self):
        """ Toggle spike count heatmap in move template view
        """
        # TODO argument should be passed instead of relying on internals
        self.render_shifted_template()

    def show_color_bar(self, show, show_labels=True):
        """ Show colorbar if show == True, else hide
        """
        if show:
            if show_labels:
                self.labelLow.setText('0')
                self.labelHigh.setText('{}'.format(int(self.activations.max())))

            cmap = cm.get_cmap(self.COLOR_MAP)
            mpl.colorbar.ColorbarBase(self.axes_color, cmap=cmap,
                                      orientation='horizontal')
            self.axes_color.figure.canvas.draw()
        else:
            self.labelLow.setText('')
            self.labelHigh.setText('')

            self.axes_color.clear()
            bg_color = self.palette().color(10).getRgbF()
            self.axes_color.figure.patch.set_facecolor(bg_color)
            self.axes_color.figure.canvas.draw()

    def calculate_move_scaling(self):
        """ Transform the given SNR into a proper template scaling
        """
        desired_scaling = np.sqrt(10**(self.spinSNR.value()/10))
        
        actual_PSNR = self.clusters[self._current_cluster].get_actual_spike_train().get_PSNR()
        actual_scaling = np.sqrt(10**(actual_PSNR/10))

        scaling_correction_factor = desired_scaling / actual_scaling

        return scaling_correction_factor

    def execute_move(self):
        """ Move shifted template in the data
        """
        if not self.import_mode_active():
            self.disable_GUI(msg='moving template')

            self.moveWorker.cluster = self.clusters[self._current_cluster]
            self.moveWorker.shifted_template = self.shifted_template

            self.moveWorker.energy_LB = self._energy_LB
            self.moveWorker.energy_UB = self._energy_UB

            if self.checkCustomSNR.isChecked():
                self.moveWorker.target_PSNR = self.spinSNR.value()
            else:
                self.moveWorker.target_PSNR = None

            self.moveWorker.start()
            return

        else:
            # insert template
            self.build_insert_dialog()

    def move_finished(self):
        """ Complete the relocation of a spike train
        """
        self.spikeTrain = self.clusters[self._current_cluster].get_actual_spike_train()

        self.reset_energy_bounds()

        self.dump_ground_truth()

        self.enable_GUI()

        self.draw_template(calcTemp=False)

        self.magicEventLoop.exit()

    def undo_move(self):
        """ Undo the last spike train relocation
        """
        # TODO implement multithreaded
        self.reset_energy_bounds()

        self.disable_GUI()

        # undo the last insertion        
        self.clusters[self._current_cluster].undo_last_operator()
        try:
            # undo the last subtraction
            self.clusters[self._current_cluster].undo_last_operator()
            deleted_cluster = False
        except IndexError:
            # remove cluster from memory
            self.clusters.remove_cluster(self._current_cluster)
            deleted_cluster = True

            list_idx = self.listClusterSelect.findText(str(int(self._current_cluster)))
            self.listClusterSelect.removeItem(list_idx)
            self.listClusterSelect.setCurrentIndex(0)

            self._current_cluster = None

        self.dump_ground_truth()

        self.enable_GUI()

        if deleted_cluster:
            self.reset_GUI_initial()
        else:
            self.draw_template(calcTemp=False)

    def enable_disable_undo(self):
        """ check whether has to be enabled/disabled for the current cluster
        """
        if self.cluster_is_hybrid() and not self.import_mode_active():
            self.btnUndo.setEnabled(True)
        else:
            self.btnUndo.setEnabled(False)

    def cluster_is_hybrid(self, cluster_idx=None):
        """ Return whether or not the given cluster is a hybrid clusters. If given
        cluster idx is None, the current cluster is checked instead.
        """
        if cluster_idx is None:
            if self._current_cluster is None:
                return False
            return self.clusters[self._current_cluster].is_hybrid()
        else:
            return self.clusters[cluster_idx].is_hybrid()

    def dump_ground_truth(self):
        """ Dump ground truth labels
        """
        csv_path = os.path.join(self._select_path, 'hybrid_GT.csv')
        print('# updated ground truth in {}'.format(csv_path))
        self.clusters.dumpCSV(csv_path)

        # also dump the history
        self.dump_history()

    def paint_cluster_list(self, cluster_idx=None):
        """ Update the color of the given cluster idx in clusters list. If given
        cluster idx is None, the current cluster is painted instead.
        """
        # convert cluster idx to list idx
        if cluster_idx is None:
            if self._current_cluster is None:
                return
            list_idx = self.listClusterSelect.findText(str(int(self._current_cluster)))
        else:
            list_idx = self.listClusterSelect.findText(str(int(cluster_idx)))

        # choose appropriate color
        if self.cluster_is_hybrid(cluster_idx=cluster_idx):
            color = QtCore.Qt.gray
        else:
            color = QtCore.Qt.black

        # change color
        self.listClusterSelect.model().item(list_idx).setForeground(color)


    """
    Methods related to exporting the active template
    """
    def export_template(self):
        """ Export template
        """
        # define channels that have to be exported
        channels = self.channels_within_lims()

        # sort these channels using the connected probe model
        channels = self.connected_probe.sort_given_channel_idx(channels)

        # convert channels to good channels space
        channels = self.recording.probe.chans_to_good_chans(channels)

        # ask feedback about number of export channels
        reply = QtWidgets.QMessageBox.question(self, 'export template', 
                                               'Export {} channels?'.format(channels.size),
                                               QtWidgets.QMessageBox.Yes,
                                               QtWidgets.QMessageBox.No)

        if reply == QtWidgets.QMessageBox.Yes:
            # open dialog to export
            self.dump_template(channels)                

    def channels_within_lims(self):
        """ Return a list of channels that are withing the main figures
        current zoom
        """
        xlims = self.axes.get_xlim()
        ylims = self.axes.get_ylim()

        min_geo = self.recording.probe.get_min_geometry()
        max_geo = self.recording.probe.get_max_geometry()

        # probe info
        x_bias = -min_geo[0]
        y_bias = -min_geo[1]

        x_range = max_geo[0] - min_geo[0]
        y_range = max_geo[1] - min_geo[1]

        if x_range == 0:
            x_range = 1

        if y_range == 0:
            y_range = 1

        channels = np.array([])

        for idx, channel in enumerate(self.recording.probe.channels):
            geo = self.recording.probe.geometry[channel]

            x_start = (geo[0] - x_bias) / x_range
            y_start = (geo[1] - y_bias) / y_range

            if (xlims[0] <= x_start) and (x_start <= xlims[1]):
                if (ylims[0] <= y_start) and (y_start <= ylims[1]):
                    channels = np.append(channels, channel)

        return channels

    def dump_template(self, channels):
        """ Export the template in CSV format
        """
        export_path, _ = \
            QtWidgets.QFileDialog.getSaveFileName(self,
                                                  'export template',
                                                  directory=self._select_path,
                                                  filter='CSV (*.csv)')

        if export_path != '':
            export_path = self.force_extension(export_path, extension='.csv')

            export_template = self.spikeTrain.template.get_template_data()[channels.astype(np.int)]

            # normalize template before exporting
            export_template = export_template / export_template.std()

            # dump csv
            np.savetxt(export_path, export_template, delimiter=',')

    def force_extension(self, path, extension='.csv'):
        """ Force a given path to a certain extension
        """
        root, _ = os.path.splitext(path)
        return root + extension


    """
    Methods related to importing a template for insertion
    """
    def build_import_dialog(self):
        """ Build the graphical template import dialog
        """
        self.importTemplateContainer = QtWidgets.QDialog(self)
        # constructor does nothing
        self.importTemplateDialog = import_template.Ui_DialogTemplateImport()
        self.importTemplateDialog.setupUi(self.importTemplateContainer)
        # prevent dialog resize
        self.importTemplateContainer.setFixedSize(self.importTemplateContainer.size())

        # template import dialog listeners
        self.importTemplateDialog.btnSelectTemplate.clicked.connect(self.select_template)
        
        self.importTemplateDialog.buttonBox.accepted.connect(self.accept_import_dialog)

        self.importTemplateContainer.exec()

    def import_template(self):
        """ Show import template dialog
        """
        # build on the fly
        self.build_import_dialog()

    def select_template(self):
        """ Click listener that opens a file dialog for selecting the template
        csv for import
        """
        # open a csv file selection dialog
        template_file, _ = QtWidgets.QFileDialog.\
            getOpenFileName(self,
                            'select template',
                            directory=self._select_path,
                            filter='CSV (*.csv)')

        if template_file != '':
            self.imported_template = np.loadtxt(template_file, delimiter=',')
            nbChannels = self.imported_template.shape[0]
            window = self.imported_template.shape[1]

            self.importTemplateDialog.nbChannels.\
                setText('detected {} channels\n'
                        'with {} samples each'.format(nbChannels,
                                                      window))

            self.importTemplateDialog.buttonBox.setEnabled(True)

    def accept_import_dialog(self):
        """ Accept import dialog
        """
        x_reach = self.importTemplateDialog.boxReach.value()
        x_offset = self.importTemplateDialog.boxOffset.value()

        print('values read')

        try:
            mapped_channels =\
                self.connected_probe.get_channels_from_zone(self.imported_template.shape[0],
                                                            x_reach, x_offset)

        except Exception as e:
            self.importTemplateContainer.close()
            QtWidgets.QMessageBox.critical(self, 'parameter error', str(e))
            self.import_template = None
            return

        mapped_channels = self.recording.probe.chans_to_good_chans(mapped_channels)

        nb_good = self.recording.get_nb_good_channels()
        window = self.imported_template.shape[1]

        self._window_samples = window

        template = np.zeros((nb_good, window))

        for idx, mapped_channel in enumerate(mapped_channels):
            if not mapped_channel is None:
                template[mapped_channel] = self.imported_template[idx]

        # set window in ms for completeness
        self.fieldWindowSize.selectAll()
        self.fieldWindowSize.del_()

        real_window = window / self.recording.sampling_rate * 1000
        self.fieldWindowSize.setText(str(real_window))

        self.draw_template(calcTemp=False, template=template)

        self.importTemplateContainer.close()

    def build_insert_dialog(self):
        """ Build the graphical template import dialog
        """
        self.insertTemplateContainer = QtWidgets.QDialog(self)
        # constructor does nothing
        self.insertTemplateDialog = insert_template.Ui_DialogInsertTemplate()
        self.insertTemplateDialog.setupUi(self.insertTemplateContainer)
        # prevent dialog resize
        self.insertTemplateContainer.setFixedSize(self.insertTemplateContainer.size())
        
        self.insertTemplateDialog.buttonBox.accepted.connect(self.insert_template)

        self.insertTemplateContainer.exec()

    def insert_template(self):
        """ Destroy the graphical template import dialog
        """
        # read desired spike train characteristics
        snr = self.insertTemplateDialog.boxSNR.value()
        rate = self.insertTemplateDialog.boxRate.value()
        refr = self.insertTemplateDialog.boxRefr.value()

        # poisson process spike times
        spike = 0
        dur = self.recording.get_duration()

        if rate > 0:
            beta = self.recording.sampling_rate / rate
            refr = int(self.recording.sampling_rate * refr / 1000) # discretize

            spikes_insert = np.array([], dtype=np.int)
            while spike < dur:
                isi = np.random.exponential(scale=beta)
                # enforce refractory period
                if isi < refr:
                    isi = refr
                spike += isi

                spikes_insert = np.append(spikes_insert, spike)
        else:
            QtWidgets.QMessageBox.critical(self, 'Invalid rate',
                                           'The given rate should be strictly positive.')
            return

        insert_waveform = self.spikeTrain.template.get_shifted_template(self.spikeTrain,
                                                                        self.x_shift,
                                                                        self.y_shift)

        within_bounds = spikes_insert <  (self.spikeTrain.recording.get_duration() - insert_waveform.window_size)
        spikes_insert = spikes_insert[within_bounds]

        insert_train = SpikeTrain(self.spikeTrain.recording, spikes_insert,
                                  template=insert_waveform,
                                  template_fitting=np.ones(spikes_insert.shape))

        insert_train.set_target_PSNR(snr)
        train_insertion = Insert(insert_train)

        # create an empty cluster
        self._import_counter = self._import_counter + 1
        self.clusters.add_empty_cluster(self._import_counter)

        self.disable_GUI()

        # insert scaled template
        self.clusters[self._import_counter].apply_operator(train_insertion)
        insert_train.recording.flush()

        self.enable_GUI()

        print('# {} spikes inserted'.format(int(spikes_insert.size)))

        self._current_cluster = self._import_counter

        # add the new cluster to the dropdown and select
        new_item = str(int(self._current_cluster))
        self.listClusterSelect.addItem(new_item)
        self.listClusterSelect.setCurrentIndex(self.listClusterSelect.findText(new_item))

        print('# Inserted spikes were assigned to new cluster', new_item)

        self.insertTemplateContainer.close()

        self.spikeTrain = self.clusters[self._current_cluster].get_actual_spike_train()
        self.reset_energy_bounds()
        self.dump_ground_truth()

        self.radioTemplate.setChecked(True)
        self.draw_template(calcTemp=False)


    def import_mode_active(self):
        """ Check whether or not the app is in import mode
        """
        try:
            is_imported = self.spikeTrain.template.imported
        except:
            is_imported = False

        return is_imported


    """
    Methods related to plotting on the GUI canvas
    """
    def plot_multichannel(self, data, color=TEMPLATE_COLOR, scaling=None,
                          activations=None):
        """ Plot multichannel data on the figure canvas
        """
        min_geo = self.recording.probe.get_min_geometry()
        max_geo = self.recording.probe.get_max_geometry()

        # probe info
        x_bias = -min_geo[0]
        y_bias = -min_geo[1]

        x_range = max_geo[0] - min_geo[0]
        y_range = max_geo[1] - min_geo[1]

        if x_range == 0:
            x_range = 1

        if y_range == 0:
            y_range = 1

        x_between = self.recording.probe.x_between
        y_between = self.recording.probe.y_between

        # data info
        min_dat = data.min()
        max_dat = data.max()

        # adapt distance between channels
        range_scaler = 1.1
        dat_range = range_scaler * (max_dat - min_dat)

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

            # if activations are available use them to pick the color
            if activations is not None:
                activation = activations[idx]
                cmap = cm.get_cmap(self.COLOR_MAP)
                tmp_color = cmap(activation)
            else:
                tmp_color = color

            if not signal.min() == signal.max():
                self.axes.plot(time, signal, color=tmp_color, linewidth=1.5)
            else: # if all zeros / flat channel
                if activations is None:
                    tmp_color = self.FLAT_COLOR
                self.axes.plot(time, signal, color=tmp_color, linewidth=0.5)

        # draw
        self.axes.figure.canvas.draw()

        return scaling

    def plot_energy(self):
        """ Plot the energy bar at the bottom of the canvas
        """
        sorted_idxs = self.spikeTrain.get_energy_sorted_idxs()
        # TODO revise this 
#        energy = self.spikeTrain._fitting_energy.copy()
        energy = self.spikeTrain._template_fitting.copy()

        energy = energy[sorted_idxs]

        # show fit
        self.labelFit.setText('factor {:.3f} '.format(energy[self._current_spike]))

        unit_line = 1 - energy.min()
        energy = energy - energy.min()

        unit_line = unit_line / energy.max()
        energy = energy / energy.max()

        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()

        y_range = (ylim[1] - ylim[0]) / 15 # arbitrary portion of the screen

        unit_line = unit_line * y_range + ylim[0]
        energy = energy * y_range + ylim[0]

        bottom = np.zeros(energy.shape) + ylim[0]

        x_range = xlim[1] - xlim[0]
        xpct = x_range * 0.005

        x = np.linspace(xlim[0]+xpct, xlim[1]-xpct, num=energy.size)

        self.axes.plot(x, energy, self.ENERGY_COLOR, zorder=1)
        self.fill = self.axes.fill_between(x, energy, bottom,
                                           color=self.ENERGY_COLOR)

        # draw lower bound graphics
        if self._energy_LB is not None:
            self.axes.plot(x[:(self._energy_LB+1)],
                           energy[:(self._energy_LB+1)],
                           self.INACTIVE_COLOR, zorder=1)
            self.fill_LB = self.axes.fill_between(x[:(self._energy_LB+1)],
                                                  energy[:(self._energy_LB+1)],
                                                  bottom[:(self._energy_LB+1)],
                                                  color=self.INACTIVE_COLOR)

        # draw upper bound graphics
        if self._energy_UB is not None:
            UB_idx = self._energy_UB - energy.size

            self.axes.plot(x[UB_idx:], energy[UB_idx:], self.INACTIVE_COLOR, zorder=1)
            self.fill_UB = self.axes.fill_between(x[UB_idx:],
                                                  energy[UB_idx:],
                                                  bottom[UB_idx:],
                                                  color=self.INACTIVE_COLOR)

        self.vline = self.axes.vlines(x[self._current_spike], ylim[0],
                                      energy[self._current_spike],
                                      color=self.MARKER_COLOR,
                                      linewidth=3, zorder=2)
        self.axes.plot(x[self._current_spike], energy[self._current_spike],
                       color=self.MARKER_COLOR, marker='|')

        self.axes.plot(x, np.ones(x.shape)*unit_line, 'k--', linewidth=0.5)

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

    def save_plot(self):
        """ Save plot
        """
        self.toolbar.save_figure()

    def lim_changed(self, ax):
        """ Callback for keeping the energy bar tidy when zooming / dragging
        """
        if self.template_fit_active:
            self.render_current_spike()


    """
    Housekeeping methods
    """
    def _get_dump_fn(self):
        """ Return full filename to dump location 
        """
        return os.path.join(self._select_path, self.HISTORY_DUMP)

    def dump_history(self):
        """ Dump the history of template moves
        """
        # forget the recording references
        self.clusters.forget_recording()

        with open(self._get_dump_fn(), 'wb') as dump_file:
            pickle.dump(self.clusters, dump_file)

        self.clusters.add_recording(self.recording)

    def load_history(self):
        """ load history
        """
        with open(self._get_dump_fn(), 'rb') as dump_file:
            self.clusters = pickle.load(dump_file)

        self.clusters.add_recording(self.recording)

    def reset_file_session_variables(self):
        """ Reset file session variables (variables related to certain file)
        """
        self.recording = None
        self.clusters = None

        self._current_cluster = None
        self.spikeTrain = None
        self._window_samples = None
        self._template_scaling = None

        self._current_spike = None

        self.activations = None
        self.sig_power = None

    def reset_GUI_initial(self, data_loaded=True):
        """ Reset GUI to initial enabled state
        """
        self.btnDraw.setEnabled(False)
        self.btnMagic.setEnabled(data_loaded)

        self.radioTemplate.setChecked(True)
        self.radioTemplate.setEnabled(False)
        self.radioFit.setEnabled(False)
        self.radioMove.setEnabled(False)

        self.set_display_template_enabled(False)
        self.set_template_fit_enabled(False)
        self.set_move_template_enabled(False)

        self.btnTemplateExport.setEnabled(False)
        self.btnTemplateImport.setEnabled(data_loaded)

        self.enable_disable_undo()

        self.plotTitle.setText('')

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
        self.GUI_status['horizontalSlider'] = self.horizontalSlider.isEnabled()
        self.GUI_status['checkBoxLower'] = self.checkBoxLower.isEnabled()
        self.GUI_status['checkBoxUpper'] = self.checkBoxUpper.isEnabled()
        self.GUI_status['btnMove'] = self.btnMove.isEnabled()
        self.GUI_status['btnResetZoom'] = self.btnResetZoom.isEnabled()
        self.GUI_status['btnZoom'] = self.btnZoom.isEnabled()
        self.GUI_status['btnPan'] = self.btnPan.isEnabled()
        self.GUI_status['btnSave'] = self.btnSave.isEnabled()
        self.GUI_status['checkHeatMap'] = self.checkHeatMap.isEnabled()
        self.GUI_status['btnTemplateImport'] = self.btnTemplateImport.isEnabled()
        self.GUI_status['btnTemplateExport'] = self.btnTemplateExport.isEnabled()
        self.GUI_status['zeroForceFraction'] = self.zeroForceFraction.isEnabled()
        self.GUI_status['btnMagic'] = self.btnMagic.isEnabled()
        self.GUI_status['zeroForceLabel'] = self.zeroForceLabel.isEnabled()
        self.GUI_status['btnUndo'] = self.btnUndo.isEnabled()
        self.GUI_status['checkCustomSNR'] = self.checkCustomSNR.isEnabled()
        self.GUI_status['spinSNR'] = self.spinSNR.isEnabled()

    def disable_GUI(self, msg=None):
        """ Disable GUI
        """
        if self.GUI_enabled:
            # prevent losing status dict
            self.GUI_enabled = False

            self.progressBar.setMaximum(0)
            self.progressBar.setEnabled(True)

            if msg is not None:
                self.progressLabel.setText(msg)

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
            self.horizontalSlider.setEnabled(False)
            self.checkBoxLower.setEnabled(False)
            self.checkBoxUpper.setEnabled(False)
            self.btnMove.setEnabled(False)
            self.btnResetZoom.setEnabled(False)
            self.btnZoom.setEnabled(False)
            self.btnPan.setEnabled(False)
            self.btnSave.setEnabled(False)
            self.checkHeatMap.setEnabled(False)
            self.btnTemplateImport.setEnabled(False)
            self.btnTemplateExport.setEnabled(False)
            self.zeroForceFraction.setEnabled(False)
            self.btnMagic.setEnabled(False)
            self.zeroForceLabel.setEnabled(False)
            self.btnUndo.setEnabled(False)
            self.checkCustomSNR.setEnabled(False)
            self.spinSNR.setEnabled(False)

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
        self.btnMove.setEnabled(self.GUI_status['btnMove'])
        self.horizontalSlider.setEnabled(self.GUI_status['horizontalSlider'])
        self.btnResetZoom.setEnabled(self.GUI_status['btnResetZoom'])
        self.btnZoom.setEnabled(self.GUI_status['btnZoom'])
        self.btnPan.setEnabled(self.GUI_status['btnPan'])
        self.btnSave.setEnabled(self.GUI_status['btnSave'])
        self.checkHeatMap.setEnabled(self.GUI_status['checkHeatMap'])
        self.btnTemplateImport.setEnabled(self.GUI_status['btnTemplateImport'])
        self.btnTemplateExport.setEnabled(self.GUI_status['btnTemplateExport'])
        self.zeroForceFraction.setEnabled(self.GUI_status['zeroForceFraction'])
        self.btnMagic.setEnabled(self.GUI_status['btnMagic'])
        self.zeroForceLabel.setEnabled(self.GUI_status['zeroForceLabel'])
        self.btnUndo.setEnabled(self.GUI_status['btnUndo'])
        self.checkCustomSNR.setEnabled(self.GUI_status['checkCustomSNR'])
        self.spinSNR.setEnabled(self.GUI_status['spinSNR'])

        self.progressBar.setMaximum(1)
        self.progressBar.setEnabled(False)
        self.progressLabel.setText('')

        self.GUI_enabled = True


def main():
    """ Main program
    """
    app = QtWidgets.QApplication(sys.argv)

    # classic style
    form = ShyBride()
    form.show()

    app.exec_()


if __name__ == '__main__':
    """ Run main program
    """
    main()
