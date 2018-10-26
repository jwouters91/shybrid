# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'hybridizer.ui'
#
# Created by: PyQt5 UI code generator 5.5.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Pybridizer(object):
    def setupUi(self, Pybridizer):
        Pybridizer.setObjectName("Pybridizer")
        Pybridizer.resize(1124, 639)
        Pybridizer.setTabShape(QtWidgets.QTabWidget.Rounded)
        Pybridizer.setUnifiedTitleAndToolBarOnMac(False)
        self.centralwidget = QtWidgets.QWidget(Pybridizer)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout.setObjectName("verticalLayout")
        spacerItem = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout.addItem(spacerItem)
        self.btnDataSelect = QtWidgets.QPushButton(self.centralwidget)
        self.btnDataSelect.setObjectName("btnDataSelect")
        self.verticalLayout.addWidget(self.btnDataSelect)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.listClusterSelect = QtWidgets.QComboBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.listClusterSelect.sizePolicy().hasHeightForWidth())
        self.listClusterSelect.setSizePolicy(sizePolicy)
        self.listClusterSelect.setObjectName("listClusterSelect")
        self.listClusterSelect.addItem("")
        self.horizontalLayout.addWidget(self.listClusterSelect)
        self.fieldWindowSize = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fieldWindowSize.sizePolicy().hasHeightForWidth())
        self.fieldWindowSize.setSizePolicy(sizePolicy)
        self.fieldWindowSize.setText("")
        self.fieldWindowSize.setObjectName("fieldWindowSize")
        self.horizontalLayout.addWidget(self.fieldWindowSize)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.btnDraw = QtWidgets.QPushButton(self.centralwidget)
        self.btnDraw.setEnabled(False)
        self.btnDraw.setAutoDefault(False)
        self.btnDraw.setDefault(False)
        self.btnDraw.setObjectName("btnDraw")
        self.verticalLayout.addWidget(self.btnDraw)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem1)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setEnabled(False)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.verticalLayout.addLayout(self.horizontalLayout_8)
        self.radioTemplate = QtWidgets.QRadioButton(self.centralwidget)
        self.radioTemplate.setEnabled(False)
        self.radioTemplate.setMouseTracking(True)
        self.radioTemplate.setCheckable(True)
        self.radioTemplate.setChecked(True)
        self.radioTemplate.setAutoExclusive(True)
        self.radioTemplate.setObjectName("radioTemplate")
        self.verticalLayout.addWidget(self.radioTemplate)
        self.radioFit = QtWidgets.QRadioButton(self.centralwidget)
        self.radioFit.setEnabled(False)
        self.radioFit.setObjectName("radioFit")
        self.verticalLayout.addWidget(self.radioFit)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.btnLeftSpike = QtWidgets.QPushButton(self.centralwidget)
        self.btnLeftSpike.setEnabled(False)
        self.btnLeftSpike.setObjectName("btnLeftSpike")
        self.horizontalLayout_4.addWidget(self.btnLeftSpike)
        self.btnRightSpike = QtWidgets.QPushButton(self.centralwidget)
        self.btnRightSpike.setEnabled(False)
        self.btnRightSpike.setObjectName("btnRightSpike")
        self.horizontalLayout_4.addWidget(self.btnRightSpike)
        self.labelSpike = QtWidgets.QLabel(self.centralwidget)
        self.labelSpike.setEnabled(False)
        self.labelSpike.setText("")
        self.labelSpike.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.labelSpike.setObjectName("labelSpike")
        self.horizontalLayout_4.addWidget(self.labelSpike)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem2 = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem2)
        self.checkBoxLower = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBoxLower.setEnabled(False)
        self.checkBoxLower.setObjectName("checkBoxLower")
        self.horizontalLayout_2.addWidget(self.checkBoxLower)
        self.labelLB = QtWidgets.QLabel(self.centralwidget)
        self.labelLB.setEnabled(False)
        self.labelLB.setText("")
        self.labelLB.setAlignment(QtCore.Qt.AlignCenter)
        self.labelLB.setObjectName("labelLB")
        self.horizontalLayout_2.addWidget(self.labelLB)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        spacerItem3 = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem3)
        self.checkBoxUpper = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBoxUpper.setEnabled(False)
        self.checkBoxUpper.setObjectName("checkBoxUpper")
        self.horizontalLayout_5.addWidget(self.checkBoxUpper)
        self.labelUB = QtWidgets.QLabel(self.centralwidget)
        self.labelUB.setEnabled(False)
        self.labelUB.setText("")
        self.labelUB.setAlignment(QtCore.Qt.AlignCenter)
        self.labelUB.setObjectName("labelUB")
        self.horizontalLayout_5.addWidget(self.labelUB)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.radioMove = QtWidgets.QRadioButton(self.centralwidget)
        self.radioMove.setEnabled(False)
        self.radioMove.setCheckable(True)
        self.radioMove.setObjectName("radioMove")
        self.verticalLayout.addWidget(self.radioMove)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.btnMoveLeft = QtWidgets.QPushButton(self.centralwidget)
        self.btnMoveLeft.setEnabled(False)
        self.btnMoveLeft.setObjectName("btnMoveLeft")
        self.horizontalLayout_6.addWidget(self.btnMoveLeft)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.btnMoveUp = QtWidgets.QPushButton(self.centralwidget)
        self.btnMoveUp.setEnabled(False)
        self.btnMoveUp.setObjectName("btnMoveUp")
        self.verticalLayout_2.addWidget(self.btnMoveUp)
        self.btnReset = QtWidgets.QPushButton(self.centralwidget)
        self.btnReset.setEnabled(False)
        self.btnReset.setObjectName("btnReset")
        self.verticalLayout_2.addWidget(self.btnReset)
        self.btnMoveDown = QtWidgets.QPushButton(self.centralwidget)
        self.btnMoveDown.setEnabled(False)
        self.btnMoveDown.setObjectName("btnMoveDown")
        self.verticalLayout_2.addWidget(self.btnMoveDown)
        self.horizontalLayout_6.addLayout(self.verticalLayout_2)
        self.btnMoveRight = QtWidgets.QPushButton(self.centralwidget)
        self.btnMoveRight.setEnabled(False)
        self.btnMoveRight.setObjectName("btnMoveRight")
        self.horizontalLayout_6.addWidget(self.btnMoveRight)
        self.verticalLayout.addLayout(self.horizontalLayout_6)
        self.btnMove = QtWidgets.QPushButton(self.centralwidget)
        self.btnMove.setEnabled(False)
        self.btnMove.setObjectName("btnMove")
        self.verticalLayout.addWidget(self.btnMove)
        spacerItem4 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem4)
        self.btnExport = QtWidgets.QPushButton(self.centralwidget)
        self.btnExport.setEnabled(False)
        self.btnExport.setObjectName("btnExport")
        self.verticalLayout.addWidget(self.btnExport)
        spacerItem5 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout.addItem(spacerItem5)
        self.horizontalLayout_3.addLayout(self.verticalLayout)
        self.plotLayout = QtWidgets.QVBoxLayout()
        self.plotLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.plotLayout.setObjectName("plotLayout")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        spacerItem6 = QtWidgets.QSpacerItem(5, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_9.addItem(spacerItem6)
        self.labelPlotControl = QtWidgets.QLabel(self.centralwidget)
        self.labelPlotControl.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labelPlotControl.sizePolicy().hasHeightForWidth())
        self.labelPlotControl.setSizePolicy(sizePolicy)
        self.labelPlotControl.setObjectName("labelPlotControl")
        self.horizontalLayout_9.addWidget(self.labelPlotControl)
        self.btnResetZoom = QtWidgets.QToolButton(self.centralwidget)
        self.btnResetZoom.setObjectName("btnResetZoom")
        self.horizontalLayout_9.addWidget(self.btnResetZoom)
        self.btnZoom = QtWidgets.QToolButton(self.centralwidget)
        self.btnZoom.setObjectName("btnZoom")
        self.horizontalLayout_9.addWidget(self.btnZoom)
        self.btnPan = QtWidgets.QToolButton(self.centralwidget)
        self.btnPan.setObjectName("btnPan")
        self.horizontalLayout_9.addWidget(self.btnPan)
        self.plotTitle = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.plotTitle.sizePolicy().hasHeightForWidth())
        self.plotTitle.setSizePolicy(sizePolicy)
        self.plotTitle.setText("")
        self.plotTitle.setAlignment(QtCore.Qt.AlignCenter)
        self.plotTitle.setObjectName("plotTitle")
        self.horizontalLayout_9.addWidget(self.plotTitle)
        self.plotLayout.addLayout(self.horizontalLayout_9)
        self.plotCanvas = QtWidgets.QGridLayout()
        self.plotCanvas.setObjectName("plotCanvas")
        self.plotLayout.addLayout(self.plotCanvas)
        self.horizontalSlider = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider.setEnabled(False)
        self.horizontalSlider.setMaximum(1)
        self.horizontalSlider.setTracking(False)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.plotLayout.addWidget(self.horizontalSlider)
        self.horizontalLayout_3.addLayout(self.plotLayout)
        Pybridizer.setCentralWidget(self.centralwidget)

        self.retranslateUi(Pybridizer)
        QtCore.QMetaObject.connectSlotsByName(Pybridizer)

    def retranslateUi(self, Pybridizer):
        _translate = QtCore.QCoreApplication.translate
        Pybridizer.setWindowTitle(_translate("Pybridizer", "SHY BRIDE - Spike HYBRIDizer for Extracellular recordings"))
        self.btnDataSelect.setText(_translate("Pybridizer", "select data"))
        self.listClusterSelect.setCurrentText(_translate("Pybridizer", "select cluster"))
        self.listClusterSelect.setItemText(0, _translate("Pybridizer", "select cluster"))
        self.fieldWindowSize.setPlaceholderText(_translate("Pybridizer", "window size (ms)"))
        self.btnDraw.setText(_translate("Pybridizer", "calculate template"))
        self.label.setText(_translate("Pybridizer", "Display options"))
        self.radioTemplate.setText(_translate("Pybridizer", "disp&lay template"))
        self.radioFit.setText(_translate("Pybridizer", "inspect &template fit"))
        self.btnLeftSpike.setText(_translate("Pybridizer", "←"))
        self.btnRightSpike.setText(_translate("Pybridizer", "→"))
        self.checkBoxLower.setText(_translate("Pybridizer", "set energy lower bound"))
        self.checkBoxUpper.setText(_translate("Pybridizer", "set energy upper bound"))
        self.radioMove.setText(_translate("Pybridizer", "move template"))
        self.btnMoveLeft.setText(_translate("Pybridizer", "←"))
        self.btnMoveUp.setText(_translate("Pybridizer", "↑"))
        self.btnReset.setText(_translate("Pybridizer", "reset"))
        self.btnMoveDown.setText(_translate("Pybridizer", "↓"))
        self.btnMoveRight.setText(_translate("Pybridizer", "→"))
        self.btnMove.setText(_translate("Pybridizer", "execute move"))
        self.btnExport.setText(_translate("Pybridizer", "export data"))
        self.labelPlotControl.setText(_translate("Pybridizer", "plot control"))
        self.btnResetZoom.setText(_translate("Pybridizer", "reset"))
        self.btnZoom.setText(_translate("Pybridizer", "zoom"))
        self.btnPan.setText(_translate("Pybridizer", "move"))

