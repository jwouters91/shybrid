# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'import_template.ui'
#
# Created by: PyQt5 UI code generator 5.5.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_DialogTemplateImport(object):
    def setupUi(self, DialogTemplateImport):
        DialogTemplateImport.setObjectName("DialogTemplateImport")
        DialogTemplateImport.setEnabled(True)
        DialogTemplateImport.resize(402, 186)
        self.buttonBox = QtWidgets.QDialogButtonBox(DialogTemplateImport)
        self.buttonBox.setEnabled(False)
        self.buttonBox.setGeometry(QtCore.QRect(200, 140, 176, 27))
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setCenterButtons(False)
        self.buttonBox.setObjectName("buttonBox")
        self.btnSelectTemplate = QtWidgets.QPushButton(DialogTemplateImport)
        self.btnSelectTemplate.setGeometry(QtCore.QRect(30, 20, 141, 27))
        self.btnSelectTemplate.setAutoDefault(False)
        self.btnSelectTemplate.setObjectName("btnSelectTemplate")
        self.nbChannels = QtWidgets.QLabel(DialogTemplateImport)
        self.nbChannels.setGeometry(QtCore.QRect(180, 20, 151, 31))
        self.nbChannels.setText("")
        self.nbChannels.setObjectName("nbChannels")
        self.boxReach = QtWidgets.QSpinBox(DialogTemplateImport)
        self.boxReach.setGeometry(QtCore.QRect(200, 70, 51, 27))
        self.boxReach.setMinimum(1)
        self.boxReach.setMaximum(99)
        self.boxReach.setProperty("value", 1)
        self.boxReach.setObjectName("boxReach")
        self.label_2 = QtWidgets.QLabel(DialogTemplateImport)
        self.label_2.setGeometry(QtCore.QRect(40, 70, 121, 21))
        self.label_2.setObjectName("label_2")
        self.label_4 = QtWidgets.QLabel(DialogTemplateImport)
        self.label_4.setGeometry(QtCore.QRect(280, 70, 121, 21))
        self.label_4.setObjectName("label_4")
        self.label_6 = QtWidgets.QLabel(DialogTemplateImport)
        self.label_6.setGeometry(QtCore.QRect(40, 100, 121, 21))
        self.label_6.setObjectName("label_6")
        self.boxOffset = QtWidgets.QSpinBox(DialogTemplateImport)
        self.boxOffset.setGeometry(QtCore.QRect(200, 100, 51, 27))
        self.boxOffset.setMinimum(0)
        self.boxOffset.setMaximum(99)
        self.boxOffset.setProperty("value", 0)
        self.boxOffset.setObjectName("boxOffset")
        self.label_7 = QtWidgets.QLabel(DialogTemplateImport)
        self.label_7.setGeometry(QtCore.QRect(280, 100, 121, 21))
        self.label_7.setObjectName("label_7")

        self.retranslateUi(DialogTemplateImport)
        QtCore.QMetaObject.connectSlotsByName(DialogTemplateImport)

    def retranslateUi(self, DialogTemplateImport):
        _translate = QtCore.QCoreApplication.translate
        DialogTemplateImport.setWindowTitle(_translate("DialogTemplateImport", "Import template"))
        self.btnSelectTemplate.setText(_translate("DialogTemplateImport", "Select template"))
        self.label_2.setText(_translate("DialogTemplateImport", "horizontal reach"))
        self.label_4.setText(_translate("DialogTemplateImport", "(channels)"))
        self.label_6.setText(_translate("DialogTemplateImport", "horizontal offset"))
        self.label_7.setText(_translate("DialogTemplateImport", "(channels)"))

