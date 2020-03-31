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

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_DialogInsertTemplate(object):
    def setupUi(self, DialogInsertTemplate):
        DialogInsertTemplate.setObjectName("DialogInsertTemplate")
        DialogInsertTemplate.resize(400, 186)
        self.boxSNR = QtWidgets.QDoubleSpinBox(DialogInsertTemplate)
        self.boxSNR.setGeometry(QtCore.QRect(190, 30, 71, 27))
        self.boxSNR.setMinimum(-99.99)
        self.boxSNR.setSingleStep(0.5)
        self.boxSNR.setProperty("value", 10.0)
        self.boxSNR.setObjectName("boxSNR")
        self.label_9 = QtWidgets.QLabel(DialogInsertTemplate)
        self.label_9.setGeometry(QtCore.QRect(50, 90, 121, 21))
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(DialogInsertTemplate)
        self.label_10.setGeometry(QtCore.QRect(290, 60, 121, 21))
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(DialogInsertTemplate)
        self.label_11.setGeometry(QtCore.QRect(290, 90, 121, 21))
        self.label_11.setObjectName("label_11")
        self.label_3 = QtWidgets.QLabel(DialogInsertTemplate)
        self.label_3.setGeometry(QtCore.QRect(50, 30, 121, 21))
        self.label_3.setObjectName("label_3")
        self.boxRate = QtWidgets.QDoubleSpinBox(DialogInsertTemplate)
        self.boxRate.setGeometry(QtCore.QRect(190, 60, 71, 27))
        self.boxRate.setMinimum(0.0)
        self.boxRate.setMaximum(9999.0)
        self.boxRate.setSingleStep(5.0)
        self.boxRate.setProperty("value", 10.0)
        self.boxRate.setObjectName("boxRate")
        self.label_5 = QtWidgets.QLabel(DialogInsertTemplate)
        self.label_5.setGeometry(QtCore.QRect(290, 30, 121, 21))
        self.label_5.setObjectName("label_5")
        self.label_8 = QtWidgets.QLabel(DialogInsertTemplate)
        self.label_8.setGeometry(QtCore.QRect(50, 60, 121, 21))
        self.label_8.setObjectName("label_8")
        self.boxRefr = QtWidgets.QDoubleSpinBox(DialogInsertTemplate)
        self.boxRefr.setGeometry(QtCore.QRect(190, 90, 71, 27))
        self.boxRefr.setMinimum(0.0)
        self.boxRefr.setMaximum(9999.0)
        self.boxRefr.setSingleStep(5.0)
        self.boxRefr.setProperty("value", 2.0)
        self.boxRefr.setObjectName("boxRefr")
        self.buttonBox = QtWidgets.QDialogButtonBox(DialogInsertTemplate)
        self.buttonBox.setEnabled(True)
        self.buttonBox.setGeometry(QtCore.QRect(190, 140, 176, 27))
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setCenterButtons(False)
        self.buttonBox.setObjectName("buttonBox")

        self.retranslateUi(DialogInsertTemplate)
        QtCore.QMetaObject.connectSlotsByName(DialogInsertTemplate)

    def retranslateUi(self, DialogInsertTemplate):
        _translate = QtCore.QCoreApplication.translate
        DialogInsertTemplate.setWindowTitle(_translate("DialogInsertTemplate", "insert template"))
        self.label_9.setText(_translate("DialogInsertTemplate", "refractory period"))
        self.label_10.setText(_translate("DialogInsertTemplate", "(Hz)"))
        self.label_11.setText(_translate("DialogInsertTemplate", "(ms)"))
        self.label_3.setText(_translate("DialogInsertTemplate", "template SNR"))
        self.label_5.setText(_translate("DialogInsertTemplate", "(dB)"))
        self.label_8.setText(_translate("DialogInsertTemplate", "spike rate"))

