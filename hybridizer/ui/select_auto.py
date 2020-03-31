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


class Ui_AutoSelect(object):
    def setupUi(self, AutoSelect):
        AutoSelect.setObjectName("AutoSelect")
        AutoSelect.resize(405, 300)
        self.buttonBox = QtWidgets.QDialogButtonBox(AutoSelect)
        self.buttonBox.setGeometry(QtCore.QRect(40, 240, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.listWidget = QtWidgets.QListWidget(AutoSelect)
        self.listWidget.setGeometry(QtCore.QRect(30, 61, 351, 171))
        self.listWidget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.listWidget.setObjectName("listWidget")
        self.label = QtWidgets.QLabel(AutoSelect)
        self.label.setGeometry(QtCore.QRect(70, 10, 281, 17))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(AutoSelect)
        self.label_2.setGeometry(QtCore.QRect(120, 30, 171, 17))
        self.label_2.setObjectName("label_2")

        self.retranslateUi(AutoSelect)
        self.buttonBox.accepted.connect(AutoSelect.accept)
        self.buttonBox.rejected.connect(AutoSelect.reject)
        QtCore.QMetaObject.connectSlotsByName(AutoSelect)

    def retranslateUi(self, AutoSelect):
        _translate = QtCore.QCoreApplication.translate
        AutoSelect.setWindowTitle(_translate("AutoSelect", "cluster select"))
        self.label.setText(_translate("AutoSelect", "Select the clusters for auto hybridization"))
        self.label_2.setText(_translate("AutoSelect", "(all selected by default)"))
