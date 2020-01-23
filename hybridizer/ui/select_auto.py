# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'select_auto.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


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
