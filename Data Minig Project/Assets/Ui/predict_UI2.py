# Form implementation generated from reading ui file 'predict.ui'
#
# Created by: PyQt6 UI code generator 6.5.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_PredictWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("PredictWindow")
        MainWindow.resize(1344, 900)
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalSlider = QtWidgets.QSlider(parent=self.centralwidget)
        self.horizontalSlider.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.gridLayout.addWidget(self.horizontalSlider, 10, 7, 1, 1)
        self.acousLabel = QtWidgets.QLabel(parent=self.centralwidget)
        self.acousLabel.setObjectName("acousLabel")
        self.gridLayout.addWidget(self.acousLabel, 5, 6, 1, 1)
        self.horizontalSlider_4 = QtWidgets.QSlider(parent=self.centralwidget)
        self.horizontalSlider_4.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.horizontalSlider_4.setObjectName("horizontalSlider_4")
        self.gridLayout.addWidget(self.horizontalSlider_4, 11, 7, 1, 1)
        self.spchLabel = QtWidgets.QLabel(parent=self.centralwidget)
        self.spchLabel.setObjectName("spchLabel")
        self.gridLayout.addWidget(self.spchLabel, 9, 2, 1, 1)
        self.titleLineEdit = QtWidgets.QLineEdit(parent=self.centralwidget)
        self.titleLineEdit.setObjectName("titleLineEdit")
        self.gridLayout.addWidget(self.titleLineEdit, 1, 3, 1, 1)
        self.bpmLabel = QtWidgets.QLabel(parent=self.centralwidget)
        self.bpmLabel.setObjectName("bpmLabel")
        self.gridLayout.addWidget(self.bpmLabel, 10, 2, 1, 1)
        self.dBLabel = QtWidgets.QLabel(parent=self.centralwidget)
        self.dBLabel.setObjectName("dBLabel")
        self.gridLayout.addWidget(self.dBLabel, 11, 2, 1, 1)
        self.valLabel = QtWidgets.QLabel(parent=self.centralwidget)
        self.valLabel.setObjectName("valLabel")
        self.gridLayout.addWidget(self.valLabel, 12, 2, 1, 1)
        self.horizontalSlider_6 = QtWidgets.QSlider(parent=self.centralwidget)
        self.horizontalSlider_6.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.horizontalSlider_6.setObjectName("horizontalSlider_6")
        self.gridLayout.addWidget(self.horizontalSlider_6, 12, 7, 1, 1)
        self.dnceLabel = QtWidgets.QLabel(parent=self.centralwidget)
        self.dnceLabel.setObjectName("dnceLabel")
        self.gridLayout.addWidget(self.dnceLabel, 5, 2, 1, 1)
        self.liveLabel = QtWidgets.QLabel(parent=self.centralwidget)
        self.liveLabel.setObjectName("liveLabel")
        self.gridLayout.addWidget(self.liveLabel, 6, 2, 1, 1)
        self.top_genreComboBox = QtWidgets.QComboBox(parent=self.centralwidget)
        self.top_genreComboBox.setObjectName("top_genreComboBox")
        self.gridLayout.addWidget(self.top_genreComboBox, 1, 7, 1, 1)
        self.bpmSpinBox = QtWidgets.QSpinBox(parent=self.centralwidget)
        self.bpmSpinBox.setObjectName("bpmSpinBox")
        self.gridLayout.addWidget(self.bpmSpinBox, 10, 3, 1, 1)
        self.titleLabel = QtWidgets.QLabel(parent=self.centralwidget)
        self.titleLabel.setObjectName("titleLabel")
        self.gridLayout.addWidget(self.titleLabel, 1, 2, 1, 1)
        self.spchSpinBox = QtWidgets.QSpinBox(parent=self.centralwidget)
        self.spchSpinBox.setObjectName("spchSpinBox")
        self.gridLayout.addWidget(self.spchSpinBox, 9, 3, 1, 1)
        self.nrgySpinBox = QtWidgets.QSpinBox(parent=self.centralwidget)
        self.nrgySpinBox.setObjectName("nrgySpinBox")
        self.gridLayout.addWidget(self.nrgySpinBox, 4, 7, 1, 1)
        self.line = QtWidgets.QFrame(parent=self.centralwidget)
        self.line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line.setObjectName("line")
        self.gridLayout.addWidget(self.line, 13, 2, 1, 7)
        self.nrgyLabel = QtWidgets.QLabel(parent=self.centralwidget)
        self.nrgyLabel.setObjectName("nrgyLabel")
        self.gridLayout.addWidget(self.nrgyLabel, 4, 6, 1, 1)
        self.durLabel = QtWidgets.QLabel(parent=self.centralwidget)
        self.durLabel.setObjectName("durLabel")
        self.gridLayout.addWidget(self.durLabel, 6, 6, 1, 1)
        self.logoLabel = QtWidgets.QLabel(parent=self.centralwidget)
        self.logoLabel.setObjectName("logoLabel")
        self.gridLayout.addWidget(self.logoLabel, 0, 2, 1, 5)
        self.yearLabel = QtWidgets.QLabel(parent=self.centralwidget)
        self.yearLabel.setObjectName("yearLabel")
        self.gridLayout.addWidget(self.yearLabel, 4, 2, 1, 1)
        self.acousSpinBox = QtWidgets.QSpinBox(parent=self.centralwidget)
        self.acousSpinBox.setObjectName("acousSpinBox")
        self.gridLayout.addWidget(self.acousSpinBox, 5, 7, 1, 1)
        self.valSpinBox = QtWidgets.QSpinBox(parent=self.centralwidget)
        self.valSpinBox.setObjectName("valSpinBox")
        self.gridLayout.addWidget(self.valSpinBox, 12, 3, 1, 1)
        self.durSpinBox = QtWidgets.QSpinBox(parent=self.centralwidget)
        self.durSpinBox.setObjectName("durSpinBox")
        self.gridLayout.addWidget(self.durSpinBox, 6, 7, 1, 1)
        self.songPopLabel = QtWidgets.QLabel(parent=self.centralwidget)
        self.songPopLabel.setObjectName("songPopLabel")
        self.gridLayout.addWidget(self.songPopLabel, 14, 3, 1, 1)
        self.yearSpinBox = QtWidgets.QSpinBox(parent=self.centralwidget)
        self.yearSpinBox.setObjectName("yearSpinBox")
        self.gridLayout.addWidget(self.yearSpinBox, 4, 3, 1, 1)
        self.liveSpinBox = QtWidgets.QSpinBox(parent=self.centralwidget)
        self.liveSpinBox.setObjectName("liveSpinBox")
        self.gridLayout.addWidget(self.liveSpinBox, 6, 3, 1, 1)
        self.dBSpinBox = QtWidgets.QSpinBox(parent=self.centralwidget)
        self.dBSpinBox.setObjectName("dBSpinBox")
        self.gridLayout.addWidget(self.dBSpinBox, 11, 3, 1, 1)
        self.dnceSpinBox = QtWidgets.QSpinBox(parent=self.centralwidget)
        self.dnceSpinBox.setObjectName("dnceSpinBox")
        self.gridLayout.addWidget(self.dnceSpinBox, 5, 3, 1, 1)
        self.top_genreLabel = QtWidgets.QLabel(parent=self.centralwidget)
        self.top_genreLabel.setObjectName("top_genreLabel")
        self.gridLayout.addWidget(self.top_genreLabel, 1, 6, 1, 1)
        self.clearButton = QtWidgets.QPushButton(parent=self.centralwidget)
        self.clearButton.setObjectName("clearButton")
        self.gridLayout.addWidget(self.clearButton, 15, 3, 1, 1)
        self.predictButton = QtWidgets.QPushButton(parent=self.centralwidget)
        self.predictButton.setObjectName("predictButton")
        self.gridLayout.addWidget(self.predictButton, 15, 7, 1, 1)
        self.predictLabel = QtWidgets.QLabel(parent=self.centralwidget)
        self.predictLabel.setObjectName("predictLabel")
        self.gridLayout.addWidget(self.predictLabel, 14, 7, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Song Popularity Prediction"))
        self.acousLabel.setText(_translate("MainWindow", "Acousticness"))
        self.spchLabel.setText(_translate("MainWindow", "Speechiness"))
        self.bpmLabel.setText(_translate("MainWindow", "BPM"))
        self.dBLabel.setText(_translate("MainWindow", "Loudness (dB)"))
        self.valLabel.setText(_translate("MainWindow", "Valence"))
        self.dnceLabel.setText(_translate("MainWindow", "Danceability"))
        self.liveLabel.setText(_translate("MainWindow", "Liveliness"))
        self.titleLabel.setText(_translate("MainWindow", "Title"))
        self.nrgyLabel.setText(_translate("MainWindow", "Energy"))
        self.durLabel.setText(_translate("MainWindow", "Duration"))
        self.logoLabel.setText(_translate("MainWindow", "Pop-ify"))
        self.yearLabel.setText(_translate("MainWindow", "Year"))
        self.songPopLabel.setText(_translate("MainWindow", "Song Popularity"))
        self.top_genreLabel.setText(_translate("MainWindow", "Top Genre"))
        self.clearButton.setText(_translate("MainWindow", "Clear"))
        self.predictButton.setText(_translate("MainWindow", "Predict"))
        self.predictLabel.setText(_translate("MainWindow", "-"))