# Form implementation generated from reading ui file 'train_UI.ui'
#
# Created by: PyQt6 UI code generator 6.5.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtGui import QIcon


class Ui_TrainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setFixedSize(640, 450)
        MainWindow.setWindowIcon(QIcon("Assets/Images/fire.png"))
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayoutWidget = QtWidgets.QWidget(parent=self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(0, 0, 641, 211))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.dataLabel = QtWidgets.QLabel(parent=self.gridLayoutWidget)
        self.dataLabel.setObjectName("dataLabel")
        self.gridLayout.addWidget(self.dataLabel, 1, 0, 1, 1)
        self.modelLabel = QtWidgets.QLabel(parent=self.gridLayoutWidget)
        self.modelLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.modelLabel.setObjectName("modelLabel")
        self.gridLayout.addWidget(self.modelLabel, 0, 0, 1, 2)
        self.mseValueLabel = QtWidgets.QLabel(parent=self.gridLayoutWidget)
        self.mseValueLabel.setText("")
        self.mseValueLabel.setObjectName("mseValueLabel")
        self.gridLayout.addWidget(self.mseValueLabel, 2, 1, 1, 1)
        self.mseLabel = QtWidgets.QLabel(parent=self.gridLayoutWidget)
        self.mseLabel.setObjectName("mseLabel")
        self.gridLayout.addWidget(self.mseLabel, 2, 0, 1, 1)
        self.dataValueLabel = QtWidgets.QLabel(parent=self.gridLayoutWidget)
        self.dataValueLabel.setText("")
        self.dataValueLabel.setObjectName("dataValueLabel")
        self.gridLayout.addWidget(self.dataValueLabel, 1, 1, 1, 1)
        self.rmseLabel = QtWidgets.QLabel(parent=self.gridLayoutWidget)
        self.rmseLabel.setObjectName("rmseLabel")
        self.gridLayout.addWidget(self.rmseLabel, 3, 0, 1, 1)
        self.rmseValueLabel = QtWidgets.QLabel(parent=self.gridLayoutWidget)
        self.rmseValueLabel.setText("")
        self.rmseValueLabel.setObjectName("rmseValueLabel")
        self.gridLayout.addWidget(self.rmseValueLabel, 3, 1, 1, 1)
        self.gridLayoutWidget_2 = QtWidgets.QWidget(parent=self.centralwidget)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(0, 260, 641, 75))
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.uploadButton = QtWidgets.QPushButton(parent=self.gridLayoutWidget_2)
        self.uploadButton.setObjectName("uploadButton")
        self.gridLayout_2.addWidget(self.uploadButton, 0, 0, 1, 1)
        self.fileNameLabel = QtWidgets.QLabel(parent=self.gridLayoutWidget_2)
        self.fileNameLabel.setObjectName("fileNameLabel")
        self.gridLayout_2.addWidget(self.fileNameLabel, 0, 1, 1, 1)
        self.fileDataLabel = QtWidgets.QLabel(parent=self.gridLayoutWidget_2)
        self.fileDataLabel.setObjectName("fileDataLabel")
        self.gridLayout_2.addWidget(self.fileDataLabel, 1, 0, 1, 1)
        self.fileDataValueLabel = QtWidgets.QLabel(parent=self.gridLayoutWidget_2)
        self.fileDataValueLabel.setText("")
        self.fileDataValueLabel.setObjectName("fileDataValueLabel")
        self.gridLayout_2.addWidget(self.fileDataValueLabel, 1, 1, 1, 1)
        self.verticalLayoutWidget = QtWidgets.QWidget(parent=self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 210, 641, 51))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.retrainLabel = QtWidgets.QLabel(parent=self.verticalLayoutWidget)
        self.retrainLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.retrainLabel.setObjectName("retrainLabel")
        self.verticalLayout.addWidget(self.retrainLabel)
        self.gridLayoutWidget_3 = QtWidgets.QWidget(parent=self.centralwidget)
        self.gridLayoutWidget_3.setGeometry(QtCore.QRect(0, 350, 641, 80))
        self.gridLayoutWidget_3.setObjectName("gridLayoutWidget_3")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.gridLayoutWidget_3)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.trainButton = QtWidgets.QPushButton(parent=self.gridLayoutWidget_3)
        self.trainButton.setObjectName("trainButton")
        self.gridLayout_3.addWidget(self.trainButton, 0, 0, 1, 1)
        self.tunedTrainButton = QtWidgets.QPushButton(parent=self.gridLayoutWidget_3)
        self.tunedTrainButton.setObjectName("tunedTrainButton")
        self.gridLayout_3.addWidget(self.tunedTrainButton, 1, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Train Model"))
        self.dataLabel.setText(_translate("MainWindow", "No. of Data"))
        self.modelLabel.setText(_translate("MainWindow", "Current Model"))
        self.mseLabel.setText(_translate("MainWindow", "MSE"))
        self.rmseLabel.setText(_translate("MainWindow", "RMSE"))
        self.uploadButton.setText(_translate("MainWindow", "Upload File"))
        self.fileNameLabel.setText(_translate("MainWindow", "File Name"))
        self.fileDataLabel.setText(_translate("MainWindow", "No. of Data"))
        self.retrainLabel.setText(_translate("MainWindow", "Retrain Model"))
        self.trainButton.setText(_translate("MainWindow", "Train Model"))
        self.tunedTrainButton.setText(_translate("MainWindow", "Train Model (With Tuning)"))

    def UI_sytleSheet(window):
        window.setStyleSheet('''
            QMainWindow{
                background-color: QLinearGradient(x0: 0, y0: 0, x1: 1, y1: 1, stop: 0 #2EBD59, stop: 1 #000000);
            }
            QPushButton{
                background-color: #FFFFFF;
                color: #000000;
                width: 100%;
                border-radius: 1px;
                margin-left: 10px;
                margin-right: 10px;
            }
            QPushButton:hover{
                background-color: #2EBD59;
            }
            #windowLabel{
                font-size: 30px;
                color: #2EBD59;
                margin-bottom: 20px;
            }
            #modelLabel, #retrainLabel{
                font-size: 18px;
                font-weight: bold;
                text-decoration: underline;
                color: #2EBD59;
            }
            #dataLabel, #mseLabel, #rmseLabel, #fileDataLabel{
                margin-left: 180px;
            }
            QLabel{
                color: #FFFFFF;
                width: 50px;
            }
            *{
                font-family: "Gotham";
                font-size: 15px;
                padding-top: 5px;
                padding-bottom: 5px;
                color: #000000;
            }
        ''')
