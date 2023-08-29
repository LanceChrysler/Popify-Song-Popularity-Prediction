from Assets.Ui.main_UI import Ui_MainWindow
from Assets.Ui.predict_UI import Ui_PredictWindow
from Assets.Ui.train_UI import Ui_TrainWindow
from Model.retrain_Model import retrain_Model_Tuned
from Model.retrain_Model_fixed import retrain_Model
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt6.QtGui import QIcon
import pandas as pd
import joblib

class Window(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.predictButton.clicked.connect(self.openPredictWindow)
        self.trainButton.clicked.connect(self.openTrainWindow)

    def openPredictWindow(self):
        self.window = QMainWindow()
        self.ui = Ui_PredictWindow()
        self.ui.setupUi(self.window)
        #window.hide()
        Ui_PredictWindow.UI_sytleSheet(self.window)
        self.window.show() 
        
        self.ui.uploadButton.clicked.connect(self.openSampleFileDialog)
        self.ui.clearButton.clicked.connect(self.clearFields)
        self.ui.predictButton.clicked.connect(self.predict)

    def openSampleFileDialog(self):
        dialog = QFileDialog()
        dialog.setNameFilter("*.csv")
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialogSuccessful = dialog.exec()

        if dialogSuccessful:
            fileLocation = dialog.selectedFiles()[0]
            features_df = pd.read_csv(fileLocation)

            self.ui.fileLineEdit.setText(fileLocation)
            self.ui.titleLineEdit.setText(features_df.at[0, 'title'])
            self.ui.topGenreLineEdit.setText(features_df.at[0, 'top genre'])
            self.ui.yearSpinBox.setValue(features_df.at[0, 'year'])
            self.ui.bpmSpinBox.setValue(features_df.at[0, 'bpm'])
            self.ui.nrgySpinBox.setValue(features_df.at[0, 'nrgy'])
            self.ui.dnceSpinBox.setValue(features_df.at[0, 'dnce'])
            self.ui.dBSpinBox.setValue(features_df.at[0, 'dB'])
            self.ui.liveSpinBox.setValue(features_df.at[0, 'live'])
            self.ui.valSpinBox.setValue(features_df.at[0, 'val'])
            self.ui.durSpinBox.setValue(features_df.at[0, 'dur'])
            self.ui.acousSpinBox.setValue(features_df.at[0, 'acous'])
            self.ui.spchSpinBox.setValue(features_df.at[0, 'spch'])
    
    def clearFields(self):
        #Line Edit
        fileLocation = self.ui.fileLineEdit.text()
        if fileLocation.strip() != "":
            self.ui.fileLineEdit.setText("")
        title = self.ui.titleLineEdit.text()
        if title.strip() != "":
            self.ui.titleLineEdit.setText("")
        topGenre = self.ui.topGenreLineEdit.text()
        if topGenre.strip() != "":
            self.ui.topGenreLineEdit.setText("")
        #Spin Box
        self.ui.yearSpinBox.clear()
        self.ui.dnceSpinBox.clear()
        self.ui.liveSpinBox.clear()
        self.ui.spchSpinBox.clear()
        self.ui.bpmSpinBox.clear()
        self.ui.dBSpinBox.clear()
        self.ui.valSpinBox.clear()
        self.ui.nrgySpinBox.clear()
        self.ui.acousSpinBox.clear()
        self.ui.durSpinBox.clear()
        #Prediction Label
        predictionValue_str = self.ui.predictionValueLabel.text()
        if predictionValue_str.strip() != "":
            self.ui.predictionValueLabel.setText("")

    def predict(self):
        if self.ui.topGenreLineEdit.text().strip() == "":
            self.showWarningDialog("fill in the required fields.")
            return

        features_df = pd.read_csv("predictors.csv")

        features_df.iloc[0, :] = 0

        features_df.loc[0, 'year'] = self.ui.yearSpinBox.value()
        features_df.loc[0, 'bpm'] = self.ui.bpmSpinBox.value()
        features_df.loc[0, 'nrgy'] = self.ui.nrgySpinBox.value()
        features_df.loc[0, 'dnce'] = self.ui.dnceSpinBox.value()
        features_df.loc[0, 'dB'] = self.ui.dBSpinBox.value()
        features_df.loc[0, 'live'] = self.ui.liveSpinBox.value()
        features_df.loc[0, 'val'] = self.ui.valSpinBox.value()
        features_df.loc[0, 'dur'] = self.ui.durSpinBox.value()
        features_df.loc[0, 'acous'] = self.ui.acousSpinBox.value()
        features_df.loc[0, 'spch'] = self.ui.spchSpinBox.value()
        features_df.loc[0, f"top genre_{self.ui.topGenreLineEdit.text()}"] = 1

        features_df.to_csv("predictors.csv", index=False)
        #remove feature names
        features_df = features_df.values

        regressor = joblib.load("Decision_Tree_Regressor")

        result = regressor.predict(features_df)
        print(result[0].astype(int))
        self.ui.predictionValueLabel.setText(str(result[0].astype(int)))

    def openTrainWindow(self):
        self.window = QMainWindow()
        self.ui = Ui_TrainWindow()
        self.ui.setupUi(self.window)
        #window.hide()
        Ui_TrainWindow.UI_sytleSheet(self.window)
        self.window.show() 

        dataset = pd.read_csv("Model/Model_History.csv")
        self.ui.dataValueLabel.setText(str(dataset.at[len(dataset) - 1, 'data']))
        self.ui.mseValueLabel.setText(str(dataset.at[len(dataset) - 1, 'mse']))
        self.ui.rmseValueLabel.setText(str(dataset.at[len(dataset) - 1, 'rmse']))

        self.ui.uploadButton.clicked.connect(self.openDatasetFileDialog)
        self.ui.trainButton.clicked.connect(self.retainModel)
        self.ui.tunedTrainButton.clicked.connect(self.retainModel_Tuned)
        
    def openDatasetFileDialog(self):
        dialog = QFileDialog()
        dialog.setNameFilter("*.csv")
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialogSuccessful = dialog.exec()

        if dialogSuccessful:
            fileLocation = dialog.selectedFiles()[0]
            self.ui.fileNameLabel.setText(fileLocation)

            dataset = pd.read_csv(fileLocation)
            self.ui.fileDataValueLabel.setText(str(len(dataset)))
            
    def retainModel_Tuned(self):
        if self.ui.fileNameLabel.text() == 'File Name':
            self.showWarningDialog("upload a file.")
            return
        
        dataset = pd.read_csv(self.ui.fileNameLabel.text())

        retrain_Model_Tuned(dataset)
        self.showSuccessDialog()

    def retainModel(self):
        if self.ui.fileNameLabel.text() == 'File Name':
            self.showWarningDialog("upload a file.")
            return
        
        dataset = pd.read_csv(self.ui.fileNameLabel.text())
        
        retrain_Model(dataset)
        self.showSuccessDialog()

    def showWarningDialog(self, text):
        dialog = QMessageBox()
        dialog.setText(f"Please {text}")
        dialog.setWindowTitle("Warning")
        dialog.setWindowIcon(QIcon("Assets/Images/fire.png"))
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.exec()

    def showSuccessDialog(self):
        dialog = QMessageBox()
        dialog.setText("Model Training Successful")
        dialog.setWindowTitle("Model Update")
        dialog.setWindowIcon(QIcon("Assets/Images/fire.png"))
        dialog.setIcon(QMessageBox.Icon.Information)
        dialog.buttonClicked.connect(self.dialog_clicked)
        dialog.exec()
        
    def dialog_clicked(self):
        self.window.close()
        self.openTrainWindow()

app = QApplication([])
window = Window()

window.show()
app.exec()