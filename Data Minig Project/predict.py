from Assets.Ui.predict_UI import Ui_PredictWindow
from PyQt6.QtWidgets import QApplication, QMainWindow

def openPredictWindow(self):
    self.predictWindow = QMainWindow()
    self.ui = Ui_PredictWindow()
    self.ui.setupUi(self.window)
    #window.hide()
    self.window.show() 
    self.ui.clearButton.clicked.connect(self.clearFields)

def clearFields(self):
    print("Clicked")
    title_str = self.lineEdit.text()

app = QApplication([])