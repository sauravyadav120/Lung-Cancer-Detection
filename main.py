import os
import sys
import numpy as np

import cv2
import qdarkstyle
from PIL import Image

from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi

from tensorflow.keras.models import load_model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ['benign', 'malignant', 'non-nodule']

DISEASE_CLASSES = {
    0: 'benign',
    1: 'malignant',
    2: 'non-nodule'
}


class LungCancer_Prediction(QMainWindow):
    def __init__(self):
        super(LungCancer_Prediction, self).__init__()

        loadUi('MainWindow.ui', self)
        os.system('cls')

        self.train_algo_comboBox.activated.connect(self.Show_training_results)
        self.browse_pushButton.clicked.connect(self.BrowseFileDialog)
        self.Prediction_pushButton.clicked.connect(self.Classification_Function)

        self.train_algo = str(self.train_algo_comboBox.currentText())
        self.prediction_algo = str(self.prediction_algo_comboBox.currentText())

        self.qm = QMessageBox()

    @pyqtSlot()
    def Show_training_results(self):

        self.train_algo = str(self.train_algo_comboBox.currentText())

        if self.train_algo == 'DenseNet':
            img_1 = cv2.imread('./models/densenet/Model_evaluation.png')
            self.DisplayImage(img_1, 1)
            img_2 = cv2.imread('./models/densenet/conf_mat.png')
            self.DisplayImage(img_2, 2)

            text = open('./models/densenet/classification_report.txt').read()
            self.plainTextEdit.setPlainText(text)

        elif self.train_algo == 'ResNet':
            img_1 = cv2.imread('./models/resnet/Model_evaluation.png')
            self.DisplayImage(img_1, 1)
            img_2 = cv2.imread('./models/resnet/conf_mat.png')
            self.DisplayImage(img_2, 2)

            text = open('./models/resnet/classification_report.txt').read()
            self.plainTextEdit.setPlainText(text)

        elif self.train_algo == 'VGG':
            img_1 = cv2.imread('./models/vgg/Model_evaluation.png')
            self.DisplayImage(img_1, 1)
            img_2 = cv2.imread('./models/vgg/conf_mat.png')
            self.DisplayImage(img_2, 2)

            text = open('./models/vgg/classification_report.txt').read()
            self.plainTextEdit.setPlainText(text)

    @pyqtSlot()
    def BrowseFileDialog(self):
        self.fname, filter = QFileDialog.getOpenFileName(self, 'Open image File', '.\\', "image Files (*.*)")
        if self.fname:
            self.LoadImageFunction(self.fname)
        else:
            print("No Valid File selected.")

    def LoadImageFunction(self, fname):
        self.image = cv2.imread(fname)
        self.DisplayImage(self.image, 0)

    def DisplayImage(self, img, window):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if (img.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImg = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        outImg = outImg.rgbSwapped()

        if window == 0:
            self.query_imglabel.setPixmap(QPixmap.fromImage(outImg))
            self.query_imglabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
            self.query_imglabel.setScaledContents(True)
        elif window == 1:
            self.model_eval_imglabel.setPixmap(QPixmap.fromImage(outImg))
            self.model_eval_imglabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
            self.model_eval_imglabel.setScaledContents(False)
        elif window == 2:
            self.confusion_matrix_imglabel.setPixmap(QPixmap.fromImage(outImg))
            self.confusion_matrix_imglabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
            self.confusion_matrix_imglabel.setScaledContents(False)

    @pyqtSlot()
    def Classification_Function(self):
        self.prediction_algo = str(self.prediction_algo_comboBox.currentText())

        if self.prediction_algo == 'DenseNet':
            self.DenseNet_Prediction()
        elif self.prediction_algo == 'ResNet':
            self.ResNet_Prediction()
        elif self.prediction_algo == 'VGG':
            self.VGG_Prediction()
        else:
            ret = self.qm.information(self, 'Error !', 'No Algo Selected !\nPlease Select an algorithm', self.qm.Close)

    # ----------------------------------------------------------------------------------------------------------------------

    def Predict_Test_Image_File(self, model):

        print(self.fname)

        # image = Image.open(self.fname)

        image = cv2.imread(self.fname)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))

        test_data = np.array(image) / 255.0
        test_data = np.expand_dims(test_data, axis=0)

        predIdxs = model.predict(test_data)
        index = np.argmax(predIdxs, axis=1)[0]

        disease_name = DISEASE_CLASSES[index]
        print(disease_name)
        self.prediction_result_label.setText(disease_name)

    # ----------------------------------------------------------------------

    def DenseNet_Prediction(self):
        new_model = load_model('./models/densenet/DenseNet201_Lung_Cancer_model.h5')
        self.Predict_Test_Image_File(new_model)

    # ----------------------------------------------------------------------

    def ResNet_Prediction(self):
        new_model = load_model('./models/resnet/Resnet50_model.h5')
        self.Predict_Test_Image_File(new_model)

    # ----------------------------------------------------------------------

    def VGG_Prediction(self):
        new_model = load_model('./models/vgg/VGG_model.h5')
        self.Predict_Test_Image_File(new_model)


# ----------------------------------------------------------------------


''' ------------------------ MAIN Function ------------------------- '''

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window = LungCancer_Prediction()
    window.show()
    sys.exit(app.exec_())
