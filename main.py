import Image_classify
from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt
from PyQt5.QtCore import *
import sys
from predict import predict_
from PIL import Image

class MainDialog(QDialog):
    def __init__(self, parent=None):
        super(QDialog, self).__init__(parent)
        self.ui = Image_classify.Ui_Dialog()
        self.ui.setupUi(self)
        self.setWindowTitle("CLFAR-10 十分类")
        self.setWindowIcon(QIcon('1.png'))

    def openImage(self):
        global fname
        imgName, imgType = QFileDialog.getOpenFileName(self, "选择图片", "", "*.jpg;;*.png;;All Files(*)")
        jpg = QPixmap(imgName).scaled(self.ui.label_image.width(), self.ui.label_image.height())
        self.ui.label_image.setPixmap(jpg)
        fname = imgName

    def run(self):
        global fname
        file_name = str(fname)
        img = Image.open(file_name)

        a, b = predict_(img)
        self.ui.display_result.setText(a)
        self.ui.disply_acc.setText(str(b))

if __name__ == '__main__':
    myapp = QApplication(sys.argv)
    myDlg =MainDialog()
    myDlg.show()
    sys.exit(myapp.exec_())

