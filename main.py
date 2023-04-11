#ui widget window from ui file
from PyQt5 import QtWidgets
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QPixmap
from PyQt5 import QtCore
import sys
import os
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
import cv2
import matplotlib.pyplot as plt
import numpy as np
#keyboard input library
sam_checkpoint = "models/sam_vit_b_01ec64.pth"
model_type = "vit_b"

device = "cpu"
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

#load ui file
qtCreatorFile = "window.ui" # Enter file here.
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.show()
        self.openFile.clicked.connect(self.open_file)
        self.addClass.clicked.connect(self.add_class)
        self.removeClass.clicked.connect(self.delete_class)
        self.startButton.clicked.connect(self.start)
        self.classWidget.itemClicked.connect(self.class_selected)
        self.input_point = np.array([])
        self.input_label = np.array([])
        self.this_mask = {}
        self.image = None


    def delete_class(self):
        #delete class from classWidget
        class_name = self.classWidget.currentItem().text()
        self.classWidget.takeItem(self.classWidget.currentRow())
        #delete mask file if it exists
        if os.path.exists(f"masks/{class_name}.png"):
            os.remove(f"masks/{class_name}.png")
        #delete mask from this_mask if it exists
        if class_name in self.this_mask.keys():
            del self.this_mask[class_name]
        #clear mask label
        self.maskLabel.clear()

    def class_selected(self):
        #get selected class name if there is one
        if self.classWidget.currentItem() == None:
            return
        class_name = self.classWidget.currentItem().text()
        if class_name == "":
            return
        #load mask from file
        #check if class name in this mask keys
        if self.this_mask is not None:
            if class_name in self.this_mask.keys():
                mask = self.this_mask[class_name]
                #save mask to file
                cv2.imwrite(f"masks/{class_name}.png", mask*255)
                #show mask in maskLabel
                pixmap = QPixmap(f"masks/{class_name}.png")
                #resize image to fit in label
                pixmap = pixmap.scaled(self.maskLabel.width(), self.maskLabel.height(), QtCore.Qt.KeepAspectRatio)
                self.maskLabel.setPixmap(pixmap)




    def start(self):
        #get selected class name if there is one
        if self.classWidget.count() == 0:
            #no classes added message box
            QtWidgets.QMessageBox.about(self, "Error", "No classes added")
            return
        if self.classWidget.currentItem() == None:
            #no class selected message box
            QtWidgets.QMessageBox.about(self, "Error", "No class selected")
            return
        class_name = self.classWidget.currentItem().text()
        if class_name == "":
            #no class selected message box
            QtWidgets.QMessageBox.about(self, "Error", "No class selected")

            return
        #clear points
        self.input_point = np.array([])
        self.input_label = np.array([])

        plt.figure(figsize=(10,10))
        plt.imshow(self.image)
        #show_points(self.input_point, self.input_label, plt)
        plt.axis('on')
        #on key or mouse click
        cid = plt.gcf().canvas.mpl_connect('button_press_event', self.onclick)
        plt.show()

        


        #save the mask to file with class name

        #cv2.imwrite(f"{class_name}.png", self.this_mask*255)
        
    
    def add_class(self):
        #add class to classWidget
        class_name = self.classEdit.text()
        self.classWidget.addItem(class_name)
        self.classEdit.setText("")


    def open_file(self):

        print("open file")
        #get file path only image files. gif, jpg, png, bmp,tiff, tif, jpeg
        file_path = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '.', "Image files (*.jpg *.gif *.png *.bmp *.tiff *.tif *.jpeg)")[0]
        print(file_path)
        self.fileEdit.setText(file_path)
        #load image and show in imageLabel
        pixmap = QPixmap(file_path)
        #resize image to fit in label
        pixmap = pixmap.scaled(self.imageLabel.width(), self.imageLabel.height(), QtCore.Qt.KeepAspectRatio)

        self.imageLabel.setPixmap(pixmap)
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image = image
        #show wait message with progress bar
        

        predictor.set_image(image)
        
        #show final message

#get mouse click points
    def onclick(self,event):
        x, y = event.xdata, event.ydata
        print(f"Clicked at {x:.0f}, {y:.0f}")
        #check which mouse button was pressed
        if event.button == 2:

            #left click
            #check if ctrl key is pressed
            
            print("Removing point")
            #remove the point
            i = np.argmin(np.sum(np.square(self.input_point - np.array([[x, y]])), axis=1))
            #check if array is empty
            if self.input_point.shape[0] == 1:
                self.input_point = np.array([])
                self.input_label = np.array([])
            else:
                self.input_point = np.delete(self.input_point, i, axis=0)
                self.input_label = np.delete(self.input_label, i, axis=0)
        elif event.button == 1:         
            print("Adding positive point")
            if self.input_point.shape[0] == 0:
                self.input_point = np.array([[x, y]])
                self.input_label = np.array([1])
            else:
                self.input_label = np.concatenate([self.input_label, [1]], axis=0)
                self.input_point = np.concatenate([self.input_point, [[x, y]]], axis=0)

        elif event.button == 3:
            #right click
            print("Adding negative point")
            if self.input_point.shape[0] == 0:
                self.input_point = np.array([[x, y]])
                self.input_label = np.array([0])
            else:
                self.input_label = np.concatenate([self.input_label, [0]], axis=0)
                self.input_point = np.concatenate([self.input_point, [[x, y]]], axis=0)

        #predict
        
        masks, scores, logits = predictor.predict(
            point_coords=self.input_point,
            point_labels=self.input_label,
            multimask_output=True,)
        print(masks.shape, scores.shape, logits.shape)
        #get the best mask

        i = np.argmax(scores)
        mask = masks[i]
        score = scores[i]
        print(f"Best mask score: {score:.3f}")
        #clear the plot
        plt.cla()
        image =self.image.copy()
        plt.imshow(image)
        show_mask(mask, plt.gca())
        #get class name from classWidget
        class_name = self.classWidget.currentItem().text()

        self.this_mask[class_name] = mask
        show_points(self.input_point, self.input_label, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    sys.exit(app.exec_())