import keras
import numpy as np
from keras import models, layers
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt
from tkinter import *
import tkinter as tk
from tkinter.ttk import Scale
from tkinter import colorchooser, filedialog, messagebox
from PIL import Image, ImageDraw
import PIL.ImageGrab as ImageGrab

class LeNet:

    def __init__(self, epoch):
        (xTrain,yTrain), (xTest,yTest) = mnist.load_data()
        xTrain = self.inputNormalizer(xTrain)
        xTest = self.inputNormalizer(xTest)

        # Transform labels 
        yTrain = np_utils.to_categorical(yTrain,10)
        yTest = np_utils.to_categorical(yTest,10)

        # Reshape into 4D array
        xTrain = xTrain.reshape(xTrain.shape[0],28,28,1)
        xTest = xTest.reshape(xTest.shape[0],28,28,1)

        # Creating the model using keras
        model = models.Sequential()
        
        # Setup the Layers of the Model
        model.add(layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=(28, 28, 1), padding='same'))
        model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
        model.add(layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
        model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        model.add(layers.Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
        model.add(layers.Flatten())
        model.add(layers.Dense(84, activation='tanh'))
        model.add(layers.Dense(10, activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer='SGD', metrics=['accuracy'])

        # Trains the Model
        hist = model.fit(x=xTrain, y=yTrain, epochs=epoch, batch_size=128, validation_data=(xTest,yTest), verbose=1)

        # Evaluates the Model's accuracy
        testScore = model.evaluate(xTest,yTest)

        # Saves the Model for use
        model.save('./4p80Deep/Model/')

        # Display the accuracy of the Model
        self.displayPlots(hist,testScore)
    
    def inputNormalizer(self,input):
        input = input.astype('float32') #convert numeric type to float32
        input /= 255 #normalize to [0,1]
        
        return input

    def displayPlots(self,hist,testScore):
        print(f'Test loss: {testScore[0]}, accuracy: {testScore[1] * 100}')

        f, ax = plt.subplots()
    
        ax.plot([None] + hist.history['accuracy'], 'o-')
        ax.plot([None] + hist.history['val_accuracy'], 'x-')

        ax.legend(['Train accuracy', 'Validation accuracy'], loc=0)
        ax.set_title('Training/Validation acc per Epoch')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')

        plt.show()

class Draw():
    def __init__(self, root, title):
        self.root = root
        self.root.title(title)
        self.root.geometry('300x330')
        self.root.configure(background='white')
        self.root.resizable(0, 0)

        # variables for pointer and Eraser   
        self.pointer = "black"
        self.erase = "white"

        # Draw Button
        self.drawer_btn = Button(self.root, text='Draw', bd=4, bg='green', relief=RIDGE, width=7, command=lambda col='black' : self.select_Color(col))
        self.drawer_btn.place(x=5, y=295)

        # Erase Button
        self.eraser_btn = Button(self.root, text="Eraser", bd=4, bg='green', command=self.eraser, width=7, relief=RIDGE)
        self.eraser_btn.place(x=85, y=295)

        # Reset Button to clear the entire screen 
        self.clear_screen = Button(self.root, text="Clear", bd=4, bg='green', command=self.clearScreen, width=7, relief=RIDGE)
        self.clear_screen.place(x=160, y=295)

        # Button to save number
        self.save_number = Button(self.root, text='Predict', bd=4, bg='green', command=self.save_Number, width=7, relief=RIDGE)
        self.save_number.place(x=235, y=295)

        # Creating the pointer
        self.pointer_frame = LabelFrame(self.root, text='size', bd=5, bg='white', font=('arial', 15, 'bold'), relief=RIDGE)
        self.pointer_size = Scale(self.pointer_frame, orient=VERTICAL, from_=48, to=0, length=168)
        self.pointer_size.set(10)

        # Defining a background color for the Canvas 
        self.background = Canvas(self.root, bg='white', bd=5, relief=GROOVE, height=280, width=280)
        self.background.place(x=10, y=0)

        # Image for number drawn
        self.image = Image.new('L', (280, 280), 'white')
        self.draw = ImageDraw.Draw(self.image)

        #Bind the background Canvas with mouse click
        self.background.bind("<B1-Motion>", self.paint)

    # Paint Function for Drawing on the Canvas
    def paint(self, event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)

        # Draws on Canvas
        self.background.create_oval(x1, y1, x2, y2, fill=self.pointer, outline=self.pointer, width=self.pointer_size.get())
        # Draws on Image Canvas
        self.draw.ellipse((x1, y1, x2, y2), 'black')

    # Function to set pointer color to black
    def select_Color(self, col):
        self.pointer = col

    # Function for defining the eraser
    def eraser(self):
        self.pointer = self.erase

    def clearScreen(self):
        self.background.delete('all')
        self.draw.rectangle((0, 0, 280, 280), 'white')

    def save_Number(self):
        resizedNumber = self.image.resize((28, 28))

        print(resizedNumber)
        img_array = np.array(resizedNumber)

        print(img_array)

        img_array = img_array.reshape(28, 28)
        print(img_array)

        img_array = img_array.astype('float32')
        print(img_array)

        img_array = 255 - img_array
        print(img_array)

        img_array /= 255
        print(img_array)
        print(img_array.size)

def main():
    # leNet = LeNet(10)

    root = Tk()
    p = Draw(root, 'Number Canvas')
    root.mainloop()
    

if __name__ == '__main__':
    main()
