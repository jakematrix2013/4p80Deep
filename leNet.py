import keras
from keras import models, layers
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt
from tkinter import *
from tkinter.ttk import Scale
from tkinter import colorchooser, filedialog, messagebox
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

        ax.legend(['Train acc', 'Validation acc'], loc=0)
        ax.set_title('Training/Validation acc per Epoch')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('acc')

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
        self.drawer_btn = Button(self.root, text='Draw', bd=4, bg='green', relief=RIDGE, width=9, command=lambda col='black' : self.select_color(col))
        self.drawer_btn.place(x=5, y=295)

        # Erase Button
        self.eraser_btn = Button(self.root, text="Eraser", bd=4, bg='green', command=self.eraser, width=9, relief=RIDGE)
        self.eraser_btn.place(x=105, y=295)

        # Reset Button to clear the entire screen 
        self.clear_screen= Button(self.root, text="Clear Screen", bd=4, bg='green', command=lambda : self.background.delete('all'), width=9, relief=RIDGE)
        self.clear_screen.place(x=215, y=295)

        # Creating the pointer
        self.pointer_frame = LabelFrame(self.root, text='size', bd=5, bg='white', font=('arial', 15, 'bold'), relief=RIDGE)
        self.pointer_size = Scale(self.pointer_frame, orient=VERTICAL, from_=48, to=0, length=168)
        self.pointer_size.set(1)

        # Defining a background color for the Canvas 
        self.background = Canvas(self.root, bg='white', bd=5, relief=GROOVE, height=280, width=280)
        self.background.place(x=10, y=0)

        #Bind the background Canvas with mouse click
        self.background.bind("<B1-Motion>", self.paint)

    # Paint Function for Drawing the lines on Canvas
    def paint(self, event):
        x1, y1 = (event.x - 2), (event.y - 2)
        x2, y2 = (event.x + 2), (event.y + 2)

        self.background.create_oval(x1, y1, x2, y2, fill=self.pointer, outline=self.pointer, width=self.pointer_size.get())

    def select_color(self,col):
        self.pointer = col

    # Function for defining the eraser
    def eraser(self):
        self.pointer= self.erase

def main():
    # leNet = LeNet(10)

    root = Tk()
    p = Draw(root, 'Canvas')
    root.mainloop()


if __name__ == '__main__':
    main()
