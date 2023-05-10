import keras
from keras import models, layers
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt

class LeNet:
    def __init__(self, epoch, modelFilePath):
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
        model.save(modelFilePath)

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
