from keras.datasets import mnist
from keras.utils import np_utils

class LeNet:
    

    def __init__(self):
        (xTrain,yTrain), (xTest,yTest) = mnist.load_data()
        xTrain = self.inputNormalizer(xTrain)
        xTest = self.inputNormalizer(xTest)

        #Transform labels 
        yTrain = np_utils.to_categorical(yTrain,10)
        yTest = np_utils.to_categorical(yTest,10)

        #Reshape into 4D array
        xTrain = xTrain.reshape(xTrain.shape[0],28,28,1)
        xTest = xTest.reshape(xTest.shape[0],28,28,1)
    
    def inputNormalizer(self,input):
        input = input.astype('float32') #convert numeric type to float32
        input /= 255 #normalize to [0,1]
        return input

    


net = LeNet()
