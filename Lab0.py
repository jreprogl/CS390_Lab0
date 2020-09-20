
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random

from math import e
import pdb

# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

# Disable some troublesome logging.
#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
#ALGORITHM = "guesser"
ALGORITHM = "custom_net"
#ALGORITHM = "tf_net"





class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate = 0.1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)

    # Activation function.
    def __sigmoid(self, x):
        exponent = -1 * x
        exponent = np.squeeze(np.asarray(exponent))
        res = 1 / (1 + e**exponent)
        return res

    # Activation prime function.
    def __sigmoidDerivative(self, x):
        exponent = -1 * x
        res = (e**exponent) / ((1 + e**exponent)**2)
        return res

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]
    
    def __backProp(self, inputs, layer1, yTrain, yTest):
#        pdb.set_trace()
        inputs = np.matrix(inputs)
        l2e = yTest - yTrain
        l2d = l2e * self.__sigmoidDerivative(yTrain)
        l1e = np.dot(l2d, np.matrix(self.W2).T)
        layer1deriv = self.__sigmoidDerivative(layer1)
        l1e = np.squeeze(np.asarray(l1e))
        layer1deriv = np.squeeze(np.asarray(layer1deriv))
        l1d = l1e * layer1deriv
        l1d = np.matrix(l1d)
        l1a = np.dot(inputs.T, l1d) * self.lr
        l2a = np.dot(np.matrix(layer1).T, np.matrix(l2d)) * self.lr
        self.W1 = self.W1 + l1a
        self.W2 = self.W2 + l2a

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs = 1, minibatches = True, mbs = 100):
        batches = self.__batchGenerator(xVals, mbs)
        x = 0
        batchNum = 1
 #       pdb.set_trace()
        for i in range(epochs):
            for batch in batches:
                print("Batch " + str(batchNum) + "/" + str(len(xVals) / mbs))
                for image in batch:
                    image = image.flatten()
                    for i in range(0, epochs):
                        layer1, layer2 = self.__forward(image)
                        self.__backProp(image, layer1, layer2, yVals[x]) 
                    x += 1
                batchNum += 1
                


    # Forward pass.
    def __forward(self, input):
        layer1 = self.__sigmoid(np.dot(input, self.W1))
        layer2 = self.__sigmoid(np.dot(layer1, self.W2))
        return layer1, layer2

    # Predict.
    def predict(self, xVals):
        _, layer2 = self.__forward(xVals)
        return layer2



# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)



#=========================<Pipeline Functions>==================================

def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))



def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw            #TODO: Add range reduction here (0-255 ==> 0.0-1.0).
    
    # Range Reduction
    xTrain = xTrain / 255
    xTest = xTest / 255
    
#    xTrain = xTrain[0:30000] #Change input size for testing 
#    yTrain = yTrain[0:30000]


    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
   # pdb.set_trace()
    return ((xTrain, yTrainP), (xTest, yTestP))

def trainKeras(model, x, y, eps=6):
    x = x.reshape(60000, 784)
    model.fit(x, y, epochs=eps, batch_size=100)
    return model


#https://towardsdatascience.com/building-our-first-neural-network-in-keras-bdc8abbc17f5
def buildKeras(xTrain, yTrain):
    model = keras.Sequential()
    lossType = keras.losses.categorical_crossentropy
    opt = tf.keras.optimizers.Adam()
    inputShape = (IMAGE_SIZE,)
    model.add(keras.layers.Dense(IMAGE_SIZE, input_shape=inputShape, activation=tf.nn.relu))  #was 2 layer, relu then sigmoid, 100 batch, 100 eps
    model.add(keras.layers.Dense(NUM_CLASSES, activation=tf.nn.sigmoid))
    model.compile(optimizer = opt, loss = lossType)
    model = trainKeras(model, xTrain, yTrain, 100)
    return model


def trainModel(data):
    xTrain, yTrain = data
    
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        print("Building and training Custom_NN.")
        model = NeuralNetwork_2Layer(IMAGE_SIZE, NUM_CLASSES, IMAGE_SIZE, .03)
        model.train(xTrain, yTrain)
        return model
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")                         #TODO: Write code to build and train your keras neural net.
        model = buildKeras(xTrain, yTrain)
        return model
    else:
        raise ValueError("Algorithm not recognized.")



def processPrediction(pred):
    index = 0
    highVal = -100
    for i in range(0, len(pred)):
        if (pred[i] > highVal):
            index = i
            highVal = pred[index]
        pred[i] = 0
    pred[index] = 1
    return pred
    

def runKeras(model, x):
    x = x.reshape(len(x), IMAGE_SIZE)
    preds = model.predict(x)
    processedPreds = []
    for pred in preds:
        processedPreds.append(processPrediction(pred))
    return np.array(processedPreds)

def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        print("Testing Custom_NN.")
        predictions = []
        for image in data:                               #TODO: Write code to run your custon neural net.
            image = image.flatten()
            pred = model.predict(image)
            pred = processPrediction(pred)
            predictions.append(pred)
        return np.array(predictions)
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")                           #TODO: Write code to run your keras neural net.
        preds = runKeras(model, data)
        return preds
    else:
        raise ValueError("Algorithm not recognized.")



def evalResults(data, preds):   #TODO: Add F1 score confusion matrix here.
    xTest, yTest = data
    acc = 0
    confusionMatrix = np.zeros((11,11), int)
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
        predIndex = preds[i].tolist().index(1)
        actIndex = yTest[i].tolist().index(1)
        confusionMatrix[predIndex][actIndex] += 1
    accuracy = acc / preds.shape[0]
    # Add totals to the confustion matrix
    # Get right side totals
    for i in range(10):
        sum = 0
        for x in range(10):
            sum += confusionMatrix[i][x]
        confusionMatrix[i][10] = sum
    # Get bottome totals
    for i in range(10):
        sum = 0
        for x in range(10):
            sum += confusionMatrix[x][i]
        confusionMatrix[10][i] = sum
    #Get Total
    sum = 0
    for i in range(10):
        sum += confusionMatrix[i][10]
    confusionMatrix[10][10] = sum
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()
    print("Confusion Matix:\n")
    print(confusionMatrix)




#=========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)



if __name__ == '__main__':
    main()
