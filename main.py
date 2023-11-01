# Group Members: Amogha Pokkulandra, Keshav Santhanam

import sys
import pandas as pd
import numpy as np

dtrain = pd.read_csv(sys.argv[1], delim_whitespace=True)
dtest = pd.read_csv(sys.argv[2], delim_whitespace=True)
testIn = dtest.iloc[:, :-1]
testOut = dtest.iloc[:, -1]


HLC = int(sys.argv[3]) # Hidden Layer Count
NPL = int(sys.argv[4]) # Nodes Per Layer
LR = float(sys.argv[5]) # Learning Rate
ITER = int(sys.argv[6]) # Iterations

def sigmoid(x): # classification function
    return 1 / (1 + np.exp(-1 * x))

class Node:
    def __init__(self, numPrevLayer):
        self.weights = [0]*numPrevLayer # Weights coming in
        self.error = 0 # Error
        self.value = 0 # Output calculated

    def toString(self):
        return str(self.value) # for debugging

    def computeError(self, idx, nextLayer):
        expectedErrorNext = 0
        if (len(nextLayer) == 1):
            expectedErrorNext = nextLayer[0].weights[idx] * nextLayer[0].error
        else:
            for i in range(len(nextLayer) - 1):
                expectedErrorNext += nextLayer[i].weights[idx] * nextLayer[i].error
        self.error = expectedErrorNext * (1 - self.value) * self.value
    def newOutput(self, prevLayer):
        self.value = 0
        for i in range(len(prevLayer)-1): # for all incoming nodes
            self.value += prevLayer[i].value * self.weights[i] # basically a dot product
        self.value += self.weights[-1] # bias weight
        self.value = sigmoid(self.value) # Call sigmoid function

    def modWeights(self, prevLayer):
        for w in range(len(self.weights)-1): # for all incoming weights
            self.weights[w] += LR * self.error * prevLayer[w].value # modifying weight based on the formula
        self.weights[-1] += LR * self.error # bias weight modification since we know the value is 1
class NeuralNet:
    def __init__(self, trainData):
        self.data = trainData
        self.trainIn = trainData.iloc[:, :-1] # just inputs
        self.trainOut = trainData.iloc[:, -1] # just outputs (class)
        self.hiddenLayers = [] # INITIALIZED WITH INPUT
        self.hiddenLayers.append([Node(0) for r in range(len(self.trainIn.columns) + 1)]) # input layer is the hiddenLayers[0]
        self.hiddenLayers[0][len(self.trainIn.columns)].value = 1 # adding in bias
        for h in range(HLC):
            arr = []
            if h == 0:
                arr = [Node(len(self.trainIn.columns)+1) for r in range(NPL)] # First hidden layer stuff
            else:
                arr = [Node(NPL+1) for r in range(NPL)] # Every other hidden layer
            arr.append(Node(0))  # Bias
            arr[NPL].value = 1 # setting Bias
            self.hiddenLayers.append(arr) # Add onto matrix

        if (HLC != 0):
            self.result = Node(NPL+1)
        else:
            self.result = Node(len(self.trainIn.columns) + 1)# not part of self.hiddenLayers, but simply our last Output

    # def toString(self): # printing out the network values
    #     print("Length of inputs is... " + str(len(self.trainIn.columns)))
    #     print("Number of Hidden Layers (including inputs) is..." + str(len(self.hiddenLayers)))
    #     print("INPUT LAYER VALUES")
    #     for i in self.hiddenLayers[0]:
    #         print(i.toString(), ":", end='')
    #         print(i.weights)
    #     print()
    #     print("HIDDEN LAYER VALUES")
    #     for r in self.hiddenLayers[1:]:
    #         for c in r:
    #             print(c.toString(), ":", end='')
    #             print(c.weights)
    #         print()
    #     print()
    #     print(self.result.toString())
    #     print(self.result.weights)
    def forPass(self, start):
        for s in range(len(self.hiddenLayers[0])-1): # reading in the training instance as input Layer
            self.hiddenLayers[0][s].value = start.iloc[s]
        prev = self.hiddenLayers[0]

        for i in range(1, len(self.hiddenLayers)): # forward passing values
            for n in range(0, len(self.hiddenLayers[i])-1):
                self.hiddenLayers[i][n].newOutput(prev) # update this value w/ prior nodes and their weights
            prev = self.hiddenLayers[i] # now we set curr to prev

        self.result.newOutput(prev) # atp prev is the last hidden layer, so compute final result
    def backPropAndUpdate(self, desired):
        self.result.error = (1 - self.result.value) * self.result.value * (desired - self.result.value) # computing output node error
        next = []
        next.append(self.result)  # next layer's node

        for i in range(len(self.hiddenLayers)-1, 0, -1): # computing errors for each layer
            for n in range(0, len(self.hiddenLayers[i])-1):
                self.hiddenLayers[i][n].computeError(n, next) # passing in n, so we know which of next layers' node weights to use
            next = self.hiddenLayers[i]

        prev = self.hiddenLayers[0] # finally updating weights
        for i in range(1, len(self.hiddenLayers)):
            for n in range(0, len(self.hiddenLayers[i])-1): # for every node except the last
                self.hiddenLayers[i][n].modWeights(prev)
            prev = self.hiddenLayers[i]


        self.result.modWeights(prev)
    def ASE(self, inputs, outputs):
        size = len(outputs)
        sumSqError = 0
        for row in range(len(inputs)):
            instance = inputs.iloc[row]
            self.forPass(instance)
            sumSqError += (self.result.value - outputs.iloc[row]) ** 2

        avgSqError = sumSqError/size
        return str("{:.4f}".format(round(avgSqError, 4)))

network = NeuralNet(dtrain)
size = len(dtrain)
for i in range(ITER):
    print("In iteration " + str(i+1) + ":")
    starting = network.trainIn.iloc[i%size]
    network.forPass(starting)
    print("Forward pass output: " + str("{:.4f}".format(round(network.result.value, 4))))
    network.backPropAndUpdate(network.trainOut.iloc[i%size])
    print("Average squared error on training set (" + str(len(network.trainOut)) +
          " instances): " + network.ASE(network.trainIn, network.trainOut))
    print("Average squared error on test set (" + str(len(testOut)) +
          " instances): " + network.ASE(testIn, testOut))
