#class Neuron
import math, random

class Dendrite:
    def __init__(self,connectedNeuron) -> None:
        self.m_connectedNeuron = connectedNeuron
        self.m_weight = random.random() #0 # a random value?
        self.m_dWeight = 0 # back propagation weight?
    
    def fwdPropagation(self) -> float:
        return self.m_connectedNeuron.getOutput() * self.m_weight
    
    def bckPropagation(self,eta,gradient,alpha) -> float:
        self.updateWeights(eta, gradient, alpha)
        self.m_connectedNeuron.addError(self.m_weight * gradient)

    def updateWeights(self, eta, gradient, alpha) -> None:
        
        self.m_dWeight = eta * gradient * self.m_connectedNeuron.m_output + alpha * self.m_dWeight 
        self.m_weight += self.m_dWeight
       # print("UP",gradient, eta, alpha, self.m_weight, self.m_dWeight)

        

class Neuron:
    def __init__(self, neuronsInLayer = [], eta = .1, alpha = .01, activationFunction = "sigmoid") -> None:
    
        #neuron parameters
        self.m_eta = eta
        self.m_alpha = alpha
        self.m_stepActivation = .5
        self.m_output = 0
        self.m_error = 0
        self.m_gradient = 0
        self.m_dendrites = []

        #creating connections between neurons of layer n and n-1       
        for neuron in neuronsInLayer:
                self.m_dendrites.append(Dendrite(neuron))

        #selecting activation functions:
        if activationFunction == "sigmoid":
            self.activationFunction = self.sigmoid
        elif activationFunction == "ReLU":
            self.activationFunction = self.relu
        else:
            self.activationFunction = self.sigmoid #default
    
    def setOutput(self,val):
        self.m_output = val

    def sigmoid(self, val) -> float:
        return 1 / (1 + math.exp(-val))

    def relu(self,val) -> float:
        if val < 0: return 0
        else: return val 
        
    def dSigmoid(self, val) -> float:
        return val * (1 - val)

    def setError(self, val) -> None:    
        self.m_error = val
    
    def addError(self, val) -> None:
        self.m_error += val

    def setOutput(self, val) -> None:
        self.m_output = val

    def getOutput(self) -> float:
        return self.m_output

    def fwdPropagation(self) -> None:
        sum = 0
        for dendrite in self.m_dendrites:
            sum += dendrite.fwdPropagation()
        self.m_output = self.activationFunction(sum)
        #print(f"FWD: {sum} -> {self.m_output}")

    def setGradient(self) -> float:
        self.m_gradient = self.dSigmoid(self.m_output) * self.m_error

    def bckPropagation(self) -> float:
        self.setGradient()
        for dendrite in self.m_dendrites:
           # print(self.m_eta, self.m_gradient, self.m_alpha)
            dendrite.bckPropagation(self.m_eta, self.m_gradient, self.m_alpha)
        #print()
        self.setError(0)

    def printNeuron(self) -> None:
        for i, dendrite in enumerate(self.m_dendrites):
            print(f"W[{i}]: {dendrite.m_weight:.2f}",end=", ")

        # 
        # for i, dendrite in enumerate(self.m_dendrites):
        #     print(f"dW[{i}]: {dendrite.m_dWeight:.2f}",end=", ")
        print(f" Err: {self.m_error:.2f}, Grd: {self.m_gradient:.2f}, Out: {self.m_output:.2f}")
 

class MLP:
    def __init__(self, topology = []) -> None:
        self.m_layers = []
        for neuronsCount in topology:
            self.addLayer(neuronsCount)
    
    def addLayer(self,neuronsCount,activationFunction = "sigmoid") -> None:

        prevLayer = self.m_layers[-1] if self.m_layers else []
        currentLayer = []
        for i in range(neuronsCount):
            currentLayer.append(Neuron(prevLayer,activationFunction=activationFunction))

            #bias??
        self.m_layers.append(currentLayer)

    
    def setInputs(self, inputs) -> None:
        if len(self.m_layers[0]) != len(inputs): 
            print("Size of inputs data does not match to size of input layer!")
            return

        for i,n in zip(inputs, self.m_layers[0]):
            #print(f"{i:.2f}",end=", ")
            n.setOutput(i)  #not sure!
            #print()

    def fwdPropagation(self) -> float:
        # print("FWD:")
        # self.printNetwork()
        for layer in self.m_layers[1:]:
            for neuron in layer:
                neuron.fwdPropagation()
        # self.printNetwork()        
        return list(i.m_output for i in self.m_layers[-1])
    
    def bckPropagation(self, target = []) -> float:
        # print("BCK:")
        # self.printNetwork()
        for neuron, val in zip(self.m_layers[-1],target):
            neuron.setError(val - neuron.getOutput())

        for layer in self.m_layers[::-1]:
            for neuron in layer:
                neuron.bckPropagation()
        # self.printNetwork()

    def calcError(self, target = []):
        e = 0
        for val, out in zip(target,self.m_layers[-1]):
            e += (val - out.getOutput()) ** 2
        return (e / len(target)) ** .5


    def printNetwork(self) -> None:
        print("----------------------------------------------------------------------")
       
        for i,lr in enumerate(self.m_layers):
            print(f"Layer #{i+1}:")
            
            for j,n in enumerate(lr):
                print(f"- Neuron #{j+1}: ",end="")
                n.printNeuron()
                
        print("----------------------------------------------------------------------") 
