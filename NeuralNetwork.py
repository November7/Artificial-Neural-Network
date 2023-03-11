#class Neuron
import math, random

class Dendrite:
    def __init__(self,connectedNeuron) -> None:
        self.m_connectedNeuron = connectedNeuron
        self.m_weight = random.random() #0 # a random value?
        self.m_dweight = 0 # back propagation weight?
    
    def fwdPropagation(self) -> float:
        return self.m_connectedNeuron.getOutput() * self.m_weight
    
    def bckPropagation(self,eta,gradient,alpha) -> float:
        self.updateWeights(eta, gradient, alpha)
        self.m_connectedNeuron.addError(self.m_weight * gradient)

    def updateWeights(self, eta, gradient, alpha) -> None:
        self.m_dweight = eta * gradient * self.m_connectedNeuron.m_output + alpha * self.m_dweight 
        self.m_weight += self.m_dweight

        

class Neuron:
    def __init__(self, neuronsInLayer = [], eta = .001, alpha = .01, activationFunction = "sigmoid") -> None:
    
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
        elif activationFunction == "step":
            self.activationFunction = self.step
        elif activationFunction == "ReLU":
            pass
        else:
            self.activationFunction = self.sigmoid #default
    
    def setOutput(self,val):
        self.m_output = val

    def sigmoid(self, val) -> float:
        return 1 / (1 + math.exp(-val))

    def step(self,val) -> int: #float??
        if val > self.m_stepActivation: return 1
        else: return 0

    def relu(self,val) -> float:
        pass


    def dSigmoid(self, val) -> float:
        return val * (1 - val)

    def setError(self, val) -> None:    
        print(f"Error przed: {self.m_error} i po: {val}")    
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
        for dendrite in self.m_dendrites:
            dendrite.bckPropagation(self.m_eta, self.m_gradient, self.m_alpha)
        self.setError(0)

    def printNeuron(self) -> None:
        for dendrite in self.m_dendrites:
            print(f"{dendrite.m_weight:.3f}",end=", ")
        print(f"Error: {self.m_error}, Gradient: {self.m_gradient}, Output: {self.m_output}")
 

class MLP:
    def __init__(self, topology) -> None:
        self.m_layers = []
        for neuronsCount in topology:
            currentLayer = []
            #add last layer or none if its first layer
            prevLayer = self.m_layers[-1] if self.m_layers else []
            #add neurons to layer
            for i in range(neuronsCount):
                currentLayer.append(Neuron(prevLayer))

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
        for layer in self.m_layers[1:]:
            for neuron in layer:
                neuron.fwdPropagation()
        return list(i.m_output for i in self.m_layers[-1])
    
    def bckPropagation(self, target = []) -> float:
        for neuron, val in zip(self.m_layers[-1],target):
            print("bkg")
            neuron.setError(val - neuron.getOutput())

        for layer in self.m_layers:
            for neuron in layer:
                neuron.bckPropagation()

    def calcError(self, target = []):
        e = 0
        for val, out in zip(target,self.m_layers[-1]):
            e += (val - out.getOutput()) ** 2
        return (e / len(target)) ** .5


    def printNetwork(self) -> None:
        i = 1
        for neurons in self.m_layers:
            print(f"Layer-{i} [{len(neurons)} neuron(s)]:")
            j = 1
            for neuron in neurons:
                print(f" - Neuron #{j} (input weights): ",end="")
                neuron.printNeuron()
                print()
                j+=1
            print()
            i+=1
            
          
            

network = MLP([2,5,1])

dane_we = [0,1]
dane_wy = [1]

for i in range(1,3):
    print(f"Epoka #{i}")
    print("-------------------------------")
    network.setInputs(dane_we)
    result = network.fwdPropagation()
   # print(f"Wynik: {result}")
    network.bckPropagation(dane_wy)
    network.printNetwork()


    

    


