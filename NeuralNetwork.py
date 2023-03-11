#class Neuron
import math, random

class Dendrite:
    def __init__(self,connectedNeuron) -> None:
        self.m_connectedNeuron = connectedNeuron
        self.m_weight = random.random() #0 # a random value?
        self.m_dweight = 0 # back propagation weight?
    
    def forwardPropagation(self) -> float:
        return self.m_connectedNeuron.getOutput() * self.m_weight
    
    def backwardPropagation(self) -> float:
        pass
        

class Neuron:
    def __init__(self, layer, eta = .001, alpha = .01, activationFunction = "sigmoid") -> None:
    
        #neuron parameters
        self.m_eta = eta
        self.m_alpha = alpha
        self.m_stepActivation = .5
        self.m_output = 0
        self.m_dentrites = []

        #creating connections between neurons of layer n and n-1        
        if layer!=None:
            for neuron in layer:
                self.m_dentrites.append(Dendrite(neuron))

        #selecting activation functions:
        if activationFunction == "sigmoid":
            self.activationFunction = self.sigmoid
        elif activationFunction == "step":
            self.activationFunction = self.step
        elif activationFunction == "ReLU":
            pass
        else:
            self.activationFunction = self.sigmoid #default
    
    
    def sigmoid(self, val) -> float:
        return 1 / (1 + math.exp(-val))

    def step(self,val) -> int:
        if val > self.m_stepActivation: return 1
        else: return 0

    def setOutput(self, val) -> None:
        self.m_output = val

    def getOutput(self) -> float:
        return 0

    def forwardPropagation(self) -> float:
        sum = 0
        for dentrite in self.m_dentrites:
            sum += dentrite.forwardPropagation()
        self.m_output = self.activationFunction(sum)

    def backwardPropagation(self) -> float:
        pass

    def Print(self) -> None:
        for dentrite in self.m_dentrites:
            print(dentrite.m_weight)
 

        
class MLP:
    def __init__(self, topology) -> None:
        self.m_layers = []
        for neuronsCount in topology:
            currentLayer = []
            if len(self.m_layers) == 0:
                prevLayer = None
            else:
                prevLayer = self.m_layers[-1]
            
            for i in range(neuronsCount):
                currentLayer.append(Neuron(prevLayer))

            #bias??

            self.m_layers.append(currentLayer)

    def Print(self) -> None:
        for neurons in self.m_layers:
            for neuron in neurons:
                neuron.Print()
            
          
            

network = MLP([2,3,1])

network.Print()
    

    

