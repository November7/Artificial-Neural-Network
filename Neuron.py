#class Layer ver 2.0

import math, random

class Dendrite:
    def __init__(self,connectedNeuron) -> None:
        self.m_connectedNeuron = connectedNeuron
        self.m_weight = random.random()
        self.m_dWeight = 0

class Neuron:
    def __init__(self,n):
        self.m_dendriteCount = n
        pass

class Input(Neuron):
    pass

class Relu(Neuron):
    pass

class Sigmoid(Neuron):
    pass

class SoftMax(Neuron):
    pass

class Layer:
    pass

class Network:
    def addLayer(self,layerClass,neuronCount=1):
        if not issubclass(layerClass,Layer): return
        
        self.m_neurons = [layerClass] * neuronCount

        




net = Network()
net.addLayer(Input,5)