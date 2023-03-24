#class Layer ver 2.0

import math, random

class Dendrite:
    def __init__(self,connectedNeuron) -> None:
        self.m_connectedNeuron = connectedNeuron
        self.m_weight = random.random()
        self.m_dWeight = 0

class Layer:
    def __init__(self,n):
        self.m_NeuronCount = n
        pass


class Input(Layer):
    pass

class Relu(Layer):
    pass

class Sigmoid(Layer):
    pass

class SoftMax(Layer):
    pass

class Network:
    def addLayer(self,layerClass,neuronCount=1):
        if not issubclass(layerClass,Layer): return
        
        self.m_neurons = [layerClass] * neuronCount

        




net = Network()
net.addLayer(Input,5)