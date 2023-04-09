from random import random
import numpy as np


class Perceptron:
    def __init__(self,size) -> None:
        self.m_inputs = np.zeros(size)
        self.m_weights = np.random.rand(size)
        print(self.m_inputs)
        print(self.m_weights)

    def fit(self, X, y, epochs = 100):
       
        print(X.shape)
    
    def predict(self, X):
        pass




def main():

    from sklearn import datasets
    X, y = datasets.make_blobs(
        n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2
    )
    print(type(X))
    network = Perceptron(5)
    network.fit(X,y)

if __name__ == "__main__":
      main()