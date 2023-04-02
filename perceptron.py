from random import random


class Perceptron:
    def __init__(self,size) -> None:
        self.m_inputs = [0]*size
        self.m_weights = random()
        print(self.m_inputs)

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