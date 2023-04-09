import numpy as np

class Perceptron:
    def __init__(self,size) -> None:
        self.m_inputs = None 
        self.m_weights = None
        self.m_bias = None
        self.m_features = 0

    def stepFunction(self,x):
        return 1 if x > 0 else 0

    def fit(self, X, y, epochs = 100, alpha = .01):
        samples, self.m_features = X.shape
        print(samples,self.m_features)
        self.m_weights = np.random.rand(self.m_features)
        self.m_bias = np.random.rand()

        yn = np.where(y > 0 , 1, 0)

        for ep in range(epochs):
            for i, x in enumerate(X):     
                yd = self.stepFunction(np.dot(x,self.m_weights) + self.m_bias) 
                w = alpha * (y[i] - yd)
                self.m_bias += w
                self.m_weights += w*x
      
        
        print(self.m_weights, self.m_bias)

        

    def predict(self, X):
        pass




def main():

    from sklearn import datasets
    X, y = datasets.make_blobs(
        n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2
    )
    network = Perceptron(5)
    network.fit(X,y)

if __name__ == "__main__":
      main()