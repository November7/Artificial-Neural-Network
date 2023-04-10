import numpy as np

class Perceptron:
    def __init__(self,size) -> None:
        self.m_inputs = None 
        self.m_weights = None
        self.m_bias = None
        self.m_features = 0

    def stepFunction(self,x):
        return np.where(x > 0 , 1, 0)

    def fit(self, X, y, epochs = 100, alpha = .01):
        samples, self.m_features = X.shape
        print(samples,self.m_features)
        self.m_weights = np.random.rand(self.m_features)
        self.m_bias = np.random.rand()

        yn = np.where(y > 0 , 1, 0)

        for ep in range(epochs):
            for i, x in enumerate(X):  

                yd = self.predict(x) 
                w = alpha * (y[i] - yd)
                self.m_bias += w
                self.m_weights += w*x


        print(self.m_weights, self.m_bias)

        

    def predict(self, X):
        
        return self.stepFunction(np.dot(X, self.m_weights) + self.m_bias)





def main():
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    X, y = datasets.make_blobs(
        n_samples=200, n_features=2, centers=2, cluster_std=1.05, random_state=2
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=170
    )

    network = Perceptron(5)
    network.fit(X_train,y_train)

    predictions = network.predict(X_test)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker=".", c=y_train)

    x0_1 = np.amin(X_train[:, 0])
    x0_2 = np.amax(X_train[:, 0])

    x1_1 = (-network.m_weights[0] * x0_1 - network.m_bias) / network.m_weights[1]
    x1_2 = (-network.m_weights[0] * x0_2 - network.m_bias) / network.m_weights[1]
    ax.plot([x0_1, x0_2], [x1_1, x1_2])

    plt.show()

if __name__ == "__main__":
      main()