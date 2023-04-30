import numpy as np


class Perceptron:
    def __init__(self) -> None:
        self.m_inputs = None 
        self.m_features = 0
        self.m_bias = None
        self.trained = None #????

    @staticmethod
    def stepFunction(x):
        return np.where(x > 0, 1, 0) 
    
    def train(self, X, y, epochs = 100, alpha = .01, af = stepFunction) -> float:
        samples, self.m_features = X.shape
        self.m_weights = np.random.rand(self.m_features)
        self.m_bias = np.random.rand()
        for _ in range(epochs):
            for _x,_y in zip(X,y):  

                e = _y - self.predict(_x) 
                w = alpha * e
                self.m_bias += w
                self.m_weights += w*_x

    def predict(self,x,af = stepFunction) -> float:
        return af(np.dot(x, self.m_weights) + self.m_bias)

#-----------------------------------------------------------------------------------------


def main():
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    X, y = datasets.make_blobs(n_samples=200, n_features=2, centers=2, cluster_std=2.5)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3)

    network = Perceptron()
    network.train(X_train,y_train)

    predictions = network.predict(X_test)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    plt.scatter(X_train[:, 0], X_train[:, 1], marker=".", c = y_train, cmap="viridis")
    # plt.scatter(X_test[:, 0],X_test[:, 1], marker="*",c = -predictions, cmap="coolwarm")
    x0_1 = np.amin(X_train[:, 0])
    x0_2 = np.amax(X_train[:, 0])

    x1_1 = (-network.m_weights[0] * x0_1 - network.m_bias) / network.m_weights[1]
    x1_2 = (-network.m_weights[0] * x0_2 - network.m_bias) / network.m_weights[1]
    ax.plot([x0_1, x0_2], [x1_1, x1_2])


    plt.show()

if __name__ == "__main__":
    main()
