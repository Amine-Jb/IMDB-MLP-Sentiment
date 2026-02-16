import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def binary_cross_entropy(y_true, y_pred):
    eps = 1e-9
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred)).mean()

class MLP:
    def __init__(self, input_size=2, hidden_size=8, lr=0.1):
        np.random.seed(42)

        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, 1) * 0.1
        self.b2 = np.zeros((1,1))

        self.lr = lr

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = np.tanh(self.Z1)

        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = sigmoid(self.Z2)

        return self.A2

    def backward(self, X, y):
        m = X.shape[0]

        dZ2 = (self.A2 - y) / m
        dW2 = self.A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (1 - np.tanh(self.Z1)**2)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y, epochs=50):
        losses = []
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = binary_cross_entropy(y, y_pred)
            losses.append(float(loss))
            self.backward(X, y)

            if epoch % 5 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

        return losses

    def predict(self, X):
        probs = self.forward(X)
        return (probs >= 0.5).astype(int)


