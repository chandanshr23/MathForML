import numpy as np
from layers import linear_forward, linear_backward
from activations import relu_forward, relu_backward
from batchnorm import BatchNorm
from dropout import Dropout

class MLP:
    def __init__(self, input_size=784, hidden_size=128, output_size=10):

        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))

        self.bn1 = BatchNorm(hidden_size)

        self.dropout = Dropout(p=0.5)

        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, X, training=True):
        z1, linear1_cache = linear_forward(X, self.W1, self.b1)

        bn_out = self.bn1.forward(z1, training=training)

        a1, relu_cache = relu_forward(bn_out)

        dropout_out = self.dropout.forward(a1, training=training)

        logits, linear2_cache = linear_forward(dropout_out, self.W2, self.b2)

        cache = (linear1_cache, relu_cache, linear2_cache)

        return logits, cache

    def backward(self, dlogits, cache):
        linear1_cache, relu_cache, linear2_cache = cache

        ddropout, dW2, db2 = linear_backward(dlogits, linear2_cache)
        da1 = self.dropout.backward(ddropout)
        dbn = relu_backward(da1, relu_cache)
        dz1, dgamma, dbeta = self.bn1.backward(dbn)
        dX, dW1, db1 = linear_backward(dz1, linear1_cache)

        gradients = {"W1": dW1, "b1": db1, "gamma": dgamma, "beta": dbeta, "W2": dW2, "b2": db2}
        return gradients

    def parameters(self):
        return {
            "W1": self.W1, "b1": self.b1,
            "gamma": self.bn1.gamma, "beta": self.bn1.beta,
            "W2": self.W2, "b2": self.b2
        }

    def save(self, path):
        """Save everything needed for inference: weights + BatchNorm running stats."""
        np.savez(
            path,
            W1=self.W1, b1=self.b1,
            gamma=self.bn1.gamma, beta=self.bn1.beta,
            running_mean=self.bn1.running_mean, running_var=self.bn1.running_var,
            W2=self.W2, b2=self.b2
        )

    def load(self, path):
        data = np.load(path)
        self.W1, self.b1 = data["W1"], data["b1"]
        self.bn1.gamma, self.bn1.beta = data["gamma"], data["beta"]
        self.bn1.running_mean, self.bn1.running_var = data["running_mean"], data["running_var"]
        self.W2, self.b2 = data["W2"], data["b2"]