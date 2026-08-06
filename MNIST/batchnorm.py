import numpy as np

class BatchNorm:

    def __init__(self, dim, momentum=0.9):

        self.gamma = np.ones((1, dim))
        self.beta = np.zeros((1, dim))

        # running stats, updated during training, used (frozen) at test time
        self.running_mean = np.zeros((1, dim))
        self.running_var = np.ones((1, dim))
        self.momentum = momentum

        self.cache = None

    def forward(self, x, training=True):

        eps = 1e-5

        if training:
            mean = np.mean(x, axis=0, keepdims=True)
            var = np.var(x, axis=0, keepdims=True)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

            x_hat = (x - mean) / np.sqrt(var + eps)
            self.cache = (x, x_hat, mean, var, eps)
        else:
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + eps)
            self.cache = (x, x_hat, self.running_mean, self.running_var, eps)

        out = self.gamma * x_hat + self.beta
        return out

    def backward(self, dout):

        if self.cache is None:
            raise RuntimeError("BatchNorm cache is empty; call forward before backward")

        x, x_hat, mean, var, eps = self.cache

        N = x.shape[0]

        dgamma = np.sum(dout * x_hat, axis=0, keepdims=True)
        dbeta = np.sum(dout, axis=0, keepdims=True)

        dx_hat = dout * self.gamma

        dvar = np.sum(
            dx_hat * (x - mean) * -0.5 * (var + eps) ** (-1.5),
            axis=0, keepdims=True
        )

        dmean = np.sum(
            dx_hat * -1 / np.sqrt(var + eps), axis=0, keepdims=True
        )

        dx = (dx_hat / np.sqrt(var + eps) + dvar * 2 * (x - mean) / N + dmean / N)

        return dx, dgamma, dbeta