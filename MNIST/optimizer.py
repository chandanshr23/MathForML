import numpy as np
class SGD:

    def __init__(self, params, lr=0.1):
        self.params = params
        self.lr = lr

    def step(self, grads):

      for name in self.params:
         self.params[name] -= self.lr * grads[name]

class Adam:
    def __init__(self, params, lr = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.t = 0

        self.m = {}

        self.v = {}

        for name, param in self.params.items():
            self.m[name] = np.zeros_like(param)
            self.v[name] = np.zeros_like(param)

    def step(self, grads):
        self.t += 1

        for name in self.params:

            self.m[name] = (
                self.beta1 * self.m[name] + (1 - self.beta1) * grads[name]
            )

            self.v[name] = ( 
                self.beta2 * self.v[name] + (1 - self.beta2) * (grads[name] ** 2)
            )

            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 -self.beta2 ** self.t)

            self.params[name] -= (
                self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            )