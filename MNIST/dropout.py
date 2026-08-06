import numpy as np


class Dropout:

    def __init__(self, p=0.5):
        self.p = p
        self.mask = None
        self.training = True

    def forward(self, x, training=True):
        self.training = training
        if training:

            self.mask = (
                np.random.rand(*x.shape)
                > self.p
            )

            out = (
                x * self.mask
            ) / (1 - self.p)

            return out

        else:
            return x



    def backward(self, dout):

        if self.mask is None:
            return dout

        dx = (
            dout * self.mask
        ) / (1 - self.p)

        return dx