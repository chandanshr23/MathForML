import numpy as np
from batchnorm import BatchNorm

np.random.seed(0)

x = np.random.randn(4, 3)

bn =BatchNorm(3)

out = bn.forward(x)
loss = np.sum(out)

dout = np.ones_like(out)

dx, dgamma, dbeta = bn.backward(dout)

def numerical_gradient(x, bn, h=1e-5):

    grad = np.zeros_like(x)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):

            old = x[i,j]

            x[i,j] = old + h
            out1 = bn.forward(x)
            loss1 = np.sum(out1)

            x[i,j] = old -h
            out2 = bn.forward(x)
            loss2 = np.sum(out2)

            x[i,j] = old
            grad[i,j] = (loss1 - loss2)/(2*h)

    return grad

num_dx = numerical_gradient(
    x,
    bn
)


difference = np.linalg.norm(dx-num_dx) / (
    np.linalg.norm(dx)
    +
    np.linalg.norm(num_dx)
)


print(
    "BatchNorm gradient difference:",
    difference
)
