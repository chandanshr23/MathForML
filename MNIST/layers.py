import numpy as np

np.random.seed(0)

def linear_forward(x , W, b):
    y = x @ W + b
    cache = (x, W, b)
    return y, cache

def linear_backward(dy, cache):
    x, W, b = cache
    dx = dy @ W.T
    dW = x.T @ dy
    db = dy.sum(axis=0)
    return dx, dW, db
