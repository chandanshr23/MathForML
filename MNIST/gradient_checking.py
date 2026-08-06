import numpy as np

from model import MLP
from loss import (softmax_forward, cross_entropy_loss)


def numerical_gradient(model, X, y, param_name, h=1e-5):
    param = model.parameters()[param_name]
    grad = np.zeros_like(param)

    iterator = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
    while not iterator.finished:
        idx = iterator.multi_index
        original_value = param[idx]

        param[idx] = original_value + h
        logits, _ = model.forward(X, training=True)
        probs = softmax_forward(logits)
        loss_plus = cross_entropy_loss(probs, y)

        param[idx] = original_value - h
        logits, _ = model.forward(X, training=True)
        probs = softmax_forward(logits)
        loss_minus = cross_entropy_loss(probs, y)

        param[idx] = original_value
        grad[idx] = (loss_plus - loss_minus) / (2 * h)
        iterator.iternext()

    return grad


np.random.seed(0)

X = np.random.randn(5, 784)
y = np.array([1, 2, 3, 4, 5])

model = MLP()
model.dropout.p = 0.0  # disable dropout's randomness for this check only —
                        # keeps training=True (so BatchNorm caches correctly for
                        # backward) while still being deterministic across the
                        # +eps / -eps forward calls

logits, cache = model.forward(X, training=True)
probs = softmax_forward(logits)

dlogits = probs.copy()
dlogits[np.arange(5), y] -= 1
dlogits /= 5

grads = model.backward(dlogits, cache)

for name in ["W1", "b1", "gamma", "beta", "W2", "b2"]:
    num_grad = numerical_gradient(model, X, y, name)
    diff = np.linalg.norm(grads[name] - num_grad) / (
        np.linalg.norm(grads[name]) + np.linalg.norm(num_grad) + 1e-8
    )
    status = "PASS" if diff < 1e-5 else "FAIL"
    print(f"{name}: gradient difference = {diff:.2e}  [{status}]")