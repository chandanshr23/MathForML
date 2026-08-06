import numpy as np
from model import MLP 

X = np.random.randn(5, 784)

model = MLP(input_size=784, hidden_size=128, output_size=10)

logits, cache = model.forward(X, training=False)

print(logits.shape)

dlogits = np.random.randn(5, 10)

grads = model.backward(dlogits, cache)

for key, value in grads.items():
    print(key, value.shape)