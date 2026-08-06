import numpy as np

from model import MLP
from loss import (softmax_forward, cross_entropy_loss, softmax_cross_entropy_backward)
from optimizer import Adam

data = np.load('mnist_data.npz')

X_train = data["X_train"]
y_train = data["y_train"]
X_test = data["X_test"]
y_test = data["y_test"]

epochs = 20
batch_size = 64

model = MLP(input_size=784, hidden_size=128, output_size=10)
optimizer = Adam(model.parameters(), lr=0.001)


def accuracy(X, y):
    logits, _ = model.forward(X, training=False)  # dropout off, BatchNorm uses running stats
    predictions = np.argmax(logits, axis=1)
    return np.mean(predictions == y)


for epoch in range(epochs):
    indices = np.random.permutation(len(X_train))
    X_train = X_train[indices]
    y_train = y_train[indices]

    total_loss = 0

    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]

        logits, cache = model.forward(X_batch, training=True)
        probs = softmax_forward(logits)
        loss = cross_entropy_loss(probs, y_batch)
        total_loss += loss

        dlogits = softmax_cross_entropy_backward(probs, y_batch)
        grads = model.backward(dlogits, cache)
        optimizer.step(grads)

    train_acc = accuracy(X_train[:5000], y_train[:5000])
    test_acc = accuracy(X_test, y_test)

    print(
        f"Epoch {epoch+1}/{epochs} "
        f"Loss: {total_loss:.4f} "
        f"Train Acc: {train_acc:.4f} "
        f"Test Acc: {test_acc:.4f}"
    )

model.save("mnist_weights.npz")
print("\nSaved trained weights to mnist_weights.npz")